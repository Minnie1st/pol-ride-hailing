#!/usr/bin/env python3
"""
03_extract_traces.py
====================
Extract PoL protocol events from SUMO simulations for Tartu experiments.

Each valid trip produces exactly four protocol events in the output CSV:
    1. driver_arrival          — vehicle first appears (pickup location, time t1)
    2. rider_pickup_presence   — rider at pickup      (same location, t1 + Δ1)
    3. trip_completion         — vehicle last seen     (dropoff location, time t2)
    4. rider_dropoff_presence  — rider at dropoff     (same location, t2 + Δ2)

Rider timing model (stated simplification):
    SUMO models vehicles only, not passengers. Rider presence times are sampled
    uniformly within the protocol window [0, DELTA_T_*]. This is documented in
    the thesis as a conservative simplification; real rider arrival times within
    the window do not affect F1 witness-coverage results (which depend only on
    location), but do affect F2 request-timing distributions.

Usage (default — F1 baseline):
    python3 03_extract_traces.py

Usage (F2 concurrency levels):
    python3 03_extract_traces.py --routes ../tartu/routes_c10.xml  --label c10
    python3 03_extract_traces.py --routes ../tartu/routes_c50.xml  --label c50
    python3 03_extract_traces.py --routes ../tartu/routes_c100.xml --label c100
    python3 03_extract_traces.py --routes ../tartu/routes_c200.xml --label c200

Output:
    ../results/trip_events_<label>.csv
    ../results/sim_summary_<label>.json

The summary JSON records peak concurrency, per-event counts, and concurrent
history for F2/F3.
"""

import os
import sys
import csv
import json
import math
import random
import argparse
import statistics
from collections import Counter, defaultdict

# ── SUMO path setup ────────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    sys.exit("ERROR: SUMO_HOME is not set. Please source your SUMO environment.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ── Protocol parameters ────────────────────────────────────────────────────────
# These must match the values used in protocol_simulator.py and 05_run_experiments.py
DELTA_T_PICKUP  = 120    # seconds: max rider-pickup delay after driver_arrival
DELTA_T_DROPOFF = 120    # seconds: max rider-dropoff delay after trip_completion

# ── Trip quality filters ───────────────────────────────────────────────────────
MIN_TRIP_DIST_M = 200    # minimum straight-line pickup→dropoff distance
MIN_TRIP_DUR_S  = 30     # minimum trip duration (avoids vehicles that barely move)

# ── Reproducibility ────────────────────────────────────────────────────────────
RIDER_TIMING_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PoL trip events from a SUMO simulation."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "tartu", "tartu.sumocfg"),
        help="Path to SUMO .sumocfg file",
    )
    parser.add_argument(
        "--routes",
        default=os.path.join(PROJECT_ROOT, "tartu", "routes_main.xml"),
        help=(
            "Route file to simulate. Defaults to tartu/routes_main.xml so that "
            "running Script 03 after Script 02 consumes the newly generated "
            "main workload."
        ),
    )
    parser.add_argument(
        "--label",
        default="main",
        help="Label appended to output filenames (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory for output files",
    )
    return parser.parse_args()


def resolve_project_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


# ──────────────────────────────────────────────────────────────────────────────
# SUMO simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(config: str, routes_override: str | None):
    """
    Run the SUMO simulation via TraCI.

    Returns
    -------
    first_seen : dict[str, tuple[float, float, float]]
        {vehicle_id: (lat, lon, time)} at first appearance — driver arrival.
    last_seen  : dict[str, tuple[float, float, float]]
        {vehicle_id: (lat, lon, time)} at last appearance — trip completion.
    peak_concurrent : int
        Maximum number of simultaneously active vehicles (for F2 summary).
    concurrent_history : list[tuple[float, int]]
        [(time, active_count), ...] sampled every second (for F2 load analysis).
    """
    cmd = ["sumo", "-c", config]
    if routes_override:
        cmd += ["--route-files", routes_override]

    traci.start(cmd)

    first_seen: dict[str, tuple[float, float, float]] = {}
    last_seen:  dict[str, tuple[float, float, float]] = {}
    peak_concurrent = 0
    concurrent_history: list[tuple[float, int]] = []

    step = 0
    last_report = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t = traci.simulation.getTime()

        active_ids = traci.vehicle.getIDList()
        n_active = len(active_ids)
        peak_concurrent = max(peak_concurrent, n_active)

        # Record concurrent count every second (integer time steps)
        if int(t) != last_report:
            concurrent_history.append((t, n_active))
            last_report = int(t)

        for veh_id in active_ids:
            # getPosition() returns SUMO internal Cartesian (x, y) — NOT lat/lon.
            x, y = traci.vehicle.getPosition(veh_id)
            # convertGeo(x, y) returns (lon, lat) in WGS-84.
            lon, lat = traci.simulation.convertGeo(x, y)

            if veh_id not in first_seen:
                first_seen[veh_id] = (lat, lon, t)

            # Overwrite every step — keeps the most recent position as dropoff.
            last_seen[veh_id] = (lat, lon, t)

        step += 1
        if step % 500 == 0:
            print(
                f"  [step {step:5d} | t={t:6.0f}s] "
                f"active={n_active:3d}  total_seen={len(first_seen):4d}"
            )

    traci.close()
    return first_seen, last_seen, peak_concurrent, concurrent_history


# ──────────────────────────────────────────────────────────────────────────────
# Event construction
# ──────────────────────────────────────────────────────────────────────────────

def build_events(
    first_seen: dict,
    last_seen:  dict,
    rng: random.Random,
) -> tuple[list[dict], dict, dict, dict]:
    """
    Construct the four PoL protocol events for each valid trip.

    Stated simplification:
        Rider presence times are sampled uniformly within [0, DELTA_T_*]
        after the corresponding driver event. This is documented in the thesis.
        It does not affect F1 coverage results (which depend on location only),
        but does affect the temporal distribution of requests in F2.

    Returns
    -------
    events   : list of event dicts (4 per valid trip)
    skipped  : dict with skip reason counts
    """
    events: list[dict] = []
    skipped = defaultdict(int)
    event_counts = Counter()
    trip_dists: list[float] = []
    trip_durs:  list[float] = []

    for veh_id in sorted(first_seen):
        pickup_lat,  pickup_lon,  t_arrival    = first_seen[veh_id]
        dropoff_lat, dropoff_lon, t_completion = last_seen[veh_id]

        # ── Quality filters ────────────────────────────────────────────────────
        dist = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        dur  = t_completion - t_arrival

        if dist < MIN_TRIP_DIST_M:
            skipped["dist_too_short"] += 1
            continue
        if dur < MIN_TRIP_DUR_S:
            skipped["dur_too_short"] += 1
            continue

        trip_dists.append(dist)
        trip_durs.append(dur)

        # ── Rider timing (uniform within protocol window) ──────────────────────
        t_rider_pickup  = t_arrival    + rng.uniform(0, DELTA_T_PICKUP)
        t_rider_dropoff = t_completion + rng.uniform(0, DELTA_T_DROPOFF)

        base = {
            "order_id":    veh_id,
            "trip_dist_m": round(dist, 1),
            "trip_dur_s":  round(dur,  1),
        }

        # Event 1 — driver_arrival
        events.append({
            **base,
            "event": "driver_arrival",
            "lat":   round(pickup_lat,  7),
            "lon":   round(pickup_lon,  7),
            "time":  round(t_arrival,   2),
        })
        event_counts["driver_arrival"] += 1

        # Event 2 — rider_pickup_presence
        events.append({
            **base,
            "event": "rider_pickup_presence",
            "lat":   round(pickup_lat,      7),
            "lon":   round(pickup_lon,      7),
            "time":  round(t_rider_pickup,  2),
        })
        event_counts["rider_pickup_presence"] += 1

        # Event 3 — trip_completion
        events.append({
            **base,
            "event": "trip_completion",
            "lat":   round(dropoff_lat,  7),
            "lon":   round(dropoff_lon,  7),
            "time":  round(t_completion, 2),
        })
        event_counts["trip_completion"] += 1

        # Event 4 — rider_dropoff_presence
        events.append({
            **base,
            "event": "rider_dropoff_presence",
            "lat":   round(dropoff_lat,      7),
            "lon":   round(dropoff_lon,      7),
            "time":  round(t_rider_dropoff,  2),
        })
        event_counts["rider_dropoff_presence"] += 1

    stats = {}
    if trip_dists:
        stats["trip_dist_m"] = {
            "min":  round(min(trip_dists), 1),
            "mean": round(statistics.mean(trip_dists), 1),
            "max":  round(max(trip_dists), 1),
        }
        stats["trip_dur_s"] = {
            "min":  round(min(trip_durs), 1),
            "mean": round(statistics.mean(trip_durs), 1),
            "max":  round(max(trip_durs), 1),
        }

    return events, dict(skipped), stats, dict(event_counts)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

FIELDS = ["order_id", "event", "lat", "lon", "time", "trip_dist_m", "trip_dur_s"]


def save_events(events: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(events)


def save_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.config = resolve_project_path(args.config)
    args.routes = resolve_project_path(args.routes)
    args.output_dir = resolve_project_path(args.output_dir)
    rng  = random.Random(RIDER_TIMING_SEED)

    events_path  = os.path.join(args.output_dir, f"trip_events_{args.label}.csv")
    summary_path = os.path.join(args.output_dir, f"sim_summary_{args.label}.json")

    print("=" * 54)
    print(" PoL Trace Extractor")
    print("=" * 54)
    print(f"  Config   : {args.config}")
    print(f"  Routes   : {args.routes or '(from config)'}")
    print(f"  Label    : {args.label}")
    print(f"  Events → : {events_path}")
    print(f"  Summary→ : {summary_path}")
    print()

    # ── Run simulation ─────────────────────────────────────────────────────────
    print("Running SUMO simulation...")
    first_seen, last_seen, peak_concurrent, concurrent_history = run_simulation(
        args.config, args.routes
    )
    n_vehicles = len(first_seen)
    print(f"  Done. Vehicles seen: {n_vehicles},  peak concurrent: {peak_concurrent}")
    print()

    # ── Build events ───────────────────────────────────────────────────────────
    print("Building protocol events...")
    events, skipped, trip_stats, event_counts = build_events(
        first_seen, last_seen, rng
    )
    n_valid = len(events) // 4

    print(f"  Valid trips     : {n_valid}")
    if skipped:
        for reason, count in skipped.items():
            print(f"  Skipped ({reason}): {count}")
    print(f"  Total events    : {len(events)}  ({n_valid} × 4)")

    # ── Save events CSV ────────────────────────────────────────────────────────
    save_events(events, events_path)
    print(f"\n  Saved: {events_path}")

    # ── Build and save summary JSON (used by F3 for real workload data) ────────
    sim_duration_s = (
        max(t for t, _ in concurrent_history) if concurrent_history else 3600
    )
    proofs_per_hour = round(len(events) / (sim_duration_s / 3600), 1)

    summary = {
        "label":            args.label,
        "config":           args.config,
        "routes_override":  args.routes,
        "simulation_params": {
            "DELTA_T_PICKUP_s":  DELTA_T_PICKUP,
            "DELTA_T_DROPOFF_s": DELTA_T_DROPOFF,
            "MIN_TRIP_DIST_m":   MIN_TRIP_DIST_M,
            "MIN_TRIP_DUR_s":    MIN_TRIP_DUR_S,
            "rider_timing_seed": RIDER_TIMING_SEED,
        },
        "results": {
            "total_vehicles":        n_vehicles,
            "valid_trips":           n_valid,
            "total_events":          len(events),
            "skipped":               skipped,
            "event_counts_by_type":  event_counts,
            "peak_concurrent_vehs":  peak_concurrent,
            "sim_duration_s":        round(sim_duration_s, 1),
            "proofs_per_hour":       proofs_per_hour,
            "trip_statistics":       trip_stats,
        },
        "concurrent_history": [
            {"time_s": round(t, 1), "active_vehicles": n}
            for t, n in concurrent_history
        ],
    }
    save_summary(summary, summary_path)
    print(f"  Saved: {summary_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 54)
    print(" Summary")
    print("=" * 54)
    if trip_stats:
        d = trip_stats["trip_dist_m"]
        t = trip_stats["trip_dur_s"]
        print(f"  Trip distance : min={d['min']:.0f}m  "
              f"mean={d['mean']:.0f}m  max={d['max']:.0f}m")
        print(f"  Trip duration : min={t['min']:.0f}s  "
              f"mean={t['mean']:.0f}s  max={t['max']:.0f}s")
    print(f"  Peak concurrent vehicles : {peak_concurrent}")
    print(f"  Event counts            : {event_counts}")
    print(f"  Proofs per hour (× 4 events): {proofs_per_hour}")
    print()
    print("  → Run 04_deploy_witnesses.py to place witnesses")
    print("  → Run 05_run_experiments.py for F1/F2/F3 analysis")


if __name__ == "__main__":
    main()
