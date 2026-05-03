"""
PoL Simulation Dashboard
========================
Run with Streamlit.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ── path setup so we can import protocol_simulator ──────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
sys.path.insert(0, SCRIPTS_DIR)

from protocol_simulator import (  # noqa: E402
    ProtocolParams,
    WitnessNode,
    generate_proof,
    haversine,
    verify_proof,
)

# ── shared protocol parameters (same as experiment suite) ───────────────────
PARAMS = ProtocolParams(
    d_ble=100.0,
    d_th=100.0,
    delta_t_max=0.035,
    tau_rssi=-90.0,
    n_w=4,
)

TARTU_CENTER = [58.378, 26.729]

# ── colors ───────────────────────────────────────────────────────────────────
C_TEAL = "#009B9E"
C_MAGENTA = "#C75DAB"
C_MINT = "#A7D3D4"
C_RED = "#E05555"
C_ORANGE = "#E07830"
WITNESS_PREVIEW_DEFAULT = 800
WITNESS_PREVIEW_MAX = 3000
WITNESS_PREVIEW_SEED = 42
OVERVIEW_RADIUS_M = 180.0


# ════════════════════════════════════════════════════════════════════════════
# Data loaders (cached)
# ════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_witnesses() -> list[WitnessNode]:
    path = os.path.join(RESULTS_DIR, "witnesses.csv")
    df = pd.read_csv(path)
    return [
        WitnessNode(id=row["id"], lat=float(row["lat"]), lon=float(row["lon"]))
        for _, row in df.iterrows()
    ]


@st.cache_data
def load_events() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "trip_events_main.csv")
    return pd.read_csv(path)


@st.cache_data
def load_f1_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "f1_psr_summary.csv"))


@st.cache_data
def load_s1_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "s1_gps_spoofing.csv"))


@st.cache_data
def load_s2_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "s2_replay_summary.csv"))


@st.cache_data
def load_s3_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "s3_relay_dt_tradeoff.csv"))


@st.cache_data
def load_f3_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "f3_batch_anchoring_tradeoff.csv"))


@st.cache_data
def load_sim_summary() -> dict:
    path = os.path.join(RESULTS_DIR, "sim_summary_main.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_witnesses_summary() -> dict:
    path = os.path.join(RESULTS_DIR, "witnesses_summary.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_s1_distance_options() -> list[int]:
    fallback = [5, 20, 50, 70, 80, 90, 95, 100, 105, 110, 120, 200, 500, 1000, 5000]
    try:
        s1 = load_s1_csv().copy()
        distances = sorted({int(float(v)) for v in s1["actual_distance_m"].dropna().tolist()})
        return distances or fallback
    except Exception:
        return fallback


# ════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ════════════════════════════════════════════════════════════════════════════


def offset_point(
    lat: float, lon: float, distance_m: float, bearing_rad: float
) -> tuple[float, float]:
    R = 6_371_000
    phi1, lam1 = math.radians(lat), math.radians(lon)
    ang = distance_m / R
    phi2 = math.asin(
        math.sin(phi1) * math.cos(ang)
        + math.cos(phi1) * math.sin(ang) * math.cos(bearing_rad)
    )
    lam2 = lam1 + math.atan2(
        math.sin(bearing_rad) * math.sin(ang) * math.cos(phi1),
        math.cos(ang) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), math.degrees(lam2)


def nearby_witnesses(
    lat: float, lon: float, witnesses: list[WitnessNode], radius_m: float
) -> list[WitnessNode]:
    return [w for w in witnesses if haversine(lat, lon, w.lat, w.lon) <= radius_m]


# ════════════════════════════════════════════════════════════════════════════
# Page 0 — Data Pipeline
# ════════════════════════════════════════════════════════════════════════════


def _load_all_sim_summaries() -> dict:
    """Load sim_summary for all 5 concurrency labels."""
    labels = ["main", "c10", "c50", "c100", "c200"]
    result = {}
    for label in labels:
        path = os.path.join(RESULTS_DIR, f"sim_summary_{label}.json")
        if os.path.exists(path):
            with open(path) as f:
                result[label] = json.load(f)
    return result


def page_pipeline() -> None:
    st.title("Simulation Data Pipeline")
    st.markdown(
        "Every number in this application comes from a four-step reproducible pipeline. "
        "This page explains exactly what each script does and shows the actual output data."
    )

    sim = load_sim_summary()
    wsm = load_witnesses_summary()
    events_df = load_events()
    sim_r = sim["results"]
    sim_p = sim["simulation_params"]
    all_sims = _load_all_sim_summaries()

    # ── Overview flow ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        ```
        tartu.net.xml  (OpenStreetMap road network, imported into SUMO)
              │
              ▼
        02_generate_trips.sh
              │  randomTrips.py -p 7 --validate --random-depart --seed 4201
              │  → trips_main.xml   (OD pairs, ~{round(3600 / 7)} entries)
              │  → routes_main.xml  (routed paths, after discarding unroutable ODs)
              │
              │  (also generates routes_c10/c50/c100/c200.xml for F2)
              ▼
        03_extract_traces.py
              │  TraCI controls SUMO step-by-step (headless, 1 s per step)
              │  Loop until getMinExpectedNumber() == 0  (all vehicles finished)
              │  first_seen[veh] = position at step vehicle first appears  → pickup
              │  last_seen[veh]  = position at LAST step vehicle is active → dropoff
              │  → trip_events_main.csv   (4 events × {sim_r['valid_trips']} valid trips = {sim_r['total_events']} rows)
              │  → sim_summary_main.json  (stats, concurrent_history)
              ▼
        04_deploy_witnesses.py
              │  sumolib reads tartu.net.xml
              │  Keeps every junction with incoming_degree ≥ 1
              │  convertXY2LonLat → WGS-84 coordinates
              │  → witnesses.csv  (candidate pool, {wsm['candidate_count_after_sampling']:,} entries)
              ▼
        05_run_experiments.py
              │  Subsamples witnesses for each strategy (random / grid / hotspot)
              │  Calls generate_proof() + verify_proof() for every event
              │  → results/  (S1–S5 security, F1–F3 feasibility)
        ```
        """
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 1 — Trip generation
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Step 1 — Trip Generation  (`02_generate_trips.sh`)")

    sim_dur_s = sim_r["sim_duration_s"]
    expected_departures = round(3600 / 7)   # SIM_DURATION / p

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulation window", "3 600 s", "vehicle departures [0, 3600 s]")
    c2.metric("Expected departures", f"≈ {expected_departures}", "3600 / p  (p = 7 s)")
    c3.metric("Vehicles in simulation", f"{sim_r['total_vehicles']:,}",
              f"after --validate filter")
    c4.metric("Valid trips (PoL events)", f"{sim_r['valid_trips']:,}",
              f"skipped: {sum(sim_r['skipped'].values())}")

    with st.expander("What randomTrips.py does — key flags explained", expanded=False):
        st.markdown(
            f"""
            `randomTrips.py` generates **random origin-destination (OD) pairs** on the
            road network and routes each trip using SUMO's `duarouter` internally.

            | Flag | Value | Meaning |
            |---|---|---|
            | `-n tartu.net.xml` | — | road network to place trips on |
            | `-e 3600` | 3600 s | vehicles may depart within [0, 3600 s] |
            | `-p 7` | 7 s | mean inter-departure interval → ~3600/7 ≈ **{expected_departures} trips** defined |
            | `--random-depart` | — | departure times are uniformly random in [0, e], not on a fixed grid |
            | `--validate` | — | internally runs `duarouter`; OD pairs that cannot be routed are **silently discarded** |
            | `--seed 4201` | — | fixed random seed for reproducibility |

            **Why do we get {sim_r['total_vehicles']:,} vehicles, not exactly {expected_departures}?**
            `--validate` discards any OD pair that `duarouter` cannot route
            (e.g. isolated network edges). A small number of trips are dropped this way.

            **What are the "Teleporting vehicle" warnings in SUMO-GUI?**
            SUMO teleports a vehicle when it waits too long at a junction
            (`waited too long (yield/jam)`) — it jumps over the blocked segment so
            the simulation can keep running. In the Tartu network this affected
            a small number of vehicles on congested edges.
            Teleportation only affects the **intermediate path**; the vehicle continues
            normally afterward. Since `03_extract_traces.py` only records the
            `first_seen` position (pickup) and `last_seen` position (dropoff),
            the endpoints remain physically valid.
            So a teleported vehicle can still remain in the valid trip set if its
            pickup and dropoff endpoints pass the quality filters.

            **Why does SUMO-GUI show only ~{sim_r['peak_concurrent_vehs']} vehicles at once?**
            All {sim_r['total_vehicles']:,} vehicles exist across the full
            {sim_dur_s:.0f} s run, but the mean trip duration is
            ~{sim_r['trip_statistics']['trip_dur_s']['mean']:.0f} s.
            At any given moment: `concurrent ≈ T_avg / p ≈
            {sim_r['trip_statistics']['trip_dur_s']['mean']:.0f} / 7 ≈
            {sim_r['trip_statistics']['trip_dur_s']['mean']/7:.0f}`.
            The measured peak was **{sim_r['peak_concurrent_vehs']} vehicles**.

            **Five route files are generated** (one per experiment):
            """
        )
        route_rows = [
            ("routes_main.xml", "7", "4201", "F1",
             f"{all_sims.get('main',{}).get('results',{}).get('total_vehicles','—')} veh, "
             f"peak {all_sims.get('main',{}).get('results',{}).get('peak_concurrent_vehs','—')}"),
            ("routes_c10.xml",  "60", "4202", "F2 ~10 concurrent",
             f"{all_sims.get('c10',{}).get('results',{}).get('total_vehicles','—')} veh, "
             f"peak {all_sims.get('c10',{}).get('results',{}).get('peak_concurrent_vehs','—')}"),
            ("routes_c50.xml",  "12", "4203", "F2 ~50 concurrent",
             f"{all_sims.get('c50',{}).get('results',{}).get('total_vehicles','—')} veh, "
             f"peak {all_sims.get('c50',{}).get('results',{}).get('peak_concurrent_vehs','—')}"),
            ("routes_c100.xml", "6",  "4204", "F2 ~100 concurrent",
             f"{all_sims.get('c100',{}).get('results',{}).get('total_vehicles','—')} veh, "
             f"peak {all_sims.get('c100',{}).get('results',{}).get('peak_concurrent_vehs','—')}"),
            ("routes_c200.xml", "3",  "4205", "F2 ~200 concurrent",
             f"{all_sims.get('c200',{}).get('results',{}).get('total_vehicles','—')} veh, "
             f"peak {all_sims.get('c200',{}).get('results',{}).get('peak_concurrent_vehs','—')}"),
        ]
        st.dataframe(
            pd.DataFrame(route_rows,
                         columns=["Route file", "p (s)", "Seed", "Used in", "Actual result"]),
            use_container_width=True, hide_index=True,
        )

    dist_stats = sim_r["trip_statistics"]["trip_dist_m"]
    dur_stats   = sim_r["trip_statistics"]["trip_dur_s"]
    st.dataframe(
        pd.DataFrame({
            "Statistic":        ["Min", "Mean", "Max"],
            "Trip distance (m)": [f"{dist_stats['min']:.0f}", f"{dist_stats['mean']:.0f}", f"{dist_stats['max']:.0f}"],
            "Trip duration (s)": [f"{dur_stats['min']:.0f}",  f"{dur_stats['mean']:.0f}",  f"{dur_stats['max']:.0f}"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 2 — PoL event extraction
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Step 2 — PoL Event Extraction  (`03_extract_traces.py`)")

    e_counts = sim_r["event_counts_by_type"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PoL events", f"{sim_r['total_events']:,}", "4 per valid trip")
    c2.metric("driver_arrival",          f"{e_counts['driver_arrival']:,}")
    c3.metric("rider_pickup_presence",   f"{e_counts['rider_pickup_presence']:,}")
    c4.metric("trip_completion",         f"{e_counts['trip_completion']:,}")
    c5.metric("rider_dropoff_presence",  f"{e_counts['rider_dropoff_presence']:,}")

    with st.expander("How TraCI extracts events — step by step", expanded=False):
        st.markdown(
            f"""
            `03_extract_traces.py` controls SUMO **headlessly** via TraCI
            (Traffic Control Interface).

            #### TraCI simulation loop
            ```python
            traci.start(["sumo", "-c", "tartu.sumocfg",
                         "--route-files", "routes_main.xml"])

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()          # advance 1 second
                for veh_id in traci.vehicle.getIDList():
                    x, y   = traci.vehicle.getPosition(veh_id)   # SUMO internal (m)
                    lon, lat = traci.simulation.convertGeo(x, y)  # → WGS-84

                    if veh_id not in first_seen:
                        first_seen[veh_id] = (lat, lon, t)        # pickup
                    last_seen[veh_id] = (lat, lon, t)             # overwrite every step → dropoff
            ```

            **Why does the simulation run to {sim_dur_s:.0f} s, not 3600 s?**
            The loop condition `getMinExpectedNumber() > 0` keeps running until
            every vehicle has finished — including those that departed close to t = 3600 s
            and need up to ~{dur_stats['max']:.0f} s to complete their route.
            The last vehicle exited at t = {sim_dur_s:.0f} s.

            #### Quality filters (applied to first_seen / last_seen after simulation)

            `build_events()` iterates all {sim_r['total_vehicles']:,} vehicles seen by TraCI
            and drops any trip that fails either threshold:

            | Filter | Threshold | Trips dropped | What it catches |
            |---|---|---|---|
            | Straight-line pickup → dropoff distance | ≥ {sim_p['MIN_TRIP_DIST_m']} m | **{sim_r['skipped'].get('dist_too_short', 0)}** | Vehicles that barely moved (routing edge cases) |
            | Trip duration (last_seen − first_seen) | ≥ {sim_p['MIN_TRIP_DUR_s']} s | **{sim_r['skipped'].get('dur_too_short', 0)}** | Vehicles that appeared and disappeared immediately |

            **Accounting for all {sim_r['total_vehicles']:,} vehicles:**
            ```
            {sim_r['total_vehicles']:,} vehicles seen by TraCI
             −  {sim_r['skipped'].get('dist_too_short', 0)} dropped (dist_too_short)
             −  {sim_r['skipped'].get('dur_too_short', 0)} dropped (dur_too_short)
            ─────────────────────────────────────────
             = {sim_r['valid_trips']:,} valid trips  →  {sim_r['total_events']:,} PoL events
            ```

            Note: all 17 SUMO-teleported vehicles passed both filters and are included
            in the valid trip set. Teleportation only affects the intermediate path, not
            the pickup/dropoff endpoints that PoL records.

            #### Four events per valid trip
            | Event | Location | Timestamp |
            |---|---|---|
            | `driver_arrival` | **pickup** lat/lon (vehicle first_seen) | t₁ = first_seen time |
            | `rider_pickup_presence` | **same pickup** lat/lon | t₁ + U[0, {sim_p['DELTA_T_PICKUP_s']} s]  (seed={sim_p['rider_timing_seed']}) |
            | `trip_completion` | **dropoff** lat/lon (vehicle last_seen) | t₂ = last_seen time |
            | `rider_dropoff_presence` | **same dropoff** lat/lon | t₂ + U[0, {sim_p['DELTA_T_DROPOFF_s']} s]  (seed={sim_p['rider_timing_seed']}) |

            Events 1 & 2 always share identical coordinates (pickup point).
            Events 3 & 4 always share identical coordinates (dropoff point).
            Rider timing is a stated simplification — SUMO does not model passengers.
            It does not affect F1 (witness coverage depends only on location).
            """
        )

    with st.expander("CSV output preview (trip_events_main.csv)", expanded=False):
        st.caption(
            f"{sim_r['total_events']:,} rows · "
            f"simulation duration {sim_dur_s:.0f} s · "
            f"≈ {sim_r['proofs_per_hour']:.0f} proofs / hour"
        )
        st.dataframe(events_df.head(12), use_container_width=True, hide_index=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 3 — Witness deployment
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Step 3 — Witness Candidate Pool  (`04_deploy_witnesses.py`)")

    itc = wsm["incoming_threshold_counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Network nodes total",     f"{wsm['network_node_count']:,}",
              "tartu.net.xml (sumolib)")
    c2.metric("Candidate witnesses",     f"{wsm['candidate_count_after_sampling']:,}",
              f"incoming_degree ≥ {wsm['min_incoming']}")
    c3.metric("Nodes discarded",
              f"{wsm['network_node_count'] - wsm['candidate_count_after_sampling']:,}",
              "dead ends (0 incoming)")
    sampling_needed = wsm["candidate_count_before_sampling"] > wsm["max_witnesses"]
    c4.metric(
        "Sampling needed?",
        "Yes" if sampling_needed else "No",
        (
            f"{wsm['candidate_count_before_sampling']:,} > max_witnesses={wsm['max_witnesses']:,}"
            if sampling_needed
            else f"{wsm['candidate_count_before_sampling']:,} <= max_witnesses={wsm['max_witnesses']:,}"
        ),
    )

    with st.expander("How witness candidates are generated", expanded=False):
        st.markdown(
            f"""
            `04_deploy_witnesses.py` uses **`sumolib`** to parse the road network XML
            directly — no SUMO simulation is run.

            ```python
            net = sumolib.net.readNet("tartu.net.xml")
            for node in net.getNodes():
                if len(node.getIncoming()) >= 1:        # has at least one incoming road
                    x, y = node.getCoord()              # SUMO internal Cartesian (metres)
                    lon, lat = net.convertXY2LonLat(x, y)  # → WGS-84
                    candidates.append({{"id": f"W_{{node.getID()}}", "lat": lat, "lon": lon}})
            ```

            **Why junctions with incoming_degree ≥ 1?**
            A BLE beacon placed at a dead-end source node (0 incoming edges) would never
            have passing traffic, making it useless as a witness.

            #### Incoming-degree distribution
            | Min incoming edges | Qualifying junctions |
            |---|---|
            | ≥ 1 | **{itc['1']:,}** (all candidates) |
            | ≥ 2 | {itc['2']:,} |
            | ≥ 3 | {itc['3']:,} |
            | ≥ 4 | {itc['4']:,} |

            #### This is a candidate POOL, not a deployment
            `witnesses.csv` contains **all** {wsm['candidate_count_after_sampling']:,}
            physically plausible witness locations.  Experiment F1 compares three
            strategies that **subsample** this pool differently inside
            `05_run_experiments.py`:

            | Strategy | How N witnesses are selected from the pool |
            |---|---|
            | **Random** | `random.sample(candidates, N)` — uniform random |
            | **Uniform Grid** | divide bounding box into N cells; pick closest candidate to each cell centre |
            | **Heatmap-guided** | rank candidates by proximity to historical trip event locations; pick top-N |
            """
        )

    all_witnesses = load_witnesses()
    m = folium.Map(location=TARTU_CENTER, zoom_start=13, tiles="CartoDB positron")
    eligible_witness_count = len(all_witnesses)
    if eligible_witness_count <= 200:
        preview_target = eligible_witness_count
    else:
        preview_target = st.slider(
            "Junction witnesses shown on city map",
            min_value=200,
            max_value=eligible_witness_count,
            value=min(WITNESS_PREVIEW_DEFAULT, eligible_witness_count),
            step=100,
            help=(
                "The slider range follows the eligible road-junction witness count."
            ),
        )
    preview_count = min(preview_target, eligible_witness_count)
    preview_rng = random.Random(WITNESS_PREVIEW_SEED)
    sample_w = preview_rng.sample(all_witnesses, preview_count)
    for w in sample_w:
        folium.CircleMarker(
            location=[w.lat, w.lon],
            radius=2,
            color=C_TEAL,
            fill=True,
            fill_opacity=0.55,
            popup=f"Witness {w.id}",
        ).add_to(m)
    st.caption(
        f"City-wide candidate pool preview: selected {preview_target:,} out of {eligible_witness_count:,} eligible road-junction witness candidates. "
        f"{preview_count:,} are currently drawn. "
        f"Each teal point is a road junction that can host a witness. Bounding box: "
        f"lat [{wsm['bounding_box']['lat_min']:.4f}, {wsm['bounding_box']['lat_max']:.4f}], "
        f"lon [{wsm['bounding_box']['lon_min']:.4f}, {wsm['bounding_box']['lon_max']:.4f}]."
    )
    st_folium(m, width=900, height=400)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 4 — Experiments
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Step 4 — Experiment Suite  (`05_run_experiments.py`)")
    st.markdown(
        "Takes `trip_events_main.csv` and `witnesses.csv` as inputs. "
        "Calls `generate_proof()` + `verify_proof()` from `protocol_simulator.py` "
        "for every event under each experimental condition."
    )
    exp_rows = [
        ("S1", "GPS Spoofing",        "BLE unreachability + RSSI check + verifier spatial check"),
        ("S2", "Location Replay",     "Context binding + event-state registry"),
        ("S3", "Relay / Collusion",   "BLE round-trip timing bound Δt_max, sweep over dt values"),
        ("S4", "On-chain Audit",      "Merkle path verify against anchored root; tamper detection"),
        ("S5", "Baseline Comparison", "GPS-only vs PoL: attack success rate side-by-side"),
        ("F1", "Witness Density",     "PSR vs witness count for 3 deployment strategies × 6 levels"),
        ("F2", "Witness Load",        "Max/P95/Mean requests per witness under 4 concurrency levels"),
        ("F3", "Batch Anchoring",     "Gas cost per proof & Merkle build time vs batch interval"),
    ]
    st.dataframe(
        pd.DataFrame(exp_rows, columns=["ID", "Experiment", "What it measures"]),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Page 1 — System Overview
# ════════════════════════════════════════════════════════════════════════════


def page_overview() -> None:
    st.title("Proof-of-Location in Ride-hailing")
    st.markdown(
        """
        This application accompanies the thesis:
        **"Trustworthy Shared Mobility through Consensus-Based Proof-of-Location"**

        The system lets any party independently verify **where** a ride-hailing participant
        was at a given moment — without trusting GPS coordinates alone.
        """
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### Prover")
        st.markdown(
            "Driver or rider who needs to prove their physical location "
            "(e.g., *I was at the pickup point at 14:02*)."
        )
    with col_b:
        st.markdown("#### Witness")
        st.markdown(
            "Roadside IoT device (BLE beacon) that independently attests the prover's "
            "proximity via a challenge-response exchange."
        )
    with col_c:
        st.markdown("#### Verifier")
        st.markdown(
            "Platform or auditor that checks the collected attestations and the "
            "on-chain Merkle anchor."
        )

    st.divider()
    st.subheader("Protocol Lifecycle — 4 Ride-hailing Events")
    steps = [
        ("1", "driver_arrival", "Driver arrives at pickup point", "Pickup"),
        ("2", "rider_pickup_presence", "Rider arrives at pickup point", "Rider"),
        ("3", "trip_completion", "Driver reaches dropoff point", "Dropoff"),
        ("4", "rider_dropoff_presence", "Rider exits at dropoff point", "Exit"),
    ]
    st.markdown(
        """
        <style>
        .protocol-event-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-top: 8px;
        }
        .protocol-event-card {
            border: 1px solid #D9D9D9;
            border-radius: 12px;
            background: #FAFAFA;
            padding: 14px 14px 12px 14px;
            min-width: 0;
        }
        .protocol-event-kicker {
            color: #009B9E;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 8px;
            line-height: 1.2;
        }
        .protocol-event-name {
            color: #2E2E3A;
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.25;
            word-break: break-word;
            overflow-wrap: anywhere;
            margin-bottom: 8px;
        }
        .protocol-event-desc {
            color: #4A4A58;
            font-size: 0.9rem;
            line-height: 1.35;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        @media (max-width: 980px) {
            .protocol-event-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .protocol-event-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    event_cards_html = '<div class="protocol-event-grid">'
    for num, event, desc, icon in steps:
        event_cards_html += (
            '<div class="protocol-event-card">'
            f'<div class="protocol-event-kicker">{icon} Event {num}</div>'
            f'<div class="protocol-event-name">{event}</div>'
            f'<div class="protocol-event-desc">{desc}</div>'
            '</div>'
        )
    event_cards_html += "</div>"
    st.markdown(event_cards_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("For each event, the protocol runs:")

    st.markdown(
        """
        ```
        Prover  ──BLE challenge──►  Witness
                ◄──signed response──

        if response_time ≤ Δt_max  AND
           claimed_distance ≤ d_th  AND
           RSSI ≥ τ_rssi:
               Witness issues attestation

        After N_w ≥ 4 attestations collected:
               LocationProof is built

        Verifier checks:
           • context binding      (proof is for the right event)
           • temporal validity    (proof is not stale)
           • spatial validity     (claimed location ≈ event location)
           • replay detection     (proof not previously used)

        Platform batches proofs → Merkle root → anchorBatch(root, batchId)
        ```
        """
    )

    st.divider()
    st.subheader("Security Threats Addressed")
    threat_data = {
        "Threat": [
            "GPS Spoofing",
            "Location Replay",
            "Relay / Collusion",
            "On-chain Tampering",
        ],
        "Example in Ride-hailing": [
            "Driver fakes being at pickup when 5 km away",
            "Driver reuses yesterday's valid proof",
            "Two accomplices forward BLE challenges across city",
            "Platform edits proof database after dispute",
        ],
        "Defence Mechanism": [
            "BLE reachability + RSSI check at witness",
            "Context-binding + event-state registry at verifier",
            "BLE round-trip timing bound Δt_max",
            "Merkle audit path vs. on-chain root",
        ],
    }
    st.dataframe(pd.DataFrame(threat_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Junction-Based Witness Placement")
    all_witnesses = load_witnesses()
    events_df = load_events()
    overview_event = events_df[events_df["event"] == "driver_arrival"].iloc[20]
    overview_lat = float(overview_event["lat"])
    overview_lon = float(overview_event["lon"])
    nearby_candidates = sorted(
        nearby_witnesses(overview_lat, overview_lon, all_witnesses, OVERVIEW_RADIUS_M),
        key=lambda w: haversine(overview_lat, overview_lon, w.lat, w.lon),
    )
    within_ble_candidates = []
    outer_ring_candidates = []
    for witness in nearby_candidates:
        dist_m = haversine(overview_lat, overview_lon, witness.lat, witness.lon)
        if dist_m <= PARAMS.d_ble:
            within_ble_candidates.append((witness, dist_m))
        else:
            outer_ring_candidates.append((witness, dist_m))

    st.caption(
        "Purpose of this map: show the difference between the local road-junction candidate pool "
        "and the smaller set of witnesses that can directly attest the event over BLE."
    )
    metric_o1, metric_o2, metric_o3 = st.columns(3)
    metric_o1.metric(
        f"Junction candidates within {OVERVIEW_RADIUS_M:.0f} m",
        f"{len(nearby_candidates):,}",
    )
    metric_o2.metric(
        f"Inside BLE radius ({PARAMS.d_ble:.0f} m)",
        f"{len(within_ble_candidates):,}",
        "can directly witness this event",
    )
    metric_o3.metric(
        f"Outside BLE but still nearby",
        f"{len(outer_ring_candidates):,}",
        f"between {PARAMS.d_ble:.0f} m and {OVERVIEW_RADIUS_M:.0f} m",
    )

    st.caption(
        f"Blue marks one real trip event. All colored dots are witness candidates placed on road junctions. "
        f"Green dots are the {len(within_ble_candidates):,} junction candidates inside the BLE attestation radius; "
        f"gray dots are the remaining {len(outer_ring_candidates):,} candidates that are nearby but too far to witness directly."
    )

    m = folium.Map(location=[overview_lat, overview_lon], zoom_start=16, tiles="CartoDB positron")
    folium.Circle(
        location=[overview_lat, overview_lon],
        radius=OVERVIEW_RADIUS_M,
        color="#A0A6AD",
        fill=True,
        fill_color="#A0A6AD",
        fill_opacity=0.04,
        weight=1.4,
        dash_array="3 6",
        tooltip=f"Local junction-candidate view: {OVERVIEW_RADIUS_M:.0f} m",
    ).add_to(m)
    folium.Circle(
        location=[overview_lat, overview_lon],
        radius=PARAMS.d_ble,
        color=C_TEAL,
        fill=True,
        fill_color=C_TEAL,
        fill_opacity=0.09,
        weight=1.8,
        dash_array="6 4",
        tooltip=f"BLE attestation radius: {PARAMS.d_ble:.0f} m",
    ).add_to(m)
    folium.Marker(
        location=[overview_lat, overview_lon],
        tooltip="Example trip event",
        popup=folium.Popup(
            (
                f"Example trip event<br>"
                f"Event: {overview_event['event']}<br>"
                f"Order ID: {overview_event['order_id']}"
            ),
            max_width=280,
        ),
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)

    for w, dist_m in outer_ring_candidates:
        folium.CircleMarker(
            location=[w.lat, w.lon],
            radius=3.6,
            color="#8F969E",
            fill=True,
            fill_color="#B7BEC6",
            fill_opacity=0.65,
            weight=1.0,
            popup=folium.Popup(
                f"Witness {w.id}<br>Road junction candidate<br>"
                f"Distance to event: {dist_m:.1f} m<br>"
                f"Status: nearby, but outside BLE radius",
                max_width=300,
            ),
            tooltip=f"Nearby candidate outside BLE<br>{w.id[:12]}…",
        ).add_to(m)

    for w, dist_m in within_ble_candidates:
        folium.PolyLine(
            [[overview_lat, overview_lon], [w.lat, w.lon]],
            color="#27AE60",
            weight=1.2,
            opacity=0.45,
        ).add_to(m)
        folium.CircleMarker(
            location=[w.lat, w.lon],
            radius=5.2,
            color="#1F8F4E",
            fill=True,
            fill_color="#33B864",
            fill_opacity=0.92,
            weight=1.2,
            popup=folium.Popup(
                f"Witness {w.id}<br>Road junction candidate<br>"
                f"Distance to event: {dist_m:.1f} m<br>"
                f"Status: inside BLE radius, can attest directly",
                max_width=300,
            ),
            tooltip=f"BLE-reachable witness candidate<br>{w.id[:12]}…",
        ).add_to(m)

    latitudes = [overview_lat] + [w.lat for w in nearby_candidates]
    longitudes = [overview_lon] + [w.lon for w in nearby_candidates]
    m.fit_bounds(
        [
            [min(latitudes) - 0.0008, min(longitudes) - 0.0008],
            [max(latitudes) + 0.0008, max(longitudes) + 0.0008],
        ]
    )

    legend_html = (
        "<div style='font-size:13px;line-height:1.7'>"
        "<span style='color:#2980b9'>●</span> Example event &nbsp;"
        "<span style='color:#33B864'>●</span> Junction candidate inside BLE radius &nbsp;"
        "<span style='color:#8F969E'>●</span> Junction candidate outside BLE radius"
        "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st_folium(m, width=900, height=400)


# ════════════════════════════════════════════════════════════════════════════
# Page 2 — Protocol Walkthrough
# ════════════════════════════════════════════════════════════════════════════


def page_protocol() -> None:
    st.title("Live Protocol Walkthrough")
    st.markdown(
        "Pick a ride-hailing trip from the simulated Tartu dataset and step through "
        "the complete Proof-of-Location protocol."
    )

    all_witnesses = load_witnesses()
    events_df = load_events()

    # ── Trip selector ──────────────────────────────────────────────────────
    order_ids = events_df["order_id"].unique().tolist()
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_order = st.selectbox(
            "Select a trip (order_id):",
            options=order_ids[:80],
            format_func=lambda x: f"Trip #{x}",
        )
    with col2:
        selected_event_name = st.selectbox(
            "Event to prove:",
            options=["driver_arrival", "rider_pickup_presence",
                     "trip_completion", "rider_dropoff_presence"],
        )

    trip_events = events_df[events_df["order_id"] == selected_order]
    event_row = trip_events[trip_events["event"] == selected_event_name]
    if event_row.empty:
        st.warning("This event is not available for the selected trip.")
        return

    ev = event_row.iloc[0]
    ev_lat, ev_lon, ev_time = float(ev["lat"]), float(ev["lon"]), float(ev["time"])
    trip_dist = float(ev["trip_dist_m"])
    trip_dur = float(ev["trip_dur_s"])

    # ── Protocol parameters panel ──────────────────────────────────────────
    with st.expander("Protocol Parameters", expanded=False):
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        d_ble_val = col_p1.slider("BLE radius d_ble (m)", 50, 200, 100, 10)
        d_th_val = col_p2.slider("Claim threshold d_th (m)", 50, 200, 100, 10)
        dt_max_val = col_p3.slider("Δt_max (ms)", 10, 120, 35, 5)
        n_w_val = col_p4.slider("Min witnesses N_w", 1, 6, 4, 1)

    params = ProtocolParams(
        d_ble=float(d_ble_val),
        d_th=float(d_th_val),
        delta_t_max=dt_max_val / 1000.0,
        tau_rssi=-90.0,
        n_w=n_w_val,
    )

    # ── Run protocol ───────────────────────────────────────────────────────
    rng_seed = int(selected_order) * 100 + ["driver_arrival", "rider_pickup_presence",
        "trip_completion", "rider_dropoff_presence"].index(selected_event_name)
    random.seed(rng_seed)

    ctx = {
        "order_id": str(selected_order),
        "event": selected_event_name,
        "session_id": f"session_{rng_seed}",
    }
    local_witnesses = nearby_witnesses(ev_lat, ev_lon, all_witnesses, params.d_ble + 20)
    proof_result = generate_proof(
        prover_id=str(selected_order),
        actual_lat=ev_lat,
        actual_lon=ev_lon,
        claim_lat=ev_lat,
        claim_lon=ev_lon,
        timestamp=ev_time,
        ctx=ctx,
        witnesses=local_witnesses,
        params=params,
        candidate_mode="actual",
    )

    verify_accepted, verify_reason = False, "no_proof"
    if proof_result.success:
        verify_accepted, verify_reason = verify_proof(
            proof_result.proof,
            ev_lat,
            ev_lon,
            ev_time - 5.0,
            ev_time + 5.0,
            params,
            set(),
            expected_ctx=ctx,
        )

    checked_ids = {chk.witness_id for chk in proof_result.witness_checks}
    accepted_count = sum(1 for chk in proof_result.witness_checks if chk.accepted)
    rejected_count = len(proof_result.witness_checks) - accepted_count
    not_contacted_count = max(0, len(local_witnesses) - len(checked_ids))

    # ── Map ────────────────────────────────────────────────────────────────
    st.subheader("Map View")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Accepted", accepted_count)
    col_m2.metric("Rejected", rejected_count)
    col_m3.metric("Not contacted", not_contacted_count)
    show_not_contacted = st.checkbox(
        "Show not-contacted nearby witnesses",
        value=False,
        help=(
            "Gray witnesses are nearby candidates, but the protocol did not contact them. "
            "This often happens because the protocol stops after it gets enough accepted attestations."
        ),
    )
    st.caption(
        "Green and red links are shown for all checked witnesses. "
        "Gray witnesses have no accept or reject result because they were not contacted."
    )

    m = folium.Map(location=[ev_lat, ev_lon], zoom_start=16, tiles="CartoDB positron")

    # BLE radius circle
    folium.Circle(
        location=[ev_lat, ev_lon],
        radius=params.d_ble,
        color=C_TEAL,
        fill=True,
        fill_opacity=0.08,
        weight=1.5,
        dash_array="6 4",
        tooltip=f"BLE range: {params.d_ble:.0f} m",
    ).add_to(m)

    # Witness markers and contact links
    for chk in proof_result.witness_checks:
        color = "#27ae60" if chk.accepted else "#e74c3c"
        label = "Accepted" if chk.accepted else f"Rejected: {chk.reason}"
        rt_ms = f"{chk.response_time_s * 1000:.1f} ms" if chk.response_time_s else "N/A"

        w_node = next((w for w in all_witnesses if w.id == chk.witness_id), None)
        if w_node:
            folium.PolyLine(
                locations=[[ev_lat, ev_lon], [w_node.lat, w_node.lon]],
                color=color,
                weight=3,
                opacity=0.85,
                dash_array=None if chk.accepted else "6 4",
                tooltip=f"{label} link",
            ).add_to(m)
            folium.CircleMarker(
                location=[w_node.lat, w_node.lon],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(
                    (
                        f"Witness {chk.witness_id}<br>"
                        f"Status: {label}<br>"
                        f"Actual dist: {chk.actual_distance_m:.1f} m<br>"
                        f"Response time: {rt_ms}"
                    ),
                    max_width=320,
                ),
                tooltip=(
                    f"{'Accepted' if chk.accepted else 'Rejected'} witness<br>"
                    f"{chk.witness_id[:12]}…"
                ),
            ).add_to(m)

    # show all nearby witnesses (not contacted) in gray
    if show_not_contacted:
        for w in local_witnesses:
            if w.id not in checked_ids:
                folium.CircleMarker(
                    location=[w.lat, w.lon],
                    radius=4,
                    color="#AAAAAA",
                    fill=True,
                    fill_opacity=0.5,
                    popup=folium.Popup(f"Witness {w.id}<br>Status: not contacted", max_width=280),
                    tooltip=f"Not-contacted witness<br>{w.id[:12]}…",
                ).add_to(m)

    # Event marker (prover)
    folium.Marker(
        location=[ev_lat, ev_lon],
        popup=folium.Popup(
            (
                f"Prover event<br>"
                f"Event: {selected_event_name}<br>"
                f"Trip dist: {trip_dist:.0f} m<br>"
                f"Duration: {trip_dur:.0f} s"
            ),
            max_width=320,
        ),
        tooltip=f"Prover event<br>{selected_event_name}",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)

    legend_html = (
        "<div style='font-size:13px;line-height:1.7'>"
        "<span style='color:#27ae60'>●</span> Accepted witness &nbsp;"
        "<span style='color:#e74c3c'>●</span> Rejected witness &nbsp;"
        "<span style='color:#AAAAAA'>●</span> Not contacted &nbsp;"
        "<span style='color:#2980b9'>●</span> Prover event location"
        "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st_folium(m, width=900, height=440)

    # ── Step-by-step log ───────────────────────────────────────────────────
    st.subheader("Step-by-step Protocol Trace")

    with st.expander("Step 1 — Event Triggered", expanded=True):
        st.markdown(
            f"""
            | Field | Value |
            |---|---|
            | Order ID | `{selected_order}` |
            | Event | `{selected_event_name}` |
            | Location | `{ev_lat:.6f}, {ev_lon:.6f}` |
            | Timestamp | `{ev_time:.1f} s` |
            | Context hash (ctx) | `{ctx}` |
            """
        )

    with st.expander("Step 2 — Witness Discovery & BLE Challenge-Response"):
        st.markdown(
            f"Found **{len(local_witnesses)}** witness candidates within "
            f"`d_ble + 20 = {params.d_ble + 20:.0f} m`."
        )
        st.markdown(f"Protocol needs **N_w = {params.n_w}** accepted attestations.")

        if proof_result.witness_checks:
            rows = []
            for chk in proof_result.witness_checks:
                rt = f"{chk.response_time_s * 1000:.2f} ms" if chk.response_time_s else "—"
                rssi = f"{chk.rssi_dbm:.1f} dBm" if chk.rssi_dbm else "—"
                rows.append({
                    "Witness": chk.witness_id[:18] + "…",
                    "Actual dist (m)": f"{chk.actual_distance_m:.1f}",
                    "Response time": rt,
                    "RSSI": rssi,
                    "Result": "Accept" if chk.accepted else f"Reject: {chk.reason}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No witnesses were in BLE range.")

    with st.expander("Step 3 — Proof Assembly"):
        if proof_result.success:
            st.success(
                f"LocationProof assembled with "
                f"**{proof_result.attestations_obtained}** attestations "
                f"(need ≥ {params.n_w})."
            )
            st.json(
                {
                    "ctx": ctx,
                    "claim": {
                        "lat": round(ev_lat, 6),
                        "lon": round(ev_lon, 6),
                        "timestamp": ev_time,
                    },
                    "attestations_count": proof_result.attestations_obtained,
                }
            )
        else:
            reasons = ", ".join(proof_result.rejection_reasons) or "insufficient_witnesses"
            st.error(
                f"Proof generation FAILED — only "
                f"{proof_result.attestations_obtained}/{params.n_w} attestations "
                f"obtained. Rejection reasons: **{reasons}**"
            )

    with st.expander("Step 4 — Verifier Check"):
        if proof_result.success:
            if verify_accepted:
                st.success("Verifier **ACCEPTED** the proof.")
                st.markdown(
                    "Checks passed: context binding | temporal | spatial | no replay"
                )
            else:
                st.error(f"Verifier **REJECTED** — reason: `{verify_reason}`")
        else:
            st.warning("Proof generation failed; verifier step skipped.")

    # ── Summary badge ──────────────────────────────────────────────────────
    st.divider()
    final_ok = proof_result.success and verify_accepted
    if final_ok:
        st.success(
            f"### Location Proof Issued\n"
            f"Trip **#{selected_order}** — event `{selected_event_name}` "
            f"successfully proved at `({ev_lat:.5f}, {ev_lon:.5f})`."
        )
    else:
        st.error(
            f"### Location Proof FAILED\n"
            f"Trip **#{selected_order}** — event `{selected_event_name}` "
            f"could not be proved. Check witness density or protocol parameters."
        )


# ════════════════════════════════════════════════════════════════════════════
# Page 3 — Attack Scenarios
# ════════════════════════════════════════════════════════════════════════════


def page_attacks() -> None:
    st.title("Attack Scenarios")

    all_witnesses = load_witnesses()
    events_df = load_events()
    ev = events_df[events_df["event"] == "driver_arrival"].iloc[20]
    ev_lat, ev_lon, ev_time = float(ev["lat"]), float(ev["lon"]), float(ev["time"])

    attack_tab, replay_tab, relay_tab = st.tabs([
        "GPS Spoofing", "Location Replay", "Relay / Collusion"
    ])

    # ── GPS Spoofing ────────────────────────────────────────────────────────
    with attack_tab:
        attack_distance_options = load_s1_distance_options()
        default_attack_distance = 100 if 100 in attack_distance_options else attack_distance_options[0]
        st.subheader("GPS Spoofing: Driver fakes being at pickup point")
        st.markdown(
            "This page uses the same tested distance points as S1, from 5 m to 5000 m. "
            "At short distance, the attacker may still be inside BLE range. "
            "Near 100 m, the result can change. Far outside BLE range, the spoof usually fails."
        )

        col_s1, col_s2 = st.columns(2)
        attack_dist = col_s1.select_slider(
            "Attacker's actual distance from event (m):",
            options=attack_distance_options,
            value=default_attack_distance,
        )
        bearing_deg = col_s2.slider("Direction (bearing °)", 0, 359, 45, 15)

        actual_lat, actual_lon = offset_point(
            ev_lat, ev_lon, float(attack_dist), math.radians(bearing_deg)
        )

        random.seed(77)
        ctx = {"order_id": str(ev["order_id"]), "event": "driver_arrival", "session_id": "spoof_case"}
        claim_candidates = nearby_witnesses(ev_lat, ev_lon, all_witnesses, PARAMS.d_ble)
        actual_side_witnesses = nearby_witnesses(actual_lat, actual_lon, all_witnesses, PARAMS.d_ble)
        spoof_result = generate_proof(
            prover_id="attacker",
            actual_lat=actual_lat,
            actual_lon=actual_lon,
            claim_lat=ev_lat,
            claim_lon=ev_lon,
            timestamp=ev_time,
            ctx=ctx,
            witnesses=claim_candidates,
            params=PARAMS,
            candidate_mode="claim",
        )

        checked_by_id = {chk.witness_id: chk for chk in spoof_result.witness_checks}
        checked_witnesses = [w for w in claim_candidates if w.id in checked_by_id]
        uncontacted_witnesses = [w for w in claim_candidates if w.id not in checked_by_id]
        accepted_count = sum(1 for chk in spoof_result.witness_checks if chk.accepted)
        rejected_count = len(spoof_result.witness_checks) - accepted_count
        reason_labels = {
            "ble_unreachable": "BLE unreachable",
            "time_bound": "Round-trip too slow",
            "distance": "Claimed distance too large",
            "rssi": "RSSI too weak",
            "pass": "Accepted",
        }
        reason_text = ", ".join(
            sorted({reason_labels.get(reason, reason) for reason in spoof_result.rejection_reasons})
        ) or "Accepted"

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Claim-side junctions", f"{len(claim_candidates)}")
        metric_b.metric(
            "Checked witnesses",
            f"{len(spoof_result.witness_checks)}",
            f"{len(uncontacted_witnesses)} not contacted",
        )
        metric_c.metric(
            "Accepted / needed",
            f"{accepted_count}/{PARAMS.n_w}",
            f"{rejected_count} rejected",
        )
        metric_d.metric(
            "Near actual position",
            f"{len(actual_side_witnesses)}",
            "road-junction witnesses",
        )

        st.caption(
            "Witnesses are deployed on road junctions. In GPS spoofing, the protocol discovers "
            "witnesses around the claimed pickup point, then each checked witness tests whether "
            "the driver is physically close enough to answer over BLE."
        )

        show_uncontacted = st.checkbox(
            "Show uncontacted claim-side witness candidates",
            value=False,
            key="spoof_show_uncontacted",
        )

        map_points = [[ev_lat, ev_lon], [actual_lat, actual_lon]]
        map_points.extend([[w.lat, w.lon] for w in checked_witnesses])
        if show_uncontacted:
            map_points.extend([[w.lat, w.lon] for w in uncontacted_witnesses])

        center_lat = sum(point[0] for point in map_points) / len(map_points)
        center_lon = sum(point[1] for point in map_points) / len(map_points)
        m2 = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
        folium.Circle(
            [ev_lat, ev_lon], radius=PARAMS.d_ble,
            color=C_TEAL, fill=True, fill_color=C_TEAL, fill_opacity=0.08, weight=2, dash_array="6 4",
            tooltip="Claim-side witness discovery zone (100 m)",
        ).add_to(m2)
        folium.Circle(
            [actual_lat, actual_lon], radius=PARAMS.d_ble,
            color=C_RED, fill=True, fill_color=C_RED, fill_opacity=0.04, weight=2, dash_array="4 6",
            tooltip="Driver's true BLE reach from actual position (100 m)",
        ).add_to(m2)
        folium.Marker(
            [ev_lat, ev_lon],
            popup=folium.Popup(
                f"<b>Claimed pickup location</b><br>"
                f"Lat/Lon: {ev_lat:.6f}, {ev_lon:.6f}<br>"
                f"Event time: {ev_time:.1f} s",
                max_width=280,
            ),
            tooltip="Claimed pickup location",
            icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
        ).add_to(m2)
        folium.Marker(
            [actual_lat, actual_lon],
            popup=folium.Popup(
                f"<b>Driver's true position</b><br>"
                f"Distance from claimed pickup: {attack_dist} m<br>"
                f"Bearing: {bearing_deg} deg",
                max_width=280,
            ),
            tooltip=f"Driver's true position ({attack_dist} m away)",
            icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
        ).add_to(m2)
        folium.PolyLine(
            [[ev_lat, ev_lon], [actual_lat, actual_lon]],
            color=C_RED, weight=2, dash_array="6 4",
            tooltip=f"Spoofing distance: {attack_dist} m",
        ).add_to(m2)

        if show_uncontacted:
            for w in uncontacted_witnesses:
                folium.CircleMarker(
                    [w.lat, w.lon],
                    radius=4.5,
                    color=C_TEAL,
                    fill=True,
                    fill_color=C_TEAL,
                    fill_opacity=0.22,
                    weight=1.2,
                    tooltip="Road-junction witness candidate near claimed pickup",
                ).add_to(m2)

        for w in checked_witnesses:
            chk = checked_by_id[w.id]
            status_text = "Accepted" if chk.accepted else f"Rejected: {reason_labels.get(chk.reason, chk.reason)}"
            color = "#27AE60" if chk.accepted else C_RED
            folium.CircleMarker(
                [w.lat, w.lon],
                radius=7 if chk.accepted else 6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1.8,
                popup=folium.Popup(
                    f"<b>Road-junction witness</b><br>"
                    f"ID: {w.id}<br>"
                    f"Claimed distance: {chk.claimed_distance_m:.1f} m<br>"
                    f"Actual distance: {chk.actual_distance_m:.1f} m<br>"
                    f"Result: {status_text}",
                    max_width=320,
                ),
                tooltip=status_text,
            ).add_to(m2)
            folium.PolyLine(
                [[actual_lat, actual_lon], [w.lat, w.lon]],
                color=color,
                weight=2 if chk.accepted else 1.7,
                opacity=0.75,
                dash_array=None if chk.accepted else "4 5",
                tooltip=f"Driver-to-witness distance: {chk.actual_distance_m:.1f} m",
            ).add_to(m2)

        lat_values = [point[0] for point in map_points]
        lon_values = [point[1] for point in map_points]
        m2.fit_bounds(
            [
                [min(lat_values) - 0.0015, min(lon_values) - 0.0025],
                [max(lat_values) + 0.0015, max(lon_values) + 0.0025],
            ]
        )

        legend_html = (
            "<div style='font-size:13px;line-height:1.7'>"
            "<span style='color:#2980b9'>●</span> Claimed pickup &nbsp;"
            "<span style='color:#e05555'>●</span> Driver's true position &nbsp;"
            "<span style='color:#27ae60'>●</span> Accepted witness &nbsp;"
            "<span style='color:#e05555'>●</span> Rejected witness &nbsp;"
            "<span style='color:#009B9E'>●</span> Uncontacted claim-side candidate"
            "</div>"
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        st_folium(m2, width=860, height=380)
        st.caption(
            "Blue marks the claimed pickup. Red marks the driver's true position. "
            "Colored witness points are road-junction witnesses that the protocol actually checked. "
            "Their color shows whether each witness accepted or rejected the claim."
        )

        if spoof_result.success:
            st.error("Spoofing SUCCEEDED — investigate parameters.")
        else:
            st.success(
                f"**Spoofing DETECTED** — proof generation failed.\n\n"
                f"Checked road-junction witnesses: **{len(spoof_result.witness_checks)}**\n\n"
                f"Main rejection reason: **`{reason_text}`**\n\n"
                f"The driver is actually {attack_dist} m away from the claimed pickup point. "
                f"The protocol queried witnesses near the claimed pickup, but those witnesses "
                f"could not validate the false location claim."
            )

        if spoof_result.witness_checks:
            attack_rows = []
            for chk in spoof_result.witness_checks:
                attack_rows.append(
                    {
                        "Witness": chk.witness_id[:18] + "…",
                        "Claimed dist (m)": f"{chk.claimed_distance_m:.1f}",
                        "Actual dist (m)": f"{chk.actual_distance_m:.1f}",
                        "Result": "Accept" if chk.accepted else f"Reject: {reason_labels.get(chk.reason, chk.reason)}",
                    }
                )
            st.dataframe(pd.DataFrame(attack_rows), use_container_width=True, hide_index=True)


    # ── Replay ─────────────────────────────────────────────────────────────
    with replay_tab:
        st.subheader("Location Replay: Reusing a valid old proof")
        st.markdown(
            "An attacker obtains a legitimate proof from a previous trip and "
            "tries to submit it again to claim payment for a ride that never happened."
        )

        random.seed(42)
        ctx_orig = {"order_id": str(ev["order_id"]), "event": "driver_arrival",
                    "session_id": "legitimate_001"}
        local_w = nearby_witnesses(ev_lat, ev_lon, all_witnesses, PARAMS.d_ble + 20)
        leg_result = generate_proof(
            prover_id=str(ev["order_id"]),
            actual_lat=ev_lat, actual_lon=ev_lon,
            claim_lat=ev_lat, claim_lon=ev_lon,
            timestamp=ev_time, ctx=ctx_orig, witnesses=local_w, params=PARAMS,
        )

        if not leg_result.success:
            st.warning("Could not generate a valid proof for this event (no witnesses nearby).")
        else:
            c_used: set[str] = set()
            accepted_events: set[str] = set()

            first_ok, first_reason = verify_proof(
                leg_result.proof, ev_lat, ev_lon,
                ev_time - 5, ev_time + 5,
                PARAMS, c_used, expected_ctx=ctx_orig, accepted_events=accepted_events,
            )
            second_ok, second_reason = verify_proof(
                leg_result.proof, ev_lat, ev_lon,
                ev_time - 5, ev_time + 5,
                PARAMS, c_used, expected_ctx=ctx_orig, accepted_events=accepted_events,
            )

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("#### Submission 1 (Legitimate)")
                if first_ok:
                    st.success(f"ACCEPTED — `{first_reason}`")
                else:
                    st.error(f"REJECTED — `{first_reason}`")

            with col_r2:
                st.markdown("#### Submission 2 (Replay Attempt)")
                if second_ok:
                    st.error("ACCEPTED — replay not detected!")
                else:
                    st.success(f"REJECTED — `{second_reason}`")
                    st.markdown(
                        "The verifier's `context_used` registry already contains "
                        "this proof's context key. Replay is blocked."
                    )

            st.markdown("---")
            st.markdown(
                "**Four anti-replay mechanisms in the protocol:**\n"
                "| Mechanism | Scenario |\n"
                "|---|---|\n"
                "| `context_replay` | Exact same proof submitted twice |\n"
                "| `event_accepted` | Same (order, event) in new session |\n"
                "| `context_binding` | Proof from event A used to claim event B |\n"
                "| `temporal` | Proof submitted after validity window expired |"
            )

    # ── Relay ────────────────────────────────────────────────────────────
    with relay_tab:
        st.subheader("Relay / Collusion: Forwarding BLE challenges across the city")
        st.markdown(
            "The attacker is physically absent from the event site. A colluder "
            "at the site forwards BLE challenges over a relay path, adding extra "
            "latency. If that latency exceeds `Δt_max`, the witness rejects the "
            "response and the proof fails."
        )

        col_ctrl1, col_ctrl2 = st.columns(2)
        dt_ms_val = col_ctrl1.slider("Δt_max threshold (ms)", 10, 120, 35, 5)
        relay_mode_label = col_ctrl2.radio(
            "Relay path",
            ["BLE + BLE", "BLE + WiFi/LTE"],
            horizontal=True,
        )
        relay_mode = "ble_ble" if relay_mode_label == "BLE + BLE" else "ble_wifi"

        relay_params = ProtocolParams(
            d_ble=100.0, d_th=100.0, delta_t_max=dt_ms_val / 1000.0,
            tau_rssi=-90.0, n_w=1,
        )
        witness = all_witnesses[5]
        actual_lat2, actual_lon2 = offset_point(witness.lat, witness.lon, 8, 0.3)
        ctx2 = {"order_id": "relay_case", "event": "driver_arrival", "session_id": "relay_001"}

        random.seed(12)
        honest_result = generate_proof(
            prover_id="honest_driver",
            actual_lat=actual_lat2, actual_lon=actual_lon2,
            claim_lat=actual_lat2, claim_lon=actual_lon2,
            timestamp=1000.0, ctx=ctx2, witnesses=[witness],
            params=relay_params, is_relay=False,
        )

        random.seed(13)
        relay_result = generate_proof(
            prover_id="relay_attacker",
            actual_lat=actual_lat2, actual_lon=actual_lon2,
            claim_lat=actual_lat2, claim_lon=actual_lon2,
            timestamp=1000.0, ctx=ctx2, witnesses=[witness],
            params=relay_params, is_relay=True, relay_mode=relay_mode,
        )

        col_relay1, col_relay2 = st.columns(2)
        with col_relay1:
            st.markdown("#### Honest Prover (direct BLE)")
            if honest_result.success:
                chk = honest_result.witness_checks[0]
                rt_ms = chk.response_time_s * 1000 if chk.response_time_s else 0.0
                st.success(f"ACCEPTED — RTT = {rt_ms:.2f} ms (≤ {dt_ms_val} ms)")
            else:
                chk = honest_result.witness_checks[0] if honest_result.witness_checks else None
                rt_ms = chk.response_time_s * 1000 if (chk and chk.response_time_s) else 0.0
                st.error(f"REJECTED — RTT = {rt_ms:.2f} ms")

        with col_relay2:
            st.markdown(f"#### Relay Attacker ({relay_mode_label})")
            if relay_result.success:
                chk = relay_result.witness_checks[0]
                rt_ms = chk.response_time_s * 1000 if chk.response_time_s else 0.0
                st.error(
                    f"NOT detected — RTT = {rt_ms:.2f} ms ≤ {dt_ms_val} ms. "
                    f"Consider lowering Δt_max."
                )
            else:
                chk = relay_result.witness_checks[0] if relay_result.witness_checks else None
                rt_ms = chk.response_time_s * 1000 if (chk and chk.response_time_s) else 0.0
                extra_ms = rt_ms - (honest_result.witness_checks[0].response_time_s * 1000
                                    if honest_result.witness_checks and
                                    honest_result.witness_checks[0].response_time_s else 0.0)
                st.success(
                    f"BLOCKED — RTT = {rt_ms:.2f} ms > {dt_ms_val} ms "
                    f"(+{extra_ms:.1f} ms relay overhead)."
                )

        st.markdown(
            "Try raising `Δt_max` above ~45 ms to see where the relay starts "
            "succeeding — this shows the security-usability trade-off."
        )

        try:
            s3_df = load_s3_csv()
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(s3_df["dt_ms"], s3_df["lar"] * 100, "o-", color=C_TEAL, label="LAR (Honest)")
            ax.plot(s3_df["dt_ms"], s3_df["arr_ble_ble"] * 100, "s--", color=C_MAGENTA, label="ARR (BLE+BLE)")
            ax.plot(s3_df["dt_ms"], s3_df["arr_ble_wifi"] * 100, "^--", color=C_MINT, label="ARR (BLE+WiFi/LTE)")
            ax.axvline(dt_ms_val, color="gray", linestyle=":", alpha=0.7, label=f"Current Δt_max = {dt_ms_val} ms")
            ax.set_xlabel("Δt_max (ms)")
            ax.set_ylabel("Rate (%)")
            ax.set_title("Security-Usability Trade-off (S3 Results)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# Page 4 — Experiment Results
# ════════════════════════════════════════════════════════════════════════════


def _derive_key_findings() -> list[tuple[str, str, str]]:
    """Read key numerical findings from CSVs — no hardcoding."""
    rows = []

    # S1 — report the farthest tested-distance detection rate from CSV
    try:
        s1 = load_s1_csv().copy()
        s1["actual_distance_m"] = s1["actual_distance_m"].astype(float)
        s1["detection_rate"] = s1["detection_rate"].astype(float)
        farthest = s1.sort_values("actual_distance_m").iloc[-1]
        rows.append((
            "GPS spoofing detection at farthest tested distance",
            f"{farthest['detection_rate']*100:.1f}% at {int(farthest['actual_distance_m']):,} m",
            "S1",
        ))
    except Exception:
        rows.append(("GPS spoofing detection rate", "CSV not found", "S1"))

    # S2 — report the worst-case replay rejection rate from CSV
    try:
        s2 = load_s2_csv().copy()
        s2["rejection_rate"] = s2["rejection_rate"].astype(float)
        worst_case = s2["rejection_rate"].min() * 100
        rows.append(("Replay rejection rate (worst case)", f"{worst_case:.1f}%", "S2"))
    except Exception:
        rows.append(("Replay rejection rate", "CSV not found", "S2"))

    # S3 — relay detection at dt_max=35ms, read from CSV
    try:
        s3 = load_s3_csv()
        row35 = s3[s3["dt_ms"] == 35]
        if not row35.empty:
            arr_val = float(row35["arr_ble_ble"].iloc[0]) * 100
            lar_val = float(row35["lar"].iloc[0]) * 100
            rows.append((
                "Relay detection BLE+BLE (Δt_max=35 ms)",
                f"{arr_val:.1f}%  (LAR={lar_val:.1f}%)",
                "S3",
            ))
        else:
            rows.append(("Relay detection BLE+BLE (Δt_max=35 ms)", "see S3 plot", "S3"))
    except Exception:
        rows.append(("Relay detection BLE+BLE (Δt_max=35 ms)", "CSV not found", "S3"))

    # F1 — min witnesses for 95% PSR (grid strategy), read from CSV
    try:
        f1 = load_f1_csv()
        grid = f1[f1["strategy"] == "grid"].copy()
        grid["overall_psr"] = grid["overall_psr"].astype(float)
        threshold = grid[grid["overall_psr"] >= 0.95]
        if not threshold.empty:
            best = threshold.sort_values("witness_count").iloc[0]
            rows.append((
                "Min witnesses for 95% PSR (Tartu, grid)",
                f"{int(best['witness_count']):,}  (PSR={best['overall_psr']*100:.1f}%)",
                "F1",
            ))
        else:
            best = grid.sort_values("overall_psr", ascending=False).iloc[0]
            rows.append((
                "Best PSR achieved (Tartu, grid)",
                f"{best['overall_psr']*100:.1f}% at {int(best['witness_count']):,} witnesses",
                "F1",
            ))
    except Exception:
        rows.append(("Min witnesses for 95% PSR (Tartu, grid)", "CSV not found", "F1"))

    # F3 — gas per proof at T_batch=15 min, read from CSV
    try:
        f3 = load_f3_csv()
        row15 = f3[f3["batch_minutes"] == 15]
        if not row15.empty:
            gas = float(row15["gas_per_proof"].iloc[0])
            rows.append((
                "Gas per proof at T_batch=15 min",
                f"{gas:.0f} gas",
                "F3",
            ))
        else:
            rows.append(("Gas per proof at T_batch=15 min", "see F3 plot", "F3"))
    except Exception:
        rows.append(("Gas per proof at T_batch=15 min", "CSV not found", "F3"))

    # S4 — Merkle audit complexity
    rows.append(("Merkle audit complexity", "O(log n)", "S4"))

    return rows


def _f3_caption() -> str:
    """Build F3 figure caption from CSV data — never hardcode build time."""
    try:
        f3 = load_f3_csv()
        max_row = f3.loc[f3["avg_build_time_ms"].idxmax()]
        min_row = f3.loc[f3["avg_build_time_ms"].idxmin()]
        max_ms = float(max_row["avg_build_time_ms"])
        max_min = int(max_row["batch_minutes"])
        min_ms = float(min_row["avg_build_time_ms"])
        min_min = int(min_row["batch_minutes"])
        above_1ms = f3[f3["avg_build_time_ms"] > 1.0]
        if above_1ms.empty:
            build_note = f"Merkle build time ranges from {min_ms:.3f} ms ({min_min} min) to {max_ms:.3f} ms ({max_min} min)."
        else:
            threshold_min = int(above_1ms["batch_minutes"].min())
            build_note = (
                f"Merkle build time ranges from {min_ms:.3f} ms ({min_min} min) to "
                f"{max_ms:.2f} ms ({max_min} min); exceeds 1 ms only at "
                f"≥ {threshold_min} min batch intervals."
            )
        return (
            "Gas cost per proof decreases as batch interval grows (more proofs amortise "
            f"the fixed anchorBatch gas). Right panel shows {build_note}"
        )
    except Exception:
        return (
            "Gas cost per proof decreases as batch interval grows (more proofs amortise "
            "the fixed anchorBatch gas). Right panel shows Merkle build time."
        )


def _f1_caption() -> str:
    """Build F1 figure caption from CSV data."""
    try:
        f1 = load_f1_csv()
        grid = f1[f1["strategy"] == "grid"].copy()
        grid["overall_psr"] = grid["overall_psr"].astype(float)
        threshold = grid[grid["overall_psr"] >= 0.95]
        if not threshold.empty:
            best = threshold.sort_values("witness_count").iloc[0]
            return (
                f"Three deployment strategies (Random / Uniform Grid / Heatmap-guided) "
                f"evaluated at {len(grid)} density levels. "
                f"Uniform Grid first reaches 95% PSR at "
                f"{int(best['witness_count']):,} witnesses "
                f"(PSR = {best['overall_psr']*100:.1f}%)."
            )
    except Exception:
        pass
    return (
        "Three deployment strategies (Random / Uniform Grid / Heatmap-guided) "
        "evaluated across density levels."
    )


def page_results() -> None:
    st.title("Experiment Results")

    FIGURES = [
        ("F1 — Witness Density vs Proof Success Rate",
         "f1_psr_vs_density.png",
         _f1_caption()),
        ("F2 — Witness Load Heatmaps",
         "f2_witness_load_heatmaps.png",
         "Spatial distribution of proof requests under 4 concurrency levels "
         "(≈10, 50, 100, 200 concurrent vehicles). Hotspots emerge near the city centre."),
        ("F2 — Load Distribution vs Concurrency",
         "f2_witness_load_distribution.png",
         "Max / P95 / Mean hourly request rates per witness as vehicle concurrency grows."),
        ("F3 — Batch Anchoring Trade-off",
         "f3_batch_anchoring_tradeoff.png",
         _f3_caption()),
        ("S1 — GPS Spoofing Resistance",
         "s1_gps_spoofing.png",
         "Distance-scan view of the BLE security boundary across 15 test distances from 5 m to 5 km. "
         "Detection stays low at short range, rises sharply near the nominal BLE edge, and approaches 100% "
         "for clearly out-of-range attackers."),
        ("S2 — Replay Resistance",
         "s2_replay_rejection.png",
         "All four replay scenarios are rejected at ≥99% rate by the corresponding "
         "protocol mechanism."),
        ("S3 — Relay Detection & Δt_max Calibration",
         "s3_relay_dt_tradeoff.png",
         "LAR vs ARR trade-off. Δt_max ≈ 30–40 ms achieves >95% LAR while "
         "blocking most BLE+BLE relay attempts."),
        ("S4 — On-chain Audit Closure",
         "s4_audit_closure.png",
         "Merkle path length grows as ⌈log₂ n⌉, confirming O(log n) audit complexity."),
        ("S5 — Baseline Comparison",
         "s5_baseline_comparison.png",
         "Proposed protocol eliminates spoofing and replay attack success entirely; "
         "GPS-only baseline is fully vulnerable."),
    ]

    for title, filename, caption in FIGURES:
        img_path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(img_path):
            with st.expander(title, expanded=False):
                st.image(img_path, use_container_width=True)
                st.caption(caption)
        else:
            with st.expander(f"{title} (not yet generated)", expanded=False):
                st.warning(
                    f"`{filename}` not found. Run `python scripts/05_run_experiments.py` first."
                )

    st.divider()
    st.subheader("Key Numerical Findings")
    findings = _derive_key_findings()
    st.dataframe(
        pd.DataFrame(findings, columns=["Metric", "Value", "Source"]),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main — Navigation
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(
        page_title="PoL Simulation Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("PoL Simulation Dashboard")
    st.sidebar.caption("Proof-of-Location · Ride-hailing · Tartu, Estonia")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Data Pipeline",
            "System Overview",
            "Protocol Walkthrough",
            "Attack Scenarios",
            "Experiment Results",
        ],
    )

    if page == "Data Pipeline":
        page_pipeline()
    elif page == "System Overview":
        page_overview()
    elif page == "Protocol Walkthrough":
        page_protocol()
    elif page == "Attack Scenarios":
        page_attacks()
    elif page == "Experiment Results":
        page_results()

    st.sidebar.divider()
    st.sidebar.caption(
        "Built with [Streamlit](https://streamlit.io) + "
        "[Folium](https://python-visualization.github.io/folium/)"
    )


if __name__ == "__main__":
    main()
