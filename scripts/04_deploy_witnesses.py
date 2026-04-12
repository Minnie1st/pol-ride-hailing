#!/usr/bin/env python3
"""
04_deploy_witnesses.py
======================
Generate the reusable witness candidate pool for PoL experiments.

This script does not implement the deployment strategies compared in F1.
Instead, it produces a reproducible witness candidate set from the road network.
The experiment runner (05_run_experiments.py) then derives the compared
deployment strategies from this shared candidate pool:

    - random
    - uniform grid
    - heatmap-guided

Candidate-generation simplification:
    We place witness candidates on road-network junctions with at least a
    configurable number of incoming edges. This is a deployment abstraction,
    not a measured infrastructure inventory, and should be described as such in
    the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

if "SUMO_HOME" not in os.environ:
    sys.exit("ERROR: SUMO_HOME is not set. Please source your SUMO environment.")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_NET = os.path.join(PROJECT_ROOT, "tartu", "tartu.net.xml")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "results", "witnesses.csv")
DEFAULT_SUMMARY = os.path.join(PROJECT_ROOT, "results", "witnesses_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible witness candidate pool from the SUMO road network."
    )
    parser.add_argument(
        "--net-file",
        default=DEFAULT_NET,
        help="Path to the SUMO .net.xml road network.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path for witness candidates.",
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
        help="Output JSON path for deployment metadata.",
    )
    parser.add_argument(
        "--max-witnesses",
        type=int,
        default=15000,
        help="Maximum number of witness candidates to keep after filtering.",
    )
    parser.add_argument(
        "--min-incoming",
        type=int,
        default=1,
        help="Minimum number of incoming edges required for a node to become a candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used only when candidate down-sampling is needed.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def deploy_on_junctions(
    net_file: str,
    max_witnesses: int,
    min_incoming: int,
    seed: int,
) -> tuple[list[dict], dict]:
    net = sumolib.net.readNet(net_file)
    incoming_threshold_counts = {str(threshold): 0 for threshold in range(1, 5)}

    all_candidates = []
    for node in net.getNodes():
        incoming_degree = len(node.getIncoming())
        for threshold in range(1, 5):
            if incoming_degree >= threshold:
                incoming_threshold_counts[str(threshold)] += 1

        if incoming_degree < min_incoming:
            continue
        x, y = node.getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        all_candidates.append(
            {
                "id": f"W_{node.getID()}",
                "lat": lat,
                "lon": lon,
                "incoming_degree": incoming_degree,
            }
        )

    all_candidates.sort(key=lambda row: (row["lat"], row["lon"], row["id"]))
    print(f"Found {len(all_candidates)} candidate junctions after filtering")

    if max_witnesses < len(all_candidates):
        rng = random.Random(seed)
        selected = rng.sample(all_candidates, max_witnesses)
        selected.sort(key=lambda row: (row["lat"], row["lon"], row["id"]))
        sampling_mode = "seeded_random_downsample"
    else:
        selected = list(all_candidates)
        sampling_mode = "all_candidates"

    lats = [row["lat"] for row in selected]
    lons = [row["lon"] for row in selected]
    summary = {
        "net_file": net_file,
        "seed": seed,
        "min_incoming": min_incoming,
        "max_witnesses": max_witnesses,
        "network_node_count": len(net.getNodes()),
        "incoming_threshold_counts": incoming_threshold_counts,
        "candidate_count_before_sampling": len(all_candidates),
        "candidate_count_after_sampling": len(selected),
        "sampling_mode": sampling_mode,
        "bounding_box": {
            "lat_min": min(lats) if lats else None,
            "lat_max": max(lats) if lats else None,
            "lon_min": min(lons) if lons else None,
            "lon_max": max(lons) if lons else None,
        },
    }
    return selected, summary


def save_witnesses(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "lat", "lon"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                }
            )


def save_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    args.net_file = resolve_path(args.net_file)
    args.output = resolve_path(args.output)
    args.summary = resolve_path(args.summary)

    if not os.path.exists(args.net_file):
        sys.exit(f"ERROR: Network file not found: {args.net_file}")

    print("=" * 58)
    print(" Witness Candidate Generator")
    print("=" * 58)
    print(f"  Network       : {args.net_file}")
    print(f"  Min incoming  : {args.min_incoming}")
    print(f"  Max witnesses : {args.max_witnesses}")
    print(f"  Seed          : {args.seed}")
    print(f"  Output CSV    : {args.output}")
    print(f"  Output JSON   : {args.summary}")
    print()

    witnesses, summary = deploy_on_junctions(
        net_file=args.net_file,
        max_witnesses=args.max_witnesses,
        min_incoming=args.min_incoming,
        seed=args.seed,
    )

    save_witnesses(witnesses, args.output)
    save_summary(summary, args.summary)

    print(f"Saved {len(witnesses)} witness candidates to {args.output}")
    print(f"Saved candidate summary to {args.summary}")


if __name__ == "__main__":
    main()
