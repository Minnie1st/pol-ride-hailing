#!/bin/bash
# =============================================================================
# 02_generate_trips.sh
#
# Generates SUMO trip/route files for PoL experiments in Tartu.
#
# Modeling note:
#   SUMO's randomTrips.py generates synthetic origin-destination pairs rather
#   than empirically calibrated ride-hailing demand. For F1/F2 this is an
#   accepted simplification: it yields a reproducible city-wide workload, but
#   it does not capture hotspot-heavy demand such as city-centre concentration.
#   This limitation should be stated explicitly in the paper.
#
# Output files and their experiment roles:
#   routes_main.xml   → F1: fixed trip set, varying witness density
#   routes_c10.xml    → F2: ~10  concurrent active vehicles
#   routes_c50.xml    → F2: ~50  concurrent active vehicles
#   routes_c100.xml   → F2: ~100 concurrent active vehicles
#   routes_c200.xml   → F2: ~200 concurrent active vehicles
#
# Concurrency estimation:
#   Concurrent vehicles ≈ T_avg / p
#   where T_avg = average trip duration (assumed ~600 s in Tartu urban network)
#   and p = inter-departure period in seconds (-p flag).
#
#   p=60  → ~10  concurrent
#   p=12  → ~50  concurrent
#   p=6   → ~100 concurrent
#   p=3   → ~200 concurrent
#
# The actual peak concurrency is reported by 03_extract_traces.py.
# Adjust -p if the reported peak deviates significantly from the target.
# Each file is generated with a fixed seed to keep the experiment inputs stable
# across reruns and across witness deployment strategies.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARTU_DIR="$PROJECT_ROOT/tartu"
NET_FILE="$TARTU_DIR/tartu.net.xml"
SIM_DURATION=3600   # seconds (1 hour)

# Verify SUMO_HOME is set
if [ -z "${SUMO_HOME:-}" ]; then
    echo "ERROR: SUMO_HOME is not set. Please source your SUMO environment."
    exit 1
fi

if [ ! -f "$NET_FILE" ]; then
    echo "ERROR: Network file not found: $NET_FILE"
    exit 1
fi

RANDOM_TRIPS="$SUMO_HOME/tools/randomTrips.py"

echo "=============================================="
echo " PoL Trip Generator — Tartu"
echo "=============================================="
echo " Network : $NET_FILE"
echo " Duration: ${SIM_DURATION}s"
echo ""

# ------------------------------------------------------------------------------
# Helper: generate one route file
# Args: label  p_value  seed  description
# ------------------------------------------------------------------------------
generate() {
    local label="$1"
    local p_val="$2"
    local seed="$3"
    local desc="$4"

    local trips_out="$TARTU_DIR/trips_${label}.xml"
    local routes_out="$TARTU_DIR/routes_${label}.xml"

    echo "[${label}] ${desc} (p=${p_val}s, seed=${seed})..."

    python3 "$RANDOM_TRIPS" \
        -n  "$NET_FILE"   \
        -o  "$trips_out"  \
        -r  "$routes_out" \
        -e  "$SIM_DURATION" \
        -p  "$p_val"      \
        --seed "$seed"    \
        --validate        \
        --random-depart   \
        2>/dev/null       \
    && echo "    -> $routes_out  [OK]" \
    || { echo "    -> FAILED (see above)"; exit 1; }
}

# ------------------------------------------------------------------------------
# F1 baseline: denser fixed trip set for witness deployment experiments
# p=7 → about 514 departures over 1 hour before trip-quality filtering
# ------------------------------------------------------------------------------
generate "main" 7 4201 "F1 baseline (~500 departures, fixed trip set)"

# ------------------------------------------------------------------------------
# F2 concurrency levels
# ------------------------------------------------------------------------------
generate "c10"  60 4202 "F2 ~10  concurrent vehicles"
generate "c50"  12 4203 "F2 ~50  concurrent vehicles"
generate "c100" 6  4204 "F2 ~100 concurrent vehicles"
generate "c200" 3  4205 "F2 ~200 concurrent vehicles"

echo ""
echo "=============================================="
echo " All route files generated:"
echo "   F1  : tartu/routes_main.xml"
echo "   F2  : tartu/routes_c10/c50/c100/c200.xml"
echo ""
echo " Next step: python3 scripts/03_extract_traces.py"
echo "=============================================="
