# Proof-of-Location Simulation — Tartu

Simulation and evaluation suite for a Proof-of-Location (PoL) protocol,
evaluated on a SUMO-based ride-hailing workload over the road network of
Tartu, Estonia. The suite covers five security experiments (S1–S5) and three
feasibility experiments (F1–F3), as reported in Chapter 4 of the thesis.

---

## Requirements

**Python 3.10 or later**

```bash
pip install -r requirements.txt
```

**SUMO 1.18 or later** — required for steps 1–3 (network build, trip
generation, trace extraction). Step 5 and the dashboard use `sumolib` for
geometry only and do not run a live SUMO instance.

```bash
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

---

## Quick Start

All simulation inputs (network, routes, traces, witness pool) are already
committed in `tartu/` and `results/`. To reproduce every figure in the thesis,
only step 5 is needed:

```bash
python3 scripts/05_run_experiments.py
```

Figures and data are written to `results/`.

---

## Full Pipeline

| Step | Script | Requires SUMO | Can skip |
|------|--------|---------------|----------|
| 1. Build road network | `scripts/01_build_network.sh` | yes | yes — `tartu/tartu.net.xml` already present |
| 2. Generate trip routes | `scripts/02_generate_trips.sh` | yes | yes — route files already present |
| 3. Extract event traces | `scripts/03_extract_traces.py` | yes | yes — trace CSVs already present |
| 4. Deploy witness pool | `scripts/04_deploy_witnesses.py` | geometry only | yes — `results/witnesses.csv` already present |
| 5. Run experiments | `scripts/05_run_experiments.py` | no | no |

### Commands (if re-running from scratch)

```bash
# Step 1 — build road network
bash scripts/01_build_network.sh

# Step 2 — generate trip routes
bash scripts/02_generate_trips.sh

# Step 3 — extract event traces (one command per scenario)
python3 scripts/03_extract_traces.py
python3 scripts/03_extract_traces.py --routes tartu/routes_c10.xml  --label c10
python3 scripts/03_extract_traces.py --routes tartu/routes_c50.xml  --label c50
python3 scripts/03_extract_traces.py --routes tartu/routes_c100.xml --label c100
python3 scripts/03_extract_traces.py --routes tartu/routes_c200.xml --label c200

# Step 4 — generate witness candidate pool
python3 scripts/04_deploy_witnesses.py
```

### Step 5 options

Run a subset of experiments:

```bash
python3 scripts/05_run_experiments.py --experiments s1 s2 s3 s4 s5
python3 scripts/05_run_experiments.py --experiments f1 f2 f3
```

Key parameters and defaults:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Base random seed |
| `--s1-distances-m` | 5 20 … 5000 | Attacker distances for S1 (m) |
| `--s1-repeats` | 30 | Repeated runs per S1 distance |
| `--s1-repeat-trials` | 30 | Trials per S1 repeated run |
| `--security-trials` | 200 | Trials for S2, S3, S4, S5 |
| `--dt-values-ms` | 10 15 … 120 | Delta_t_max sweep for S3 (ms) |
| `--f1-counts` | 2000 … 12000 | Witness counts for F1 |
| `--batch-minutes` | 1 5 15 30 60 | Batch intervals for F3 (minutes) |

---

## Experiments

### Security

| ID | Description | Key output |
|----|-------------|------------|
| S1 | GPS spoofing resistance — detection rate vs attacker distance (5 m to 5 km), 30 x 30 repeated trials, repeat-based 95% CI | `s1_gps_spoofing.pdf` |
| S2 | Replay resistance — identical-context, shifted-context, and shifted-spatial replay attempts | `s2_replay_rejection.pdf` |
| S3 | Relay attack — LAR and ARR across Delta_t_max 10–120 ms for BLE+BLE and BLE+WiFi/LTE | `s3_relay_dt_tradeoff.pdf` |
| S4 | On-chain tamper detection — six audit scenarios; Merkle path length vs ceil(log2 n) | `s4_audit_closure.pdf` |
| S5 | Baseline comparison — PoL vs GPS-only and simple PoL | `s5_baseline_comparison.pdf` |

### Feasibility

| ID | Description | Key output |
|----|-------------|------------|
| F1 | Proof success rate vs witness count (2 k–12 k), three deployment strategies | `f1_psr_vs_density.pdf` |
| F2 | Per-witness request load under four concurrency levels (c10/c50/c100/c200) | `f2_witness_load_heatmaps.pdf` |
| F3 | Batch anchoring gas cost vs latency across batch intervals 1–60 min | `f3_batch_anchoring_tradeoff.pdf` |

---

## Reproducibility

All random processes are seeded deterministically via a SHA-256-based function
keyed on the base seed, experiment name, and trial index. Results are identical
across platforms for the same seed.

The only exception is `avg_elapsed_ms` in `results/s4_audit_complexity.csv`,
which measures wall-clock audit time and varies with CPU performance.

---

## Dashboard

The dashboard is an interactive companion to the experiment suite, intended
for protocol exploration and result inspection without re-running the full
pipeline. It has five pages:

- **Data Pipeline** — walkthrough of each script, showing actual output data
  and explaining every processing step.
- **System Overview** — map of the Tartu witness deployment, concurrency
  statistics, and trip-event summaries.
- **Protocol Walkthrough** — live proof generation with adjustable parameters
  (d_ble, d_th, Delta_t_max, N_w). Shows which witnesses are queried and
  whether each attestation passes.
- **Attack Scenarios** — interactive simulation of GPS spoofing and relay
  attacks at user-defined distances and delay values.
- **Experiment Results** — charts for S1, S2, S3, F1, F3 rendered from the
  pre-computed result files in `results/`.

```bash
cd simulator_dashboard
streamlit run app.py
```

---

## Simulation Notes

- Rider presence times at pickup and dropoff are sampled uniformly within the
  protocol window. SUMO models vehicles only, not passengers.
- Witness candidates are placed on road-network junctions. This is a
  deployment abstraction, not a measured infrastructure inventory.
- Trip demand uses SUMO's `randomTrips.py` with uniform origin-destination
  sampling and does not reflect hotspot-heavy ride-hailing patterns.
- S1 does not inject GPS noise. Residual detection below the BLE boundary
  arises from RSSI variability and the hard BLE reachability cutoff.
