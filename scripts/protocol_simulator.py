import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class ProtocolParams:
    d_ble: float = 100.0
    d_th: float = 100.0
    delta_t_max: float = 0.035
    tau_rssi: float = -90.0
    n_w: int = 4
    delta_t_valid: float = 300.0
    r: float = 100.0
    ble_roundtrip_mean: float = 0.015
    ble_roundtrip_std: float = 0.005
    sign_time: float = 0.003
    relay_wifi_min: float = 0.020
    relay_wifi_max: float = 0.100


@dataclass
class WitnessNode:
    id: str
    lat: float
    lon: float


@dataclass
class LocationClaim:
    prover_id: str
    lat: float
    lon: float
    timestamp: float


@dataclass
class WitnessAttestation:
    witness_id: str
    witness_lat: float
    witness_lon: float
    nonce: str
    timestamp: float
    ctx: dict
    h_lc: str


@dataclass
class LocationProof:
    ctx: dict
    claim: LocationClaim
    attestations: List[WitnessAttestation]


@dataclass
class WitnessCheck:
    witness_id: str
    accepted: bool
    reason: str
    actual_distance_m: float
    claimed_distance_m: float
    response_time_s: Optional[float] = None
    rssi_dbm: Optional[float] = None


@dataclass
class VerificationResult:
    accepted: bool
    reason: str


@dataclass
class ProofResult:
    success: bool
    candidate_count: int = 0
    nearby_count: int = 0
    attestations_obtained: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    witness_checks: List[WitnessCheck] = field(default_factory=list)
    proof: Optional[LocationProof] = None


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sample_truncated_normal(mean: float, std: float, lower: float = 0.0) -> float:
    while True:
        value = random.gauss(mean, std)
        if value >= lower:
            return value


def sample_ble_roundtrip(params: ProtocolParams) -> float:
    return sample_truncated_normal(
        params.ble_roundtrip_mean,
        params.ble_roundtrip_std,
        lower=0.0,
    )


def sample_relay_extra_delay(
    params: ProtocolParams,
    relay_mode: str = "ble_ble",
    relay_extra_delay: Optional[float] = None,
) -> float:
    if relay_extra_delay is not None:
        return relay_extra_delay
    if relay_mode == "ble_ble":
        return sample_ble_roundtrip(params) + sample_ble_roundtrip(params)
    if relay_mode == "ble_wifi":
        return random.uniform(params.relay_wifi_min, params.relay_wifi_max)
    raise ValueError(f"Unsupported relay mode: {relay_mode}")


def simulate_rssi(distance: float) -> float:
    if distance < 1:
        distance = 1
    base_rssi = -40
    path_loss_exp = 2.5
    rssi = base_rssi - 10 * path_loss_exp * math.log10(distance)
    rssi += random.gauss(0, 3)
    return rssi


def simulate_ble_delay(
    params: ProtocolParams,
    is_relay: bool = False,
    relay_mode: str = "ble_ble",
    relay_extra_delay: Optional[float] = None,
) -> float:
    # Physical propagation is negligible at these distances. The timing budget is
    # modeled as BLE RTT plus signing time, with relay overhead added on top.
    delay = sample_ble_roundtrip(params) + params.sign_time
    if is_relay:
        delay += sample_relay_extra_delay(params, relay_mode, relay_extra_delay)
    return delay


def _candidate_witnesses(
    actual_lat: float,
    actual_lon: float,
    claim_lat: float,
    claim_lon: float,
    witnesses: Sequence[WitnessNode],
    params: ProtocolParams,
    candidate_mode: str,
) -> List[WitnessNode]:
    if candidate_mode == "actual":
        center_lat, center_lon = actual_lat, actual_lon
    elif candidate_mode == "claim":
        center_lat, center_lon = claim_lat, claim_lon
    elif candidate_mode == "union":
        candidates = []
        for witness in witnesses:
            actual_dist = haversine(actual_lat, actual_lon, witness.lat, witness.lon)
            claim_dist = haversine(claim_lat, claim_lon, witness.lat, witness.lon)
            if min(actual_dist, claim_dist) <= params.d_ble:
                candidates.append((min(actual_dist, claim_dist), witness))
        return [witness for _, witness in sorted(candidates, key=lambda item: item[0])]
    else:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    candidates = []
    for witness in witnesses:
        distance = haversine(center_lat, center_lon, witness.lat, witness.lon)
        if distance <= params.d_ble:
            candidates.append((distance, witness))
    return [witness for _, witness in sorted(candidates, key=lambda item: item[0])]


def witness_verify(
    witness: WitnessNode,
    claim_lat: float,
    claim_lon: float,
    actual_lat: float,
    actual_lon: float,
    params: ProtocolParams,
    is_relay: bool = False,
    relay_mode: str = "ble_ble",
    relay_extra_delay: Optional[float] = None,
) -> Tuple[bool, WitnessCheck]:
    actual_distance = haversine(actual_lat, actual_lon, witness.lat, witness.lon)
    claimed_distance = haversine(claim_lat, claim_lon, witness.lat, witness.lon)

    if actual_distance > params.d_ble:
        return False, WitnessCheck(
            witness_id=witness.id,
            accepted=False,
            reason="ble_unreachable",
            actual_distance_m=actual_distance,
            claimed_distance_m=claimed_distance,
        )

    response_time = simulate_ble_delay(
        params,
        is_relay=is_relay,
        relay_mode=relay_mode,
        relay_extra_delay=relay_extra_delay,
    )
    if response_time > params.delta_t_max:
        return False, WitnessCheck(
            witness_id=witness.id,
            accepted=False,
            reason="time_bound",
            actual_distance_m=actual_distance,
            claimed_distance_m=claimed_distance,
            response_time_s=response_time,
        )

    if claimed_distance > params.d_th:
        return False, WitnessCheck(
            witness_id=witness.id,
            accepted=False,
            reason="distance",
            actual_distance_m=actual_distance,
            claimed_distance_m=claimed_distance,
            response_time_s=response_time,
        )

    rssi = simulate_rssi(actual_distance)
    if rssi < params.tau_rssi:
        return False, WitnessCheck(
            witness_id=witness.id,
            accepted=False,
            reason="rssi",
            actual_distance_m=actual_distance,
            claimed_distance_m=claimed_distance,
            response_time_s=response_time,
            rssi_dbm=rssi,
        )

    return True, WitnessCheck(
        witness_id=witness.id,
        accepted=True,
        reason="pass",
        actual_distance_m=actual_distance,
        claimed_distance_m=claimed_distance,
        response_time_s=response_time,
        rssi_dbm=rssi,
    )


def generate_proof(
    prover_id: str,
    actual_lat: float,
    actual_lon: float,
    claim_lat: float,
    claim_lon: float,
    timestamp: float,
    ctx: dict,
    witnesses: Sequence[WitnessNode],
    params: ProtocolParams,
    is_relay: bool = False,
    relay_extra_delay: Optional[float] = None,
    relay_mode: str = "ble_ble",
    candidate_mode: str = "actual",
) -> ProofResult:
    result = ProofResult(success=False)

    candidates = _candidate_witnesses(
        actual_lat,
        actual_lon,
        claim_lat,
        claim_lon,
        witnesses,
        params,
        candidate_mode=candidate_mode,
    )
    result.candidate_count = len(candidates)

    nearby = []
    for witness in witnesses:
        if haversine(actual_lat, actual_lon, witness.lat, witness.lon) <= params.d_ble:
            nearby.append(witness)
    result.nearby_count = len(nearby)

    attestations = []
    for witness in candidates:
        passed, check = witness_verify(
            witness,
            claim_lat,
            claim_lon,
            actual_lat,
            actual_lon,
            params,
            is_relay=is_relay,
            relay_mode=relay_mode,
            relay_extra_delay=relay_extra_delay,
        )
        result.witness_checks.append(check)
        if passed:
            attestations.append(
                WitnessAttestation(
                    witness_id=witness.id,
                    witness_lat=witness.lat,
                    witness_lon=witness.lon,
                    nonce=f"nc_{random.randint(0, 999999):06d}",
                    timestamp=timestamp + random.uniform(0.005, 0.030),
                    ctx=dict(ctx),
                    h_lc=f"hlc_{random.randint(0, 999999):06d}",
                )
            )
        else:
            result.rejection_reasons.append(check.reason)

        if len(attestations) >= params.n_w:
            break

    result.attestations_obtained = len(attestations)

    if len(attestations) < params.n_w and not result.rejection_reasons:
        result.rejection_reasons.append("insufficient_witnesses")

    if len(attestations) >= params.n_w:
        result.success = True
        result.proof = LocationProof(
            ctx=dict(ctx),
            claim=LocationClaim(
                prover_id=prover_id,
                lat=claim_lat,
                lon=claim_lon,
                timestamp=timestamp,
            ),
            attestations=attestations,
        )

    return result


def verify_proof(
    proof: LocationProof,
    expected_lat: float,
    expected_lon: float,
    expected_time_min: float,
    expected_time_max: float,
    params: ProtocolParams,
    c_used: Set[str],
    expected_ctx: Optional[dict] = None,
    accepted_events: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    claim = proof.claim
    proof_ctx = json.dumps(proof.ctx, sort_keys=True, separators=(",", ":"))

    for attestation in proof.attestations:
        attestation_ctx = json.dumps(attestation.ctx, sort_keys=True, separators=(",", ":"))
        if attestation_ctx != proof_ctx:
            return False, "context_mismatch"

    if expected_ctx is not None:
        expected_ctx_key = json.dumps(expected_ctx, sort_keys=True, separators=(",", ":"))
    else:
        expected_ctx_key = proof_ctx

    event_ctx = expected_ctx if expected_ctx is not None else proof.ctx
    event_key = json.dumps(
        {
            "order_id": event_ctx.get("order_id"),
            "event": event_ctx.get("event"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if accepted_events is not None and event_key in accepted_events:
        return False, "event_accepted"

    if proof_ctx != expected_ctx_key:
        return False, "context_binding"

    if not (expected_time_min <= claim.timestamp <= expected_time_max):
        return False, "temporal"

    for attestation in proof.attestations:
        if abs(claim.timestamp - attestation.timestamp) > params.delta_t_valid:
            return False, "temporal_consistency"

    dist_to_event = haversine(claim.lat, claim.lon, expected_lat, expected_lon)
    if dist_to_event > params.r:
        return False, "spatial"

    if proof_ctx in c_used:
        return False, "context_replay"

    witness_ids = [attestation.witness_id for attestation in proof.attestations]
    if len(set(witness_ids)) < len(witness_ids):
        return False, "duplicate_witness"

    c_used.add(proof_ctx)
    if accepted_events is not None:
        accepted_events.add(event_key)
    return True, "accept"


def count_reasons(checks: Iterable[WitnessCheck]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for check in checks:
        counts[check.reason] = counts.get(check.reason, 0) + 1
    return counts


def proof_to_payload(proof: LocationProof) -> Dict[str, Any]:
    return {
        "ctx": proof.ctx,
        "claim": {
            "prover_id": proof.claim.prover_id,
            "lat": round(proof.claim.lat, 7),
            "lon": round(proof.claim.lon, 7),
            "timestamp": round(proof.claim.timestamp, 3),
        },
        "attestations": [
            {
                "witness_id": attestation.witness_id,
                "witness_lat": round(attestation.witness_lat, 7),
                "witness_lon": round(attestation.witness_lon, 7),
                "nonce": attestation.nonce,
                "timestamp": round(attestation.timestamp, 3),
                "ctx": attestation.ctx,
                "h_lc": attestation.h_lc,
            }
            for attestation in proof.attestations
        ],
    }


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def leaf_hash(payload: Any) -> str:
    if isinstance(payload, str):
        blob = payload
    else:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_hex(blob)


def proof_hash(proof: LocationProof) -> str:
    return leaf_hash(proof_to_payload(proof))


def merkle_parent(left: str, right: str) -> str:
    return sha256_hex(left + right)


def build_merkle_levels(leaves: Sequence[str]) -> List[List[str]]:
    if not leaves:
        return [[leaf_hash("empty-batch")]]

    levels = [list(leaves)]
    current = list(leaves)
    while len(current) > 1:
        padded = list(current)
        if len(padded) % 2 == 1:
            padded.append(padded[-1])
        nxt = []
        for idx in range(0, len(padded), 2):
            nxt.append(merkle_parent(padded[idx], padded[idx + 1]))
        levels.append(nxt)
        current = nxt
    return levels


def merkle_root(leaves: Sequence[str]) -> str:
    return build_merkle_levels(leaves)[-1][0]


def merkle_path(leaves: Sequence[str], index: int) -> List[Tuple[str, str]]:
    levels = build_merkle_levels(leaves)
    path: List[Tuple[str, str]] = []
    current_index = index
    for level in levels[:-1]:
        sibling_index = current_index ^ 1
        if sibling_index >= len(level):
            sibling_hash = level[current_index]
        else:
            sibling_hash = level[sibling_index]
        direction = "right" if current_index % 2 == 0 else "left"
        path.append((direction, sibling_hash))
        current_index //= 2
    return path


def verify_merkle_path(leaf: str, path: Sequence[Tuple[str, str]], root: str) -> bool:
    digest = leaf
    for direction, sibling in path:
        if direction == "right":
            digest = merkle_parent(digest, sibling)
        else:
            digest = merkle_parent(sibling, digest)
    return digest == root


def anchor_batch(batch_registry: Dict[int, str], batch_id: int, merkle_root_hash: str) -> None:
    batch_registry[batch_id] = merkle_root_hash


def audit_batch_membership(
    batch_registry: Dict[int, str],
    batch_id: int,
    leaves: Sequence[str],
    index: int,
    tampered_leaf: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    stored_root = batch_registry.get(batch_id)
    if stored_root is None:
        return {
            "accepted": False,
            "reason": "missing_anchor",
            "path_length": 0,
            "elapsed_ms": 0.0,
        }

    leaf = tampered_leaf if tampered_leaf is not None else leaves[index]
    path = merkle_path(leaves, index)
    accepted = verify_merkle_path(leaf, path, stored_root)
    reason = "accept" if accepted else "merkle_mismatch"
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "accepted": accepted,
        "reason": reason,
        "path_length": len(path),
        "elapsed_ms": elapsed_ms,
    }
