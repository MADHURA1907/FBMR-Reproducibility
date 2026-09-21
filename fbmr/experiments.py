"""
FBMR 4-Condition Benchmark Runner
=================================
Runs reproducible protocol-level empirical evaluations comparing:
  Condition A: Fixed Utkarsh CWS weights (baseline)
  Condition B: Non-federated (local) CWS calibration
  Condition C: Federated CWS calibration without DP
  Condition D: Federated CWS calibration with DP + Secure Aggregation

Evaluated across 3 distinct random seeds: 42, 123, 999.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

# Ensure SafeMind src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, ".."))
safemind_src = os.path.join(workspace_root, "privacy-preserving-fl-mental-health-master", "src")
if safemind_src not in sys.path:
    sys.path.insert(0, safemind_src)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from privacy.privacy_accountant import RDPAccountant
from privacy.noise_scheduler import NoiseScheduler
from fl.secure_aggregation import SecureAggregator

from fbmr.boundary_guard import UnifiedDataBoundaryEnforcer
from fbmr.consent_bridge import ConsentVerificationBridge
from fbmr.cws_calibrator import (
    FederatedCWSCalibrator,
    UTKARSH_BASELINE_CWS_WEIGHTS,
    project_to_simplex,
)
from fbmr.client_bridge import FBMRBridge
from fbmr.college_aggregator import IntraCollegeAggregator
from fbmr.federated_server import FBMRServer
from fbmr.synthetic_data import (
    SYNTHETIC_TRUE_WEIGHTS,
    generate_synthetic_college_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FBMR-Experiment")


def run_condition_a_fixed(
    colleges_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    num_rounds: int,
) -> Dict:
    """Condition A: Fixed Utkarsh weights (Static w0)."""
    calibrator = FederatedCWSCalibrator(UTKARSH_BASELINE_CWS_WEIGHTS)
    fixed_weights = np.copy(UTKARSH_BASELINE_CWS_WEIGHTS)

    # Compute loss on all student samples
    all_X = []
    all_y = []
    for c_students in colleges_data.values():
        for X, y in c_students:
            all_X.append(X)
            all_y.append(y)
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    loss = calibrator.compute_loss(fixed_weights, X_all, y_all)
    dist_to_true = float(np.linalg.norm(fixed_weights - SYNTHETIC_TRUE_WEIGHTS))

    losses = [loss] * num_rounds
    distances = [dist_to_true] * num_rounds

    return {
        "condition": "A (Fixed Utkarsh)",
        "final_weights": fixed_weights,
        "round_losses": losses,
        "round_distances": distances,
        "final_loss": loss,
        "final_distance": dist_to_true,
        "epsilon": 0.0,
        "total_bytes": 0,
        "total_time_ms": 0.5,
    }


def run_condition_b_local(
    colleges_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    num_rounds: int,
    learning_rate: float = 0.10,
) -> Dict:
    """Condition B: Non-federated (local) calibration per college (isolated)."""
    start_time = time.time()
    calibrator = FederatedCWSCalibrator(UTKARSH_BASELINE_CWS_WEIGHTS)
    college_weights = {
        cid: np.copy(UTKARSH_BASELINE_CWS_WEIGHTS) for cid in colleges_data.keys()
    }

    round_losses = []
    round_distances = []

    for r in range(1, num_rounds + 1):
        round_loss_accum = []
        for cid, students in colleges_data.items():
            w = college_weights[cid]
            # Average gradient across local students
            grads = [
                calibrator.compute_gradient(w, X, y) for X, y in students
            ]
            avg_grad = np.mean(grads, axis=0)
            # Local gradient descent step + simplex projection
            w_new = project_to_simplex(w - learning_rate * avg_grad)
            college_weights[cid] = w_new

            # Loss on college data
            c_X = np.vstack([X for X, _ in students])
            c_y = np.concatenate([y for _, y in students])
            round_loss_accum.append(calibrator.compute_loss(w_new, c_X, c_y))

        round_losses.append(float(np.mean(round_loss_accum)))
        avg_w = np.mean(list(college_weights.values()), axis=0)
        round_distances.append(float(np.linalg.norm(avg_w - SYNTHETIC_TRUE_WEIGHTS)))

    elapsed = (time.time() - start_time) * 1000.0

    return {
        "condition": "B (Non-Federated)",
        "final_weights": np.mean(list(college_weights.values()), axis=0),
        "round_losses": round_losses,
        "round_distances": round_distances,
        "final_loss": round_losses[-1],
        "final_distance": round_distances[-1],
        "epsilon": 0.0,
        "total_bytes": 0,
        "total_time_ms": elapsed,
    }


def run_condition_c_federated_nodp(
    colleges_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    num_rounds: int,
    learning_rate: float = 0.10,
) -> Dict:
    """Condition C: Federated CWS calibration without DP."""
    start_time = time.time()
    boundary = UnifiedDataBoundaryEnforcer()
    consent_bridge = ConsentVerificationBridge()
    calibrator = FederatedCWSCalibrator()

    num_colleges = len(colleges_data)
    server = FBMRServer(
        num_colleges=num_colleges,
        learning_rate=learning_rate,
        boundary_enforcer=boundary,
        consent_bridge=consent_bridge,
    )

    # Setup clients and aggregators
    aggregators = {}
    clients = {}
    total_bytes = 0

    for cid, students in colleges_data.items():
        aggregators[cid] = IntraCollegeAggregator(
            college_id=cid,
            min_cohort_size=3,
            clipping_bound_ci=1.0,
            boundary_enforcer=boundary,
        )
        c_clients = []
        for sid, _ in enumerate(students):
            client_id = f"{cid}_s{sid}"
            consent_bridge.register_consent(client_id, cid)
            c_clients.append(
                FBMRBridge(
                    client_id=client_id,
                    college_id=cid,
                    consent_bridge=consent_bridge,
                    boundary_enforcer=boundary,
                )
            )
        clients[cid] = c_clients

    # All data for evaluation
    all_X = np.vstack([X for c_studs in colleges_data.values() for X, _ in c_studs])
    all_y = np.concatenate([y for c_studs in colleges_data.values() for _, y in c_studs])

    round_losses = []
    round_distances = []

    for r in range(1, num_rounds + 1):
        global_w = server.weights
        college_payloads = []

        for cid, c_clients in clients.items():
            students = colleges_data[cid]
            client_updates = []
            for i, client in enumerate(c_clients):
                X, y = students[i]
                update = client.prepare_update(global_w, X, y, r)
                client_updates.append(update)
                if update is not None:
                    total_bytes += 7 * 8 + 64  # float64 payload + metadata

            col_payload = aggregators[cid].aggregate_client_updates(client_updates, r)
            if col_payload is not None:
                college_payloads.append(col_payload)
                total_bytes += 7 * 8 + 64

        # Server update
        new_w = server.aggregate_and_update(college_payloads, r)
        loss = calibrator.compute_loss(new_w, all_X, all_y)
        dist = float(np.linalg.norm(new_w - SYNTHETIC_TRUE_WEIGHTS))
        round_losses.append(loss)
        round_distances.append(dist)

    elapsed = (time.time() - start_time) * 1000.0

    return {
        "condition": "C (Federated No-DP)",
        "final_weights": server.weights,
        "round_losses": round_losses,
        "round_distances": round_distances,
        "final_loss": round_losses[-1],
        "final_distance": round_distances[-1],
        "epsilon": 0.0,
        "total_bytes": total_bytes,
        "total_time_ms": elapsed,
    }


def run_condition_d_federated_dp_secagg(
    colleges_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    num_rounds: int,
    learning_rate: float = 0.10,
    target_epsilon: float = 4.0,
    target_delta: float = 1e-5,
) -> Dict:
    """Condition D: Full FBMR protocol (Federated + DP + Secure Aggregation)."""
    start_time = time.time()
    boundary = UnifiedDataBoundaryEnforcer()
    consent_bridge = ConsentVerificationBridge()
    calibrator = FederatedCWSCalibrator()

    num_colleges = len(colleges_data)

    # SafeMind DP Configuration
    safemind_cfg = {
        "privacy": {
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "max_grad_norm": 1.0,
            "noise_multiplier": 0.85,
            "noise_schedule": "uniform",
            "per_client_budget": True,
        },
        "fl": {"num_rounds": num_rounds},
        "partitioning": {"num_clients": num_colleges},
        "secure_aggregation": {
            "enabled": True,
            "threshold_ratio": 0.5,
            "quantization_bits": 32,
            "sparsification_topk": 1.0,
        },
    }

    scheduler = NoiseScheduler(safemind_cfg)
    accountant = RDPAccountant(safemind_cfg)
    sec_agg = SecureAggregator(safemind_cfg)

    server = FBMRServer(
        num_colleges=num_colleges,
        learning_rate=learning_rate,
        secure_aggregator=sec_agg,
        boundary_enforcer=boundary,
        consent_bridge=consent_bridge,
    )

    aggregators = {}
    clients = {}
    total_bytes = 0

    for cid, students in colleges_data.items():
        aggregators[cid] = IntraCollegeAggregator(
            college_id=cid,
            min_cohort_size=3,
            clipping_bound_ci=1.0,
            noise_scheduler=scheduler,
            rdp_accountant=accountant,
            boundary_enforcer=boundary,
        )
        c_clients = []
        for sid, _ in enumerate(students):
            client_id = f"{cid}_s{sid}"
            consent_bridge.register_consent(client_id, cid)
            c_clients.append(
                FBMRBridge(
                    client_id=client_id,
                    college_id=cid,
                    consent_bridge=consent_bridge,
                    boundary_enforcer=boundary,
                )
            )
        clients[cid] = c_clients

    all_X = np.vstack([X for c_studs in colleges_data.values() for X, _ in c_studs])
    all_y = np.concatenate([y for c_studs in colleges_data.values() for _, y in c_studs])

    round_losses = []
    round_distances = []
    secagg_errors = []

    for r in range(1, num_rounds + 1):
        global_w = server.weights
        college_payloads = []

        for cid, c_clients in clients.items():
            students = colleges_data[cid]
            client_updates = []
            for i, client in enumerate(c_clients):
                X, y = students[i]
                update = client.prepare_update(global_w, X, y, r)
                client_updates.append(update)
                if update is not None:
                    total_bytes += 7 * 8 + 64

            col_payload = aggregators[cid].aggregate_client_updates(client_updates, r)
            if col_payload is not None:
                college_payloads.append(col_payload)
                # SecAgg payload overhead: masked update + ECDH public key share
                total_bytes += 7 * 8 + 128

        # Calculate exact aggregation error to verify mask cancellation
        raw_updates = [p["update"] for p in college_payloads]
        true_mean = np.mean(raw_updates, axis=0)

        # Server updates weights via Secure Aggregator
        new_w = server.aggregate_and_update(college_payloads, r)

        # Verify mask cancellation mathematically
        rec_grad = server.round_history[-1]["global_grad"]
        err = float(np.max(np.abs(true_mean - rec_grad)))
        secagg_errors.append(err)

        loss = calibrator.compute_loss(new_w, all_X, all_y)
        dist = float(np.linalg.norm(new_w - SYNTHETIC_TRUE_WEIGHTS))
        round_losses.append(loss)
        round_distances.append(dist)

    elapsed = (time.time() - start_time) * 1000.0

    return {
        "condition": "D (FBMR: DP + SecAgg)",
        "final_weights": server.weights,
        "round_losses": round_losses,
        "round_distances": round_distances,
        "final_loss": round_losses[-1],
        "final_distance": round_distances[-1],
        "epsilon": accountant.get_cumulative_epsilon(),
        "secagg_max_error": float(np.max(secagg_errors)),
        "total_bytes": total_bytes,
        "total_time_ms": elapsed,
    }


def run_all_experiments(
    seeds: List[int] = [42, 123, 999],
    num_rounds: int = 15,
    num_colleges: int = 4,
    students_per_college: int = 15,
) -> Dict:
    """Execute complete 4-condition experiment over all requested seeds."""
    logger.info(f"Starting FBMR benchmark across seeds {seeds} ({num_rounds} rounds)...")
    results_by_condition = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    for seed in seeds:
        logger.info(f"--- Running Seed {seed} ---")
        colleges_data = generate_synthetic_college_data(
            num_colleges=num_colleges,
            students_per_college=students_per_college,
            samples_per_student=40,
            seed=seed,
        )

        res_a = run_condition_a_fixed(colleges_data, num_rounds)
        res_b = run_condition_b_local(colleges_data, num_rounds)
        res_c = run_condition_c_federated_nodp(colleges_data, num_rounds)
        res_d = run_condition_d_federated_dp_secagg(colleges_data, num_rounds)

        results_by_condition["A"].append(res_a)
        results_by_condition["B"].append(res_b)
        results_by_condition["C"].append(res_c)
        results_by_condition["D"].append(res_d)

    return results_by_condition


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n=== EXPERIMENT RESULTS SUMMARY ===")
    for cond_key, cond_runs in results.items():
        cond_name = cond_runs[0]["condition"]
        losses = [r["final_loss"] for r in cond_runs]
        dists = [r["final_distance"] for r in cond_runs]
        eps = cond_runs[0]["epsilon"]
        times = [r["total_time_ms"] for r in cond_runs]
        print(f"Condition {cond_name}:")
        print(f"  Final Loss: {np.mean(losses):.6f} +/- {np.std(losses):.6f}")
        print(f"  Distance to Target: {np.mean(dists):.6f} +/- {np.std(dists):.6f}")
        print(f"  Epsilon: {eps:.4f}")
        print(f"  Avg Runtime: {np.mean(times):.2f} ms")
        print(f"  Final Weights: {np.round(cond_runs[0]['final_weights'], 4)}")
