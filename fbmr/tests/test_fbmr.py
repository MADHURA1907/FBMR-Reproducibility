"""
Unit and Integration Test Suite for FBMR
=========================================
Tests all 8 required components and mathematical properties:
1. Simplex projection (weights >= 0, sum == 1)
2. Local L2 gradient clipping (norm <= C_L)
3. Differential privacy Gaussian noise and RDP accounting
4. Intra-college aggregation (k-anonymity cohort enforcement, mean preservation)
5. Secure aggregation mask cancellation (exact ECDH mask sum = 0)
6. Blockchain consent verification and rejection of unconsented clients
7. Data boundary enforcement (rejection of raw texts, surveys, PHQ/GAD, clinical data, identifiers)
8. End-to-end multi-round calibration cycle (simplex preservation, finite loss, convergence)
"""

import os
import sys
import numpy as np
import pytest

# Ensure safe mind source is discoverable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAFEMIND_SRC = os.path.join(PROJECT_ROOT, "privacy-preserving-fl-mental-health-master", "src")
if SAFEMIND_SRC not in sys.path:
    sys.path.insert(0, SAFEMIND_SRC)

from fbmr.boundary_guard import UnifiedDataBoundaryEnforcer, DataBoundaryViolationError
from fbmr.consent_bridge import ConsentVerificationBridge, ConsentStatus
from fbmr.cws_calibrator import (
    FederatedCWSCalibrator,
    project_onto_simplex,
    UTKARSH_BASELINE_CWS_WEIGHTS,
)
from fbmr.client_bridge import FBMRBridge
from fbmr.college_aggregator import IntraCollegeAggregator
from fbmr.federated_server import FBMRServer

from privacy.noise_scheduler import NoiseScheduler
from privacy.privacy_accountant import RDPAccountant
from fl.secure_aggregation import SecureAggregator


def test_simplex_projection():
    """Verify that project_onto_simplex maps arbitrary vectors to Delta^6."""
    calibrator = FederatedCWSCalibrator()

    # Case 1: All negative components
    v1 = np.array([-1.0, -2.0, -0.5, -3.0, -0.1, -10.0, -0.01])
    w1 = calibrator.project(v1)
    assert np.all(w1 >= -1e-9), f"Weights must be non-negative, got {w1}"
    assert np.isclose(np.sum(w1), 1.0, atol=1e-6), f"Weights must sum to 1.0, got sum {np.sum(w1)}"

    # Case 2: Extreme unnormalized vector
    v2 = np.array([100.0, 0.0, -50.0, 25.0, 0.0, 10.0, 5.0])
    w2 = calibrator.project(v2)
    assert np.all(w2 >= -1e-9)
    assert np.isclose(np.sum(w2), 1.0, atol=1e-6)

    # Case 3: Already on simplex
    v3 = np.copy(UTKARSH_BASELINE_CWS_WEIGHTS)
    w3 = calibrator.project(v3)
    np.testing.assert_allclose(w3, UTKARSH_BASELINE_CWS_WEIGHTS, atol=1e-6)

    # Case 4: Uniform vector
    v4 = np.ones(7)
    w4 = calibrator.project(v4)
    np.testing.assert_allclose(w4, np.full(7, 1.0 / 7.0), atol=1e-6)


def test_l2_clipping():
    """Verify gradient is scaled to have norm <= C_L."""
    clipping_bound = 1.0
    calibrator = FederatedCWSCalibrator(clipping_bound_cl=clipping_bound)

    # Small gradient (norm < C_L) -> unchanged
    g_small = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    clipped_small = calibrator.clip_gradient(g_small)
    np.testing.assert_allclose(clipped_small, g_small, atol=1e-7)

    # Large gradient (norm >> C_L) -> norm equals C_L
    g_large = np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0])
    clipped_large = calibrator.clip_gradient(g_large)
    norm_clipped = np.linalg.norm(clipped_large)
    assert np.isclose(norm_clipped, clipping_bound, atol=1e-6)
    # Direction should be preserved
    unit_original = g_large / np.linalg.norm(g_large)
    unit_clipped = clipped_large / norm_clipped
    np.testing.assert_allclose(unit_clipped, unit_original, atol=1e-6)


def test_dp_gaussian_noise():
    """
    Verify:
    - Gaussian noise has approximately expected mean and variance within statistical tolerance.
    - RDP accountant increments correctly.
    """
    cfg = {
        "privacy": {
            "noise_schedule": "uniform",
            "target_epsilon": 10.0,
            "target_delta": 1e-5,
            "max_grad_norm": 1.0,
            "noise_multiplier": 1.0,
            "per_client_budget": False,
        },
        "partitioning": {
            "num_clients": 5,
        },
        "fl": {
            "num_rounds": 100,
        },
    }
    scheduler = NoiseScheduler(cfg)
    accountant = RDPAccountant(cfg)

    agg = IntraCollegeAggregator(
        college_id="college_0",
        min_cohort_size=3,
        clipping_bound_ci=1.0,
        noise_scheduler=scheduler,
        rdp_accountant=accountant,
    )

    # Collect multiple noise samples
    samples = []
    np.random.seed(42)
    for r in range(1, 101):
        dummy_payloads = [
            {"client_id": f"c_{i}", "college_id": "college_0", "round_id": r, "update": np.zeros(7)}
            for i in range(5)
        ]
        out = agg.aggregate_client_updates(dummy_payloads, round_id=r)
        samples.append(out["update"])

    samples = np.vstack(samples)
    emp_mean = np.mean(samples)
    emp_var = np.var(samples)

    # Expected mean ~0.0, expected var ~sigma^2 = 1.0 (within statistical tolerance for 700 draws)
    assert abs(emp_mean) < 0.15, f"Expected mean near 0, got {emp_mean}"
    assert abs(emp_var - 1.0) < 0.25, f"Expected variance near 1.0, got {emp_var}"
    assert accountant.get_cumulative_epsilon() > 0.0, "RDP accountant should track positive epsilon"


def test_intra_college_aggregation():
    """Verify average matches mathematical expectation, and raises error / returns None if cohort < k_min."""
    agg = IntraCollegeAggregator(college_id="college_0", min_cohort_size=3, clipping_bound_ci=10.0)

    # Test valid cohort
    p1 = {"client_id": "s1", "college_id": "college_0", "round_id": 1, "update": np.full(7, 1.0)}
    p2 = {"client_id": "s2", "college_id": "college_0", "round_id": 1, "update": np.full(7, 2.0)}
    p3 = {"client_id": "s3", "college_id": "college_0", "round_id": 1, "update": np.full(7, 3.0)}

    out = agg.aggregate_client_updates([p1, p2, p3], 1)
    assert out is not None
    np.testing.assert_allclose(out["update"], np.full(7, 2.0), atol=1e-7)

    # Test rejection when cohort < min_cohort_size (e.g. only 2 clients)
    out_rejected = agg.aggregate_client_updates([p1, p2], 1)
    assert out_rejected is None, "Should return None when participant count < min_cohort_size"


def test_secure_aggregation():
    """Verify pairwise ECDH masks cancel identically to zero across colleges."""
    cfg = {
        "secure_aggregation": {
            "enabled": True,
            "threshold_ratio": 0.5,
            "quantization_bits": 32,
            "sparsification_topk": 1.0,
        },
        "partitioning": {"num_clients": 3},
    }
    sec_agg = SecureAggregator(cfg)

    # Generate keys for 3 colleges
    for i in range(3):
        pub = sec_agg.generate_keypair(i)
        sec_agg.register_public_key(i, pub)

    # Raw updates
    u0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    u1 = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    u2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    raw_mean = (u0 + u1 + u2) / 3.0

    # Mask updates
    m0 = sec_agg.mask_update(u0, 0, [0, 1, 2], round_number=1)
    m1 = sec_agg.mask_update(u1, 1, [0, 1, 2], round_number=1)
    m2 = sec_agg.mask_update(u2, 2, [0, 1, 2], round_number=1)

    # Masks must significantly alter individual vectors
    assert not np.allclose(m0, u0, atol=1e-3)

    # When aggregated, masks must cancel identically
    agg_result = sec_agg.aggregate_masked_updates({0: m0, 1: m1, 2: m2})
    np.testing.assert_allclose(agg_result, raw_mean, atol=1e-6)


def test_consent_rejection():
    """Verify that unconsented or revoked client is strictly blocked."""
    consent = ConsentVerificationBridge()
    # Register client 1 as active
    consent.register_consent("client_1", "college_0", ConsentStatus.ACTIVE)
    # Register client 2 as revoked
    consent.register_consent("client_2", "college_0", ConsentStatus.REVOKED)

    bridge_1 = FBMRBridge("client_1", "college_0", consent)
    bridge_2 = FBMRBridge("client_2", "college_0", consent)
    bridge_3 = FBMRBridge("client_unregistered", "college_0", consent)

    X = np.eye(7)
    y = np.ones(7)

    assert bridge_1.prepare_update(UTKARSH_BASELINE_CWS_WEIGHTS, X, y, 1) is not None
    assert bridge_2.prepare_update(UTKARSH_BASELINE_CWS_WEIGHTS, X, y, 1) is None
    assert bridge_3.prepare_update(UTKARSH_BASELINE_CWS_WEIGHTS, X, y, 1) is None


def test_data_boundary_rejection():
    """Verify DataBoundaryViolationError triggers on prohibited keys or invalid shapes."""
    enforcer = UnifiedDataBoundaryEnforcer()

    # Valid payload passes
    valid_payload = {
        "client_id": "c1",
        "college_id": "col1",
        "round_id": 1,
        "update": np.zeros(7, dtype=np.float64),
    }
    enforcer.validate_client_payload(valid_payload)

    # Prohibited key (e.g. phq9_score)
    with pytest.raises(DataBoundaryViolationError):
        enforcer.validate_client_payload({
            "client_id": "c1",
            "phq9_score": 15,
            "update": np.zeros(7),
        })

    # Prohibited key (e.g. user_id or message)
    with pytest.raises(DataBoundaryViolationError):
        enforcer.validate_client_payload({
            "user_id": "student_123",
            "update": np.zeros(7),
        })

    # Invalid tensor shape (e.g. length 8 instead of 7)
    with pytest.raises(DataBoundaryViolationError):
        enforcer.validate_client_payload({
            "client_id": "c1",
            "update": np.zeros(8),
        })


def test_end_to_end_flow():
    """
    Verify full FBMR 2-round cycle:
    - Valid simplex weights after every round
    - Finite loss values
    - Overall optimization progress from initial to final loss
    - Successful consent, intra-college aggregation, DP, Secure Aggregation, and server update.
    """
    consent = ConsentVerificationBridge()
    boundary = UnifiedDataBoundaryEnforcer()
    calibrator = FederatedCWSCalibrator()

    # Create 2 colleges with 5 students each
    colleges = ["college_0", "college_1"]
    client_bridges = []
    data_dict = {}

    target_w = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
    rng = np.random.default_rng(42)

    for col in colleges:
        data_dict[col] = []
        for s in range(5):
            cid = f"{col}_s{s}"
            consent.register_consent(cid, col)
            client_bridges.append(FBMRBridge(cid, col, consent, boundary))
            X = rng.uniform(0.1, 0.9, size=(30, 7))
            y = np.dot(X, target_w) + rng.normal(0, 0.005, size=30)
            data_dict[col].append((X, y))

    # Aggregators
    aggregators = {col: IntraCollegeAggregator(col, min_cohort_size=3) for col in colleges}

    # Server
    server = FBMRServer(num_colleges=2, learning_rate=0.2, boundary_enforcer=boundary)

    # Initial loss
    all_X = np.vstack([X for c in colleges for X, _ in data_dict[c]])
    all_y = np.concatenate([y for c in colleges for _, y in data_dict[c]])
    initial_loss = calibrator.compute_loss(server.weights, all_X, all_y)

    round_losses = []

    for r in range(1, 4):
        col_payloads = []
        for col in colleges:
            stud_updates = []
            for i, (X, y) in enumerate(data_dict[col]):
                bridge = [b for b in client_bridges if b.client_id == f"{col}_s{i}"][0]
                upd = bridge.prepare_update(server.weights, X, y, r)
                stud_updates.append(upd)
            col_upd = aggregators[col].aggregate_client_updates(stud_updates, r)
            col_payloads.append(col_upd)

        w_new, loss = server.server_round_update(col_payloads, r, all_X, all_y)
        round_losses.append(loss)

        # Simplex verification
        assert np.all(w_new >= -1e-6), f"Weights must be non-negative: {w_new}"
        assert np.isclose(np.sum(w_new), 1.0, atol=1e-5), f"Weights must sum to 1.0: {np.sum(w_new)}"
        assert np.isfinite(loss), f"Round {r} loss is not finite: {loss}"

    # Overall optimization progress from initial to final loss
    final_loss = round_losses[-1]
    assert final_loss < initial_loss, f"Optimization progress failed: initial={initial_loss}, final={final_loss}"
