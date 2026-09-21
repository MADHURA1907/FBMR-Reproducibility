"""
FBMR Federated Server (Tier 4 Central Coordinator)
==================================================
Coordinates cross-college federated learning rounds:
- Coordinates Secure Aggregation using SafeMind's SecureAggregator
- Reconstructs global aggregate update
- Performs projected gradient descent onto probability simplex Pi_W
- Updates and validates global CWS weights
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .boundary_guard import UnifiedDataBoundaryEnforcer
from .cws_calibrator import (
    FederatedCWSCalibrator,
    UTKARSH_BASELINE_CWS_WEIGHTS,
    project_to_simplex,
)


class FBMRServer:
    """
    Federated server orchestrating cross-college CWS weight refinement.
    """

    def __init__(
        self,
        num_colleges: int,
        learning_rate: float = 0.10,
        initial_weights: np.ndarray = UTKARSH_BASELINE_CWS_WEIGHTS,
        secure_aggregator=None,
        boundary_enforcer: Optional[UnifiedDataBoundaryEnforcer] = None,
        consent_bridge=None,
    ):
        self.num_colleges = num_colleges
        self.learning_rate = float(learning_rate)
        self.weights = project_to_simplex(np.copy(initial_weights))
        self.secure_aggregator = secure_aggregator
        self.boundary_enforcer = boundary_enforcer or UnifiedDataBoundaryEnforcer()
        self.consent_bridge = consent_bridge
        self.round_history: List[Dict] = []

    def aggregate_and_update(
        self,
        college_payloads: List[Dict],
        round_id: int,
    ) -> np.ndarray:
        """
        Aggregate protected updates from participating colleges and update global weights.

        Args:
            college_payloads: List of college payload dicts containing 'update'.
            round_id: Current communication round.

        Returns:
            Updated global weights array satisfying simplex constraints.
        """
        valid_payloads = [p for p in college_payloads if p is not None]
        if not valid_payloads:
            return self.weights

        unmasked_updates = {i: p["update"] for i, p in enumerate(valid_payloads)}

        # Secure Aggregation Phase (if enabled)
        if self.secure_aggregator is not None:
            all_ids = list(unmasked_updates.keys())
            # Ensure keypairs exist
            for cid in all_ids:
                if cid not in self.secure_aggregator._public_keys:
                    pub = self.secure_aggregator.generate_keypair(cid)
                    self.secure_aggregator.register_public_key(cid, pub)

            # Mask each update
            masked_updates = {}
            for cid, upd in unmasked_updates.items():
                masked = self.secure_aggregator.mask_update(
                    update=upd.astype(np.float32),
                    client_id=cid,
                    all_client_ids=all_ids,
                    round_number=round_id,
                )
                masked_updates[cid] = masked

            # Server aggregates masked updates (pairwise masks cancel identically)
            global_grad = self.secure_aggregator.aggregate_masked_updates(masked_updates)
            global_grad = np.asarray(global_grad, dtype=np.float64)
        else:
            # Standard mean aggregation
            global_grad = np.mean(list(unmasked_updates.values()), axis=0)

        # Projected Gradient Descent Step:
        # w^(t+1) = Pi_W ( w^(t) - eta * G^(t) )
        w_new = self.weights - self.learning_rate * global_grad
        self.weights = project_to_simplex(w_new)

        # Enforce boundary check on global weights
        self.boundary_enforcer.validate_global_weights(self.weights)

        # Record history
        self.round_history.append({
            "round_id": round_id,
            "weights": np.copy(self.weights),
            "global_grad": np.copy(global_grad),
            "global_grad_norm": float(np.linalg.norm(global_grad)),
            "num_colleges_reporting": len(valid_payloads),
        })

        return self.weights

    def server_round_update(
        self,
        college_payloads: List[Dict],
        round_id: int,
        eval_X: Optional[np.ndarray] = None,
        eval_y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Executes aggregation, simplex projection, and computes validation loss.
        """
        w_new = self.aggregate_and_update(college_payloads, round_id)
        loss = 0.0
        if eval_X is not None and eval_y is not None:
            calibrator = FederatedCWSCalibrator()
            loss = calibrator.compute_loss(w_new, eval_X, eval_y)
        return w_new, loss
