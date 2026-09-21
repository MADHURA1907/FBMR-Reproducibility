"""
FBMR Client Bridge (Tier 3 Leaf Node)
=====================================
Executes on the simulated student device:
- Evaluates consent gating
- Computes local gradients from local features
- Applies per-device L2 norm clipping (Cu)
- Validates data boundary before transmission
"""

from typing import Dict, Optional
import numpy as np

from .boundary_guard import UnifiedDataBoundaryEnforcer, DataBoundaryViolationError
from .consent_bridge import ConsentVerificationBridge
from .cws_calibrator import FederatedCWSCalibrator


class FBMRBridge:
    """
    On-device client bridge representing a student node in FBMR.
    """

    def __init__(
        self,
        client_id: str,
        college_id: str,
        consent_bridge: ConsentVerificationBridge,
        boundary_enforcer: Optional[UnifiedDataBoundaryEnforcer] = None,
        clipping_bound_cu: float = 1.0,
    ):
        self.client_id = client_id
        self.college_id = college_id
        self.consent_bridge = consent_bridge
        self.boundary_enforcer = boundary_enforcer or UnifiedDataBoundaryEnforcer()
        self.clipping_bound = float(clipping_bound_cu)
        self.calibrator = FederatedCWSCalibrator()

    def prepare_update(
        self,
        global_weights: np.ndarray,
        local_X: np.ndarray,
        local_y: np.ndarray,
        round_id: int,
    ) -> Optional[Dict]:
        """
        Compute and prepare clipped gradient update for the CWS weights.

        Returns:
            Sanitized payload dictionary, or None if consent is not active.
        """
        # Step 1: Consent Verification
        if not self.consent_bridge.is_consent_valid(self.client_id):
            return None

        # Step 2: Compute local gradient on synthetic features
        grad = self.calibrator.compute_gradient(global_weights, local_X, local_y)

        # Step 3: Per-device L2 norm clipping (Equation 8 of Springer paper)
        # ar{g} = g * min(1, Cu / ||g||_2)
        norm = np.linalg.norm(grad)
        if norm > self.clipping_bound and norm > 0:
            clipped_grad = grad * (self.clipping_bound / norm)
        else:
            clipped_grad = np.copy(grad)

        # Step 4: Construct sanitized payload with allowed simulation metadata only
        payload = {
            "client_id": self.client_id,
            "college_id": self.college_id,
            "round_id": round_id,
            "num_samples": len(local_X),
            "clipping_bound": self.clipping_bound,
            "update": clipped_grad.astype(np.float64),
        }

        # Step 5: Enforce strict data boundary before passing payload
        sanitized = self.boundary_enforcer.validate_client_payload(payload)
        return sanitized
