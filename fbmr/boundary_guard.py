"""
Unified Data Boundary Enforcer for FBMR
=======================================
Enforces strict deterministic non-disclosure boundaries:
- Raw text, messages, audio, and transcripts MUST NOT leave the device.
- Raw PHQ-9, GAD-7, and clinical screening scores MUST NOT leave the device.
- Individual CWS scores, risk levels, and user identities MUST NOT leave the device.
- Only permitted model/update tensors and pseudonymous protocol metadata may cross.
"""

from typing import Any, Dict, Set
import numpy as np


class DataBoundaryViolationError(Exception):
    """Raised when an unpermitted artifact attempts to cross the FBMR boundary."""
    pass


class UnifiedDataBoundaryEnforcer:
    """
    Validates and sanitizes outgoing payloads at the device-to-institution
    and institution-to-server boundaries.
    """

    ALLOWED_CLIENT_METADATA: Set[str] = {
        "client_id",
        "college_id",
        "round_id",
        "num_samples",
        "clipping_bound",
        "timestamp",
    }

    ALLOWED_COLLEGE_METADATA: Set[str] = {
        "college_id",
        "round_id",
        "num_participating_clients",
        "clipping_bound",
        "noise_sigma",
        "epsilon_spent",
        "timestamp",
    }

    PROHIBITED_KEY_SUBSTRINGS: Set[str] = {
        "phq",
        "gad",
        "cws",
        "score",
        "text",
        "raw",
        "audio",
        "message",
        "chat",
        "user_id",
        "name",
        "email",
        "student_name",
        "roll_number",
        "ip_address",
        "device_id",
        "token",
        "transcript",
        "survey",
        "questionnaire",
    }

    EXPECTED_TENSOR_DIM: int = 7

    def validate_client_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate outgoing update payload from a student device to college aggregator.

        Args:
            payload: Dictionary containing 'update' and allowed metadata.

        Returns:
            Sanitized payload dictionary.

        Raises:
            DataBoundaryViolationError: If unauthorized data or metadata is present.
        """
        if not isinstance(payload, dict):
            raise DataBoundaryViolationError("Client payload must be a dictionary.")

        if "update" not in payload:
            raise DataBoundaryViolationError("Client payload missing required 'update' tensor.")

        # Check for prohibited keys or non-whitelisted keys
        for key in payload.keys():
            if key != "update" and key not in self.ALLOWED_CLIENT_METADATA:
                raise DataBoundaryViolationError(
                    f"Unauthorized metadata key '{key}' in client payload. Only whitelisted simulation metadata is allowed."
                )

        # Validate update tensor
        update = payload["update"]
        if not isinstance(update, np.ndarray):
            raise DataBoundaryViolationError("Client update must be a numpy.ndarray.")

        if update.shape != (self.EXPECTED_TENSOR_DIM,):
            raise DataBoundaryViolationError(
                f"Client update shape must be ({self.EXPECTED_TENSOR_DIM},), got {update.shape}."
            )

        if not np.issubdtype(update.dtype, np.floating):
            raise DataBoundaryViolationError("Client update array must have floating-point dtype.")

        if not np.all(np.isfinite(update)):
            raise DataBoundaryViolationError("Client update array contains non-finite values (NaN/Inf).")

        # Validate string metadata values do not contain sensitive leaks
        for k, v in payload.items():
            if k == "update":
                continue
            if isinstance(v, str):
                v_lower = v.lower()
                for prohibited in ["phq", "gad", "depression", "anxiety", "@", "http"]:
                    if prohibited in v_lower:
                        raise DataBoundaryViolationError(
                            f"Prohibited sensitive value pattern '{prohibited}' detected in metadata '{k}'."
                        )

        return payload

    def validate_college_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate outgoing update payload from college aggregator to federated server.

        Args:
            payload: Dictionary containing 'update' and allowed college metadata.

        Returns:
            Sanitized payload dictionary.
        """
        if not isinstance(payload, dict):
            raise DataBoundaryViolationError("College payload must be a dictionary.")

        if "update" not in payload:
            raise DataBoundaryViolationError("College payload missing required 'update' tensor.")

        for key in payload.keys():
            if key != "update" and key not in self.ALLOWED_COLLEGE_METADATA:
                raise DataBoundaryViolationError(
                    f"Unauthorized metadata key '{key}' in college payload."
                )

        update = payload["update"]
        if not isinstance(update, np.ndarray) or update.shape != (self.EXPECTED_TENSOR_DIM,):
            raise DataBoundaryViolationError(
                f"College update must be a 1D array of shape ({self.EXPECTED_TENSOR_DIM},)."
            )

        if not np.all(np.isfinite(update)):
            raise DataBoundaryViolationError("College update array contains non-finite values (NaN/Inf).")

        return payload

    def validate_global_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Validate global CWS weights sent from server to clients.
        """
        if not isinstance(weights, np.ndarray):
            raise DataBoundaryViolationError("Global weights must be a numpy.ndarray.")

        if weights.shape != (self.EXPECTED_TENSOR_DIM,):
            raise DataBoundaryViolationError(
                f"Global weights shape must be ({self.EXPECTED_TENSOR_DIM},), got {weights.shape}."
            )

        if np.any(weights < -1e-6):
            raise DataBoundaryViolationError("Global CWS weights contain negative components.")

        sum_weights = float(np.sum(weights))
        if abs(sum_weights - 1.0) > 1e-4:
            raise DataBoundaryViolationError(
                f"Global CWS weights must sum to 1.0, got {sum_weights:.6f}."
            )

        return weights
