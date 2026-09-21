"""
Federated CWS Calibrator & Simplex Projection
=============================================
Implements:
1. Exact Utkarsh CWS formulation (weights, 7-day smoothing, non-negativity).
2. O(d log d) Euclidean Simplex Projection (Pi_W).
3. Synthetic protocol-validation objective loss and gradient calculation.
"""

from typing import List, Tuple
import numpy as np


# Exact baseline weights from Utkarsh CWSEngine (cws_engine.dart lines 8-16)
UTKARSH_BASELINE_CWS_WEIGHTS = np.array(
    [0.25, 0.15, 0.15, 0.10, 0.10, 0.15, 0.10], dtype=np.float64
)

CWS_FEATURE_NAMES: List[str] = [
    "Emotion",
    "Stress",
    "Tasks",
    "Activity",
    "Routine",
    "Behavior",
    "Growth",
]


def project_to_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the probability simplex:
    Delta^d = { w in R^d | w >= 0, sum(w) = s }

    Uses the optimal O(d log d) algorithm by Wang & Carreira-Perpinan (2013).

    Args:
        v: Arbitrary 1D numpy array of shape (d,)
        s: Simplex radius (default 1.0 for probability simplex)

    Returns:
        Projected vector w satisfying w_i >= 0 and sum(w) == s.
    """
    v = np.asarray(v, dtype=np.float64)
    d = len(v)
    if d == 0:
        return v

    # Sort v into descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_indices = np.nonzero(u * np.arange(1, d + 1) > (cssv - s))[0]

    if len(rho_indices) == 0:
        # Fallback in degenerate case
        return np.full(d, s / d, dtype=np.float64)

    rho = rho_indices[-1]
    theta = (cssv[rho] - s) / (rho + 1)
    w = np.maximum(v - theta, 0.0)

    # Numerical normalization to ensure exact unit sum
    total = np.sum(w)
    if total > 0:
        w = (w / total) * s
    else:
        w = np.full(d, s / d, dtype=np.float64)

    return w


project_onto_simplex = project_to_simplex


class FederatedCWSCalibrator:
    """
    Mathematical engine for on-device CWS scoring and federated calibration.
    """

    def __init__(
        self,
        initial_weights: np.ndarray = UTKARSH_BASELINE_CWS_WEIGHTS,
        clipping_bound_cl: float = 1.0,
    ):
        self.clipping_bound_cl = float(clipping_bound_cl)
        self.weights = project_to_simplex(np.copy(initial_weights))

    def project(self, v: np.ndarray, s: float = 1.0) -> np.ndarray:
        return project_to_simplex(v, s)

    def clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(grad)
        if norm > self.clipping_bound_cl and norm > 0:
            return grad * (self.clipping_bound_cl / norm)
        return np.copy(grad)

    def compute_cws(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute CWS score for normalized feature vectors.
        CWS = X @ weights
        """
        return np.dot(features, weights)

    def apply_smoothing(self, current_cws: float, historical_cws: float) -> float:
        """
        Exact Utkarsh 7-day smoothing formulation (cws_engine.dart line 97):
        smoothedCWS = (0.7 * cws) + (0.3 * historyAvg)
        """
        return 0.7 * current_cws + 0.3 * historical_cws

    def compute_loss(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate synthetic protocol-validation MSE loss:
        L(w; X, y) = 1/(2m) * ||Xw - y||_2^2
        """
        m = len(X)
        if m == 0:
            return 0.0
        preds = np.dot(X, weights)
        diff = preds - y
        return float(np.mean(diff ** 2) * 0.5)

    def compute_gradient(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of synthetic objective w.r.t. weights w:
        grad = 1/m * X.T @ (Xw - y)
        """
        m = len(X)
        if m == 0:
            return np.zeros_like(weights)
        preds = np.dot(X, weights)
        diff = preds - y
        grad = np.dot(X.T, diff) / m
        return grad.astype(np.float64)
