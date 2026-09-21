"""
FBMR (Federated Behavioral Model Refinement) Module
===================================================
A unified bridge connecting on-device behavioral intelligence
with cross-institutional federated learning under differential
privacy, secure aggregation, and consent governance.
"""

from .boundary_guard import UnifiedDataBoundaryEnforcer, DataBoundaryViolationError
from .consent_bridge import ConsentVerificationBridge, ConsentStatus
from .cws_calibrator import (
    FederatedCWSCalibrator,
    project_to_simplex,
    UTKARSH_BASELINE_CWS_WEIGHTS,
    CWS_FEATURE_NAMES,
)
from .client_bridge import FBMRBridge
from .college_aggregator import IntraCollegeAggregator
from .federated_server import FBMRServer

__all__ = [
    "UnifiedDataBoundaryEnforcer",
    "DataBoundaryViolationError",
    "ConsentVerificationBridge",
    "ConsentStatus",
    "FederatedCWSCalibrator",
    "project_to_simplex",
    "UTKARSH_BASELINE_CWS_WEIGHTS",
    "CWS_FEATURE_NAMES",
    "FBMRBridge",
    "IntraCollegeAggregator",
    "FBMRServer",
]
