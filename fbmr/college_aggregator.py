"""
Intra-College Aggregator (Tier 3-4 Institutional Node)
=====================================================
Executes on the simulated participating college node:
- Verifies minimum cohort participation threshold (n_k >= k_min)
- Computes intra-college mean of clipped client updates
- Applies institutional clipping (CI)
- Injects Gaussian differential privacy noise using SafeMind's NoiseScheduler
- Tracks privacy expenditure via SafeMind's RDPAccountant
"""

from typing import Dict, List, Optional
import numpy as np

from .boundary_guard import UnifiedDataBoundaryEnforcer, DataBoundaryViolationError


class IntraCollegeAggregator:
    """
    Aggregates student contributions within a single college and applies institutional DP.
    """

    def __init__(
        self,
        college_id: str,
        min_cohort_size: int = 5,
        clipping_bound_ci: float = 1.0,
        noise_scheduler=None,
        rdp_accountant=None,
        boundary_enforcer: Optional[UnifiedDataBoundaryEnforcer] = None,
    ):
        self.college_id = college_id
        self.min_cohort_size = min_cohort_size
        self.clipping_bound_ci = float(clipping_bound_ci)
        self.noise_scheduler = noise_scheduler
        self.rdp_accountant = rdp_accountant
        self.boundary_enforcer = boundary_enforcer or UnifiedDataBoundaryEnforcer()

    def aggregate_client_updates(
        self,
        client_payloads: List[Optional[Dict]],
        round_id: int,
    ) -> Optional[Dict]:
        """
        Aggregate client updates within the college and add institutional DP noise.

        Args:
            client_payloads: List of sanitized client payloads (some may be None if unconsented).
            round_id: Current communication round (1-indexed).

        Returns:
            Protected college update payload, or None if cohort size < min_cohort_size.
        """
        # Filter valid consented client payloads
        valid_payloads = [p for p in client_payloads if p is not None]
        n_k = len(valid_payloads)

        # Enforce minimum participation threshold
        if n_k < self.min_cohort_size:
            return None

        # Intra-college aggregation (Equation 9 of Springer paper)
        # g_k = (1 / n_k) * sum_{i=1}^{n_k} ar{g}_{k,i}
        updates = [p["update"] for p in valid_payloads]
        gk = np.mean(updates, axis=0)

        # Institutional sensitivity clipping (CI)
        norm_gk = np.linalg.norm(gk)
        if norm_gk > self.clipping_bound_ci and norm_gk > 0:
            gk_clipped = gk * (self.clipping_bound_ci / norm_gk)
        else:
            gk_clipped = np.copy(gk)

        # Institutional Gaussian DP Noise (Equation 10 of Springer paper)
        # 	ilde{g}_k = gk_clipped + N(0, \sigma_I^2 * C_I^2 * I)
        sigma = 0.0
        if self.noise_scheduler is not None:
            sigma = float(self.noise_scheduler.get_sigma(round_id))
            noise_std = sigma * self.clipping_bound_ci
            dp_noise = np.random.normal(0.0, noise_std, size=gk_clipped.shape)
            gk_protected = gk_clipped + dp_noise
        else:
            gk_protected = gk_clipped

        # Track privacy budget in RDPAccountant if present
        epsilon_spent = 0.0
        if self.rdp_accountant is not None and sigma > 0:
            # Subsampling rate = 1.0 at college level (all participating colleges contribute)
            self.rdp_accountant.track_round(
                noise_multiplier=sigma,
                sample_rate=1.0,
                num_steps=1,
            )
            epsilon_spent = self.rdp_accountant.get_cumulative_epsilon()

        college_payload = {
            "college_id": self.college_id,
            "round_id": round_id,
            "num_participating_clients": n_k,
            "clipping_bound": self.clipping_bound_ci,
            "noise_sigma": sigma,
            "epsilon_spent": epsilon_spent,
            "update": gk_protected.astype(np.float64),
        }

        # Validate through data boundary enforcer
        sanitized = self.boundary_enforcer.validate_college_payload(college_payload)
        return sanitized
