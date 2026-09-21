"""
Synthetic Non-IID Data Generator for FBMR Protocol Validation
==============================================================
Generates synthetic student feature matrices and synthetic supervision targets.
Adheres strictly to zero-real-student-data ethics:
- Features are synthetic normalized floats in [0, 1]^7
- Supervision label y is generated mathematically: y = X @ w*_true + noise
- Evaluates protocol convergence without real mental health or clinical datasets.
"""

from typing import Dict, List, Tuple
import numpy as np

from .cws_calibrator import project_to_simplex


# Ground-truth synthetic target for protocol convergence verification
SYNTHETIC_TRUE_WEIGHTS = project_to_simplex(
    np.array([0.30, 0.20, 0.10, 0.15, 0.05, 0.10, 0.10], dtype=np.float64)
)


def generate_synthetic_college_data(
    num_colleges: int = 4,
    students_per_college: int = 15,
    samples_per_student: int = 40,
    synthetic_target_weights: np.ndarray = SYNTHETIC_TRUE_WEIGHTS,
    seed: int = 42,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate non-IID synthetic college populations.

    Args:
        num_colleges: Number of simulated colleges
        students_per_college: Number of simulated student devices per college
        samples_per_student: Number of interaction rows per student
        synthetic_target_weights: Ground-truth synthetic weight vector in Simplex
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping college_id to list of (X, y) client tuples.
    """
    rng = np.random.default_rng(seed)
    colleges_data = {}

    # Dirichlet concentration vectors to generate non-IID shifts per college
    # e.g., College 0 has higher stress/tasks, College 1 has higher routine/growth
    college_dirichlet_alphas = [
        [3.0, 4.0, 4.0, 1.5, 1.5, 2.0, 1.0],  # College 0: Academic pressure / high task load
        [1.5, 2.0, 2.0, 4.0, 3.5, 3.0, 2.0],  # College 1: High activity and routine
        [2.5, 2.5, 2.0, 2.0, 2.0, 2.5, 3.5],  # College 2: Balanced / personal growth focus
        [4.0, 2.0, 1.5, 3.0, 2.0, 3.0, 1.5],  # College 3: High emotional variation
    ]

    for cid in range(num_colleges):
        college_key = f"college_{cid}"
        alpha = college_dirichlet_alphas[cid % len(college_dirichlet_alphas)]
        students_list = []

        for sid in range(students_per_college):
            # Generate feature proportions from Dirichlet, scaled to [0, 1]
            student_alpha = np.maximum(0.5, rng.dirichlet(alpha) * 10.0)
            X = rng.dirichlet(student_alpha, size=samples_per_student)

            # Synthetic supervision target generated strictly mathematically:
            # y = X @ w_true + Gaussian noise
            noise = rng.normal(0.0, 0.01, size=samples_per_student)
            y = np.dot(X, synthetic_target_weights) + noise
            y = np.clip(y, 0.0, 1.0)

            students_list.append((X.astype(np.float64), y.astype(np.float64)))

        colleges_data[college_key] = students_list

    return colleges_data
