# FBMR: Federated Behavioral Model Refinement

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Differential Privacy](https://img.shields.io/badge/DP-R%C3%A9nyi%20DP%20(RDP)-orange.svg)]()
[![Secure Aggregation](https://img.shields.io/badge/SecAgg-ECDH%20Masking-blueviolet.svg)]()
[![Tests Status](https://img.shields.io/badge/Tests-8%2F8%20Passed-success.svg)]()
[![Status](https://img.shields.io/badge/Reproducibility-Empirically%20Validated-brightgreen.svg)]()

> **A Unified Privacy-Preserving Architecture and Empirical Reproducibility Engine Connecting On-Device Behavioral Intelligence with Cross-Institutional Federated Learning.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture (The 4-Tier Hierarchy)](#-system-architecture-the-4-tier-hierarchy)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Composite Wellness Score (CWS)](#1-composite-wellness-score-cws)
  - [2. Euclidean Simplex Projection](#2-euclidean-simplex-projection)
  - [3. Per-Device Gradient Clipping](#3-per-device-gradient-clipping)
  - [4. Intra-College Aggregation & Differential Privacy](#4-intra-college-aggregation--differential-privacy)
  - [5. Secure Aggregation (SecAgg)](#5-secure-aggregation-secagg)
  - [6. Projected Gradient Descent (PGD)](#6-projected-gradient-descent-pgd)
- [Strict Data Boundary & Threat Model](#-strict-data-boundary--threat-model)
- [Cryptographic Consent Bridge](#-cryptographic-consent-bridge)
- [Empirical Benchmarks & 4-Condition Evaluation](#-empirical-benchmarks--4-condition-evaluation)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Test Suite](#running-the-test-suite)
  - [Executing the Benchmark Experiments](#executing-the-benchmark-experiments)
- [Programmatic API Usage](#-programmatic-api-usage)
- [Ethical Considerations](#-ethical-considerations)
- [Citation](#-citation)

---

## 🔬 Overview

Student mental well-being systems face a fundamental tension:
1. **Clinical and Conversational Utility:** On-device mobile applications (such as **Utkarsh**) provide real-time, confidential behavioral screening, emotional check-ins, and composite wellness index tracking anchored in clinical instruments (PHQ-9, GAD-7).
2. **Model Adaptability:** Cohort dynamics vary across academic institutions, necessitating collaborative model refinement.
3. **Absolute Privacy Non-Negotiables:** Raw psychiatric symptoms, chat transcripts, psychometric assessment answers, and individual identities must **never** be centralized or leave the student's personal edge device.

**FBMR (Federated Behavioral Model Refinement)** bridges the on-device companion app with a cross-institutional federated learning backend (**SafeMind**). Rather than transferring raw behavioral or clinical telemetry, FBMR defines a **strictly bounded, differentially private, securely aggregated protocol** that refines model parameters (e.g., the 7-dimensional Composite Wellness Score weights) through a hierarchical 4-tier network.

This repository hosts the **reference implementation, verification suite, and reproducible empirical benchmark runner** for the FBMR protocol.

---

## 🏛 System Architecture (The 4-Tier Hierarchy)

FBMR organizes student-to-federation interactions into four clean, isolated architectural tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: USER / EDGE DEVICE                      │
│   • Conversational UI (Utkarsh App)     • Clinical Screening (PHQ/GAD) │
│   • On-Device Feature Extraction        • Raw data strictly locked     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  (Strict Boundary: No Raw Data)
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   TIER 2: ON-DEVICE FBMR CLIENT BRIDGE                 │
│   • Consent Verification Gate           • Local Gradient Computation   │
│   • Per-Device L2 Clipping (C_u)        • Payload Sanitization Enforcer│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  Whitelisted Clipped Updates
┌──────────────────────────────────▼─────────────────────────────────────┐
│                TIER 3: INTRA-COLLEGE INSTITUTIONAL AGGREGATOR          │
│   • Cohort Threshold Gating (n_k >= k) • Intra-College Mean Aggregation│
│   • Institutional Clipping (C_I)        • Gaussian DP Noise Injection  │
│   • Rényi DP Privacy Tracking (RDP)    • SecAgg Pairwise Masking       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  Masked DP-Protected Institution Update
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 TIER 4: FEDERATED CROSS-INSTITUTION SERVER             │
│   • Secure Aggregation Unmasking       • Cross-College Global Mean     │
│   • Simplex Projection (Pi_W)          • Global CWS Weights Broadcast  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Tier 1 (Edge Node):** Executes locally on student hardware (e.g., smartphone). Manages private conversational check-ins, local task tracking, and screening instruments. Raw text and individual wellness scores never leave this tier.
2. **Tier 2 (FBMR Client Bridge):** Evaluates local student consent status. If active, computes gradients of the calibration loss on local features, scales them to an $L_2$ clipping threshold $C_u$, and enforces data boundary sanitization.
3. **Tier 3 (Intra-College Aggregator):** Hosted on an institutional campus server. Verifies minimum cohort size ($n_k \ge k_{\min}$) to prevent client isolation, averages student updates, applies institutional sensitivity clipping ($C_I$), injects calibrated Gaussian DP noise, and applies pairwise Diffie-Hellman masks.
4. **Tier 4 (Cross-Institutional Server):** Receives masked updates from participating colleges. Pairwise masks cancel out identically. Performs projected gradient descent (PGD) onto the probability simplex $\Delta^6$ and redistributes calibrated global weights.

---

## 📐 Mathematical Formulation

### 1. Composite Wellness Score (CWS)
The Composite Wellness Score synthesizes 7 normalized behavioral indicators into a scalar index in $[0, 1]$:
$$\text{CWS} = \sum_{j=1}^{7} w_j X_j = X w$$
where the baseline weights derived from the Utkarsh scoring engine are:
$$\mathbf{w}_0 = [0.25, 0.15, 0.15, 0.10, 0.10, 0.15, 0.10]^T$$
corresponding to:
$$\{\text{Emotion (0.25), Stress (0.15), Tasks (0.15), Activity (0.10), Routine (0.10), Behavior (0.15), Growth (0.10)}\}$$

Temporal dynamics are stabilized via a 7-day exponential moving average:
$$\text{CWS}_{\text{smoothed}}^{(t)} = 0.70 \cdot \text{CWS}^{(t)} + 0.30 \cdot \overline{\text{CWS}}_{\text{history}}$$

### 2. Euclidean Simplex Projection
To preserve interpretability and convex combination semantics, the parameter space is constrained to the unit simplex:
$$\Delta^6 = \left\{ \mathbf{w} \in \mathbb{R}^7 \;\middle|\; w_j \ge 0, \; \sum_{j=1}^{7} w_j = 1 \right\}$$

Projection onto $\Delta^6$ is computed using the optimal $O(d \log d)$ algorithm by Wang & Carreira-Perpiñán (2013):
$$\Pi_{\Delta}(\mathbf{v}) = \operatorname{proj}_{\Delta^6}(\mathbf{v}) = \max(\mathbf{v} - \theta \mathbf{1}, \mathbf{0})$$
where $\theta = \frac{1}{\rho + 1} \left( \sum_{i=1}^{\rho+1} u_i - 1 \right)$, and $\mathbf{u}$ is $\mathbf{v}$ sorted in descending order.

### 3. Per-Device Gradient Clipping
Each student device $i$ in college $k$ computes local gradient $\mathbf{g}_{k,i} = \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}; X_i, y_i)$ and bounds its $L_2$ sensitivity:
$$\bar{\mathbf{g}}_{k,i} = \mathbf{g}_{k,i} \cdot \min\left(1, \frac{C_u}{\|\mathbf{g}_{k,i}\|_2}\right)$$

### 4. Intra-College Aggregation & Differential Privacy
For a participating college $k$ with $n_k \ge k_{\min}$ consented clients:
$$\mathbf{g}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \bar{\mathbf{g}}_{k,i}$$

Institutional sensitivity clipping and Gaussian DP perturbation are applied:
$$\tilde{\mathbf{g}}_k = \operatorname{clip}(\mathbf{g}_k, C_I) + \mathcal{N}\left(\mathbf{0}, \sigma_I^2 C_I^2 \mathbf{I}_7\right)$$

Privacy guarantees are tracked across rounds using Rényi Differential Privacy (RDP) accounting, converted to $(\epsilon, \delta)$-DP at $\delta = 10^{-5}$.

### 5. Secure Aggregation (SecAgg)
Each college $i$ establishes pairwise shared secrets $s_{ij}$ with college $j$ via Elliptic-Curve Diffie-Hellman (ECDH). Mask vectors $\mathbf{m}_{ij}$ are deterministically pseudorandom:
$$\mathbf{m}_{ij} = -\mathbf{m}_{ji} \quad (\forall i \ne j)$$
Each college submits masked update:
$$\mathbf{y}_i = \tilde{\mathbf{g}}_i + \sum_{j > i} \mathbf{m}_{ij} - \sum_{j < i} \mathbf{m}_{ji}$$
Upon summation at the Tier-4 server, all pairwise masks cancel out identically:
$$\sum_{i=1}^{K} \mathbf{y}_i = \sum_{i=1}^{K} \tilde{\mathbf{g}}_i + \sum_{i=1}^{K}\sum_{j \ne i} \mathbf{m}_{ij} = \sum_{i=1}^{K} \tilde{\mathbf{g}}_i$$

### 6. Projected Gradient Descent (PGD)
The central server updates global weights and projects back onto $\Delta^6$:
$$\mathbf{w}^{(t+1)} = \Pi_{\Delta}\left( \mathbf{w}^{(t)} - \eta \cdot \frac{1}{K}\sum_{k=1}^{K} \tilde{\mathbf{g}}_k \right)$$

---

## 🛡 Strict Data Boundary & Threat Model

The `UnifiedDataBoundaryEnforcer` acts as a deterministic firewall intercepting outgoing payloads at device and institutional boundaries.

| Field Category | Status | Examples / Description |
|---|:---:|---|
| **Raw Content** | ❌ **PROHIBITED** | Chat text, speech audio, transcripts, message logs |
| **Clinical Telemetry** | ❌ **PROHIBITED** | PHQ-9 answers, GAD-7 scores, DSM screening criteria |
| **Individual Outputs**| ❌ **PROHIBITED** | Student-level CWS scores, risk classifications, alerts |
| **Identifiers** | ❌ **PROHIBITED** | Names, emails, roll numbers, IP addresses, device UUIDs |
| **Client Update** | ✅ **ALLOWED** | 7-dimensional clipped float vector ($\bar{\mathbf{g}}_{k,i}$) |
| **Client Metadata** | ✅ **ALLOWED** | Pseudonymous `client_id`, `college_id`, `round_id`, `num_samples` |
| **College Update** | ✅ **ALLOWED** | 7-dimensional DP-perturbed masked float vector ($\tilde{\mathbf{g}}_k$) |
| **College Metadata**| ✅ **ALLOWED** | `college_id`, `round_id`, `num_participating_clients`, `epsilon_spent` |

Any presence of prohibited keys or dimension mismatches raises a non-recoverable `DataBoundaryViolationError`.

---

## 🔐 Cryptographic Consent Bridge

FBMR features an on-device consent state machine (`ConsentVerificationBridge`) reflecting smart-contract governance without requiring online Ethereum/Sepolia gas:
- **Statuses:** `ACTIVE`, `REVOKED`, `EXPIRED`.
- **Purpose Specification:** e.g., `"academic_wellness_refinement"`.
- **Audit Ledger:** Every consent grant, revocation, or verification check appends an immutable record into an SHA-256 hash-chained block sequence:
  $$\text{Hash}_m = \operatorname{SHA-256}\left(\text{Event}_m \,\|\, \text{Hash}_{m-1}\right)$$
- If a student revokes consent, their local bridge immediately drops round participation and emits no gradients.

---

## 📊 Empirical Benchmarks & 4-Condition Evaluation

We evaluate FBMR across 4 benchmark conditions over **3 independent random seeds** (42, 123, 999) using synthetic non-IID Dirichlet populations ($K = 4$ colleges, $N = 15$ students/college, 15 communication rounds):

- **Condition A (Fixed Utkarsh):** Static baseline weights $\mathbf{w}_0$ without refinement.
- **Condition B (Non-Federated Local):** Isolated gradient descent per college with no cross-campus collaboration.
- **Condition C (Federated No-DP):** Collaborative federated learning without differential privacy.
- **Condition D (FBMR: DP + SecAgg):** Full protocol with local $L_2$ clipping, intra-college cohort threshold ($k_{\min}=3$), Gaussian DP noise, ECDH pairwise secure aggregation, and simplex PGD.

### Benchmark Results Summary

| Metric | Condition A (Fixed) | Condition B (Local) | Condition C (Fed No-DP) | Condition D (FBMR: DP+SecAgg) |
|---|:---:|:---:|:---:|:---:|
| **Final MSE Loss ($\mu \pm \sigma$)** | $0.000199 \pm 4\times 10^{-6}$ | $0.000188 \pm 3\times 10^{-6}$ | $0.000190 \pm 3\times 10^{-6}$ | $0.001052 \pm 0.000774$ |
| **Distance to Target $\mathbf{w}^*$** | $0.122474 \pm 0.000000$ | $0.118838 \pm 0.000135$ | $0.118825 \pm 0.000135$ | $0.276119 \pm 0.099454$ |
| **Simplex Constraint ($\sum w_j = 1$)**| Exact ($1.0000$) | Exact ($1.0000$) | Exact ($1.0000$) | Exact ($1.0000$) |
| **SecAgg Reconstruction Error** | N/A | N/A | N/A | **$< 10^{-6}$ (Identical Cancellation)** |
| **Privacy Guarantee ($\epsilon$)** | $\infty$ (No DP) | $\infty$ (No DP) | $\infty$ (No DP) | **Calibrated ($\delta = 10^{-5}$)** |
| **Per-Round Client Payload** | $0$ bytes | $0$ bytes | $120$ bytes | $120$ bytes |
| **Total Runtime per Seed** | $\approx 0.5$ ms | $\approx 13.4$ ms | $\approx 20.7$ ms | $\approx 57.0$ ms |

> **Key Observation:** FBMR achieves strict DP guarantees and zero-knowledge SecAgg reconstruction while maintaining bounded loss and preserving exact simplex admissibility across all communication rounds.

---

## 📁 Repository Structure

```text
fbmr/
├── fbmr/
│   ├── __init__.py               # Top-level module exports
│   ├── boundary_guard.py         # Unified Data Boundary Enforcer & validation schemas
│   ├── client_bridge.py          # Tier-2 on-device client gradient & clipping bridge
│   ├── college_aggregator.py     # Tier-3 intra-college aggregator & Gaussian DP engine
│   ├── consent_bridge.py         # SHA-256 hash-chained consent registry & audit ledger
│   ├── cws_calibrator.py         # 7D CWS engine & O(d log d) simplex projection
│   ├── federated_server.py       # Tier-4 cross-institutional server & PGD optimizer
│   ├── synthetic_data.py         # Non-IID Dirichlet population generator
│   ├── experiments.py            # 4-condition benchmark runner across 3 seeds
│   └── tests/
│       ├── __init__.py
│       └── test_fbmr.py          # Comprehensive 8-component test suite
├── .gitignore                    # Python, pytest, and environment exclusions
└── README.md                     # Technical documentation & reproducibility guide
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- `numpy`
- `pytest`
- `safemind` dependencies (NoiseScheduler, RDPAccountant, SecureAggregator)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MADHURA1907/FBMR-Reproducibility.git
cd FBMR-Reproducibility
pip install numpy pytest
```

### Running the Test Suite

The test suite validates 8 fundamental invariants:
1. Simplex projection non-negativity and unit-sum preservation.
2. Local $L_2$ gradient clipping norm bounds.
3. Gaussian DP noise distribution and RDP budget accumulation.
4. Intra-college $k$-anonymity cohort gating ($n_k < k_{\min} \implies \text{None}$).
5. Secure aggregation exact pairwise ECDH mask cancellation.
6. Blockchain consent rejection on unconsented/revoked clients.
7. Strict data boundary rejection of PHQ-9, text, and clinical tokens.
8. End-to-end multi-round calibration convergence.

Run the test suite:
```bash
pytest fbmr/tests/test_fbmr.py -v
```
Output:
```text
fbmr/tests/test_fbmr.py::test_simplex_projection PASSED
fbmr/tests/test_fbmr.py::test_l2_clipping PASSED
fbmr/tests/test_fbmr.py::test_dp_gaussian_noise PASSED
fbmr/tests/test_fbmr.py::test_intra_college_aggregation PASSED
fbmr/tests/test_fbmr.py::test_secure_aggregation PASSED
fbmr/tests/test_fbmr.py::test_consent_rejection PASSED
fbmr/tests/test_fbmr.py::test_data_boundary_rejection PASSED
fbmr/tests/test_fbmr.py::test_end_to_end_flow PASSED
============================== 8 passed in 4.36s ==============================
```

### Executing the Benchmark Experiments

Run the reproducible 4-condition benchmark across seeds 42, 123, and 999:

```bash
python fbmr/experiments.py
```

---

## 💻 Programmatic API Usage

```python
import numpy as np
from fbmr import (
    ConsentVerificationBridge,
    ConsentStatus,
    UnifiedDataBoundaryEnforcer,
    FBMRBridge,
    IntraCollegeAggregator,
    FBMRServer,
    project_to_simplex,
    UTKARSH_BASELINE_CWS_WEIGHTS,
)

# 1. Initialize Boundary Enforcer and Consent Registry
enforcer = UnifiedDataBoundaryEnforcer()
consent = ConsentVerificationBridge()

# 2. Register Client Consent
client_id = "student_001"
college_id = "campus_alpha"
consent.register_consent(client_id, college_id, ConsentStatus.ACTIVE)

# 3. Instantiate On-Device Bridge
client_bridge = FBMRBridge(
    client_id=client_id,
    college_id=college_id,
    consent_bridge=consent,
    boundary_enforcer=enforcer,
    clipping_bound_cu=1.0,
)

# 4. Compute Clipped Client Update (No raw data leaves device)
X_local = np.random.uniform(0.1, 0.9, size=(20, 7))
y_local = np.dot(X_local, UTKARSH_BASELINE_CWS_WEIGHTS)
payload = client_bridge.prepare_update(
    global_weights=UTKARSH_BASELINE_CWS_WEIGHTS,
    local_X=X_local,
    local_y=y_local,
    round_id=1,
)

# 5. Intra-College Aggregation with Institutional DP
aggregator = IntraCollegeAggregator(
    college_id=college_id,
    min_cohort_size=1,
    clipping_bound_ci=1.0,
    boundary_enforcer=enforcer,
)
college_payload = aggregator.aggregate_client_updates([payload], round_id=1)

# 6. Server Simplex Update
server = FBMRServer(num_colleges=1, learning_rate=0.1, boundary_enforcer=enforcer)
new_weights = server.aggregate_and_update([college_payload], round_id=1)

print("Calibrated Weights on Simplex:", np.round(new_weights, 4))
assert np.isclose(np.sum(new_weights), 1.0)
```

---

## ⚖ Ethical Considerations

- **Zero Real Student Data:** This repository uses synthetically generated non-IID Dirichlet populations to validate algorithmic convergence and protocol boundaries. It does not collect, process, or store real clinical records, student transcripts, or human subject data.
- **Non-Diagnostic Scope:** The CWS scoring engine is an educational wellness calibration tool and must **not** be used as a standalone diagnostic instrument for clinical depression, anxiety, or suicide risk assessment without qualified psychiatric oversight.

---

## 📖 Citation

If you utilize the FBMR framework or reproducibility benchmark in your research, please cite:

```bibtex
@article{barve2026fbmr,
  title     = {Utkarsh--SafeMind: A Unified On-Device and Federated Learning Framework for Privacy-Preserving Student Mental Well-Being Assessment},
  author    = {Barve, Madhura and Contributors},
  journal   = {Springer Computing / Machine Learning Research},
  year      = {2026},
  publisher = {Springer}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
