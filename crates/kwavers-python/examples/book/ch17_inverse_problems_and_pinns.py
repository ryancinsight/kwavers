"""
Chapter 17 figure generation — Inverse Problems and PINNs
==========================================================

Produces publication-quality figures for docs/book/inverse_problems_and_pinns.md.

Output directory: docs/book/figures/ch17/

Figures produced
----------------
fig01  Ill-conditioning: singular value spectrum of 2D Helmholtz forward map
fig02  Tikhonov regularization: trade-off curve (L-curve) for 1D deconvolution
fig03  PINN loss landscape: physics-loss + data-loss weighting illustration
fig04  CBS FWI convergence — kwavers Born vs spectral CBS objective history
fig05  Sound speed reconstruction — kwavers Born vs CBS FWI on a 2-D phantom

References
----------
Hansen (1992) Inverse Probl. 8(6):849
Kaipio & Somersalo (2005) Statistical and Computational Inverse Problems. Springer.
Osnabrugge, Leedumrongwatthanakun & Vellekoop (2016) J. Comput. Phys. 322:113
Raissi et al. (2019) J. Comput. Phys. 378:686
"""

from __future__ import annotations

import os
import numpy as np
from scipy.linalg import svd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pykwavers as kw
    _PYKWAVERS = True
except ImportError:
    _PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch17")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch17/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


# ── Figure 01: SVD spectrum of discretised Helmholtz operator ─────────────────
def fig01_svd_spectrum() -> None:
    """
    1D Helmholtz operator: (d²/dx² + k²) discretised with second-order FD.
    Singular values σ_i of the forward map G show rapid decay → ill-conditioning.
    """
    N_arr = [32, 64, 128]
    k = 10.0  # wavenumber

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for N, col in zip(N_arr, colors):
        dx = 1.0 / N
        # Second-order FD Helmholtz matrix (Dirichlet BCs)
        diag = -2.0 / dx**2 + k**2
        offdiag = 1.0 / dx**2
        A = (
            np.diag(diag * np.ones(N))
            + np.diag(offdiag * np.ones(N - 1), 1)
            + np.diag(offdiag * np.ones(N - 1), -1)
        )
        sv = svd(A, compute_uv=False)
        sv_sorted = np.sort(np.abs(sv))[::-1]
        sv_norm = sv_sorted / sv_sorted[0]
        idx = np.arange(1, len(sv_norm) + 1)
        ax.semilogy(idx, sv_norm, color=col, label=f"$N={N}$")

    ax.axhline(1e-10, color="r", linestyle="--", linewidth=1, label="Machine epsilon level")
    ax.set_xlabel("Singular value index $i$")
    ax.set_ylabel(r"Normalised singular value $\sigma_i / \sigma_1$")
    ax.set_title("SVD spectrum of 1D Helmholtz forward operator")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig01_svd_spectrum")
    plt.close(fig)


# ── Figure 02: L-curve for Tikhonov regularization ───────────────────────────
def fig02_lcurve() -> None:
    """
    1D deconvolution: y = Ax + ε, where A is a Gaussian convolution matrix.
    Tikhonov solution: x_λ = (A^T A + λI)^{-1} A^T y.
    L-curve: ||Ax_λ - y|| vs ||x_λ|| as λ varies.
    """
    N = 64
    t = np.linspace(0, 1, N)

    # Gaussian convolution kernel
    sigma = 0.05
    A = np.exp(-0.5 * ((t[:, None] - t[None, :]) ** 2) / sigma**2) / (
        sigma * np.sqrt(2 * np.pi)
    )
    A /= N

    # True signal: sum of Gaussians
    x_true = np.exp(-0.5 * ((t - 0.3) ** 2) / 0.01) + 0.7 * np.exp(
        -0.5 * ((t - 0.7) ** 2) / 0.01
    )

    rng = np.random.default_rng(42)
    y = A @ x_true + 0.01 * rng.standard_normal(N)

    ATA = A.T @ A
    ATy = A.T @ y

    lambdas = np.logspace(-6, 2, 100)
    res_norms = []
    sol_norms = []

    for lam in lambdas:
        x_lam = np.linalg.solve(ATA + lam * np.eye(N), ATy)
        res_norms.append(np.linalg.norm(A @ x_lam - y))
        sol_norms.append(np.linalg.norm(x_lam))

    # L-curve corner (maximum curvature region)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(res_norms, sol_norms, "b-", linewidth=1.5)
    # Mark a few lambda values
    for idx, lam in [
        (10, r"$\lambda=10^{-4}$"),
        (50, r"$\lambda=10^{-1}$"),
        (90, r"$\lambda=10^{1}$"),
    ]:
        ax.scatter(res_norms[idx], sol_norms[idx], s=80, color="r", zorder=5)
        ax.annotate(
            lam,
            (res_norms[idx], sol_norms[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel(r"Residual norm $\|Ax_\lambda - y\|$")
    ax.set_ylabel(r"Solution norm $\|x_\lambda\|$")
    ax.set_title("L-curve for Tikhonov regularization\n1D Gaussian deconvolution")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig02_lcurve")
    plt.close(fig)


# ── Figure 03: PINN loss decomposition ───────────────────────────────────────
def fig03_pinn_loss() -> None:
    """
    PINN total loss: L = L_data + λ_r L_pde + λ_bc L_bc

    Illustrates how the weighting strategy λ_r affects loss convergence rate
    and final data-fit floor.  The curves are parametric exponential models
    L(epoch) = L0 * exp(-epoch / τ) + floor that match the qualitative
    behaviour reported in Raissi et al. (2019) for balanced vs physics-heavy
    vs data-heavy weighting.  The models are not the output of a kwavers PINN
    run (full neural-network training requires several minutes on GPU); they
    are used here to illustrate the sensitivity of loss balance to λ_r without
    claiming to show measured training curves.
    """
    epochs = np.arange(1, 10001)

    # Convergence model: exponential decay with additive noise floor
    def convergence(L0, tau, floor, eps):
        return L0 * np.exp(-eps / tau) + floor

    configs = [
        (r"$\lambda_r=1$ (balanced)", 0.5, 2000, 1e-4, "#1f77b4"),
        (r"$\lambda_r=10$ (physics-heavy)", 0.5, 1000, 5e-5, "#ff7f0e"),
        (r"$\lambda_r=0.1$ (data-heavy)", 0.5, 5000, 5e-4, "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lbl, L0, tau, floor, col in configs:
        L = convergence(L0, tau, floor, epochs)
        ax.semilogy(epochs, L, color=col, label=lbl)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Total PINN loss $\\mathcal{L}$")
    ax.set_title(
        r"PINN convergence illustration: $\mathcal{L} = \mathcal{L}_\mathrm{data}"
        r" + \lambda_r \mathcal{L}_\mathrm{PDE}$"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig03_pinn_loss")
    plt.close(fig)


# ── CBS FWI shared phantom setup ─────────────────────────────────────────────
#
# Small 2-D phantom: 10×10×1 grid at 4 mm spacing (40 mm × 40 mm × 4 mm).
# Ring array: 8 elements, 1 row, radius 28 mm — surrounds the inversion volume.
# True medium: 1500 m/s background with a 3×3 voxel high-speed inclusion (+40 m/s)
# at the centre.  Observations generated with the spectral CBS forward model at
# 200 kHz.  Born inversion uses the single-scatter (linearised) operator;
# CBS inversion uses the spectral periodic convergent Born-series operator
# (Osnabrugge et al. 2016, Theorem 1: contraction guaranteed for ε ≥ ‖V‖_∞).

_FWI_NX, _FWI_NY, _FWI_NZ = 10, 10, 1
_FWI_SPACING_M = 4.0e-3  # 4 mm
_FWI_FREQUENCY_HZ = 200_000.0  # 200 kHz
_FWI_RING_RADIUS_M = 0.028  # 28 mm — outside the 28.3 mm grid half-diagonal
_FWI_RING_ELEMENTS = 8
_FWI_CBS_ITERATIONS = 20  # outer gradient steps
_FWI_BORN_ITERATIONS = 8  # outer gradient steps (linear — converges quickly)


def _build_fwi_truth() -> "np.ndarray":
    truth = np.full((_FWI_NX, _FWI_NY, _FWI_NZ), 1500.0)
    cx, cy = _FWI_NX // 2, _FWI_NY // 2
    truth[cx - 1 : cx + 2, cy - 1 : cy + 2, :] = 1540.0
    return truth


def _run_fwi_pair() -> "tuple[dict, dict, np.ndarray]":
    """
    Run Born and CBS FWI on the shared phantom and return their InversionResult
    dicts plus the true sound speed volume.

    Requires pykwavers.  Callers must check ``_PYKWAVERS`` before calling.
    """
    array = kw.MultiRowRingArray(_FWI_RING_ELEMENTS, 1, _FWI_RING_RADIUS_M, 0.0)
    array = kw.snap_breast_fwi_array_to_grid(
        array, (_FWI_NX, _FWI_NY, _FWI_NZ), _FWI_SPACING_M
    )
    truth = _build_fwi_truth()
    initial = np.full((_FWI_NX, _FWI_NY, _FWI_NZ), 1500.0)

    # ── Spectral CBS config (used for both observation generation and CBS inversion)
    cbs_cfg = kw.FrequencyDomainFwiConfig(
        spacing_m=_FWI_SPACING_M,
        iterations=_FWI_CBS_ITERATIONS,
        propagation_model="spectral_convergent_born",
        cbs_iterations=64,
        cbs_relative_tolerance=1.0e-10,
        estimate_source_scaling=False,
        min_sound_speed_m_s=1450.0,
        max_sound_speed_m_s=1580.0,
    )

    # Simulate observations using the accurate CBS forward model
    obs_2d = kw.simulate_breast_fwi_frequency_observation(
        truth, array, _FWI_FREQUENCY_HZ, cbs_cfg
    )
    obs_stack = obs_2d[np.newaxis, ...]  # shape: (1 freq, transmit, receiver)

    # ── CBS inversion
    cbs_result = kw.invert_breast_fwi(
        [_FWI_FREQUENCY_HZ], obs_stack, array, initial, cbs_cfg
    )

    # ── Born (single-scatter, linearised) inversion — same observations
    born_cfg = kw.FrequencyDomainFwiConfig(
        spacing_m=_FWI_SPACING_M,
        iterations=_FWI_BORN_ITERATIONS,
        propagation_model="single_scatter_born",
        estimate_source_scaling=False,
        min_sound_speed_m_s=1450.0,
        max_sound_speed_m_s=1580.0,
    )
    born_result = kw.invert_breast_fwi(
        [_FWI_FREQUENCY_HZ], obs_stack, array, initial, born_cfg
    )

    return born_result, cbs_result, truth


# ── Figure 04: CBS FWI convergence ───────────────────────────────────────────
def fig04_convergence_comparison() -> None:
    """
    CBS FWI objective history for Born (single-scatter) and spectral CBS
    operators on a 10×10×1 phantom with a 3×3 voxel high-speed inclusion.

    The objective J(s) = (1/2) ‖F(s) - d‖² is minimised by nonlinear
    conjugate-gradient descent.  The Born operator provides a one-step
    linearised reconstruction; the spectral CBS operator (Osnabrugge et al.
    2016) iterates over the full Lippmann-Schwinger series until the CBS
    fixed-point residual falls below tolerance, then takes a gradient step.

    Data source: kwavers pykwavers CBS FWI (``invert_breast_fwi``).
    """
    if not _PYKWAVERS:
        print("  [skip] fig04 requires pykwavers — install and rebuild")
        return

    print("  running CBS FWI for fig04/fig05 (Born + spectral CBS)...")
    born_result, cbs_result, _ = _run_fwi_pair()

    born_obj = np.asarray(born_result["objective_history"], dtype=float)
    cbs_obj = np.asarray(cbs_result["objective_history"], dtype=float)

    # Normalize by initial objective so both curves start at 1.0
    born_norm = born_obj / born_obj[0] if born_obj[0] > 0 else born_obj
    cbs_norm = cbs_obj / cbs_obj[0] if cbs_obj[0] > 0 else cbs_obj

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(
        np.arange(len(born_norm)),
        born_norm,
        "s--",
        color="#1f77b4",
        markersize=4,
        label=f"Born / single-scatter ({len(born_norm)} iters)",
    )
    ax.semilogy(
        np.arange(len(cbs_norm)),
        cbs_norm,
        "o-",
        color="#d62728",
        markersize=4,
        label=f"Spectral CBS (Osnabrugge 2016, {len(cbs_norm)} iters)",
    )
    ax.set_xlabel("Gradient-descent iteration")
    ax.set_ylabel(r"Normalised objective $J / J_0$")
    ax.set_title(
        "CBS FWI convergence: Born vs spectral CBS\n"
        f"10×10×1 phantom, 200 kHz, {_FWI_RING_ELEMENTS}-element ring"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_convergence_comparison")
    plt.close(fig)


# ── Figure 05: Sound speed reconstruction ────────────────────────────────────
def fig05_sound_speed_reconstruction() -> None:
    """
    2-D sound speed phantom reconstructed by Born and spectral CBS FWI.

    True phantom: 1500 m/s background with a 3×3 voxel high-speed inclusion
    (+40 m/s) at the centre.  The figure shows the centre-row profile
    c(x, y = N/2, z = 0) for the true, Born-inverted, and CBS-inverted media.

    Born inversion uses the single-scatter (linearised) Green operator; CBS
    inversion uses the full Lippmann-Schwinger series with the spectral
    periodic Green operator.  At 40 m/s contrast the Born approximation
    underestimates the peak speed and broadens the anomaly; the CBS operator
    resolves the inclusion more faithfully.

    Data source: kwavers pykwavers CBS FWI (``invert_breast_fwi``).
    """
    if not _PYKWAVERS:
        print("  [skip] fig05 requires pykwavers — install and rebuild")
        return

    born_result, cbs_result, truth = _run_fwi_pair()

    cy = _FWI_NY // 2
    x_mm = np.arange(_FWI_NX) * _FWI_SPACING_M * 1e3  # mm

    c_true = truth[:, cy, 0]
    c_born = np.asarray(born_result["sound_speed_m_s"])[:, cy, 0]
    c_cbs = np.asarray(cbs_result["sound_speed_m_s"])[:, cy, 0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_mm, c_true, "k-", linewidth=2.5, label="True $c(x)$")
    ax.plot(
        x_mm,
        c_born,
        "--",
        color="#1f77b4",
        linewidth=2.0,
        label=f"Born inversion ({_FWI_BORN_ITERATIONS} iters)",
    )
    ax.plot(
        x_mm,
        c_cbs,
        "-.",
        color="#d62728",
        linewidth=2.0,
        label=f"Spectral CBS ({_FWI_CBS_ITERATIONS} iters)",
    )
    ax.set_xlabel("Position $x$ (mm)")
    ax.set_ylabel("Speed of sound $c$ (m/s)")
    ax.set_title(
        "Sound speed reconstruction: Born vs spectral CBS\n"
        f"centre-row profile, 10×10×1 phantom, 200 kHz"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig05_sound_speed_reconstruction")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 17 figures (Inverse Problems and PINNs)...")
    fig01_svd_spectrum()
    fig02_lcurve()
    fig03_pinn_loss()
    fig04_convergence_comparison()
    fig05_sound_speed_reconstruction()
