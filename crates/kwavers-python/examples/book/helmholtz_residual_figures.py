"""
Helmholtz residual analysis on cached PSTD kernels.

Visualises how well the per-voxel peak rarefactional pressure field
of each cached focused-bowl PSTD kernel satisfies the Helmholtz
equation `R(x,y,z) = ∇²p + k²p ≈ 0` (k = 2π·f0/c0). Small interior
residuals validate both the kernel-cube data and the residual
formulation that will drive the parameterised field-surrogate PINN's
physics-residual loss term in Phase C-2.

Outputs
-------
docs/book/figures/ch21e/helmholtz_kernel_<f0>MHz_<pnp>MPa.png
    For each cached kernel: 3-panel figure showing (a) axial
    cross-section of the envelope, (b) axial cross-section of the
    Helmholtz residual, (c) histogram of normalised interior residuals.
docs/book/figures/ch21e/helmholtz_summary.png
    Summary panel comparing the four kernels' RMS-residual ratios.
docs/book/figures/ch21e/helmholtz_summary.txt
    Tabulated stats per kernel.
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(__file__))
from kernel_loader import KERNEL_DIR

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21e")
os.makedirs(OUT_DIR, exist_ok=True)

C0_WATER = 1500.0  # m/s, matches kwavers::physics::field_surrogate::HELMHOLTZ_C0_WATER

_FILE_RE = re.compile(
    r"kernel_(?P<f0>[0-9.]+)MHz_(?P<pnp>[0-9]+)MPa_"
    r"(?P<roc>[0-9]+)roc_(?P<diam>[0-9]+)diam\.npz"
)


def helmholtz_residual_field(p: np.ndarray, dx_m: float, f0: float, c0: float) -> np.ndarray:
    """Finite-difference Helmholtz residual `R = ∇²p + k²p`.

    Mirrors `kwavers::physics::field_surrogate::helmholtz::
    helmholtz_residual_field` (the Rust implementation that drives the
    Phase-C-2 PINN training loss). Boundary voxels are zero.
    """
    nx, ny, nz = p.shape
    out = np.zeros_like(p)
    if nx < 3 or ny < 3 or nz < 3:
        return out
    inv_dx2 = 1.0 / (dx_m ** 2)
    k2 = (2.0 * np.pi * f0 / c0) ** 2
    pc = p[1:-1, 1:-1, 1:-1]
    lap = (
        (p[2:, 1:-1, 1:-1] - 2.0 * pc + p[:-2, 1:-1, 1:-1])
        + (p[1:-1, 2:, 1:-1] - 2.0 * pc + p[1:-1, :-2, 1:-1])
        + (p[1:-1, 1:-1, 2:] - 2.0 * pc + p[1:-1, 1:-1, :-2])
    ) * inv_dx2
    out[1:-1, 1:-1, 1:-1] = lap + k2 * pc
    return out


def make_per_kernel_figure(npz_path: str) -> dict:
    """Render the 3-panel figure for a single kernel and return summary stats."""
    with np.load(npz_path) as d:
        p_min = np.asarray(d["p_min"], dtype=np.float64)
        dx_m = float(d["dx"])
        focus_idx = tuple(int(v) for v in d["focus_idx"])
        f0 = float(d["f0"])
        pnp_realised = float(d["pnp_realised"])

    p_neg = -p_min  # peak rarefactional, positive Pa
    fx, fy, fz = focus_idx

    # Helmholtz residual on the envelope
    R = helmholtz_residual_field(p_neg, dx_m, f0, C0_WATER)

    # Interior stats (boundary shell excluded)
    interior = R[1:-1, 1:-1, 1:-1]
    rms = float(np.sqrt(np.mean(interior ** 2)))
    max_abs = float(np.abs(interior).max())
    p_max_abs = float(np.abs(p_neg).max())
    k = 2.0 * np.pi * f0 / C0_WATER
    norm_ratio = rms / (k * k * p_max_abs) if p_max_abs > 0 else 0.0

    # Axial cross-section through focal voxel
    p_slice = p_neg[:, fy, :].T  # (nz, nx) — display nz as vertical, nx as horizontal
    R_slice = R[:, fy, :].T
    extent = [
        0.0,
        p_neg.shape[0] * dx_m * 1e3,
        -focus_idx[2] * dx_m * 1e3,
        (p_neg.shape[2] - focus_idx[2]) * dx_m * 1e3,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A — envelope cross-section
    im_a = axes[0].imshow(
        p_slice * 1e-6, extent=extent, origin="lower", cmap="inferno",
        aspect="equal",
    )
    axes[0].set_title(
        f"Envelope $|p_-|$ axial slice (y = focus)\n"
        f"f0 = {f0/1e6:.2f} MHz, pnp_realised = {pnp_realised/1e6:.1f} MPa"
    )
    axes[0].set_xlabel("x (axial) [mm]")
    axes[0].set_ylabel("z (lateral) [mm]")
    plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04, label="$|p_-|$ [MPa]")

    # Panel B — Helmholtz residual cross-section
    R_norm = R_slice / (k * k * p_max_abs)
    vlim = max(1e-9, float(np.abs(R_norm).max()))
    im_b = axes[1].imshow(
        R_norm, extent=extent, origin="lower", cmap="seismic",
        vmin=-vlim, vmax=vlim, aspect="equal",
    )
    axes[1].set_title(
        f"Helmholtz residual $R/(k^2 |p|_\\infty)$\n"
        f"$\\lambda$ = {C0_WATER/f0*1e3:.2f} mm, k = {k:.0f} m$^{{-1}}$"
    )
    axes[1].set_xlabel("x (axial) [mm]")
    axes[1].set_ylabel("z (lateral) [mm]")
    plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04, label="normalised residual")

    # Panel C — histogram of |R|/(k²·|p|_∞) over interior voxels
    ratios = np.abs(interior).ravel() / (k * k * p_max_abs)
    axes[2].hist(ratios, bins=60, log=True, color="#1f77b4", edgecolor="black", lw=0.4)
    axes[2].axvline(norm_ratio, color="r", ls="--", lw=1.2,
                     label=f"RMS = {norm_ratio:.3g}")
    axes[2].set(
        xlabel="$|R| / (k^2 |p|_\\infty)$",
        ylabel="voxel count (log)",
        title=f"Interior residual distribution\nmax = {max_abs/(k*k*p_max_abs):.3g}",
    )
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Helmholtz residual analysis — kernel f0 = {f0/1e6:.2f} MHz, "
        f"pnp = {pnp_realised/1e6:.1f} MPa",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(
        OUT_DIR,
        f"helmholtz_kernel_{f0/1e6:.2f}MHz_{pnp_realised/1e6:.0f}MPa.png",
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": out_path,
        "f0_MHz": f0 / 1e6,
        "pnp_MPa": pnp_realised / 1e6,
        "rms_residual_Pa_per_m2": rms,
        "max_abs_Pa_per_m2": max_abs,
        "p_neg_max_MPa": p_max_abs / 1e6,
        "k_per_m": k,
        "normalised_rms_ratio": norm_ratio,
    }


def make_summary_figure(stats: list[dict]) -> str:
    """Bar chart of normalised RMS residual per kernel."""
    if not stats:
        raise SystemExit("No kernels found — run cavitation_kernel.py --sweep first.")
    stats = sorted(stats, key=lambda s: (s["f0_MHz"], s["pnp_MPa"]))
    labels = [f"{s['f0_MHz']:.2f} MHz / {s['pnp_MPa']:.0f} MPa" for s in stats]
    ratios = [s["normalised_rms_ratio"] for s in stats]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(range(len(stats)), ratios, color="#1f77b4", edgecolor="black", lw=0.6)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMS residual / $(k^2 |p|_\\infty)$")
    ax.set_title(
        "Helmholtz residual on cached PSTD kernels\n"
        "(small ratio = envelope is Helmholtz-consistent)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for b, r in zip(bars, ratios):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height(),
            f"{r:.3g}", ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "helmholtz_summary.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary_text(stats: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Helmholtz residual analysis on cached PSTD kernels\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "R(x,y,z) = (laplacian p)(x,y,z) + k^2 p(x,y,z),  "
            "k = 2*pi*f0/c0,  c0 = 1500 m/s.\n"
            "Per-kernel: RMS residual over interior voxels normalised "
            "by k^2 * max|p|.\n\n"
        )
        f.write(
            f"{'f0 (MHz)':>10}{'pnp (MPa)':>12}"
            f"{'k (1/m)':>12}{'|p|_max (MPa)':>16}"
            f"{'RMS R/(k^2|p|)':>20}{'max R/(k^2|p|)':>20}\n"
        )
        f.write("-" * 90 + "\n")
        for s in stats:
            scale = s["k_per_m"] ** 2 * s["p_neg_max_MPa"] * 1e6
            max_norm = s["max_abs_Pa_per_m2"] / scale if scale > 0 else 0.0
            f.write(
                f"{s['f0_MHz']:>10.2f}{s['pnp_MPa']:>12.1f}"
                f"{s['k_per_m']:>12.0f}{s['p_neg_max_MPa']:>16.2f}"
                f"{s['normalised_rms_ratio']:>20.4g}{max_norm:>20.4g}\n"
            )


def main() -> int:
    if not os.path.isdir(KERNEL_DIR):
        raise SystemExit(f"No kernel dir at {KERNEL_DIR}")
    stats: list[dict] = []
    for name in sorted(os.listdir(KERNEL_DIR)):
        if not _FILE_RE.match(name):
            continue
        path = os.path.join(KERNEL_DIR, name)
        print(f"[helmholtz] processing {name}")
        stats.append(make_per_kernel_figure(path))
    summary_png = make_summary_figure(stats)
    summary_txt = os.path.join(OUT_DIR, "helmholtz_summary.txt")
    write_summary_text(stats, summary_txt)
    print(f"[helmholtz] wrote {len(stats)} per-kernel figures + {summary_png} + {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
