"""
Figures for the parameterised field-surrogate PINN training (Phase C-3,
Adam + KernelCubeSampler + Helmholtz residual).

Reads training history + multi-f0 axial-line predictions written by the
Rust example binary `kwavers/examples/field_surrogate_demo.rs`. Run that
first via:

    cargo run --example field_surrogate_demo --release --features pinn

Outputs (under `docs/book/figures/ch21e/`):
    * param_pinn_loss_curves.png  — data + Helmholtz residual loss vs
      training step (log y, smoothed).
    * param_pinn_axial_line_fit.png — network prediction vs analytical
      Penttinen-Gaussian target at three f0 sweep points (corners +
      midpoint).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEMO_DIR = os.path.join(REPO_ROOT, "target", "field_surrogate_demo")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21e")
os.makedirs(OUT_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(DEMO_DIR, "training_history.csv")
AXIAL_CSV = os.path.join(DEMO_DIR, "axial_lines.csv")


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def make_loss_curve_figure() -> str:
    if not os.path.exists(HISTORY_CSV):
        raise SystemExit(
            f"Missing {HISTORY_CSV}. Run:\n"
            f"  cargo run --example field_surrogate_demo --release --features pinn"
        )
    data = np.loadtxt(HISTORY_CSV, delimiter=",", skiprows=1)
    step = data[:, 0]
    data_loss = data[:, 1]
    helm_loss = data[:, 2]
    total_loss = data[:, 3]

    window = max(5, len(step) // 50)
    smoothed_step = step[window - 1:]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(step, data_loss, color="tab:blue", alpha=0.25, lw=0.7)
    ax.plot(smoothed_step, _running_mean(data_loss, window),
            color="tab:blue", lw=2.0,
            label=f"data MSE (smoothed, w={window})")
    ax.plot(step, helm_loss, color="tab:red", alpha=0.25, lw=0.7)
    ax.plot(smoothed_step, _running_mean(helm_loss, window),
            color="tab:red", lw=2.0,
            label=f"Helmholtz residual (smoothed, w={window})")
    ax.plot(step, total_loss, color="black", alpha=0.20, lw=0.7)
    ax.plot(smoothed_step, _running_mean(total_loss, window),
            color="black", lw=1.5, ls="--",
            label="total loss (smoothed)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (dimensionless)")
    ax.set_title(
        "ParamFieldPINN training (Phase C-3)\n"
        "Burn-Autodiff Adam + KernelCubeSampler + dimensionless Helmholtz "
        "residual (NdArray<f32>)"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)

    first_avg = float(np.mean(data_loss[:20]))
    last_avg = float(np.mean(data_loss[-20:]))
    drop = first_avg / max(last_avg, 1e-30)
    ax.text(
        0.02,
        0.05,
        f"data MSE: {first_avg:.2e} -> {last_avg:.2e}  ({drop:.1f}x drop)",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5"),
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "param_pinn_loss_curves.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def make_axial_line_figure() -> str:
    if not os.path.exists(AXIAL_CSV):
        raise SystemExit(f"Missing {AXIAL_CSV}; run the Rust demo first.")
    data = np.loadtxt(AXIAL_CSV, delimiter=",", skiprows=1)
    f0_col = data[:, 0]
    x_col = data[:, 1]
    target_col = data[:, 2]
    pred_col = data[:, 3]

    f0_values = sorted(set(f0_col.tolist()))
    n_panels = len(f0_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, f0 in zip(axes, f0_values):
        mask = np.isclose(f0_col, f0)
        x = x_col[mask]
        order = np.argsort(x)
        x_sorted = x[order]
        target_sorted = target_col[mask][order]
        pred_sorted = pred_col[mask][order]
        ax.plot(x_sorted, target_sorted, color="black", lw=2.0,
                label="target (Penttinen Gaussian)")
        ax.plot(x_sorted, pred_sorted, color="tab:blue", lw=1.8, ls="--",
                label="network prediction (Adam + cosine LR, importance-sampled)")
        ax.axhline(0.0, color="gray", lw=0.6)
        err = pred_sorted - target_sorted
        rmse = float(np.sqrt(np.mean(err ** 2)))
        kind = "training corner" if f0 in (0.5, 1.0) else "held-out midpoint"
        ax.set_title(f"f0 = {f0:.2f} MHz  ({kind})\nRMSE = {rmse:.3g}", fontsize=10)
        ax.set_xlabel(r"$\hat x \in [-1, 1]$")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat p_{\max}$ (normalised)")
            ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "ParamFieldPINN axial-line prediction vs ground truth (3 f0 values)\n"
        "Lateral coordinates fixed at 0; demonstrates generalisation across the "
        "sweep f0 axis",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "param_pinn_axial_line_fit.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    paths = [make_loss_curve_figure(), make_axial_line_figure()]
    for p in paths:
        print(f"[param-pinn] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
