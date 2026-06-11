"""
Histotripsy figure generation — Cavitation-Shielding Control (book section 14.12)
=================================================================================

Produces figures for docs/book/histotripsy.md section 14.12, illustrating how
millisecond pulsing and frequency sweeping suppress cavitation shielding. All
physics is computed by kwavers (Rust); this file contains only matplotlib
rendering. Requires pykwavers to be installed.

Output directory: docs/book/figures/ch14/

References
----------
Wang M. (2017) PhD thesis, NTU — HIFU ablation using frequency-sweeping excitation.
Ultrasonics Sonochemistry (2015), PII S1350417715300419 — short frequency-sweep gaps.
LWT — Food Science and Technology (2021), PII S0023643821010756 — pulsed-mode OFF-time dissolution.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch14")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Shared exposure parameters (a partial-shielding focal regime) ------------
DRIVE_PA = 2.0e6          # surface drive pressure [Pa]
F_START, F_END = 1.2e6, 2.0e6   # sweep band [Hz]; mean (1.6 MHz) is the fixed tone
SWEEP_PERIOD = 0.5e-3     # 0.5 ms sweep period (short sweep time)
PROFILE = "triangular"
PULSE_ON, PULSE_OFF = 5.0e-3, 0.4   # 5 ms ON / 400 ms OFF
TOTAL_T, DT = 2.0, 5.0e-4
# Focal-region medium (cycling regime: partial shielding, drive keeps re-cavitating)
MED = dict(k_prod_per_s=30.0, beta_max=1.0e-2, p_threshold_pa=1.0e6, p_ref_pa=1.0e6,
           supralinearity=3.0, c_liquid=1540.0, rho_liquid=1050.0, mu_liquid=1.5e-3,
           p0_pa=101_325.0, polytropic=1.4, r0_m=1.0e-6, alpha_tissue_np_m=2.0,
           path_len_m=0.005, saturation_fraction=0.9)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch14/{name}.{{pdf,png}}")


def _trace(freq_mode: str, pulse_off: float):
    """Run a shielding trace; returns (t, beta, delivered_fraction, summary dict)."""
    (t, beta, deliv_frac, _deliv_p, peak, mean_b, mean_df, e_del, e_uns, loss) = (
        kw.simulate_shielding_trace(
            DRIVE_PA, freq_mode, 0.5 * (F_START + F_END), F_START, F_END, SWEEP_PERIOD,
            PROFILE, PULSE_ON, pulse_off, TOTAL_T, DT, **MED))
    summ = dict(peak=peak, mean_beta=mean_b, mean_df=mean_df,
                e_del=e_del, e_uns=e_uns, loss=loss)
    return np.asarray(t), np.asarray(beta), np.asarray(deliv_frac), summ


def fig_void_fraction_timeseries() -> None:
    """Void fraction and delivered transmission over time for the four exposures."""
    cases = [
        ("CW, fixed", "fixed", 0.0, "#d62728"),
        ("CW, swept", "swept", 0.0, "#ff7f0e"),
        ("pulsed, fixed", "fixed", PULSE_OFF, "#1f77b4"),
        ("pulsed, swept", "swept", PULSE_OFF, "#2ca02c"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for label, mode, off, col in cases:
        t, beta, df, _ = _trace(mode, off)
        ax1.plot(t, beta * 1e3, color=col, label=label, lw=1.4)
        ax2.plot(t, df, color=col, label=label, lw=1.4)
    ax1.set_ylabel(r"void fraction $\beta$ ($\times10^{-3}$)")
    ax1.set_title("Cavitation-shielding control: void fraction and delivered transmission")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_ylabel(r"delivered fraction $p_\mathrm{focus}/p_\mathrm{drive}$")
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig22_shielding_timeseries")
    plt.close(fig)


def fig_control_matrix() -> None:
    """2x2 control matrix: shielding loss and delivered energy per exposure."""
    vals = kw.compare_shielding_control(
        DRIVE_PA, F_START, F_END, SWEEP_PERIOD, PROFILE, PULSE_ON, PULSE_OFF,
        TOTAL_T, DT, **MED)
    rows = ["CW\nfixed", "CW\nswept", "pulsed\nfixed", "pulsed\nswept"]
    # row metrics: [peak, mean_beta, mean_delivered_fraction_on, delivered_energy, loss]
    m = np.asarray(vals).reshape(4, 5)
    loss = m[:, 4]
    mean_df = m[:, 2]  # duty-fair efficacy: transmission while driving
    x = np.arange(4)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2))
    axa.bar(x, loss, color="#d62728", alpha=0.8)
    axa.set_xticks(x); axa.set_xticklabels(rows)
    axa.set_ylabel("shielding loss fraction")
    axa.set_title("Shielding loss (lower is better)")
    axa.grid(True, axis="y", alpha=0.3)
    axb.bar(x, mean_df, color="#2ca02c", alpha=0.8)
    axb.set_xticks(x); axb.set_xticklabels(rows)
    axb.set_ylabel(r"mean delivered fraction while ON")
    axb.set_title("Duty-fair delivered transmission (higher is better)")
    axb.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig("fig23_shielding_control_matrix")
    plt.close(fig)
    print("  control-matrix mean-delivered-fraction-on:",
          ", ".join(f"{r.replace(chr(10), ' ')}={e:.3f}" for r, e in zip(rows, mean_df)))


def fig_prf_sweep() -> None:
    """Final residual void fraction vs OFF interval (the PRF control knob)."""
    offs = np.linspace(0.02, 0.8, 20)
    residual = []
    for off in offs:
        t, beta, _, _ = _trace("fixed", float(off))
        # residual sampled mid-OFF of the last full cycle to avoid burst-phase noise
        residual.append(float(beta[int(0.9 * len(beta))]))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(offs * 1e3, np.asarray(residual) * 1e3, "o-", color="#1f77b4")
    ax.set_xlabel("OFF interval (ms)")
    ax.set_ylabel(r"late residual $\beta$ ($\times10^{-3}$)")
    ax.set_title("OFF-interval clearance: longer OFF dissolves more residual cloud")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig24_prf_clearance")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating histotripsy section 14.12 figures (cavitation-shielding control)...")
    fig_void_fraction_timeseries()
    fig_control_matrix()
    fig_prf_sweep()
    print("Done. Output: docs/book/figures/ch14/")
