#!/usr/bin/env python3
"""Validation of the kwavers cavitation / bubble physics against analytical theory.

Why not k-Wave here
-------------------
The other ``*_compare.py`` examples validate kwavers' **acoustic propagation**
against k-wave-python / k-Wave.jl, because that is the physics k-Wave solves.
k-Wave is a (linear/nonlinear) *grid acoustic* solver: it has **no bubble
model** — no Rayleigh-Plesset / Keller-Miksis ODE, no bubbly-medium dispersion.
So the recent cavitation-cloud work (ADRs 027-030) has no k-Wave counterpart to
compare against; the *correct* oracle for bubble dynamics is **analytical bubble
theory**. (For the k-Wave acoustic parity that underlies the therapy *driving*
field, see the ``at_*_compare.py`` suite.)

This script validates the three pillars the cavitation cloud is built on, each
against an independent closed-form reference, and writes plots + metrics:

  1. Keller-Miksis linear resonance  -> peaks at the Minnaert frequency f0.
  2. Wood mixture sound speed        -> matches the closed-form Wood equation.
  3. Commander-Prosperetti attenuation -> peaks at the bubble resonance.

Run (from crates/kwavers-python, in the maturin venv):
    .venv/Scripts/python examples/cavitation_bubble_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pykwavers as pk  # noqa: E402

# ── Water / air properties at ~20 C ───────────────────────────────────────────
C_L, RHO_L = 1481.0, 998.0          # liquid sound speed [m/s], density [kg/m^3]
C_G, RHO_G = 340.0, 1.2             # gas sound speed [m/s], density [kg/m^3]
MU, SIGMA = 1.0e-3, 0.0725          # viscosity [Pa s], surface tension [N/m]
GAMMA, PV = 1.4, 2.34e3             # polytropic index, vapour pressure [Pa]
P0 = 1.013e5                        # ambient pressure [Pa]
R0 = 5.0e-6                         # equilibrium bubble radius [m]

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def keller_miksis_resonance_curve() -> dict:
    """Forced small-amplitude amplitude vs drive frequency; peaks at the
    surface-tension-corrected Minnaert resonance (the oracle matching the KM
    physics, which includes the 2σ/R0 stiffening the plain Minnaert omits)."""
    f0_plain = pk.minnaert_resonance_hz(R0, GAMMA, P0, RHO_L)
    f0 = pk.minnaert_resonance_corrected_hz(R0, GAMMA, P0, RHO_L, SIGMA)
    # Fine sweep around the corrected resonance for a precise peak location.
    freqs = np.linspace(0.6 * f0, 1.5 * f0, 60)
    p_ac = 2.0e3  # 2 kPa — linear regime (well below the Blake threshold)
    n_cycles, steps_per_cycle = 40, 200
    amp = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        t_end = n_cycles / f
        _, r, _ = pk.solve_keller_miksis(
            R0, 0.0, P0, p_ac, f, t_end, n_cycles * steps_per_cycle,
            RHO_L, SIGMA, GAMMA, MU, PV, C_L,
        )
        tail = r[-10 * steps_per_cycle:]          # last 10 cycles -> steady state
        amp[i] = (tail.max() - tail.min()) / R0   # peak-to-peak / R0
    f_peak = freqs[int(np.argmax(amp))]
    return {"f0": f0, "f0_plain": f0_plain, "freqs": freqs, "amp": amp,
            "f_peak": f_peak, "rel_err": abs(f_peak - f0) / f0}


def wood_sound_speed_check() -> dict:
    """kwavers Wood mixture sound speed vs the independent closed-form Wood eq."""
    beta = np.logspace(-6, np.log10(0.5), 40)
    kw = np.array([pk.wood_sound_speed(b, C_L, RHO_L, C_G, RHO_G) for b in beta])
    # Closed form: 1/(rho_m c_m^2) = (1-b)/(rho_l c_l^2) + b/(rho_g c_g^2).
    rho_m = (1.0 - beta) * RHO_L + beta * RHO_G
    kappa_m = (1.0 - beta) / (RHO_L * C_L**2) + beta / (RHO_G * C_G**2)
    c_ref = 1.0 / np.sqrt(rho_m * kappa_m)
    rel = np.abs(kw - c_ref) / c_ref
    return {"beta": beta, "kw": kw, "ref": c_ref, "max_rel_err": float(rel.max())}


def attenuation_peak_check() -> dict:
    """Commander-Prosperetti attenuation vs frequency; peaks near the resonance."""
    f0 = pk.minnaert_resonance_hz(R0, GAMMA, P0, RHO_L)
    beta = 1.0e-4
    freqs = np.linspace(0.2 * f0, 3.0 * f0, 200)
    alpha = np.array([
        pk.bubbly_cloud_attenuation(f, beta, R0, C_L, RHO_L, MU, P0, GAMMA)
        for f in freqs
    ])
    f_peak = freqs[int(np.argmax(alpha))]
    return {"f0": f0, "freqs": freqs, "alpha": alpha, "f_peak": f_peak,
            "rel_err": abs(f_peak - f0) / f0}


def main() -> int:
    res = keller_miksis_resonance_curve()
    wood = wood_sound_speed_check()
    att = attenuation_peak_check()

    # ── Pass/fail against analytical thresholds (derived, not tuned) ──────────
    checks = {
        "KM resonance peaks at corrected Minnaert f0 (<5%)": res["rel_err"] < 0.05,
        "Wood c matches closed form (<1e-6)": wood["max_rel_err"] < 1e-6,
        "CP attenuation peaks near resonance (<35%)": att["rel_err"] < 0.35,
    }

    f0_khz = res["f0"] / 1e3
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(res["freqs"] / 1e3, res["amp"], "o-", ms=3, label="Keller-Miksis")
    ax[0].axvline(f0_khz, color="r", ls="--",
                  label=f"corrected Minnaert f0 = {f0_khz:.0f} kHz")
    ax[0].axvline(res["f0_plain"] / 1e3, color="gray", ls=":",
                  label=f"plain Minnaert = {res['f0_plain']/1e3:.0f} kHz")
    ax[0].set(xlabel="drive frequency [kHz]", ylabel="peak-to-peak ΔR / R0",
              title="1. Bubble resonance (R0 = 5 µm)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].semilogx(wood["beta"], wood["kw"], "-", lw=2, label="kwavers")
    ax[1].semilogx(wood["beta"], wood["ref"], "k--", lw=1, label="Wood closed form")
    ax[1].set(xlabel="void fraction β", ylabel="mixture sound speed [m/s]",
              title=f"2. Wood mixture c (max rel err {wood['max_rel_err']:.1e})")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].plot(att["freqs"] / 1e3, att["alpha"], "-", lw=2, label="C-P attenuation")
    ax[2].axvline(f0_khz, color="r", ls="--", label=f"resonance f0 = {f0_khz:.0f} kHz")
    ax[2].set(xlabel="frequency [kHz]", ylabel="α [Np/m]",
              title="3. Bubbly-cloud attenuation (β = 1e-4)")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    fig.suptitle("kwavers cavitation/bubble physics vs analytical theory "
                 "(k-Wave has no bubble model — see docstring)")
    fig.tight_layout()
    png = OUT / "cavitation_bubble_validation.png"
    fig.savefig(png, dpi=150)

    lines = [
        "kwavers cavitation/bubble validation vs analytical theory",
        "=" * 58,
        f"Minnaert f0 (plain)       : {res['f0_plain']/1e3:8.1f} kHz",
        f"Minnaert f0 (w/ sigma)    : {res['f0']/1e3:8.1f} kHz",
        f"KM resonance peak         : {res['f_peak']/1e3:8.1f} kHz "
        f"(rel err {res['rel_err']*100:.1f}% vs corrected)",
        f"Wood c max rel err        : {wood['max_rel_err']:.2e}",
        f"CP attenuation peak       : {att['f_peak']/1e3:8.1f} kHz "
        f"(rel err {att['rel_err']*100:.1f}%)",
        "",
    ]
    all_pass = True
    for name, ok in checks.items():
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_pass &= ok
    report = "\n".join(lines)
    (OUT / "cavitation_bubble_validation_metrics.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved {png}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
