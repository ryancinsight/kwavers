#!/usr/bin/env python
"""Staged-sonication frequency sweep: a physics-driven demonstration.

Every physical quantity is computed by the kwavers-physics Rust core through the
pykwavers bindings -- nothing is hand-drawn:

  * the staged frequency program, per-pulse cavitation activity, and residual
    void-fraction evolution -> kw.staged_sonication_sweep
  * the cavitation-optimal turn frequency -> kw.cavitation_optimal_frequency
  * per-size peak expansion R_max/R0 at the current stage -> kw.chirped_peak_expansion_ratio

The drive frequency makes ONE slow up-and-down excursion across the whole
sonication (the per-spot pulse train), oriented to the cavitation-optimal
frequency at mid-sonication:

  BUILD half  (stage 0 -> 1/2): the drive moves toward the cavitation-optimal
    frequency, so cavitation activity climbs to a PEAK at mid-sonication; the
    bubble cloud / residual gas builds.
  WIND-DOWN half (stage 1/2 -> 1): the drive returns to the quiet (high-
    threshold) frequency, tapering new cavitation, and the falling sweep clears
    the residual bubbles (fragment + faster Epstein-Plesset dissolution) so the
    NEXT sonication starts from a clean, unshielded field.

Output: docs/book/figures/ch21e/ch21e_swept_frequency_demo.mp4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import lognorm  # display-only PDF; engagement physics is from Rust

import pykwavers as kw

# ── ms boiling-histotripsy regime (per-spot sonication) ──────────────────────
MEDIAN_R = 3.3e-6        # median nucleus radius [m]
SIGMA_G = 1.7            # geometric std of the log-normal nuclei population
AMPLITUDE = 0.15e6       # sub-saturation (cloud-periphery) drive amplitude [Pa]
PULSE = 10e-3            # single ms pulse duration [s]
PRF = 1.0               # per-spot PRF [Hz] -> 1 s inter-pulse interval
N_PULSES = 40           # pulses in the per-spot train (animation frames)
F_BAND = (0.4e6, 1.6e6)  # frequency band the program may use [Hz]
N_SIZE = 18             # nuclei radii for the per-size collapse panel

LN_S = np.log(SIGMA_G)
RADII = np.exp(np.linspace(np.log(MEDIAN_R) - 3 * LN_S,
                           np.log(MEDIAN_R) + 3 * LN_S, N_SIZE))

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                                       "docs", "book", "figures", "ch21e"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "ch21e_swept_frequency_demo.mp4")


def minnaert_radius(freq_hz):
    return (3.0 * 1.4 * 101_325.0 / 1050.0) ** 0.5 / (2.0 * np.pi * freq_hz)


# ── Physics from the Rust core ───────────────────────────────────────────────
print("[demo] finding cavitation-optimal turn frequency (Rust core)...")
f_peak, frac_at_peak = kw.cavitation_optimal_frequency(
    MEDIAN_R, SIGMA_G, F_BAND[0], F_BAND[1], AMPLITUDE, PULSE, n_scan=13, n_size_samples=N_SIZE)
# Quiet endpoint = the band edge with the LEAST cavitation (clean start/end).
f_lo_frac = kw.swept_vs_monochromatic_engagement(  # reuse mono fraction at the edges
    MEDIAN_R, SIGMA_G, F_BAND[0], F_BAND[0] + 1.0, 1e-3, "linear", AMPLITUDE, PULSE,
    n_size_samples=N_SIZE)[0]
f_hi_frac = kw.swept_vs_monochromatic_engagement(
    MEDIAN_R, SIGMA_G, F_BAND[1] - 1.0, F_BAND[1], 1e-3, "linear", AMPLITUDE, PULSE,
    n_size_samples=N_SIZE)[0]
f_quiet = F_BAND[1] if f_hi_frac <= f_lo_frac else F_BAND[0]
print(f"[demo] turn f_peak={f_peak/1e6:.2f} MHz (activity {frac_at_peak:.2f}), "
      f"quiet endpoint={f_quiet/1e6:.2f} MHz")

print("[demo] running staged sonication sweep (Rust core)...")
stage, freq, activity, residual, peak_stage, res_peak, res_end = kw.staged_sonication_sweep(
    MEDIAN_R, SIGMA_G, f_quiet, f_peak, AMPLITUDE, PULSE, N_PULSES, PRF,
    void_deposit_per_activity=0.02, residual_radius_m=6e-6,
    clearing_fragment_count=8.0, saturation_fraction=0.7, n_size_samples=N_SIZE)
stage = np.asarray(stage); freq = np.asarray(freq)
activity = np.asarray(activity); residual = np.asarray(residual)
print(f"[demo] cavitation peaks at stage {peak_stage:.2f}; "
      f"residual peak {res_peak:.4f} -> end {res_end:.4f} "
      f"({res_peak/max(res_end,1e-9):.0f}x cleared in wind-down)")

# Per-size R_max/R0 at each stage frequency (monochromatic = degenerate sweep).
print("[demo] per-size collapse at each stage (Rust core)...")
size_ratio = np.zeros((N_PULSES, N_SIZE))
for i in range(N_PULSES):
    f = float(freq[i])
    for j, r in enumerate(RADII):
        size_ratio[i, j] = kw.chirped_peak_expansion_ratio(
            float(r), f, f, PULSE, "linear", AMPLITUDE, PULSE, steps_per_cycle=48)

r_axis_um = RADII * 1e6
pdf = lognorm.pdf(RADII, LN_S, scale=MEDIAN_R)
pdf_disp = pdf / pdf.max()

# ── Figure / animation ───────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
(axF, axA), (axR, axN) = axes

# (0,0) staged frequency program — one up-down excursion.
axF.plot(stage, freq / 1e6, color="#3070b0", lw=1.8)
axF.axvline(0.5, color="#999999", lw=0.8, ls=":")
axF.axhline(f_peak / 1e6, color="#208040", lw=0.9, ls="--",
            label=f"cavitation-optimal {f_peak/1e6:.2f} MHz")
axF.set_xlim(0, 1); axF.set_ylim(F_BAND[0] / 1e6 * 0.9, F_BAND[1] / 1e6 * 1.05)
axF.set_xlabel("stage of sonication"); axF.set_ylabel("drive frequency  [MHz]")
axF.set_title("staged frequency program (one up-down excursion)")
axF.legend(loc="upper center", fontsize=7.5)
fmark, = axF.plot([], [], "o", color="#d04020", ms=9)

# (0,1) cavitation activity vs stage — build to peak, then wind down.
axA.plot(stage, activity, color="#208040", lw=2.2)
axA.axvline(0.5, color="#999999", lw=0.8, ls=":")
axA.set_xlim(0, 1); axA.set_ylim(0, max(activity.max() * 1.15, 0.05))
axA.set_xlabel("stage of sonication"); axA.set_ylabel("cavitation activity (engaged fraction)")
axA.set_title("cavitation: increase to a peak, then decrease")
amark, = axA.plot([], [], "o", color="#d04020", ms=9)
axA.axvspan(0, 0.5, color="#d04020", alpha=0.05)
axA.axvspan(0.5, 1, color="#208040", alpha=0.05)
axA.text(0.25, axA.get_ylim()[1] * 0.06, "BUILD", ha="center", fontsize=8, color="#a03020")
axA.text(0.75, axA.get_ylim()[1] * 0.06, "WIND-DOWN", ha="center", fontsize=8, color="#206030")

# (1,0) residual void fraction vs stage — builds then cleared.
axR.plot(stage, residual, color="#b05020", lw=2.2)
axR.axvline(0.5, color="#999999", lw=0.8, ls=":")
axR.set_xlim(0, 1); axR.set_ylim(0, max(res_peak * 1.2, 1e-4))
axR.set_xlabel("stage of sonication"); axR.set_ylabel("residual void fraction (shielding)")
axR.set_title("residual builds, then wind-down clears it for the next sonication")
rmark, = axR.plot([], [], "o", color="#d04020", ms=9)

# (1,1) nuclei engaged at the current stage frequency.
axN.set_xscale("log")
axN.plot(r_axis_um, pdf_disp, color="#888888", lw=1.4, label="nuclei population")
axN.set_xlim(r_axis_um.min(), r_axis_um.max()); axN.set_ylim(0, 1.15)
axN.set_xlabel("nucleus radius  [um]"); axN.set_ylabel("relative number density")
axN.set_title("nuclei driven to inertial collapse at the current stage")
eng_scatter = axN.scatter(r_axis_um, np.zeros(N_SIZE), s=40, c="#cccccc",
                          edgecolors="k", linewidths=0.4, zorder=5)
res_line = axN.axvline(minnaert_radius(f_peak) * 1e6, color="#d04020", lw=1.3, ls="--",
                       label="current resonance")
axN.legend(loc="upper right", fontsize=7.5)


def update(i):
    s = stage[i]
    fmark.set_data([s], [freq[i] / 1e6])
    amark.set_data([s], [activity[i]])
    rmark.set_data([s], [residual[i]])
    engaged = size_ratio[i] >= 2.0
    eng_scatter.set_color(np.where(engaged, "#208040", "#cccccc"))
    eng_scatter.set_offsets(np.column_stack([r_axis_um,
                                             np.clip(size_ratio[i] / 4.0, 0, 1.1)]))
    res_line.set_xdata([minnaert_radius(float(freq[i])) * 1e6] * 2)
    phase = "BUILD (recruit cavitation to peak)" if s <= 0.5 else \
            "WIND-DOWN (clear residual, protect next sonication)"
    fig.suptitle(f"Staged-sonication frequency sweep  -  stage {s:.2f}  -  {phase}\n"
                 "(all curves from the kwavers-physics Rust core)",
                 fontsize=12, fontweight="bold")
    return ()


print(f"[demo] rendering {N_PULSES} frames -> {OUT}")
anim = FuncAnimation(fig, update, frames=N_PULSES, blit=False)
writer = FFMpegWriter(fps=6, bitrate=2400,
                      metadata={"title": "Staged-sonication frequency sweep"})
fig.tight_layout(rect=[0, 0, 1, 0.94])
anim.save(OUT, writer=writer, dpi=120)
plt.close(fig)
print(f"[demo] saved {OUT}")
