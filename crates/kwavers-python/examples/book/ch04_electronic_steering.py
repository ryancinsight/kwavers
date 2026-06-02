"""
Chapter 4: Electronic Steering — Figure and Animation Generation Script
=======================================================================

Generates the figures and animated GIFs for the "Electronic Steering" section
of Chapter 4 (Transducer Arrays and Beamforming) of the kwavers book.

ALL physics is computed by kwavers (Rust) through the pykwavers PyO3 bindings.
This file contains ONLY matplotlib rendering — no acoustic computation is done
in Python (no array factors, directivities, delays, focal geometry, field
magnitudes, or normalisation are evaluated here):

  - linear_array_positions     element geometry (centred linear array)
  - near_field_distance        natural focus  N = D²/(4λ)
  - steering_focus_point       focus on the natural-focus arc
  - delay_law_steer_2d         steer+focus delay law on the arc (Eq. 4.11)
  - beam_pattern_magnitude     |D·AF| pattern (pattern-multiplication theorem)
  - grating_lobe_angles        grating-lobe positions (Eq. 4.8)
  - beam_pattern_2d_magnitude  normalised 2-D CW pressure magnitude field

The only numerical operations performed in Python are unit conversions for axis
labelling (rad↔deg, m↔mm, s↔µs) and the cosmetic dB log mapping for display,
both of which are plotting concerns rather than physics.

Output directory: docs/book/figures/ch04/

References
----------
- Szabo (2014) Diagnostic Ultrasound Imaging, Ch. 6–7
- Steinberg (1976) Principles of Aperture and Array System Design
- O'Neil (1949) baffled piston directivity
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, PillowWriter

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch04")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
    "figure.dpi": 150,
})

# --- Shared array definition: 64-element linear array at 2 MHz ----------------
C0 = 1540.0           # sound speed [m/s]
F0 = 2.0e6            # centre frequency [Hz]
LAM = C0 / F0         # wavelength [m]
K0 = 2.0 * np.pi / LAM
N_ELEM = 64
PITCH_HALF = LAM / 2.0          # grating-lobe-free pitch
PITCH_FULL = LAM                # grating-lobe-prone pitch
ELEM_W = 0.9 * PITCH_HALF       # element width (directivity envelope)
KA_ELEM = K0 * ELEM_W / 2.0     # element directivity parameter k·a_elem

# Natural focus (Fresnel near-field transition) of the full aperture — kwavers.
APERTURE = N_ELEM * PITCH_HALF                            # full aperture width D [m]
NATURAL_FOCUS = kw.near_field_distance(APERTURE, F0, C0)  # N = D²/(4λ) [m]


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch04/{name}.{{pdf,png}}")


def linear_db(mag: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Cosmetic dB mapping for an already-normalised (peak=1) magnitude array.

    Pure display transform; the magnitude itself is computed by kwavers.
    """
    floor = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(mag, floor))


# Shared element geometry (kwavers).
EX_H, EZ_H = (np.asarray(a) for a in kw.linear_array_positions(N_ELEM, PITCH_HALF))
WEIGHTS = np.ones(N_ELEM)


# =============================================================================
# Figure 4.12 — Steering delay laws across the aperture (natural-focus arc)
# =============================================================================
print("[fig07] Steering delay laws")

steer_angles = [0.0, 10.0, 20.0, 30.0]
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(steer_angles)))

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for ang, col in zip(steer_angles, colors):
    # Focus on the natural-focus arc at angle `ang` — delay law from kwavers.
    tau = np.asarray(
        kw.delay_law_steer_2d(EX_H, EZ_H, NATURAL_FOCUS, np.deg2rad(ang), C0)
    ) * 1e6  # → microseconds (axis units)
    ax.plot(np.arange(N_ELEM), tau, color=col, marker="o", ms=2.5,
            label=fr"$\theta_s = {ang:.0f}^\circ$")
ax.set_xlabel("Element index $n$")
ax.set_ylabel(r"Transmit delay $\tau_n$ ($\mu$s)")
ax.set_title("Linear-Array Steering Delay Law (Eq. 4.11)\n"
             f"{N_ELEM} elements, $d = \\lambda/2$, focus on natural-focus arc "
             f"$N = {NATURAL_FOCUS*1e3:.0f}$ mm")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig07_steering_delays")
plt.close()


# =============================================================================
# Figure 4.13 — Beam pattern vs steering angle (lambda/2, grating-lobe-free)
# =============================================================================
print("[fig08] Beam pattern vs steering angle")

theta = np.linspace(-90.0, 90.0, 4001)
theta_rad = np.deg2rad(theta)
steer_set = [0.0, 15.0, 30.0, 45.0]
colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(steer_set)))

fig, ax = plt.subplots(figsize=(8.0, 4.6))
for ang, col in zip(steer_set, colors):
    mag = np.asarray(
        kw.beam_pattern_magnitude(theta_rad, K0, PITCH_HALF, N_ELEM,
                                  np.deg2rad(ang), KA_ELEM)
    )
    ax.plot(theta, linear_db(mag), color=col, label=fr"$\theta_s = {ang:.0f}^\circ$")
    ax.axvline(ang, color=col, lw=0.8, ls="--", alpha=0.6)
ax.axhline(-6, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"Observation angle $\theta$ (deg)")
ax.set_ylabel("Beam pattern (dB, norm.)")
ax.set_title(r"Electronic Steering with $d=\lambda/2$: main lobe tracks $\theta_s$, "
             "no grating lobes")
ax.set_xlim(-90, 90)
ax.set_ylim(-60, 3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.legend(ncol=2)
plt.tight_layout()
savefig("fig08_beam_vs_steer")
plt.close()


# =============================================================================
# Figure 4.14 — Grating-lobe onset: d = lambda/2 vs d = lambda at theta_s = 30 deg
# =============================================================================
print("[fig09] Grating-lobe onset vs pitch")

steer = np.deg2rad(30.0)
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
for ax, pitch, tag in zip(axes, (PITCH_HALF, PITCH_FULL),
                          (r"$d=\lambda/2$ (GL-free)", r"$d=\lambda$ (grating lobe)")):
    mag = np.asarray(
        kw.beam_pattern_magnitude(theta_rad, K0, pitch, N_ELEM, steer, KA_ELEM)
    )
    ax.plot(theta, linear_db(mag), lw=1.2)
    ax.axvline(30.0, color="C1", lw=1.0, ls="--", label=r"main lobe $\theta_s=30^\circ$")
    gl = np.rad2deg(np.asarray(kw.grating_lobe_angles(K0, pitch, steer)))
    for j, g in enumerate(gl):
        ax.axvline(g, color="red", lw=1.0, ls=":",
                   label="grating lobe" if j == 0 else None)
        ax.text(g, 1.0, f"{g:.0f}°", color="red", fontsize=8, ha="center")
    ax.axhline(-6, color="k", lw=0.7, ls=":")
    ax.set_title(tag)
    ax.set_xlabel(r"Observation angle $\theta$ (deg)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.legend(loc="lower left")
axes[0].set_ylabel("Beam pattern (dB, norm.)")
fig.suptitle("Grating-Lobe Onset at $\\theta_s = 30^\\circ$ (Eq. 4.8, Corollary 4.2)", y=1.02)
plt.tight_layout()
savefig("fig09_grating_lobe_onset")
plt.close()


# =============================================================================
# Figure 4.15 — 2-D steered + focused CW pressure field (focus on natural arc)
# =============================================================================
print("[fig10] 2-D steered + focused field on natural-focus arc")

# Window spans the natural focus N so the steered focal spot is visible.
x_arr = np.linspace(-0.090, 0.090, 301)
z_arr = np.linspace(0.002, 1.25 * NATURAL_FOCUS, 361)
focus_angles = [0.0, 20.0, -20.0]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), sharey=True)
for ax, ang in zip(axes, focus_angles):
    # Focal point on the natural-focus arc and the steer+focus delays — kwavers.
    xf, zf = kw.steering_focus_point(NATURAL_FOCUS, np.deg2rad(ang))
    delays = np.asarray(
        kw.delay_law_steer_2d(EX_H, EZ_H, NATURAL_FOCUS, np.deg2rad(ang), C0)
    )
    field = np.asarray(
        kw.beam_pattern_2d_magnitude(x_arr, z_arr, EX_H, EZ_H, F0, C0, WEIGHTS, delays)
    )
    field_db = linear_db(field, floor_db=-40.0)
    im = ax.imshow(field_db.T, origin="lower", aspect="auto", cmap="inferno",
                   vmin=-40, vmax=0,
                   extent=[x_arr[0] * 1e3, x_arr[-1] * 1e3,
                           z_arr[0] * 1e3, z_arr[-1] * 1e3])
    ax.plot(xf * 1e3, zf * 1e3, "c+", ms=12, mew=2.0)
    ax.plot(EX_H * 1e3, EZ_H * 1e3, color="cyan", lw=2.5)
    ax.set_title(fr"steer $\theta_s = {ang:.0f}^\circ$")
    ax.set_xlabel("Lateral $x$ (mm)")
axes[0].set_ylabel("Axial $z$ (mm)")
cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Pressure magnitude (dB)")
fig.suptitle(f"Steered + Focused CW Field — focus on natural-focus arc "
             f"$N = {NATURAL_FOCUS*1e3:.0f}$ mm, $d=\\lambda/2$", y=1.03)
savefig("fig10_steered_focused_field")
plt.close()


# =============================================================================
# Figure 4.16 — Focusing around the natural focus (on-axis depth sweep)
# =============================================================================
print("[fig11] Focusing around the natural focus")

# On-axis CW field magnitude for focal ranges below, at, and beyond the natural
# focus N. Beyond N the aperture cannot tighten the beam, so the achievable
# focal gain saturates near N — the depth limit of electronic focusing.
z_axis = np.linspace(0.002, 0.090, 441)
x_axis = np.array([0.0])  # on-axis lateral position
focal_fracs = [0.5, 1.0, 1.5]
colors = plt.cm.cividis(np.linspace(0.1, 0.85, len(focal_fracs)))

fig, ax = plt.subplots(figsize=(8.0, 4.6))
for frac, col in zip(focal_fracs, colors):
    focal_range = frac * NATURAL_FOCUS
    # On-axis focus (theta=0) at the chosen range — delays from kwavers.
    delays = np.asarray(kw.delay_law_steer_2d(EX_H, EZ_H, focal_range, 0.0, C0))
    # Evaluate the field on a 1×Nz on-axis line (x=0) — kwavers.
    field = np.asarray(
        kw.beam_pattern_2d_magnitude(x_axis, z_axis, EX_H, EZ_H, F0, C0, WEIGHTS, delays)
    ).ravel()
    ax.plot(z_axis * 1e3, linear_db(field, floor_db=-30.0), color=col,
            label=fr"focus $= {frac:g}\,N$ ({focal_range*1e3:.0f} mm)")
    ax.axvline(focal_range * 1e3, color=col, lw=0.8, ls="--", alpha=0.6)
ax.axvline(NATURAL_FOCUS * 1e3, color="k", lw=1.2, ls=":",
           label=fr"natural focus $N = {NATURAL_FOCUS*1e3:.0f}$ mm")
ax.set_xlabel("Axial depth $z$ (mm)")
ax.set_ylabel("On-axis pressure (dB, norm.)")
ax.set_title("Focusing Around the Natural Focus $N = D^2/(4\\lambda)$\n"
             "focal gain saturates as the focal range approaches $N$")
ax.set_xlim(z_axis[0] * 1e3, z_axis[-1] * 1e3)
ax.set_ylim(-30, 3)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig11_natural_focus_sweep")
plt.close()


# =============================================================================
# Animation 1 — beam-pattern steering sweep (lambda/2 vs lambda)
# =============================================================================
print("[anim] Beam-pattern steering sweep -> anim_beam_steering.gif")

sweep = np.concatenate([np.linspace(0, 60, 49), np.linspace(60, 0, 49)[1:]])
fig, ax = plt.subplots(figsize=(8.0, 4.6))
line_half, = ax.plot([], [], color="C0", lw=1.4, label=r"$d=\lambda/2$ (GL-free)")
line_full, = ax.plot([], [], color="C3", lw=1.2, alpha=0.8, label=r"$d=\lambda$")
main_marker = ax.axvline(0, color="C1", lw=1.0, ls="--")
gl_markers = [ax.axvline(0, color="red", lw=1.0, ls=":", visible=False) for _ in range(2)]
ax.axhline(-6, color="k", lw=0.7, ls=":")
ax.set_xlim(-90, 90)
ax.set_ylim(-60, 3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.set_xlabel(r"Observation angle $\theta$ (deg)")
ax.set_ylabel("Beam pattern (dB, norm.)")
ax.legend(loc="upper right")
title = ax.set_title("")


def anim1_update(frame_angle):
    sr = np.deg2rad(frame_angle)
    mag_h = np.asarray(kw.beam_pattern_magnitude(theta_rad, K0, PITCH_HALF, N_ELEM, sr, KA_ELEM))
    mag_f = np.asarray(kw.beam_pattern_magnitude(theta_rad, K0, PITCH_FULL, N_ELEM, sr, KA_ELEM))
    line_half.set_data(theta, linear_db(mag_h))
    line_full.set_data(theta, linear_db(mag_f))
    main_marker.set_xdata([frame_angle, frame_angle])
    gl = np.rad2deg(np.asarray(kw.grating_lobe_angles(K0, PITCH_FULL, sr)))
    for m in gl_markers:
        m.set_visible(False)
    for m, g in zip(gl_markers, gl):
        m.set_xdata([g, g])
        m.set_visible(True)
    n_gl = int(np.sum(np.abs(gl) <= 90.0))
    title.set_text(fr"Steering sweep $\theta_s = {frame_angle:.0f}^\circ$ — "
                   fr"$d=\lambda$ grating lobes in field of view: {n_gl}")
    return [line_half, line_full, main_marker, title, *gl_markers]


anim1 = FuncAnimation(fig, anim1_update, frames=sweep, blit=False)
anim1.save(os.path.join(OUT_DIR, "anim_beam_steering.gif"),
           writer=PillowWriter(fps=12), dpi=90)
print("  saved: docs/book/figures/ch04/anim_beam_steering.gif")
plt.close()


# =============================================================================
# Animation 2 — 2-D field steering sweep along the natural-focus arc
# =============================================================================
print("[anim] 2-D field steering sweep -> anim_field_steering.gif")

x_arr_a = np.linspace(-0.110, 0.110, 201)
z_arr_a = np.linspace(0.002, 1.25 * NATURAL_FOCUS, 221)
sweep_field = np.concatenate([np.linspace(-30, 30, 41), np.linspace(30, -30, 41)[1:]])

fig, ax = plt.subplots(figsize=(5.6, 5.2))
zero = np.zeros((len(z_arr_a), len(x_arr_a)))
im = ax.imshow(zero, origin="lower", aspect="auto", cmap="inferno", vmin=-40, vmax=0,
               extent=[x_arr_a[0] * 1e3, x_arr_a[-1] * 1e3,
                       z_arr_a[0] * 1e3, z_arr_a[-1] * 1e3])
ax.plot(EX_H * 1e3, EZ_H * 1e3, color="cyan", lw=2.5)
# Dashed natural-focus arc the focus travels along — geometry from kwavers.
arc_ang = np.linspace(-35, 35, 121)
arc_xz = np.array([kw.steering_focus_point(NATURAL_FOCUS, np.deg2rad(a)) for a in arc_ang])
ax.plot(arc_xz[:, 0] * 1e3, arc_xz[:, 1] * 1e3, color="white", lw=0.8, ls="--", alpha=0.6)
focus_dot, = ax.plot([], [], "c+", ms=13, mew=2.2)
ax.set_xlabel("Lateral $x$ (mm)")
ax.set_ylabel("Axial $z$ (mm)")
cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
cbar.set_label("Pressure magnitude (dB)")
title2 = ax.set_title("")


def anim2_update(frame_angle):
    xf, zf = kw.steering_focus_point(NATURAL_FOCUS, np.deg2rad(frame_angle))
    delays = np.asarray(
        kw.delay_law_steer_2d(EX_H, EZ_H, NATURAL_FOCUS, np.deg2rad(frame_angle), C0)
    )
    field = np.asarray(
        kw.beam_pattern_2d_magnitude(x_arr_a, z_arr_a, EX_H, EZ_H, F0, C0, WEIGHTS, delays)
    )
    im.set_data(linear_db(field, floor_db=-40.0).T)
    focus_dot.set_data([xf * 1e3], [zf * 1e3])
    title2.set_text(fr"Steered + focused field — $\theta_s = {frame_angle:.0f}^\circ$, "
                    fr"focus on natural arc $N={NATURAL_FOCUS*1e3:.0f}$ mm")
    return [im, focus_dot, title2]


anim2 = FuncAnimation(fig, anim2_update, frames=sweep_field, blit=False)
anim2.save(os.path.join(OUT_DIR, "anim_field_steering.gif"),
           writer=PillowWriter(fps=10), dpi=90)
print("  saved: docs/book/figures/ch04/anim_field_steering.gif")
plt.close()

print("Done: Chapter 4 electronic-steering figures and animations generated.")
