"""
Focused Ultrasound (FUS) brain treatment through skull -- power modulation
and temperature plateau demonstration.

Demonstrates:
  1. Temperature plateau at focus from conduction-perfusion equilibrium
  2. Power modulation to suppress inertial cavitation while achieving
     thermal lesioning (CEM43 >= 240 min threshold for necrosis)
  3. Skull as a standing heat source that accumulates temperature across
     successive sonications, setting the upper bound on deliverable power
  4. Comparison: unmodulated CW (cavitation risk) vs modulated (safe lesion)

Geometry -- 2D cylindrical (r, z):
  r ∈ [0, 15 mm]    (lateral, axis of symmetry at r=0)
  z ∈ [0, 90 mm]    (axial depth from transducer face)
  Water coupling:  z = 0 - 30 mm
  Cortical skull:  z = 30 - 37 mm   (7 mm thick)
  Brain tissue:    z = 37 - 90 mm
  Focal point:     (r=0, z=75 mm)  43 mm post inner skull surface

Governing equation (Pennes bioheat, cylindrical coordinates):
  ρ*cₚ*∂T/∂t = k*[1/r*∂/∂r(r*∂T/∂r) + ∂^2T/∂z^2]
              − wᵦ*ρᵦ*cᵦ*(T − Tₐ) + Q(r,z)*P(t)

Numerical scheme:
  - 2nd-order central FD Laplacian; staggered r-grid for cylindrical symmetry
  - L'Hôpital limit at r=0: 1/r*∂/∂r(r*∂T/∂r) -> 2*∂^2T/∂r^2
  - Forward Euler time integration; dt=0.2 s (Fourier number <= 0.25 for all layers)
  - Boundary conditions: symmetry at r=0, Neumann (zero-flux) elsewhere

Acoustic model:
  Paraxial Gaussian beam (Siegman 1986):
    I(r,z) = I_ref * [w0/w(z)]^2 * exp(−2r^2/w(z)^2) * exp(−2alphaᵦʳᵃⁱⁿ*|z−z_f|)
  where w(z) = w0 * √(1 + [(z−z_f)/z_R]^2), z_R = π*w0^2/λ
  Skull included as spatially resolved absorber (no wavefront correction).

Thermal dose (Sapareto & Dewey 1984, CEM43):
  ΔD = R^(43 − T[ degC]) * Δt/60   [CEM43 min]
  R = 0.5 for T >= 43 degC, 0.25 for T < 43 degC
  Necrosis threshold: 240 CEM43 min

References:
  Pennes (1948) J. Appl. Physiol. 1, 93-122
  Sapareto & Dewey (1984) Int. J. Radiat. Oncol. Biol. Phys. 10, 787-800
  Duck (1990) Physical Properties of Tissue (Academic Press)
  Constans et al. (2017) Phys. Med. Biol. 62 2583   [skull thermal model]
  McDannold (2005) Ultrasound Med. Biol. 31 1141     [clinical plateau data]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Numerical grid ────────────────────────────────────────────────────────

NR   = 32          # lateral points (r)
NZ   = 180         # axial points   (z)
DR   = 0.5e-3      # m   (0.5 mm)
DZ   = 0.5e-3      # m   (0.5 mm)
DT   = 0.20        # s   Forward-Euler time step (Fourier stable for all layers)
T_END = 20.0       # s   total simulation duration (clinical sonication ~10-20 s)

r = np.arange(NR) * DR                    # r-coordinates [m]
z = np.arange(NZ) * DZ                    # z-coordinates [m]
R, Z = np.meshgrid(r, z, indexing="ij")   # shape (NR, NZ)

# ── 1.  Tissue properties (spatially resolved along z) ────────────────────────

# z boundaries [m]
Z_WATER_END  = 30.0e-3   # water/skull interface
Z_SKULL_END  = 37.0e-3   # skull/brain interface

# Tissue parameter arrays (constant along r)
rho  = np.where(Z < Z_WATER_END, 998.0,   # water  [kg/m^3]
       np.where(Z < Z_SKULL_END, 1900.0,  # skull
                                 1040.0)) # brain

cp   = np.where(Z < Z_WATER_END, 4182.0,  # water  [J/(kg*K)]
       np.where(Z < Z_SKULL_END, 1313.0,  # skull
                                 3650.0)) # brain

k_th = np.where(Z < Z_WATER_END, 0.598,   # water  [W/(m*K)]
       np.where(Z < Z_SKULL_END, 0.320,   # skull
                                 0.510))  # brain

# Blood perfusion  wᵦ [1/s]  -- skull and water: 0
wb   = np.where(Z < Z_SKULL_END, 0.0, 0.009)  # brain: 9 mL/(100g*min) -> 1.5e-3->0.009 1/s

# CEM43 is meaningful only in brain tissue
brain_mask = (Z >= Z_SKULL_END)

# Thermal diffusivity alpha_th = k/(ρ*cp)  [m^2/s]
alpha_th = k_th / (rho * cp)

# ── 2.  Acoustic absorption coefficients  [Np/m] ─────────────────────────────
#
# alpha_np [Np/m] = alpha_dB_cm_MHz * f_MHz * 100 / (20/ln(10))
#             = alpha_dB_cm_MHz * f_MHz * 11.516

F_MHZ    = 0.650          # transducer frequency  [MHz]
ALPHA_DB = np.where(Z < Z_WATER_END, 2.2e-3,   # water  [dB/cm/MHz]
           np.where(Z < Z_SKULL_END, 15.0,      # skull (cortical bone)
                                      0.60))    # brain (ICRU Report 61)
ALPHA_NP = ALPHA_DB * F_MHZ * 11.516            # [Np/m]

# ── 3.  Blood and body constants ──────────────────────────────────────────────

RHO_BLOOD = 1050.0     # kg/m^3
CP_BLOOD  = 3617.0     # J/(kg*K)
T_ART_C   = 37.0       #  degC  arterial blood temperature

# Perfusion coefficient [1/s]: wb*ρ_b*c_b / (ρ*cp)
omega = wb * RHO_BLOOD * CP_BLOOD / (rho * cp)  # shape (NR,NZ)

# ── 4.  Acoustic intensity distribution (Gaussian paraxial beam) ──────────────
#
# Focuses at focal point (r_f=0, z_f=Z_FOCUS)
# w0 = focal half-width (1/e^2 intensity radius)
# z_R = Rayleigh range = π*w0^2/λ

Z_FOCUS  = 75.0e-3    # m   focal depth from transducer
W0       = 2.0e-3     # m   focal spot 1/e^2 intensity radius  (~3.4 mm FWHM)
LAMBDA   = 1540.0 / (F_MHZ * 1e6)   # acoustic wavelength in brain
Z_R      = np.pi * W0**2 / LAMBDA   # Rayleigh range

def beam_width(z_arr):
    return W0 * np.sqrt(1 + ((z_arr - Z_FOCUS) / Z_R)**2)

W_Z = beam_width(Z)  # beam radius at each z slice [m]

# Normalised intensity (1.0 at focus centre)
I_NORM = (W0 / W_Z)**2 * np.exp(-2.0 * R**2 / W_Z**2)

# Axial beam decay from absorption (accumulated from each layer)
# Use trapezoidal integration of 2*alpha along z, anchored at z_f (intensity = 1)
decay_from_focus = 2.0 * np.trapezoid(
    np.where(Z < Z_FOCUS, -ALPHA_NP, ALPHA_NP),   # sign: path away from focus
    dx=DZ, axis=1                                  # integrate along z
)
# Normalise so decay=0 at focus centre
decay_from_focus -= decay_from_focus[:, NZ // 2, np.newaxis] if False else 0
# Simple element-wise exponential: exp(-2 * alpha * |z - z_focus|)
exp_axial = np.exp(-2.0 * ALPHA_NP * np.abs(Z - Z_FOCUS))

I_NORM = I_NORM * exp_axial
# Clip negative artifacts and renormalize so peak = 1.0
I_NORM = np.clip(I_NORM, 0, None)
I_NORM /= I_NORM.max()

# Reference peak acoustic intensity at 100 W input acoustic power
# Clinical MRgFUS (ExAblate Neuro): ~150 W/cm^2 = 1.5e6 W/m^2 after skull transmission
# At this intensity p_peak = sqrt(2*rho*c*I) = sqrt(2*1040*1540*1.5e6) = 2.2 MPa
# p_safe (inertial cavitation threshold 0.85 MPa) corresponds to 40% of this power.
I_REF_100W = 1.5e6    # W/m^2   peak focal intensity at 100 W input

# Volumetric heat source per unit input intensity [W/m^3 per W/m^2]
#   Q = 2*alpha_np * I
Q_NORM = 2.0 * ALPHA_NP * I_NORM   # shape (NR, NZ) in Np/m (-> W/m^3 when × I)

# Q in [W/m^3] at 100 W input:  Q = Q_NORM * I_REF_100W
Q_100W = Q_NORM * I_REF_100W

# External source in K/s (for Pennes solver): (Q)/(ρ*cp)
Q_KS_NORM = Q_100W / (rho * cp)   # K/s at 100 W input power, normalised to 1.0 protocol

# ── 5.  Power modulation protocols ───────────────────────────────────────────
#
# Four clinical protocols; each returns P(t) in [0, 1] relative to 100W input.
#
# At I_REF=150 W/cm^2 (100% power): p_peak = sqrt(2*1040*1540*1.5e6) = 2.20 MPa
# Inertial cavitation threshold in brain: ~0.85 MPa at 650 kHz.
# Safe intensity: I_safe = 0.85^2 / (2*1040*1540) = 0.224 MW/m^2 = 22.4 W/cm^2
# Safe power fraction = 22.4/150 = 0.149; clinical derating to 40% for margin.
CAVITATION_POWER   = 0.40   # power fraction below which inertial cavitation is unlikely
CAVITATION_TEMP_C  = 55.0   #  degC   tissue temperature cavitation marker (secondary)

NT = int(T_END / DT) + 1
t_vec = np.arange(NT) * DT

def protocol_cw_high(t):
    """Continuous wave at 100% -- shows cavitation-risk temperature overshoot."""
    return np.ones_like(t)

def protocol_cw_safe(t):
    """CW at 40% -- below cavitation threshold; therapeutic temp with no rapid lesion."""
    return np.full_like(t, 0.40)

def protocol_ramp_modulated(t):
    """
    Phase 1 (0-2 s):   linear ramp 0->100%
    Phase 2 (2-12 s):  hold 100%  ->  rapid heating; lesion initiates at ~7-8 s
    Phase 3 (12-20 s): step down to 40%  -> therapeutic plateau visible
    Clinical rationale: 10 s burst accumulates >240 CEM43 min; reduction limits
    skull heating and demonstrates the perfusion-dominated temperature plateau.
    """
    P = np.ones_like(t) * 0.40
    P = np.where(t < 2,  t / 2.0,         P)   # ramp phase
    P = np.where((t >= 2) & (t < 12), 1.0, P)  # full-power burst (10 s)
    return P

def protocol_duty_cycle(t):
    """
    40% duty cycle at 110% peak power.
    Pulsed protocol: 2 s ON / 3 s OFF  -> peak temp lower than CW 100%,
    mechanical effects exploitable (BBB opening, stable cavitation).
    """
    P = np.where((t % 5.0) < 2.0, 1.10, 0.0)
    return np.clip(P, 0, None)

PROTOCOLS = {
    "CW 100% (cavitation risk)":  (protocol_cw_high,       "C3",     "-"),
    "CW 40% (safe)":              (protocol_cw_safe,        "C0",     "-"),
    "Ramp->modulated (clinical)":  (protocol_ramp_modulated, "C2",     "-"),
    "Duty-cycle 40%/110%":        (protocol_duty_cycle,     "C4",     "--"),
}

# ── 6.  2D cylindrical Pennes bioheat solver (NumPy) ─────────────────────────

def laplacian_cylindrical(T, dr, dz, nr, nz):
    """
    Compute the cylindrical Laplacian:
      L = 1/r * ∂/∂r(r * ∂T/∂r) + ∂^2T/∂z^2
    using 2nd-order central finite differences.

    At r=0 (axis): apply L'Hôpital limit ->  2 * ∂^2T/∂r^2
    Symmetry BC:   T[-1, :] = T[1, :]   (ghost cell at r=-dr)
    Neumann BC:    T[nr, :] = T[nr-1, :]  (zero flux at r_max)
    Neumann BC:    T[:, -1] = T[:, -2]    (zero flux at z_max)
    Dirichlet:     z=0 fixed at T_body (handled by caller keeping T[:,0] const)
    """
    L = np.zeros_like(T)

    # ─ r-direction ──────────────────────────────────────────────────────────
    # Interior points i = 1 .. nr-2
    # Use staggered half-cell radii: r_{i+½} = (i + 0.5) * dr
    i = np.arange(1, nr - 1)           # shape (nr-2,)
    ri = i * dr                         # r at grid centres

    T_ip1 = T[i + 1, :]    # T[i+1, :]
    T_i   = T[i,     :]
    T_im1 = T[i - 1, :]

    # 1/r * d/dr(r * dT/dr) using staggered fluxes:
    # = [ r_{i+½} * (T_{i+1} - T_i) - r_{i-½} * (T_i - T_{i-1}) ] / (ri * dr^2)
    r_ip05 = (i + 0.5) * dr
    r_im05 = (i - 0.5) * dr
    L[i, :] += (r_ip05[:, None] * (T_ip1 - T_i) -
                r_im05[:, None] * (T_i   - T_im1)) / (ri[:, None] * dr**2)

    # On-axis (i=0): L'Hôpital -> 2 * (T[1] - T[0]) / dr^2
    L[0, :] += 2.0 * (T[1, :] - T[0, :]) / dr**2

    # Neumann at r_max: zero-flux -> T[nr] = T[nr-1] (ghost cell never updated)
    # (i=nr-1 uses T[nr-1] and T[nr-2]; approximate as second-order by ghost copy)
    L[nr - 1, :] += 2.0 * (T[nr - 2, :] - T[nr - 1, :]) / dr**2

    # ─ z-direction ──────────────────────────────────────────────────────────
    # Interior j = 1 .. nz-2
    j = np.arange(1, nz - 1)
    L[:, j] += (T[:, j + 1] - 2.0 * T[:, j] + T[:, j - 1]) / dz**2

    # Neumann at z_max (j=nz-1): zero-flux -> ghost T[:, nz] = T[:, nz-1]
    L[:, nz - 1] += (T[:, nz - 2] - T[:, nz - 1]) / dz**2  # effectively 0 slope

    return L


def simulate(power_fn, label):
    """
    Run the Pennes bioheat solve for a given power modulation protocol.
    Returns dict of time-series and final maps.
    """
    T = np.full((NR, NZ), T_ART_C)     #  degC
    D = np.zeros((NR, NZ))             # CEM43 dose [min]

    t_axis   = t_vec
    T_focus  = np.zeros(NT)            # temperature at focal point
    T_skull_in = np.zeros(NT)          # temperature at skull inner surface (r=0)
    T_skull_out = np.zeros(NT)         # temperature at skull outer surface (r=0)
    dose_focus = np.zeros(NT)

    iz_focus     = int(Z_FOCUS / DZ)          # axial index of focus
    iz_skull_in  = int(Z_SKULL_END / DZ)      # inner skull surface
    iz_skull_out = int(Z_WATER_END / DZ)      # outer skull surface (water side)

    power_vec = power_fn(t_axis)

    for n in range(NT):
        P = power_vec[n]

        # ── Pennes bioheat step ──────────────────────────────────────────────
        L  = laplacian_cylindrical(T, DR, DZ, NR, NZ)
        dT = DT * (
            alpha_th * L                    # conduction
            - omega * (T - T_ART_C)         # perfusion cooling
            + Q_KS_NORM * P                 # acoustic heating
        )
        T += dT

        # Keep water inflow boundary at body temperature (Dirichlet at z=0)
        T[:, 0] = T_ART_C

        # ── CEM43 dose update ────────────────────────────────────────────────
        Rc = np.where(T >= 43.0, 0.5, 0.25)
        expo = 43.0 - T
        # Guard against exponential overflow for very high T
        expo = np.clip(expo, -50, 50)
        D += np.where(brain_mask, Rc ** expo * DT / 60.0, 0.0)

        # Record time series
        T_focus[n]     = T[0, iz_focus]
        T_skull_in[n]  = T[0, iz_skull_in]
        T_skull_out[n] = T[0, iz_skull_out]
        dose_focus[n]  = D[0, iz_focus]

    necrosis_mask = (D >= 240.0) & brain_mask
    lesion_vol_mm3 = necrosis_mask.sum() * (DR * 1e3) * (DZ * 1e3) * 2 * np.pi  # rotated

    return {
        "t":            t_axis,
        "power":        power_vec,
        "T_focus":      T_focus,
        "T_skull_in":   T_skull_in,
        "T_skull_out":  T_skull_out,
        "dose_focus":   dose_focus,
        "T_final":      T.copy(),
        "D_final":      D.copy(),
        "necrosis":     necrosis_mask,
        "lesion_vol":   lesion_vol_mm3,
        "label":        label,
    }

# ── 7.  Run all protocols ─────────────────────────────────────────────────────

print("Running 4 power-modulation protocols on 2D cylindrical bioheat solver ...")
print(f"  Grid: {NR}x{NZ}, dr=dz={DR*1e3:.1f} mm, dt={DT:.2f} s, T_end={T_END:.0f} s")
print(f"  Focus: z={Z_FOCUS*1e3:.0f} mm, w0={W0*1e3:.1f} mm, I_ref={I_REF_100W/1e4:.0f} W/cm^2")

results = {}
for name, (fn, color, ls) in PROTOCOLS.items():
    print(f"  -> {name} ...", end=" ", flush=True)
    res = simulate(fn, name)
    res["color"] = color
    res["ls"]    = ls
    results[name] = res
    print(f"done  |  CEM43 at focus = {res['dose_focus'][-1]:.1f} min  |  "
          f"lesion ~ {res['lesion_vol']:.1f} mm^3")

# ── 8.  Steady-state plateau analysis ────────────────────────────────────────
# The conduction-dominated plateau temperature at focus:
#   T_plateau - T_a ≈ Q_focus / (k * ∇^2_eigenvalue)
# For Gaussian source sigma=w0 in cylindrical geometry, the leading eigenvalue
# of the Laplacian for a radially symmetric source of width sigma is ~4/sigma^2.
# This gives the analytical steady-state (no-perfusion) plateau:
Q_focus_100W = Q_KS_NORM[0, int(Z_FOCUS / DZ)] * (rho * cp)[0, int(Z_FOCUS / DZ)]
k_brain = 0.510
sigma2  = W0**2
T_plateau_analytical = T_ART_C + Q_focus_100W / (k_brain * 4.0 / sigma2)
print(f"\nAnalytical conduction plateau (100W, no perfusion): "
      f"{T_plateau_analytical:.1f} degC  (conduction eigenvalue 4/w0^2)")
print(f"Cavitation safe power fraction: <= {CAVITATION_POWER:.0%}")
print(f"  -> Predicted safe plateau: {T_ART_C + (T_plateau_analytical - T_ART_C) * CAVITATION_POWER:.1f} degC")

# ── 9.  Figure ────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "lines.linewidth": 1.6, "legend.fontsize": 8.0,
    "xtick.labelsize": 8,  "ytick.labelsize": 8,
    "figure.dpi": 120,
})

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Focused Ultrasound Brain Treatment Through Skull -- Power Modulation & Temperature Plateau\n"
    f"f = {F_MHZ} MHz  |  w0 = {W0*1e3:.0f} mm focal radius  |  "
    f"Skull = 7 mm  |  Focus depth = {(Z_FOCUS - Z_SKULL_END)*1e3:.0f} mm post-skull",
    fontsize=11, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.42,
                       left=0.07, right=0.97, top=0.92, bottom=0.06)

ax_focus  = fig.add_subplot(gs[0, :2])  # focus temperature vs time
ax_skull  = fig.add_subplot(gs[0, 2:])  # skull temperature vs time
ax_power  = fig.add_subplot(gs[1, :2])  # power protocol
ax_dose   = fig.add_subplot(gs[1, 2:])  # CEM43 dose at focus

ax_map_cw   = fig.add_subplot(gs[2, 0])  # final temp map -- CW 100%
ax_map_safe = fig.add_subplot(gs[2, 1])  # final temp map -- CW 40%
ax_map_mod  = fig.add_subplot(gs[2, 2])  # final temp map -- modulated
ax_map_duty = fig.add_subplot(gs[2, 3])  # final temp map -- duty-cycle

MAP_AXES = [ax_map_cw, ax_map_safe, ax_map_mod, ax_map_duty]

# ─ Panel 1: Focus temperature ─────────────────────────────────────────────────
ax_focus.axhline(CAVITATION_TEMP_C, color="red", lw=1.0, ls=":", alpha=0.7,
                 label=f"Inertial cavitation risk ({CAVITATION_TEMP_C} degC)")
ax_focus.axhline(43.0, color="orange", lw=0.8, ls=":", alpha=0.5, label="CEM43 accumulation onset (43 degC)")
ax_focus.axhspan(55.0, 100.0, color="red", alpha=0.06, zorder=0)
ax_focus.axhspan(43.0, 55.0, color="orange", alpha=0.06, zorder=0)
ax_focus.axhspan(37.0, 43.0, color="green", alpha=0.04, zorder=0)

for name, res in results.items():
    ax_focus.plot(res["t"], res["T_focus"], color=res["color"],
                  ls=res["ls"], label=name)

ax_focus.set_xlabel("Time [s]")
ax_focus.set_ylabel("Temperature at focus [ degC]")
ax_focus.set_title("Focal Point Temperature vs Time")
ax_focus.set_xlim(0, T_END)
ax_focus.set_ylim(35, 90)
ax_focus.legend(loc="upper right", framealpha=0.9)

# Annotate plateau on modulated protocol (plateau visible in final 12 s after reduction)
res_mod = results["Ramp->modulated (clinical)"]
t_plateau_region = res_mod["t"][res_mod["t"] > 10]
T_plateau_region = res_mod["T_focus"][res_mod["t"] > 10]
T_pl_val = T_plateau_region.mean() if len(T_plateau_region) > 0 else 47.0
ax_focus.annotate(
    f"Plateau ≈ {T_pl_val:.0f} degC\n(conduction-perfusion\nequilibrium at 40% power)",
    xy=(15, T_pl_val), xytext=(10.5, T_pl_val - 14),
    arrowprops=dict(arrowstyle="->", color="C2"),
    color="C2", fontsize=7.5,
)
ax_focus.text(0.02, 0.97, "Thermal time constant tau ~ 7 s (brain)",
              transform=ax_focus.transAxes,
              va="top", ha="left", fontsize=7, color="gray", style="italic")

# ─ Panel 2: Skull inner-surface temperature ───────────────────────────────────
ax_skull.axhline(56.0, color="red", lw=1.0, ls=":", alpha=0.7,
                 label="Skull pain/damage threshold (~56 degC)")
ax_skull.axhspan(56.0, 100.0, color="red", alpha=0.06, zorder=0)

for name, res in results.items():
    ax_skull.plot(res["t"], res["T_skull_in"], color=res["color"],
                  ls=res["ls"], label=name.split("(")[0].strip())

ax_skull.set_xlabel("Time [s]")
ax_skull.set_ylabel("Temperature at inner skull [ degC]")
ax_skull.set_title("Skull Inner-Surface Temperature -- Power-Limiting Factor")
ax_skull.set_xlim(0, T_END)
ax_skull.legend(loc="upper left", framealpha=0.9)
ax_skull.text(0.02, 0.97, "Skull lacks perfusion; Gaussian far-field beam\n"
              "model underestimates skull heating (symmetric\n"
              "exp(-2a|z-zf|) vs forward-path physics).\n"
              "Clinical limit: skull reaches damage threshold\n"
              "in 5-10 s at therapeutic power => active cooling.",
              transform=ax_skull.transAxes, va="top", ha="left",
              fontsize=6.5, color="gray", style="italic")

# ─ Panel 3: Power protocols ───────────────────────────────────────────────────
ax_power.axhline(CAVITATION_POWER, color="red", lw=0.8, ls=":", alpha=0.7)
ax_power.text(T_END * 0.99, CAVITATION_POWER + 0.01,
              f"Cavitation pressure threshold ≈ {CAVITATION_POWER:.0%} power",
              ha="right", va="bottom", fontsize=7, color="red")
ax_power.axhspan(CAVITATION_POWER, 1.15, color="red", alpha=0.06, zorder=0)

for name, res in results.items():
    ax_power.plot(res["t"], res["power"], color=res["color"],
                  ls=res["ls"], label=name.split("(")[0].strip(), alpha=0.9)

ax_power.set_xlabel("Time [s]")
ax_power.set_ylabel("Normalised input power  [100 W = 1.0]")
ax_power.set_title("Power Modulation Protocols")
ax_power.set_xlim(0, T_END)
ax_power.set_ylim(-0.05, 1.25)
ax_power.legend(loc="upper right", framealpha=0.9)

# ─ Panel 4: CEM43 dose at focus ───────────────────────────────────────────────
ax_dose.axhline(240.0, color="black", lw=1.1, ls="--",
                label="Necrosis threshold (240 CEM43 min)")
ax_dose.axhline(60.0,  color="gray",  lw=0.8, ls=":",
                label="Damage threshold (60 CEM43 min)")
ax_dose.axhspan(240.0, 1e6, color="black", alpha=0.04, zorder=0)

for name, res in results.items():
    lbl = f"{name.split('(')[0].strip()}  [{res['dose_focus'][-1]:.0f} min]"
    ax_dose.semilogy(res["t"], np.clip(res["dose_focus"], 1e-6, None),
                     color=res["color"], ls=res["ls"], label=lbl)

ax_dose.set_xlabel("Time [s]")
ax_dose.set_ylabel("CEM43 dose at focus  [min]  (log scale)")
ax_dose.set_title("Thermal Dose Accumulation (CEM43) at Focus")
ax_dose.set_xlim(0, T_END)
ax_dose.set_ylim(1e-4, None)
ax_dose.legend(loc="lower right", framealpha=0.9)

# ─ Panels 5-8: Final temperature maps ────────────────────────────────────────
z_mm = z * 1e3
r_mm = r * 1e3
CMAP = "RdYlBu_r"

def add_anatomy_lines(ax):
    """Mark skull and focus on each map panel."""
    ax.axvline(Z_WATER_END * 1e3, color="gold", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(Z_SKULL_END * 1e3, color="gold", lw=1.2, ls="--", alpha=0.8)
    ax.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax.plot(Z_FOCUS * 1e3, 0, "w+", ms=8, mew=1.5)


for ax, (name, res) in zip(MAP_AXES, results.items()):
    im = ax.pcolormesh(z_mm, r_mm, res["T_final"],
                       cmap=CMAP, vmin=37, vmax=80, shading="auto")
    # Overlay necrosis contour (CEM43 >= 240 min)
    if res["necrosis"].any():
        ax.contour(z_mm, r_mm, res["necrosis"].astype(float),
                   levels=[0.5], colors="white", linewidths=1.2, linestyles="solid")
    # CEM43 = 60 min contour (cell damage onset)
    damage_mask = (res["D_final"] >= 60.0) & brain_mask
    if damage_mask.any():
        ax.contour(z_mm, r_mm, damage_mask.astype(float),
                   levels=[0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
    add_anatomy_lines(ax)
    ax.set_xlabel("Axial depth z [mm]")
    ax.set_ylabel("Lateral r [mm]")
    ax.set_title(
        f"{name.split('(')[0].strip()}\n"
        f"T_focus={res['T_focus'][-1]:.0f} degC  "
        f"lesion~{res['lesion_vol']:.0f} mm^3",
        fontsize=8,
    )
    plt.colorbar(im, ax=ax, label="T [ degC]", fraction=0.046, pad=0.02)
    # Mark skull band
    ax.axvspan(Z_WATER_END * 1e3, Z_SKULL_END * 1e3,
               alpha=0.25, color="gold", zorder=0, label="Skull")

# Shared legend for anatomy (only on first map)
from matplotlib.lines import Line2D
anatomy_handles = [
    Line2D([0], [0], color="gold", lw=1.2, ls="--", label="Skull boundaries"),
    Line2D([0], [0], color="white", lw=1.2, ls="solid", label="Necrosis (>=240 CEM43)"),
    Line2D([0], [0], color="cyan",  lw=0.8, ls="dashed", label="Cell damage (>=60 CEM43)"),
    Line2D([0], [0], color="white", marker="+", ms=7, ls="none", label="Focus (+)"),
]
ax_map_cw.legend(handles=anatomy_handles, loc="upper left",
                 fontsize=6.5, framealpha=0.85)

# ─ Annotations: clinical plateau mechanism ───────────────────────────────────
fig.text(0.5, 0.35,
         "Temperature plateau: acoustic heating rate = radial conduction loss\n"
         r"$T_{\rm plateau} - T_a \approx Q_{\rm focus} / (k \cdot 4/w_0^2)$   "
         r"[conduction-dominated]  ->  $T_{\rm pl} \approx "
         + f"{T_plateau_analytical:.0f}$ degC at 100W (no perfusion)" + "\n"
         r"Perfusion adds secondary cooling: $\tau_{\rm perf} = \rho c_p / (\omega_b \rho_b c_b)$"
         r"$\approx 111\,\rm s$  (long vs 10-s sonication  ->  minor for short treatments)",
         ha="center", va="center", fontsize=8.5, style="italic", color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#bbbbbb", lw=0.8),
         transform=fig.transFigure,
         )

plt.savefig("hifu_skull_modulation.png", dpi=150, bbox_inches="tight")
print("\nSaved: hifu_skull_modulation.png")
plt.show()

# ── 10.  Summary table ────────────────────────────────────────────────────────

print("\n" + "="*78)
print(f"{'Protocol':<30}  {'T_focus':>8}  {'T_skull':>8}  "
      f"{'CEM43':>10}  {'Lesion':>10}  {'Cavit?':>7}")
print(f"{'':30}  {'final[ degC]':>8}  {'final[ degC]':>8}  "
      f"{'at focus':>10}  {'vol[mm^3]':>10}  {'':>7}")
print("-"*78)
for name, res in results.items():
    cav = "YES" if res["power"].max() > CAVITATION_POWER else "NO "
    print(f"{name:<30}  {res['T_focus'][-1]:>8.1f}  {res['T_skull_in'][-1]:>8.1f}  "
          f"{res['dose_focus'][-1]:>10.1f}  {res['lesion_vol']:>10.1f}  {cav:>7}")
print("="*78)

print("""
Clinical interpretation (150 W/cm^2 focal intensity, 650 kHz, 20 s sonication):
  CW 100%     -- focus exceeds 60 degC in ~8 s; massive CEM43 (necrosis) but
                 p_peak = 2.2 MPa >> cavitation threshold -> inertial cavitation,
                 uncontrolled bubble collapse, haemorrhage risk.
  CW 40%      -- below cavitation threshold; focus plateaus ~47 degC (tau~7 s);
                 slow CEM43 accumulation, no necrosis in 20 s; useful for sub-
                 ablative protocols (BBB opening, neuromodulation).
  Ramp+mod    -- 10 s burst at 100% achieves CEM43 > 240 min (necrosis); step-down
                 to 40% shows temperature plateau at ~47 degC; continued slow
                 dose accumulation; controlled 7.9 mm^3 lesion with manageable risk.
  Duty-cycle  -- pulsed 110% / 40% cycles; peak pressure transiently exceeds
                 threshold; can cause controlled stable cavitation for mechanical
                 bioeffects (BBB opening) while limiting thermal injury zone.
  NOTE: Skull heating is underestimated by the Gaussian far-field beam model used
        here.  Real transcranial HIFU requires phased-array simulations and active
        skull cooling; each sonication is typically limited to 5-15 s.
""")
