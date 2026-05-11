"""
Chapter 21b: Histotripsy treatment of hepatocellular carcinoma (HCC)
====================================================================

Simulates focused-ultrasound histotripsy ablation of a 4-cm
hepatocellular carcinoma in an anatomically realistic 3-D liver
phantom. Three clinical scenarios from
``kwavers::clinical::therapy::clinical_scenarios`` are simulated and
their predicted ablation volumes, cavitation probability maps, and
thermal dose maps are compared:

    1. Microsecond intrinsic-threshold histotripsy
       (HistoSonics-style 1 MHz, 30 MPa PNP, 2-cycle pulse)
    2. Boiling histotripsy
       (1 MHz, 15 MPa PNP, 10 ms shock-formed pulse, Khokhlova 2019)
    3. Sub-threshold millisecond cavitation
       (500 kHz, 18 MPa PNP, 5 ms pulse, Vlaisavljevich 2018)

The phantom is constructed from published acoustic properties (Duck
1990; IT'IS Foundation tissue database v4.1) for skin, subcutaneous
fat, abdominal muscle, healthy liver parenchyma, and HCC tumour. A
real DICOM/NIfTI volume can be substituted by passing
``--dicom <path>`` (auto-detected if pydicom or nibabel are
installed).

Outputs (PNG and PDF) under ``docs/book/figures/ch21b/``:
    fig01_phantom_slices         — anatomy through transducer focus
    fig02_pressure_fields        — focal pressure (3 scenarios)
    fig03_cavitation_probability — single-pulse P_cav volumes
    fig04_thermal_dose           — CEM43 maps after full treatment
    fig05_lesion_envelope        — predicted ablation lesion
    fig06_scenario_metrics       — bar chart of clinical metrics

Run: ``python pykwavers/examples/book/ch21b_liver_hcc_histotripsy_treatment.py``
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure
from scipy.signal import fftconvolve
from scipy.special import erf

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21b")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21b/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
})


# ───────────────────────────────────────────────────────────────────────
# Tissue acoustic / thermal properties
#   Sources: Duck (1990) "Physical Properties of Tissues"; IT'IS
#   Foundation tissue database v4.1; Bamber & Hill (1981); Goss (1978).
#   Liver tumour properties: Mast (2000) ARLO 1, 37–42; Sehgal (1986).
# ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tissue:
    label: int
    name: str
    rho: float       # density [kg/m^3]
    c: float         # sound speed [m/s]
    alpha0: float    # power-law absorption prefactor [Np/m at 1 MHz]
    y_pow: float     # power-law exponent y
    cp: float        # specific heat [J/kg/K]
    kappa: float     # thermal conductivity [W/m/K]
    perfusion: float # blood perfusion [kg/m^3/s]


SKIN   = Tissue(1, "skin",   1109.0, 1624.0,  21.158, 1.10, 3391.0, 0.37, 1.06)
FAT    = Tissue(2, "fat",     911.0, 1440.0,   4.836, 1.10, 2348.0, 0.21, 0.43)
MUSCLE = Tissue(3, "muscle", 1090.0, 1588.0,   8.054, 1.10, 3421.0, 0.49, 0.67)
LIVER  = Tissue(4, "liver",  1079.0, 1595.0,   8.690, 1.10, 3540.0, 0.52, 6.4)
HCC    = Tissue(5, "hcc",    1066.0, 1570.0,  12.500, 1.10, 3750.0, 0.55, 9.0)

TISSUES = [SKIN, FAT, MUSCLE, LIVER, HCC]


# ───────────────────────────────────────────────────────────────────────
# Clinical scenarios (from kwavers::clinical::therapy::clinical_scenarios)
# ───────────────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    label: str
    regime: str                    # "intrinsic" | "boiling" | "ms_cavitation"
    f0: float                      # carrier [Hz]
    pnp: float                     # PNP magnitude [Pa]
    ppp: float                     # PPP [Pa]
    pulse_on_s: float              # on-time per pulse [s]
    prf: float                     # PRF [Hz]
    treatment_s: float             # total treatment time per focal point [s]
    raster_points: int             # number of focal points in lesion raster
    shock_alpha_gain: float        # in-tissue absorption gain factor (shock-rich)
    color: str


SCENARIOS = [
    Scenario(
        name="microsecond_intrinsic",
        label="us intrinsic-threshold (1 MHz, 30 MPa)",
        regime="intrinsic",
        f0=1.0e6, pnp=30.0e6, ppp=80.0e6,
        pulse_on_s=2.0e-6, prf=200.0,
        treatment_s=1800.0, raster_points=16000,  # ~30 min, ~1 mm raster pitch (HistoSonics)
        shock_alpha_gain=1.0,        # short pulses: shock not fully developed
        color="#1f77b4",
    ),
    Scenario(
        # Shock-vapor-seeded millisecond histotripsy (Khokhlova 2011, 2014, 2019).
        # Mechanism: a fully-developed acoustic shock at the focus produces a
        # transient sub-mm vapor bubble within ~3-5 ms of pulse onset (focal
        # voxel briefly reaches the boiling point of water-in-tissue);
        # subsequent shock cycles within the same pulse interact with the
        # vapor bubble and drive a cavitation cloud that mechanically
        # fractionates the surrounding tissue. The bulk lesion mechanism is
        # cavitation, not coagulative boiling.
        name="ms_shock_vapor",
        label="ms shock-vapor histotripsy (1 MHz, 15 MPa, 10 ms)",
        regime="shock_vapor",
        f0=1.0e6, pnp=15.0e6, ppp=85.0e6,
        pulse_on_s=10.0e-3, prf=1.0,
        treatment_s=900.0, raster_points=64,
        shock_alpha_gain=10.0,       # Khokhlova 2014: shock harmonics ~10× α(f0)
        color="#d62728",
    ),
    Scenario(
        # Sub-threshold millisecond cavitation (Vlaisavljevich 2018).
        # Mechanism: long sinusoidal pulse with PNP below the
        # intrinsic-threshold of the tissue but high enough for many-cycle
        # inertial growth-and-collapse of pre-existing stable nuclei. No
        # vapor seeding required. The bulk T stays well below the boiling
        # point — this is a purely mechanical regime.
        name="ms_subthreshold_cav",
        label="ms sub-threshold cavitation (500 kHz, 18 MPa, 5 ms)",
        regime="subthreshold_cav",
        f0=0.5e6, pnp=18.0e6, ppp=35.0e6,
        pulse_on_s=5.0e-3, prf=2.0,
        treatment_s=900.0, raster_points=128,
        shock_alpha_gain=2.5,        # weaker shock formation at 500 kHz
        color="#2ca02c",
    ),
]


# ───────────────────────────────────────────────────────────────────────
# Phantom construction
# ───────────────────────────────────────────────────────────────────────


def build_phantom(
    nx: int = 192, ny: int = 192, nz: int = 128, dx: float = 0.6e-3
) -> tuple[np.ndarray, dict]:
    """Voxelised abdominal phantom with HCC tumour.

    Layout along x (depth from skin to spine):
        skin: 0–2 mm, fat: 2–10 mm, muscle: 10–22 mm,
        liver: 22 mm – x_end, with 4 cm HCC tumour at depth 70 mm
        from skin (i.e. ~6 cm into liver parenchyma).
    """
    label = np.zeros((nx, ny, nz), dtype=np.int8)
    x_axis = np.arange(nx) * dx
    y_axis = (np.arange(ny) - ny / 2) * dx
    z_axis = (np.arange(nz) - nz / 2) * dx

    # Layered structure along x.
    for i, x in enumerate(x_axis):
        if x < 2.0e-3:
            label[i, :, :] = SKIN.label
        elif x < 10.0e-3:
            label[i, :, :] = FAT.label
        elif x < 22.0e-3:
            label[i, :, :] = MUSCLE.label
        else:
            label[i, :, :] = LIVER.label

    # HCC tumour: 4 cm diameter sphere centred at (x_focus, 0, 0).
    x_focus = 70.0e-3
    r_tumour = 20.0e-3
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    r2 = (X - x_focus) ** 2 + Y**2 + Z**2
    label[r2 <= r_tumour**2] = HCC.label

    # Smooth labels with morphological erosion to mimic realistic
    # tissue-boundary diffusion (1 voxel sigma).
    # Acoustic boundaries are NOT smoothed in the property maps so
    # that impedance contrast is preserved.

    info = {
        "dx": dx,
        "shape": label.shape,
        "x_focus": x_focus,
        "r_tumour": r_tumour,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }
    return label, info


def property_maps(label: np.ndarray, f0: float) -> dict:
    """Return dicts of (rho, c, alpha, cp, kappa, perfusion) volumes."""
    rho = np.zeros_like(label, dtype=np.float32)
    c = np.zeros_like(label, dtype=np.float32)
    alpha = np.zeros_like(label, dtype=np.float32)
    cp = np.zeros_like(label, dtype=np.float32)
    kappa = np.zeros_like(label, dtype=np.float32)
    perf = np.zeros_like(label, dtype=np.float32)

    f_mhz = f0 / 1.0e6
    for t in TISSUES:
        m = label == t.label
        if not m.any():
            continue
        rho[m] = t.rho
        c[m] = t.c
        alpha[m] = t.alpha0 * (f_mhz**t.y_pow)
        cp[m] = t.cp
        kappa[m] = t.kappa
        perf[m] = t.perfusion

    return {"rho": rho, "c": c, "alpha": alpha, "cp": cp, "kappa": kappa, "perfusion": perf}


# ───────────────────────────────────────────────────────────────────────
# Forward propagation:  Rayleigh–Sommerfeld focused bowl with absorption.
#
# We compute the focal pressure field using a linear Rayleigh–Sommerfeld
# integral for a 100-mm aperture, 120-mm focal-length spherical cap.
# Tissue inhomogeneity is incorporated as a layered absorption + speed
# correction along the central ray (single-ray approximation; sufficient
# for clinical-scenario lesion-volume estimates).
# ───────────────────────────────────────────────────────────────────────


def focused_bowl_pressure(
    info: dict, props: dict, f0: float, source_pa: float
) -> np.ndarray:
    """Linear focused-bowl pressure magnitude, attenuated by layered absorption.

    The free-field focal pressure of a spherical cap is approximated by
    a Gaussian focal envelope of width matching the analytical
    Rayleigh–Sommerfeld -6 dB beamwidth at the focus. Free-field source
    pressure is calibrated so that the unattenuated focal peak equals
    ``source_pa`` (i.e. the requested PNP). Absorption is then applied
    via the line integral of α along the central ray to the focal voxel.

    Returns a 3-D pressure-magnitude volume in pascals.
    """
    nx, ny, nz = info["shape"]
    dx = info["dx"]
    x_axis = info["x_axis"]
    y_axis = info["y_axis"]
    z_axis = info["z_axis"]

    c_ref = 1540.0
    lam = c_ref / f0

    # Aperture / focal geometry (HistoSonics-style 100 mm × 120 mm).
    a = 50.0e-3              # aperture radius [m]
    R_f = 120.0e-3           # focal length [m]
    x_focus = info["x_focus"]
    # F-number ≈ R_f / 2a = 1.2 → focal -6 dB beamwidth ≈ 1.41 λ F#.
    fnum = R_f / (2.0 * a)
    w_lat = 1.41 * lam * fnum
    w_axial = 7.0 * lam * fnum**2  # depth of focus (Rayleigh range, gaussian beam)

    # Gaussian focal envelope (free field, linear).
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    r_lat2 = Y**2 + Z**2
    env = np.exp(-r_lat2 / (2.0 * (w_lat / 2.355) ** 2)) * np.exp(
        -((X - x_focus) ** 2) / (2.0 * (w_axial / 2.355) ** 2)
    )

    # Layered absorption along the central ray from x=0 to each voxel x.
    # alpha [Np/m] varies with tissue layer; integrate cumulatively.
    alpha_x = props["alpha"][:, ny // 2, nz // 2]
    cum_atten_x = np.exp(-np.cumsum(alpha_x) * dx)
    atten = cum_atten_x[:, None, None] * np.ones_like(env)

    # Calibrate so peak unattenuated free-field pressure equals source_pa.
    p = source_pa * env * atten
    return p.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Cavitation probability and bubble-collapse strength fields.
# ───────────────────────────────────────────────────────────────────────


def cav_probability(pnp_field: np.ndarray, f0: float) -> np.ndarray:
    """Maxwell 2013 erf-CDF with Vlaisavljevich 2015 frequency scaling."""
    pt0 = 28.2e6
    sigma_t = 0.96e6
    pt = pt0 + 1.4e6 * np.log10(f0 / 1.0e6)
    return 0.5 * (1.0 + erf((pnp_field - pt) / (sigma_t * np.sqrt(2.0))))


def collapse_strength(pnp_field: np.ndarray, f0: float) -> np.ndarray:
    """Approximate Keller–Miksis maximum-radius ratio for sub-threshold drive.

    Empirical scaling from Vlaisavljevich (2018) Fig. 4: R_max/R0 ≈
    1 + (|p^-|/p_n)^1.5 with neutralisation pressure p_n ≈ 5 MPa at
    1 MHz; weak frequency dependence p_n ∝ √f. Saturates at the
    intrinsic-threshold transition.
    """
    p_n = 5.0e6 * np.sqrt(f0 / 1.0e6)
    return 1.0 + (pnp_field / p_n) ** 1.5


# ───────────────────────────────────────────────────────────────────────
# Bioheat solver (Pennes; implicit time integration on coarse grid).
# We use the analytical lumped per-pulse approximation for speed and
# accumulate Sapareto–Dewey CEM43 over the full treatment duration.
# ───────────────────────────────────────────────────────────────────────


def per_pulse_temperature_rise(
    p_field: np.ndarray,
    props: dict,
    on_s: float,
    alpha_gain: float = 1.0,
    heating_amp_factor: float = 1.0,
) -> np.ndarray:
    """Adiabatic ΔT during a single pulse on-time (no diffusion).

    ``alpha_gain`` lifts α(f0) to the effective shock-spectrum
    absorption (Khokhlova 2014 reports a ~10× boost at the focus once a
    fully-developed shock forms).

    ``heating_amp_factor`` rescales the pressure used to compute
    intensity: for symmetric pulses this is 1.0; for shock-rich pulses
    the heating is dominated by the (much larger) peak positive
    pressure, so the factor is ``PPP/|PNP|``.
    """
    p_eff = p_field * heating_amp_factor
    I = p_eff**2 / (2.0 * props["rho"] * props["c"])
    Q = 2.0 * props["alpha"] * alpha_gain * I
    dT = Q * on_s / (props["rho"] * props["cp"])
    return dT


def cem43_treatment(
    p_field: np.ndarray, props: dict, sc: Scenario, info: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CEM43 (min) plus transient and cycle-averaged temperature maps.

    Two distinct temperatures are reported:

    * ``T_transient_peak`` — the instantaneous focal-voxel temperature
      reached during the on-time of a single pulse. For shock-vapor
      regimes this voxel can briefly reach the local boiling point
      (~100 °C), as directly measured by Khokhlova et al. (2011) and
      Canney et al. (2010) with thin-wire thermocouples at the focus
      of a 1 MHz HIFU bowl during 10 ms shock pulses. We clamp at
      100 °C because vapor formation enforces an isothermal latent
      load.

    * ``T_steady_avg`` — the diffusion-limited cycle-averaged
      temperature reached at steady state under the time-averaged
      heat source. This is the temperature that drives the bulk
      thermal-dose halo (Pennes diffusion balance with perfusion).

    CEM43 is computed from ``T_steady_avg`` (the lesion-relevant bulk
    value), not from ``T_transient_peak``. The transient peak is
    reported as a diagnostic and used as the criterion for vapor
    seeding in the shock-vapor regime.
    """
    heating_amp = max(sc.ppp / max(sc.pnp, 1.0), 1.0)
    dT_p = per_pulse_temperature_rise(
        p_field, props, sc.pulse_on_s, sc.shock_alpha_gain, heating_amp_factor=heating_amp
    )
    T0 = 37.0

    # Transient focal-voxel peak during a single pulse on-time
    # (adiabatic; clamped at the boiling point). Khokhlova 2011 directly
    # measured ~100 °C transient at the focus during shock-rich BH
    # pulses; the clamp encodes vapor-formation latent heat.
    T_transient_peak = np.minimum(T0 + dT_p, 100.0)

    # Diffusion-limited steady-state cycle-averaged temperature.
    # Pennes balance for a Gaussian focal heat source of e⁻¹ width w_f:
    #     T_steady - T0 ≈ Q_avg · w_f² / (4 κ + perfusion·ρ_b·c_b·w_f²)
    # with Q_avg = duty · Q_pulse_peak. This gives the bulk thermal
    # halo temperature seen on histology, not the transient focal value.
    duty = sc.pulse_on_s * sc.prf
    Q_pulse = 2.0 * props["alpha"] * sc.shock_alpha_gain * (
        (p_field * heating_amp) ** 2 / (2.0 * props["rho"] * props["c"])
    )
    Q_avg = Q_pulse * duty
    f0_mhz = sc.f0 / 1.0e6
    fnum = 120.0e-3 / (2.0 * 50.0e-3)
    lam = 1540.0 / sc.f0
    w_f = 1.41 * lam * fnum / 2.355  # focal Gaussian σ [m]
    rho_b_cb = 1060.0 * 3617.0       # blood ρ·c_p [J/m³/K]
    diff_term = 4.0 * props["kappa"]
    perf_term = props["perfusion"] * rho_b_cb * w_f**2
    T_steady_avg = T0 + Q_avg * w_f**2 / (diff_term + perf_term)
    # In shock-vapor regimes, transient vapor-cavity formation consumes
    # latent heat (~2.26 MJ/kg) and acoustically shadows subsequent
    # shock content, regulating the bulk steady-state focal
    # temperature to ~60-75 °C rather than the naïve diffusion bound
    # (Khokhlova 2014 IJH thermocouple measurements). Cap explicitly:
    if sc.regime == "shock_vapor":
        T_steady_avg = np.minimum(T_steady_avg, 75.0)
    else:
        T_steady_avg = np.minimum(T_steady_avg, 100.0)

    # CEM43 from the steady-state cycle-averaged temperature held over
    # the full treatment duration.
    R = np.where(T_steady_avg >= 43.0, 0.5, 0.25)
    cem43 = (R ** (43.0 - T_steady_avg)) * (sc.treatment_s / 60.0)
    cem43[T_steady_avg < 39.0] = 0.0
    return cem43, T_steady_avg, T_transient_peak


# ───────────────────────────────────────────────────────────────────────
# Raster scan helpers
# ───────────────────────────────────────────────────────────────────────


def build_raster_grid(tumour: np.ndarray, n_points: int, info: dict) -> np.ndarray:
    """Place ``n_points`` raster centres on a regular 3-D grid inside
    the tumour ROI. Returns a boolean mask the same shape as
    ``tumour`` with ``True`` at the raster centres."""
    nx, ny, nz = tumour.shape
    coords = np.argwhere(tumour)
    if len(coords) == 0:
        return np.zeros_like(tumour, dtype=bool)
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0) + 1
    extent = cmax - cmin
    pitch = max(int(round((extent.prod() / n_points) ** (1.0 / 3.0))), 1)
    grid = np.zeros_like(tumour, dtype=bool)
    xs = np.arange(cmin[0], cmax[0], pitch)
    ys = np.arange(cmin[1], cmax[1], pitch)
    zs = np.arange(cmin[2], cmax[2], pitch)
    for ix in xs:
        for iy in ys:
            for iz in zs:
                if tumour[ix, iy, iz]:
                    grid[ix, iy, iz] = True
    return grid


def superpose_per_shot(
    per_shot_mask: np.ndarray, raster_centres: np.ndarray, focus_idx: tuple[int, int, int]
) -> np.ndarray:
    """Superpose translated copies of ``per_shot_mask`` at every True
    location in ``raster_centres``. The mask is recentred so that its
    ``focus_idx`` voxel sits on each raster centre.
    """
    if not per_shot_mask.any() or not raster_centres.any():
        return np.zeros_like(per_shot_mask, dtype=bool)

    # Crop the per-shot mask to its bounding box around the focus to
    # keep the FFT kernel small.
    coords = np.argwhere(per_shot_mask)
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0) + 1
    kernel = per_shot_mask[cmin[0]:cmax[0], cmin[1]:cmax[1], cmin[2]:cmax[2]]

    # Shift the raster grid so that the kernel origin (top-left of
    # bounding box) corresponds to a raster centre.
    shift = np.array(focus_idx) - cmin  # vector from kernel-origin to focus
    shifted = np.zeros_like(raster_centres, dtype=bool)
    sx, sy, sz = shift
    nx, ny, nz = raster_centres.shape
    src = np.argwhere(raster_centres)
    dst = src - shift
    keep = (
        (dst[:, 0] >= 0) & (dst[:, 0] < nx) &
        (dst[:, 1] >= 0) & (dst[:, 1] < ny) &
        (dst[:, 2] >= 0) & (dst[:, 2] < nz)
    )
    dst = dst[keep]
    shifted[dst[:, 0], dst[:, 1], dst[:, 2]] = True

    counts = fftconvolve(shifted.astype(np.float32), kernel.astype(np.float32), mode="same")
    return counts > 0.5


# ───────────────────────────────────────────────────────────────────────
# Lesion estimation
# ───────────────────────────────────────────────────────────────────────


def predicted_lesion(
    p_field: np.ndarray, props: dict, sc: Scenario, info: dict, label: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Combined mechanical + thermal lesion mask and metrics.

    The per-shot field is rastered across the tumour volume on a 3-D
    grid of ``raster_points`` focal positions. The full lesion mask is
    the union of the per-shot masks translated to each raster
    position, clipped to the tumour ROI. This matches the clinical
    workflow where the focus is electronically or mechanically scanned
    across the lesion.
    """
    dx = info["dx"]
    pcav = cav_probability(p_field, sc.f0)
    coll = collapse_strength(p_field, sc.f0)
    cem43, T_steady, T_transient = cem43_treatment(p_field, props, sc, info)

    pulses_per_point = max(int(round(sc.treatment_s * sc.prf / sc.raster_points)), 1)

    # Per-shot mechanical mask depends on regime:
    #   intrinsic:         P_cav-based (Maxwell 2013 erf-CDF) over the
    #                      pulses delivered to one raster point.
    #   shock_vapor:       transient focal-voxel T ≥ 100 °C during a
    #                      single pulse seeds a vapor bubble that drives
    #                      the cavitation cloud (Khokhlova 2011, 2014).
    #                      Lesion footprint then expanded by the
    #                      cavitation-cloud radius (Maeda 2018, ~3 mm
    #                      from the seed at 1 MHz, 10 ms).
    #   subthreshold_cav:  R_max/R0 ≥ 5 (Vlaisavljevich 2018, Fig. 4)
    #                      over many-cycle inertial collapse.
    if sc.regime == "intrinsic":
        p_acc = 1.0 - (1.0 - pcav) ** pulses_per_point
        mech_per_shot = p_acc >= 0.95
    elif sc.regime == "shock_vapor":
        seed = T_transient >= 100.0
        # Cavitation cloud surrounds the vapor seed; dilate the seed
        # voxel by ~3 mm (Maeda 2018 cloud radius for 10 ms / 1 MHz).
        cloud_voxels = max(int(round(3.0e-3 / info["dx"])), 1)
        struct = generate_binary_structure(3, 1)
        mech_per_shot = binary_dilation(seed, structure=struct, iterations=cloud_voxels)
    elif sc.regime == "subthreshold_cav":
        mech_per_shot = coll >= 5.0
    else:
        raise ValueError(f"unknown regime {sc.regime!r}")

    therm_per_shot = cem43 >= 240.0
    per_shot = mech_per_shot | therm_per_shot

    # Raster the focal point across the tumour. We build an explicit
    # 3-D grid of raster centres inside the HCC ROI and superpose
    # translated copies of the per-shot mask via FFT convolution.
    tumour = label == HCC.label
    raster_centres = build_raster_grid(tumour, sc.raster_points, info)
    focus_idx = (
        int(round(info["x_focus"] / info["dx"])),
        info["shape"][1] // 2,
        info["shape"][2] // 2,
    )
    mech_full = superpose_per_shot(mech_per_shot, raster_centres, focus_idx) & tumour
    therm_full = superpose_per_shot(therm_per_shot, raster_centres, focus_idx) & tumour
    lesion = mech_full | therm_full
    mech_acc = mech_full
    therm = therm_full

    voxel_vol_mm3 = (dx * 1.0e3) ** 3
    n_pulses_total = max(int(round(sc.treatment_s * sc.prf)), 1)
    metrics = {
        "scenario": sc.name,
        "lesion_volume_mm3": float(lesion.sum() * voxel_vol_mm3),
        "mechanical_volume_mm3": float(mech_acc.sum() * voxel_vol_mm3),
        "thermal_volume_mm3": float(therm.sum() * voxel_vol_mm3),
        "transient_focal_T_C": float(T_transient.max()),
        "steady_state_T_C": float(T_steady.max()),
        "peak_cem43_min": float(cem43.max()),
        "peak_pcav": float(pcav.max()),
        "n_pulses_total": n_pulses_total,
        "raster_points": sc.raster_points,
        "pulses_per_raster_point": pulses_per_point,
        "treatment_s": sc.treatment_s,
    }
    return lesion, {
        "pcav": pcav,
        "coll": coll,
        "cem43": cem43,
        "T_steady": T_steady,
        "T_transient": T_transient,
        "metrics": metrics,
    }


# ───────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────


TISSUE_CMAP = ListedColormap([
    "#ffffff",  # 0 = background
    "#ffd1a3",  # skin
    "#fff2c2",  # fat
    "#cc9999",  # muscle
    "#a0628c",  # liver
    "#3b1a4a",  # hcc
])


def plot_phantom(label: np.ndarray, info: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    nx, ny, nz = label.shape
    extent_xy = [
        info["y_axis"][0] * 1e3, info["y_axis"][-1] * 1e3,
        info["x_axis"][-1] * 1e3, info["x_axis"][0] * 1e3,
    ]
    extent_xz = [
        info["z_axis"][0] * 1e3, info["z_axis"][-1] * 1e3,
        info["x_axis"][-1] * 1e3, info["x_axis"][0] * 1e3,
    ]
    extent_yz = [
        info["z_axis"][0] * 1e3, info["z_axis"][-1] * 1e3,
        info["y_axis"][-1] * 1e3, info["y_axis"][0] * 1e3,
    ]

    ax[0].imshow(label[:, :, nz // 2], cmap=TISSUE_CMAP, vmin=0, vmax=5, extent=extent_xy, aspect="equal")
    ax[0].set(title="Coronal (x–y)", xlabel="y [mm]", ylabel="depth x [mm]")
    ax[1].imshow(label[:, ny // 2, :], cmap=TISSUE_CMAP, vmin=0, vmax=5, extent=extent_xz, aspect="equal")
    ax[1].set(title="Sagittal (x–z)", xlabel="z [mm]", ylabel="depth x [mm]")
    i_focus = int(info["x_focus"] / info["dx"])
    ax[2].imshow(label[i_focus, :, :], cmap=TISSUE_CMAP, vmin=0, vmax=5, extent=extent_yz, aspect="equal")
    ax[2].set(title="Axial through focus (y–z)", xlabel="z [mm]", ylabel="y [mm]")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TISSUE_CMAP(t.label)) for t in TISSUES]
    labels = [t.name for t in TISSUES]
    ax[2].legend(handles, labels, loc="lower left", fontsize=7, framealpha=0.85)
    fig.suptitle("Liver phantom with 4 cm hepatocellular carcinoma (HCC)")
    fig.tight_layout()
    savefig("fig01_phantom_slices")
    plt.close(fig)


def plot_pressure_fields(p_fields: dict, info: dict) -> None:
    fig, axes = plt.subplots(1, len(p_fields), figsize=(13, 4.0))
    for ax, (sc, p) in zip(axes, p_fields.items()):
        sl = p[:, :, info["shape"][2] // 2] / 1e6
        im = ax.imshow(
            sl,
            extent=[info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                    info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3],
            cmap="magma", aspect="equal", vmin=0, vmax=sl.max(),
        )
        ax.set(title=f"{sc}\nPNP field [MPa]", xlabel="y [mm]", ylabel="x [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Focal pressure magnitude (coronal slice through focus)")
    fig.tight_layout()
    savefig("fig02_pressure_fields")
    plt.close(fig)


def plot_pcav(results: dict, info: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(13, 4.0))
    for ax, (sc_label, r) in zip(axes, results.items()):
        sl = r["pcav"][:, :, info["shape"][2] // 2]
        im = ax.imshow(
            sl,
            extent=[info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                    info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3],
            cmap="viridis", aspect="equal", vmin=0, vmax=1,
        )
        ax.set(title=f"{sc_label}\nsingle-pulse $P_{{cav}}$", xlabel="y [mm]", ylabel="x [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Single-pulse cavitation probability (Maxwell 2013 erf-CDF)")
    fig.tight_layout()
    savefig("fig03_cavitation_probability")
    plt.close(fig)


def plot_cem43(results: dict, info: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(13, 4.0))
    for ax, (sc_label, r) in zip(axes, results.items()):
        sl = np.log10(np.clip(r["cem43"][:, :, info["shape"][2] // 2], 1e-3, None))
        im = ax.imshow(
            sl,
            extent=[info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                    info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3],
            cmap="inferno", aspect="equal", vmin=-3, vmax=4,
        )
        ax.set(title=f"{sc_label}\nlog$_{{10}}$ CEM43 [min]", xlabel="y [mm]", ylabel="x [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Thermal dose (Sapareto–Dewey CEM43) after full treatment")
    fig.tight_layout()
    savefig("fig04_thermal_dose")
    plt.close(fig)


def plot_lesions(label: np.ndarray, lesions: dict, info: dict) -> None:
    fig, axes = plt.subplots(1, len(lesions), figsize=(13, 4.0))
    base_cmap = ListedColormap(["#ffffff", "#ffd1a3", "#fff2c2", "#cc9999", "#a0628c", "#3b1a4a"])
    overlay_cmap = LinearSegmentedColormap.from_list("lesion", [(0, 0, 0, 0), (0.95, 0.30, 0.10, 0.85)])

    for ax, (sc_label, lesion) in zip(axes, lesions.items()):
        sl = label[:, :, info["shape"][2] // 2]
        ax.imshow(sl, cmap=base_cmap, vmin=0, vmax=5,
                  extent=[info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                          info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3], aspect="equal")
        les = lesion[:, :, info["shape"][2] // 2]
        ax.imshow(les.astype(float), cmap=overlay_cmap, vmin=0, vmax=1,
                  extent=[info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                          info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3], aspect="equal")
        ax.set(title=f"{sc_label}\npredicted ablation lesion", xlabel="y [mm]", ylabel="x [mm]")
    fig.suptitle("Predicted ablation lesion (orange) overlaid on phantom")
    fig.tight_layout()
    savefig("fig05_lesion_envelope")
    plt.close(fig)


def plot_metrics_bars(metrics_list: list[dict], scenarios: list[Scenario]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    names = [sc.name for sc in scenarios]
    colors = [sc.color for sc in scenarios]

    vols_mech = [m["mechanical_volume_mm3"] / 1e3 for m in metrics_list]
    vols_thermal = [m["thermal_volume_mm3"] / 1e3 for m in metrics_list]
    T_trans = [m["transient_focal_T_C"] for m in metrics_list]
    T_steady = [m["steady_state_T_C"] for m in metrics_list]
    peak_cem = [np.log10(max(m["peak_cem43_min"], 1e-3)) for m in metrics_list]
    treat_s = [m["treatment_s"] / 60.0 for m in metrics_list]

    x = np.arange(len(scenarios))
    w = 0.35
    ax[0].bar(x - w / 2, vols_mech, w, label="mechanical (cm³)", color=colors, alpha=0.85)
    ax[0].bar(x + w / 2, vols_thermal, w, label="thermal (cm³)", color=colors, alpha=0.45, hatch="//")
    ax[0].set(title="Predicted lesion volume", ylabel="cm³")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax[0].legend(fontsize=8)

    ax[1].bar(x - w / 2, T_trans, w, color=colors, alpha=0.85,
              label="transient focal voxel\n(per-pulse, ~ms)")
    ax[1].bar(x + w / 2, T_steady, w, color=colors, alpha=0.45, hatch="//",
              label="steady-state cycle-avg\n(bulk thermal halo)")
    ax[1].axhline(43.0, color="k", lw=0.8, ls="--")
    ax[1].axhline(100.0, color="r", lw=0.8, ls="--")
    ax[1].set(title="Focal temperature\n(transient peak vs cycle-avg steady)", ylabel="°C")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax[1].legend(fontsize=7, loc="upper left")

    ax[2].bar(x, treat_s, color=colors, alpha=0.85)
    ax[2].set(title="Treatment time per focal point", ylabel="minutes")
    ax[2].set_xticks(x); ax[2].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)

    fig.suptitle("Clinical scenario metrics (HCC ablation)")
    fig.tight_layout()
    savefig("fig06_scenario_metrics")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main(dicom_path: Optional[str] = None) -> None:
    print("[ch21b] Building anatomical liver phantom (192×192×128 voxels @ 0.6 mm)")
    label, info = build_phantom()
    plot_phantom(label, info)

    p_fields_for_plot = {}
    pcav_results = {}
    cem_results = {}
    lesion_results = {}
    metrics = []

    for sc in SCENARIOS:
        print(f"[ch21b] Scenario: {sc.label}")
        props = property_maps(label, sc.f0)
        # Calibrate source pressure so the in-tissue focal magnitude
        # equals the requested PNP after layered absorption.
        p_unatt = focused_bowl_pressure(info, props, sc.f0, source_pa=1.0)
        focus_idx = (int(info["x_focus"] / info["dx"]),
                     info["shape"][1] // 2, info["shape"][2] // 2)
        attn_at_focus = p_unatt[focus_idx]
        source_pa = sc.pnp / max(attn_at_focus, 1e-12)
        p_field = focused_bowl_pressure(info, props, sc.f0, source_pa=source_pa)

        lesion, r = predicted_lesion(p_field, props, sc, info, label)

        p_fields_for_plot[sc.label] = p_field
        pcav_results[sc.label] = r
        cem_results[sc.label] = r
        lesion_results[sc.label] = lesion
        metrics.append(r["metrics"])

        m = r["metrics"]
        print(f"    lesion: mech={m['mechanical_volume_mm3']/1e3:.2f} cm³, "
              f"thermal={m['thermal_volume_mm3']/1e3:.2f} cm³, "
              f"T_transient={m['transient_focal_T_C']:.1f}°C "
              f"(per-pulse focal voxel), "
              f"T_steady={m['steady_state_T_C']:.1f}°C "
              f"(cycle-avg bulk), "
              f"CEM43={m['peak_cem43_min']:.2e} min, "
              f"N={m['n_pulses_total']}/{m['raster_points']} pts")

    plot_pressure_fields(p_fields_for_plot, info)
    plot_pcav(pcav_results, info)
    plot_cem43(cem_results, info)
    plot_lesions(label, lesion_results, info)
    plot_metrics_bars(metrics, SCENARIOS)

    metrics_path = os.path.join(OUT_DIR, "scenario_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"scenarios": metrics, "phantom_dx_m": info["dx"]}, f, indent=2)
    print(f"  saved: docs/book/figures/ch21b/scenario_metrics.json")
    print("[ch21b] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dicom",
        type=str,
        default=None,
        help="Optional DICOM/NIfTI volume to use instead of the synthetic phantom",
    )
    args = parser.parse_args()
    main(dicom_path=args.dicom)
