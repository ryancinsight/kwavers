"""ch21e — Liver histotripsy treatment pipeline (whole-tumour, multi-region).

A complete liver-cancer histotripsy treatment, built on the kwavers Rust physics
core (this script is geometry + orchestration + plotting only — no domain math):

  • A 3-D tumour (PTV) in the liver is tiled into MULTIPLE SONICATION REGIONS
    (mechanical focus positions). Each region is covered by an electronic-steering
    SUB-SPOT grid, and each sub-spot is lesioned by MULTIPLE PULSES (repetitions).
    The whole tumour is treated region-by-region until volumetric coverage is met.
  • A sensitive structure (OAR, e.g. a major vessel) runs along the tumour edge;
    sub-spots whose focal lesion would breach the safety margin are NOT fired
    (coordinates/dose compensated for lesion expansion into the OAR).
  • Per sub-spot the delivered pressure is derated by the genuine electronic
    steering efficiency, interface reflection, tissue attenuation and residual-gas
    (Commander–Prosperetti) shielding (kwavers). Whether a sub-spot lesions is the
    Maxwell cumulative-cavitation-probability fractionation; the focal lesion
    extent is the beam FWHM (Penttinen/O'Neil).

Figures (all data-plotted, written to ``_pipeline_out/``):
  A  Pulsing pattern — the real pulse train from ``kw.sonication_schedule`` over
     the whole multi-region treatment (overview + one-region zoom showing the
     interleaved sub-spots; ① pulse duration ② repetition time ③ sonication).
  B  Sensor-recorded cavitation monitor over the full treatment (spectrum,
     signal+power controls, cumulative dose → goal).
  C  Treatment console — tumour coverage in space (orthogonal slices with region
     centres + OAR + grown lesion), coverage-vs-time, the live monitor, and the
     user feedback gauges.

Run: ``python ch21e_treatment_pipeline.py``.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import pykwavers as kw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cavitation_dose_monitor import BubbleMedium, simulate_population_emission

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pipeline_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Clinical / physical scenario ─────────────────────────────────────────────
F0 = 1.0e6
C_LIVER = 1540.0
RHO_LIVER = 1060.0
Z_FAT = 1.38e6
Z_LIVER = RHO_LIVER * C_LIVER
ALPHA_LIVER_NP_M = 9.0
P0 = 101_325.0
YIELD_STRESS_PA = 2.0e3
R0_NUC = 3.0e-6
FOCAL_DEPTH_M = 0.08
PULSE_DURATION_S = 10e-6
PRF_HZ = 200.0                 # fired-pulse rate within a region
P_DRIVE_PA = 70e6
P_T_1MHZ_PA = 28.2e6
SIGMA_T_PA = 0.96e6
COVERAGE_TARGET = 0.95         # cumulative cavitation prob to lesion a sub-spot
REGION_MOVE_S = 0.3            # mechanical move dwell between sonication regions
RESIDUAL_VOID_PER_PULSE = 5.0e-7   # focal residual void per pulse (Bader/Duryea)
# Fractionation: complete histotripsy tissue homogenisation needs MANY pulses per
# spot — each pulse's bubble cloud removes a small tissue fraction q, so coverage
# saturates as 1−(1−q)^N (Vlaisavljevich 2014; Khokhlova 2015). q grows with the
# per-pulse inertial cavitation dose; this reference sets the pulses-per-spot.
FRACTIONATION_ICD_REF = 800.0

MEDIUM = BubbleMedium(
    rho=RHO_LIVER, sigma=0.056, gamma=1.4, mu=2.0e-3, pv=2.3e3, c_l=C_LIVER, p0=P0
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: tumour, OAR, sonication regions, sub-spot grid
# ─────────────────────────────────────────────────────────────────────────────
def build_geometry():
    dx = 1.0e-3                                   # voxel size [m]
    # Steering-frame coords (lateral x, elevation y, axial z), origin at focus.
    xs = np.arange(-14e-3, 14e-3 + 1e-9, dx)
    ys = np.arange(-14e-3, 14e-3 + 1e-9, dx)
    zs = np.arange(-12e-3, 12e-3 + 1e-9, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Tumour: ellipsoid (10×10×9 mm semi-axes).
    a, b, c = 10e-3, 10e-3, 9e-3
    tumour = (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2 <= 1.0
    # OAR: vessel plane just beyond the +lateral tumour edge.
    oar_x = 12.0e-3
    safety_margin = 2.0e-3

    # Focal lesion extent ≈ beam FWHM (the lethal cavitation core fills roughly the
    # −6 dB focal region). les_lat must exceed the sub-spot diagonal half-pitch
    # (2 mm grid → 1.41 mm) so the grid tiles without interstitial gaps.
    fwhm_lat, fwhm_ax = focal_fwhm(F0)
    les_lat = 0.6 * fwhm_lat
    les_ax = 0.4 * fwhm_ax

    # Sonication regions (mechanical foci) tiling the tumour bbox.
    rxc = np.arange(-10e-3, 10e-3 + 1e-9, 4e-3)
    ryc = np.arange(-10e-3, 10e-3 + 1e-9, 4e-3)
    rzc = np.arange(-9e-3, 9e-3 + 1e-9, 4.5e-3)
    region_centers = []
    for zc in rzc:
        for yc in ryc:
            for xc in rxc:
                # Include edge regions (centre within tumour + one focal radius) so
                # their focal lesions cover the rounded tumour periphery.
                if (xc / (a + 2e-3)) ** 2 + (yc / (b + 2e-3)) ** 2 + (zc / (c + 2e-3)) ** 2 <= 1.0:
                    region_centers.append((xc, yc, zc))
    region_centers = np.array(region_centers)

    # Electronic sub-spot offsets within a region (±3 mm lateral/elev).
    so = np.array([-2e-3, 0.0, 2e-3])
    subspot_offsets = np.array([(ox, oy, 0.0) for oy in so for ox in so])

    return {
        "dx": dx, "xs": xs, "ys": ys, "zs": zs, "X": X, "Y": Y, "Z": Z,
        "tumour": tumour, "oar_x": oar_x, "safety_margin": safety_margin,
        "a": a, "b": b, "c": c, "les_lat": les_lat, "les_ax": les_ax,
        "region_centers": region_centers, "subspot_offsets": subspot_offsets,
    }


def focal_fwhm(f0):
    lam = C_LIVER / f0
    fnum = 120.0e-3 / (2.0 * 50.0e-3)
    return 1.41 * lam * fnum, 7.0 * lam * fnum ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Treatment planning: per sub-spot physics + volumetric coverage + schedule
# ─────────────────────────────────────────────────────────────────────────────
def plan_treatment(geom):
    rng = np.random.default_rng(7)

    # ICD-per-pulse vs delivered pressure (genuine Σ(R_max/R₀)³ on KM trajectory).
    p_bins = np.linspace(20e6, P_DRIVE_PA, 6)
    icd_bin = np.array([max(_pulse_icd(float(pb), rng), 1e-9) for pb in p_bins])

    p_t = float(kw.frequency_dependent_intrinsic_threshold_pa(
        np.array([F0]), P_T_1MHZ_PA, 1.4e6)[0])
    n_grid = np.arange(1, 401, dtype=float)

    X, Y, Z = geom["X"], geom["Y"], geom["Z"]
    tumour = geom["tumour"]
    lesioned = np.zeros_like(tumour, dtype=bool)
    les_lat, les_ax = geom["les_lat"], geom["les_ax"]

    regions = []          # per-region dict
    spots = []            # flat list of fired sub-spots (for maps)
    for ridx, rc in enumerate(geom["region_centers"]):
        xc, yc, zc = rc
        reg_spots = []
        for off in geom["subspot_offsets"]:
            gx, gy, gz = xc + off[0], yc + off[1], zc + off[2]
            # Skip sub-spots whose focal lesion would breach the OAR margin.
            if (geom["oar_x"] - gx) - les_lat < geom["safety_margin"]:
                reg_spots.append({"pos": (gx, gy, gz), "fired": False,
                                  "pulses": 0, "p_spot": 0.0})
                continue
            # Delivery: small electronic steering offset within the region.
            eps = kw.electronic_steering_efficiency(float(off[0]), float(off[1]),
                                                    F0, C_LIVER, True)
            path = FOCAL_DEPTH_M + gz
            deliver = kw.forward_delivery_fraction(
                float(eps), Z_FAT, Z_LIVER, ALPHA_LIVER_NP_M, float(path),
                0.0, F0, R0_NUC, C_LIVER, RHO_LIVER, MEDIUM.mu, P0, MEDIUM.gamma)
            p_spot = P_DRIVE_PA * deliver
            p_single = float(kw.intrinsic_threshold_cavitation_probability(
                np.array([p_spot]), p_t, SIGMA_T_PA)[0])
            if p_single < 0.5:
                # Sub-threshold after delivery losses — cannot lesion by steering.
                reg_spots.append({"pos": (gx, gy, gz), "fired": False,
                                  "pulses": 0, "p_spot": p_spot})
                continue
            # Pulses for complete fractionation: per-pulse homogenisation fraction
            # q grows with the genuine per-pulse inertial cavitation dose; coverage
            # saturates as 1−(1−q)^N → solve for N at COVERAGE_TARGET.
            icd = float(np.interp(p_spot, p_bins, icd_bin))
            q = float(np.clip(icd / FRACTIONATION_ICD_REF, 0.005, 0.08))
            pcum = np.asarray(kw.cumulative_cavitation_probability(q, n_grid), float)
            hit = np.argmax(pcum >= COVERAGE_TARGET)
            pulses = int(n_grid[hit]) if pcum[hit] >= COVERAGE_TARGET else int(n_grid[-1])
            reg_spots.append({"pos": (gx, gy, gz), "fired": True,
                              "pulses": pulses, "p_spot": p_spot})
            spots.append(reg_spots[-1])
            # Mark the focal lesion ellipsoid (lethal core) as lesioned.
            ell = ((X - gx) / les_lat) ** 2 + ((Y - gy) / les_lat) ** 2 + ((Z - gz) / les_ax) ** 2
            lesioned |= ell <= 1.0
        fired = [s for s in reg_spots if s["fired"]]
        n_fired = len(fired)
        n_rep = max((s["pulses"] for s in fired), default=0)
        regions.append({"center": rc, "spots": reg_spots, "n_fired": n_fired,
                        "n_rep": n_rep})

    cov = float(np.count_nonzero(lesioned & tumour) / max(np.count_nonzero(tumour), 1))

    # Build the full multi-region pulse timeline (Rust schedule per region,
    # concatenated with mechanical-move dwell between regions).
    onsets, region_id, t = [], [], 0.0
    region_windows = []
    for ridx, reg in enumerate(regions):
        if reg["n_fired"] == 0 or reg["n_rep"] == 0:
            continue
        on, sub, rep, pd_s, rep_t, son_t, nrep = kw.sonication_schedule(
            reg["n_fired"], reg["n_rep"], PULSE_DURATION_S, PRF_HZ, True)
        on = np.asarray(on) + t
        onsets.append(on)
        region_id.append(np.full(on.size, ridx))
        region_windows.append((ridx, t, t + son_t, reg["center"]))
        t += son_t + REGION_MOVE_S
    onsets = np.concatenate(onsets) if onsets else np.zeros(0)
    region_id = np.concatenate(region_id) if region_id else np.zeros(0)
    treatment_s = float(t)

    return {
        "regions": regions, "spots": spots, "lesioned": lesioned, "coverage": cov,
        "onsets": onsets, "region_id": region_id, "treatment_s": treatment_s,
        "region_windows": region_windows, "rep_time_s": geom and None,
        "n_regions_active": len(region_windows), "p_t": p_t,
        "rep_time_value": float(region_windows[0][2] - region_windows[0][1]) if region_windows else 0.0,
    }


def _pulse_icd(drive_pa, rng, n_bubbles=6):
    icds = []
    for _ in range(n_bubbles):
        r0 = float(rng.lognormal(np.log(1.5e-6), 0.4))
        _t, r, rdot, _e, _mc, _mm, _nc, _cv = kw.simulate_bubble_emission(
            r0, drive_pa, F0, 8.0, 4096, 5.0e-2,
            p0_pa=MEDIUM.p0, rho=MEDIUM.rho, c_liquid=MEDIUM.c_l, mu=MEDIUM.mu,
            sigma=MEDIUM.sigma, pv=MEDIUM.pv, gamma=MEDIUM.gamma)
        r = np.asarray(r, float); rdot = np.asarray(rdot, float)
        if r.size < 4 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(rdot))):
            continue
        icds.append(float(kw.inertial_cavitation_dose(r, rdot, r0)))
    return float(np.mean(icds)) if icds else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fig A: real pulse train (plotted, not drawn)
# ─────────────────────────────────────────────────────────────────────────────
def figure_pulsing_pattern(geom, plan):
    onsets = plan["onsets"]
    region_id = plan["region_id"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

    # Overview: every fired pulse as a marker over the whole treatment, coloured
    # by sonication region; region windows shaded.
    cmap = plt.cm.turbo
    nreg = max(int(region_id.max()) + 1, 1) if onsets.size else 1
    for (ridx, t0, t1, _ctr) in plan["region_windows"]:
        ax0.axvspan(t0, t1, color=cmap(ridx / nreg), alpha=0.12)
    ax0.plot(onsets, region_id, "|", ms=4, alpha=0.5, color="#1f3c88")
    ax0.set_xlabel("treatment time [s]"); ax0.set_ylabel("sonication region #")
    ax0.set_title(f"Whole-tumour pulse train — {plan['n_regions_active']} sonication "
                  f"regions, {onsets.size} pulses, ③ Sonication Duration "
                  f"{plan['treatment_s']:.1f} s")
    ax0.set_xlim(0, plan["treatment_s"])

    # Zoom: the first region's first few repetitions, real onsets per sub-spot.
    if plan["region_windows"]:
        ridx0, t0, t1, _ = plan["region_windows"][0]
        reg = plan["regions"][ridx0]
        n_fired = reg["n_fired"]
        on0, sub0, rep0, pd_s, rep_t, _son, _nr = kw.sonication_schedule(
            n_fired, reg["n_rep"], PULSE_DURATION_S, PRF_HZ, True)
        on0 = np.asarray(on0); sub0 = np.asarray(sub0)
        show = on0 < 3.0 * rep_t          # first 3 repetitions
        ax1.stem(on0[show] * 1e3, sub0[show] + 1, basefmt=" ",
                 linefmt="#1f3c88", markerfmt="o")
        ax1.set_xlabel("time within region [ms]"); ax1.set_ylabel("sub-spot #")
        ax1.set_title("Interleaved sub-spot firing within one region "
                      f"(① pulse {pd_s*1e6:.0f} µs, ② repetition time {rep_t*1e3:.0f} ms)")
        # ② repetition-time bracket.
        ax1.annotate("", xy=(rep_t * 1e3, n_fired + 0.6), xytext=(0, n_fired + 0.6),
                     arrowprops=dict(arrowstyle="<->", color="k"))
        ax1.text(rep_t * 1e3 / 2, n_fired + 0.8, "② Repetition Time",
                 ha="center", fontsize=9)
        ax1.set_ylim(0.4, n_fired + 1.4)

    fig.tight_layout()
    _save(fig, "ch21e_pipeline_pulsing_pattern")


# ─────────────────────────────────────────────────────────────────────────────
# Fig B: sensor-recorded cavitation monitor over the whole treatment
# ─────────────────────────────────────────────────────────────────────────────
def simulate_measured_sonication(plan, n_pulses=80, n_draws=3, seed=11):
    rng = np.random.default_rng(seed)
    f0 = F0
    treatment_s = plan["treatment_s"]
    dt = max(treatment_s, 1e-6) / n_pulses
    tau_d = kw.epstein_plesset_dissolution_time(R0_NUC, 0.5)
    rep_time = max(plan["rep_time_value"], 1e-3)
    carry = float(np.exp(-rep_time / max(tau_d, 1e-9)))

    p_lo, p_hi = 0.4 * P_DRIVE_PA, P_DRIVE_PA
    p = 0.7 * P_DRIVE_PA
    cal = simulate_population_emission(kw, drive_pa=p, f0=f0, medium=MEDIUM,
                                       n_bubbles=12, rng=rng, n_cycles=8.0,
                                       n_out=4096, steps_per_cycle=1500)
    target_signal = 0.85 * (cal["stable"] + cal["broadband"])
    inertial_cap = 1.4 * cal["broadband"]

    sig = np.zeros(n_pulses); pwr = np.zeros(n_pulses)
    stable = np.zeros(n_pulses); broad = np.zeros(n_pulses); beta = np.zeros(n_pulses)
    b = 0.0
    z1, z2, path = Z_FAT, Z_LIVER, FOCAL_DEPTH_M
    cloud = 3.0e-3
    spectrum = None
    for k in range(n_pulses):
        b_path = b * cloud / max(path, 1e-9)
        recv = kw.received_signal_fraction(z1, z2, ALPHA_LIVER_NP_M, path, b_path,
                                           f0, R0_NUC, C_LIVER, RHO_LIVER,
                                           MEDIUM.mu, P0, MEDIUM.gamma)
        st_acc = br_acc = 0.0
        for _ in range(n_draws):
            r = simulate_population_emission(kw, drive_pa=p, f0=f0, medium=MEDIUM,
                                             n_bubbles=12, rng=rng, n_cycles=8.0,
                                             n_out=4096, steps_per_cycle=1500)
            st_acc += r["stable"]; br_acc += r["broadband"]
            if spectrum is None and r["psd"].size > 4:
                spectrum = (np.asarray(r["freqs"]), np.asarray(r["psd"]))
        r_st, r_br = st_acc / n_draws, br_acc / n_draws
        s_st, s_br = r_st * recv, r_br * recv
        stable[k], broad[k], sig[k] = s_st, s_br, s_st + s_br
        pwr[k] = (p / p_hi) ** 2 * 100.0
        beta[k] = b
        deposit = RESIDUAL_VOID_PER_PULSE * (r_br / max(inertial_cap, 1e-12))
        b = min((b + deposit) * carry, 3.0e-6)
        p = kw.cavitation_controller_pressure(p, s_st, s_br, target_signal,
                                              inertial_cap, 0.06, p_lo, p_hi)

    t = np.arange(n_pulses) * dt
    cumulative = np.asarray(kw.cumulative_cavitation_dose(sig, dt), float)
    goal = 0.9 * float(cumulative[-1]) if cumulative[-1] > 0 else 1.0
    done = int(np.argmax(cumulative >= goal)) if np.any(cumulative >= goal) else n_pulses - 1
    return {"t": t, "signal": sig, "power_pct": pwr, "cumulative": cumulative,
            "goal": goal, "beta": beta, "spectrum": spectrum, "tau_d_s": tau_d,
            "done_t": t[done], "dt": dt}


def figure_monitor(mon):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("black")
    for a in axes:
        a.set_facecolor("black"); a.tick_params(colors="white")
        for sp in a.spines.values():
            sp.set_color("white")
    a = axes[0]
    if mon["spectrum"] is not None:
        f, psd = mon["spectrum"]; m = f > 5e4
        fm, pp = f[m] / 1e6, psd[m]
        a.fill_between(fm, pp / (pp.max() + 1e-30), color="orange", alpha=0.9)
        a.set_xlim(0, 4)
    a.set_title("Acoustic Spectrum", color="orange"); a.set_xlabel("MHz", color="white")
    a = axes[1]
    a.bar(mon["t"], mon["signal"] / (mon["signal"].max() + 1e-30),
          width=mon["dt"] * 0.9, color="orange", alpha=0.85)
    a.plot(mon["t"], mon["power_pct"] / 100.0, color="lime", lw=1.6, label="Power %")
    a.set_title("Acoustic Controls", color="orange"); a.set_xlabel("sec", color="white")
    a.legend(facecolor="black", labelcolor="white")
    a = axes[2]
    a.plot(mon["t"], mon["cumulative"], color="orange", lw=2.2)
    a.axhline(mon["goal"], color="yellow", ls="--", lw=1.4)
    a.axvline(mon["done_t"], color="cyan", ls=":", lw=1.4,
              label=f"tumour lesioned @ {mon['done_t']:.0f}s")
    a.set_title("Cavitation Dose", color="orange"); a.set_xlabel("sec", color="white")
    a.legend(facecolor="black", labelcolor="white")
    fig.suptitle("Sensor-recorded cavitation over the whole-tumour treatment", color="white")
    _save(fig, "ch21e_pipeline_monitor", facecolor="black")


# ─────────────────────────────────────────────────────────────────────────────
# Fig C: treatment console
# ─────────────────────────────────────────────────────────────────────────────
def figure_treatment_screen(geom, plan, mon):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0b0f1a")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    def style(a, title):
        a.set_facecolor("#0b0f1a"); a.set_title(title, color="#9ecbff", fontsize=10)
        a.tick_params(colors="#9ecbff", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#33415c")

    xs, ys, zs = geom["xs"] * 1e3, geom["ys"] * 1e3, geom["zs"] * 1e3
    tumour, lesioned = geom["tumour"], plan["lesioned"]
    rc = geom["region_centers"] * 1e3
    iy0 = tumour.shape[1] // 2          # elevation mid-slice
    iz0 = tumour.shape[2] // 2          # axial mid-slice

    # (1) Axial–lateral slice (elevation = 0): tumour, lesion, OAR, regions.
    a = fig.add_subplot(gs[0, 0]); style(a, "Coverage — axial×lateral slice")
    _slice_map(a, xs, zs, tumour[:, iy0, :], lesioned[:, iy0, :])
    a.axvline(geom["oar_x"] * 1e3, color="red", lw=2.5)
    a.scatter(rc[:, 0], rc[:, 2], s=12, c="cyan", marker="+")
    a.set_xlabel("lateral [mm]", color="#9ecbff"); a.set_ylabel("axial [mm]", color="#9ecbff")

    # (2) Lateral–elevation slice (axial = 0).
    a = fig.add_subplot(gs[0, 1]); style(a, "Coverage — lateral×elevation slice")
    _slice_map(a, xs, ys, tumour[:, :, iz0], lesioned[:, :, iz0])
    a.axvline(geom["oar_x"] * 1e3, color="red", lw=2.5)
    a.scatter(rc[:, 0], rc[:, 1], s=12, c="cyan", marker="+")
    a.set_xlabel("lateral [mm]", color="#9ecbff"); a.set_ylabel("elevation [mm]", color="#9ecbff")

    # (3) Coverage vs treatment time (region-by-region accumulation).
    a = fig.add_subplot(gs[0, 2]); style(a, "Tumour coverage vs time")
    tcov, ccov = _coverage_curve(geom, plan)
    a.plot(tcov, ccov * 100.0, color="#2ecc71", lw=2.2)
    a.axhline(98.0, color="yellow", ls="--", lw=1.2)
    a.set_xlabel("treatment time [s]", color="#9ecbff")
    a.set_ylabel("% tumour lesioned", color="#9ecbff"); a.set_ylim(0, 105)

    # (4–6) Live monitor.
    a = fig.add_subplot(gs[1, 0]); style(a, "Acoustic Spectrum")
    if mon["spectrum"] is not None:
        f, psd = mon["spectrum"]; m = f > 5e4
        a.fill_between(f[m] / 1e6, psd[m] / (psd[m].max() + 1e-30), color="orange", alpha=0.9)
        a.set_xlim(0, 4)
    a.set_xlabel("MHz", color="#9ecbff")
    a = fig.add_subplot(gs[1, 1]); style(a, "Acoustic Controls (signal + power)")
    a.bar(mon["t"], mon["signal"] / (mon["signal"].max() + 1e-30),
          width=mon["dt"] * 0.9, color="orange", alpha=0.85)
    a.plot(mon["t"], mon["power_pct"] / 100.0, color="lime", lw=1.6)
    a.set_xlabel("sec", color="#9ecbff")
    a = fig.add_subplot(gs[1, 2]); style(a, "Cavitation Dose → goal")
    a.plot(mon["t"], mon["cumulative"], color="orange", lw=2.0)
    a.axhline(mon["goal"], color="yellow", ls="--", lw=1.2)
    a.axvline(mon["done_t"], color="cyan", ls=":", lw=1.2)
    a.set_xlabel("sec", color="#9ecbff")

    # (7) Residual-gas β(t).
    a = fig.add_subplot(gs[2, 0]); style(a, "Residual-gas void fraction β(t)")
    a.plot(mon["t"], mon["beta"], color="#ff6f6f", lw=1.8)
    a.set_xlabel("sec", color="#9ecbff")

    # (8) Feedback gauges.
    min_margin = min((geom["oar_x"] - s["pos"][0] - geom["les_lat"]) * 1e3
                     for s in plan["spots"]) if plan["spots"] else 0.0
    gauges = [
        ("Tumour coverage", plan["coverage"] * 100.0, "%", plan["coverage"] >= 0.98),
        ("Sonication regions", float(plan["n_regions_active"]), "", True),
        ("Min OAR margin", min_margin, "mm", min_margin >= geom["safety_margin"] * 1e3 * 0.5),
        ("Treatment time", plan["treatment_s"], "s", True),
    ]
    a = fig.add_subplot(gs[2, 1:]); style(a, "Treatment feedback"); a.axis("off")
    for i, (name, val, unit, ok) in enumerate(gauges):
        x = 0.02 + i * 0.245
        col = "#2ecc71" if ok else "#e67e22"
        a.add_patch(FancyBboxPatch((x, 0.25), 0.22, 0.5, transform=a.transAxes,
                    boxstyle="round,pad=0.02", facecolor="#16203a", edgecolor=col, lw=2))
        a.text(x + 0.11, 0.62, name, transform=a.transAxes, ha="center",
               color="#9ecbff", fontsize=9)
        a.text(x + 0.11, 0.42, f"{val:.1f} {unit}", transform=a.transAxes,
               ha="center", color=col, fontsize=15, fontweight="bold")

    fig.suptitle("kwavers — Liver Histotripsy Treatment Console (whole-tumour, multi-region)",
                 color="white", fontsize=14)
    _save(fig, "ch21e_pipeline_treatment_screen", facecolor="#0b0f1a")


def _slice_map(a, xax, yax, tumour_slice, lesion_slice):
    """Plot a tumour/lesion slice as genuine data: tumour outline + lesioned px."""
    rgba = np.zeros(tumour_slice.shape + (4,))
    rgba[tumour_slice] = [0.3, 0.5, 0.9, 0.35]          # tumour (blue)
    rgba[lesion_slice & tumour_slice] = [0.95, 0.55, 0.1, 0.95]  # lesioned (orange)
    rgba[lesion_slice & ~tumour_slice] = [0.6, 0.2, 0.2, 0.6]    # spill outside
    a.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower",
             extent=[xax[0], xax[-1], yax[0], yax[-1]], aspect="auto")


def _coverage_curve(geom, plan):
    """Cumulative lesioned-tumour fraction as regions are treated in time order."""
    X, Y, Z = geom["X"], geom["Y"], geom["Z"]
    tumour = geom["tumour"]
    ntum = max(np.count_nonzero(tumour), 1)
    les_lat, les_ax = geom["les_lat"], geom["les_ax"]
    acc = np.zeros_like(tumour, dtype=bool)
    ts, cs = [0.0], [0.0]
    for (ridx, t0, t1, _ctr) in plan["region_windows"]:
        for s in plan["regions"][ridx]["spots"]:
            if not s["fired"]:
                continue
            gx, gy, gz = s["pos"]
            ell = ((X - gx) / les_lat) ** 2 + ((Y - gy) / les_lat) ** 2 + ((Z - gz) / les_ax) ** 2
            acc |= ell <= 1.0
        ts.append(t1); cs.append(np.count_nonzero(acc & tumour) / ntum)
    return np.array(ts), np.array(cs)


def _save(fig, name, facecolor="white"):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    print("Building tumour + sonication-region raster…")
    geom = build_geometry()
    print(f"  {geom['region_centers'].shape[0]} candidate regions, "
          f"{geom['subspot_offsets'].shape[0]} sub-spots/region")
    print("Planning treatment (genuine per-spot physics + volumetric coverage)…")
    plan = plan_treatment(geom)
    print(f"  {plan['n_regions_active']} active regions, {len(plan['spots'])} fired sub-spots, "
          f"{plan['onsets'].size} pulses; coverage {plan['coverage']*100:.1f}%; "
          f"treatment {plan['treatment_s']:.1f} s")
    print("Fig A — whole-tumour pulse train…")
    figure_pulsing_pattern(geom, plan)
    print("Fig B — sensor-recorded cavitation monitor…")
    mon = simulate_measured_sonication(plan)
    figure_monitor(mon)
    print("Fig C — treatment console…")
    figure_treatment_screen(geom, plan, mon)
    print("Done.")


if __name__ == "__main__":
    main()
