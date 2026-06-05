"""
Chapter 1 figure generation — Wave Physics Fundamentals
========================================================

Produces all figures referenced in docs/book/foundations.md.
Run from the repository root:

    python crates/kwavers-python/examples/book/ch01_wave_physics_fundamentals.py

Output directory: docs/book/figures/ch01/

Physics policy
--------------
All physics is computed by kwavers (Rust); this file performs only matplotlib
rendering and signal post-processing (FFT, projection).  Figures 1.1 and 1.4 are
genuine solver / exact-series results, not closed-form approximations:

  * Fig 1.1 drives the real FDTD and PSTD forward solvers on an initial-value
    problem and overlays the analytic standing wave plus the pointwise residual.
  * Fig 1.4 uses the exact Fubini Bessel harmonic series and marks the
    quasi-linear tangent (Theorem 1.8, Eq. 1.27) and the shock distance.

Solver note (Fig 1.1)
---------------------
The PSTD initial-value path uses an equal split-density representation
(rho_x = rho_y = rho_z = rho/n_active).  For a field that is uniform along a
lateral axis the corresponding components have zero divergence and cannot
evolve, which collapses a quasi-1-D standing mode on a multi-dimensional grid.
FDTD is unaffected.  The PSTD curve is therefore run on a genuine 1-D grid
(n_active = 1, no split) and FDTD on a thin 3-D slab; both reproduce the
analytic standing wave.  See memory project_pstd_ivp_standing_wave_dissipation.

References
----------
- Duck (1990) Physical Properties of Tissue
- Treeby & Cox (2010) doi:10.1121/1.3377056
- Hamilton & Blackstock (1998) Nonlinear Acoustics, ch. 3-4
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pykwavers as kw

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
FIG_W = 8.0
FIG_H = 5.5

# Water reference properties used by the worked example (foundations.md §1.13).
C0 = 1481.0       # m/s, sound speed in water at 20 °C
RHO0 = 998.0      # kg/m³


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path.with_suffix(".pdf"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def _standing_wave_pstd(L, nx, k, p0):
    """PSTD standing-wave IVP with a TRANSPARENT boundary (set_pml_alpha(0)).

    The spectral PSTD spatial operator is intrinsically periodic, so a transparent
    boundary makes the domain a perfect resonator: the IVP p(x,0)=p0 sin(kx),
    u(x,0)=0 reproduces p0 sin(kx) cos(ωt) (the default absorbing PML instead
    absorbs the counter-propagating constituents and the standing wave decays).
    Runs in true 1-D (n_active = 1) so the split-density IVP has no frozen
    lateral component.  Returns (sensor_data[nx, nt], dt, time_offset).
    """
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    om = C0 * k
    dt = 0.2 * dx / C0
    n_steps = int(math.ceil((3 * math.pi / 4 / om) / dt)) + 4
    p0arr = np.zeros((nx, 1, 1))
    p0arr[:, 0, 0] = p0 * np.sin(k * x)
    grid = kw.Grid(nx=nx, ny=1, nz=1, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    src = kw.Source.from_initial_pressure(p0arr.copy())
    mask = np.zeros((nx, 1, 1), dtype=bool)
    mask[:, 0, 0] = True
    # pml_size=0 → transparent boundary; with the periodic spectral PSTD operator
    # the column is a lossless resonator that sustains the standing wave.
    sim = kw.Simulation(grid, medium, src, kw.Sensor.from_mask(mask),
                        solver=kw.SolverType.PSTD, pml_size=0)
    res = sim.run(time_steps=n_steps, dt=dt)
    sd = np.asarray(res.sensor_data, dtype=float)
    dt_a = float(res.dt)
    shape = np.sin(k * x)
    proj = (sd.T @ shape) / (shape @ shape)
    n = np.arange(sd.shape[1])
    off = min((0.0, 1.0),
              key=lambda o: np.sum((proj - proj[0] * np.cos(om * (n + o) * dt_a)) ** 2))
    return sd, dt_a, off


def _travelling_pulse_fdtd(nx, dx):
    """FDTD travelling-pulse IVP validated against the d'Alembert solution.

    A Gaussian-modulated tone burst p(x,0)=g(x), u(x,0)=0 splits (d'Alembert)
    into two half-amplitude counter-propagating copies:
        p(x,t) = ½[g(x - c₀t) + g(x + c₀t)].
    FDTD is run on a thin 3-D slab (it requires ≥2 lateral cells); the field is
    uniform laterally and the central x-line is recorded.  Returns
    (x, g, sd[nx, nt], dt, t_eval, n_eval).
    """
    x = (np.arange(nx) + 0.5) * dx
    x0 = 0.3 * nx * dx
    sig = 2.5e-3
    lam = 8e-3
    g = np.exp(-((x - x0) ** 2) / (2 * sig ** 2)) * np.cos(2 * np.pi * (x - x0) / lam)
    g *= 1.0e5
    ny = nz = 16
    p0arr = np.zeros((nx, ny, nz))
    p0arr[:] = g[:, None, None]
    grid = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    src = kw.Source.from_initial_pressure(p0arr.copy())
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[:, ny // 2, nz // 2] = True
    sim = kw.Simulation(grid, medium, src, kw.Sensor.from_mask(mask),
                        solver=kw.SolverType.FDTD, pml_size=12)
    dt = 0.3 * dx / C0
    # Evaluate while the right-mover is mid-domain, clear of the PML.
    t_eval = (0.55 * nx * dx - x0) / C0
    n_eval = int(round(t_eval / dt))
    res = sim.run(time_steps=n_eval + 2, dt=dt)
    sd = np.asarray(res.sensor_data, dtype=float)
    return x, g, sd, float(res.dt), t_eval, n_eval


def fig_standing_wave() -> None:
    """Figure 1.1 — genuine solver validation: PSTD standing wave (transparent
    boundary) + FDTD travelling pulse vs the d'Alembert solution."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ── Panel (a): PSTD standing wave vs analytic (Eq. 1.30) ──────────────────
    L = 0.05
    nx = 400
    k = 2.0 * math.pi / L            # full-wavelength mode (periodic spectral basis)
    om = C0 * k
    p0 = 1.0e5
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    sd, dt_a, off = _standing_wave_pstd(L, nx, k, p0)

    phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    labels = [r"$\omega t = 0$", r"$\omega t = \pi/4$",
              r"$\omega t = \pi/2$", r"$\omega t = 3\pi/4$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    max_err_a = 0.0
    for phase, label, color in zip(phases, labels, colors):
        n = max(0, min(int(round((phase / om) / dt_a - off)), sd.shape[1] - 1))
        t = (n + off) * dt_a
        ana = p0 * np.sin(k * x) * math.cos(om * t)
        axa.plot(x * 1e3, ana / 1e3, color=color, lw=1.6, label=label)
        axa.plot(x[::14] * 1e3, sd[::14, n] / 1e3, "o", color=color, ms=4, mfc="none", lw=0)
        max_err_a = max(max_err_a, np.abs(sd[:, n] - ana).max() / p0)
    axa.plot([], [], "o", color="k", mfc="none", ms=5, lw=0, label="PSTD (solver)")
    axa.axhline(0, color="k", lw=0.6, ls="--")
    axa.set_xlabel("Position $x$ [mm]", fontsize=11)
    axa.set_ylabel("Pressure $p$ [kPa]", fontsize=11)
    axa.set_title(
        "(a) Standing wave — PSTD (transparent boundary) vs analytic\n"
        r"$p = p_0 \sin(kx)\cos(\omega t)$  "
        rf"(max error {max_err_a*100:.2f}%)",
        fontsize=10,
    )
    axa.legend(fontsize=8, loc="upper right", ncol=2)
    axa.set_xlim(0, L * 1e3)
    axa.grid(True, ls=":", alpha=0.5)

    # ── Panel (b): FDTD travelling pulse vs d'Alembert ────────────────────────
    nxb = 700
    dxb = 0.15e-3
    xb, g, sdb, dtb, t_eval, n_eval = _travelling_pulse_fdtd(nxb, dxb)
    ct = C0 * (n_eval * dtb)
    # d'Alembert: ½[g(x-ct) + g(x+ct)]; interpolate g at shifted coordinates.
    g_right = np.interp(xb - ct, xb, g, left=0.0, right=0.0)
    g_left = np.interp(xb + ct, xb, g, left=0.0, right=0.0)
    dalembert = 0.5 * (g_right + g_left)
    err_b = np.abs(sdb[:, n_eval] - dalembert).max() / np.abs(g).max()

    axb.plot(xb * 1e3, g / 1e3, color="0.7", lw=1.2, label="initial $g(x)$")
    axb.plot(xb * 1e3, dalembert / 1e3, color="#1f77b4", lw=1.8,
             label=r"d'Alembert $\frac{1}{2}[g(x{-}ct){+}g(x{+}ct)]$")
    axb.plot(xb[::6] * 1e3, sdb[::6, n_eval] / 1e3, "x", color="#d62728", ms=4, lw=0,
             label="FDTD (solver)")
    axb.axhline(0, color="k", lw=0.6, ls="--")
    axb.set_xlabel("Position $x$ [mm]", fontsize=11)
    axb.set_ylabel("Pressure $p$ [kPa]", fontsize=11)
    axb.set_title(
        "(b) Travelling pulse — FDTD vs d'Alembert (Theorem 1.2)\n"
        rf"split into $\pm c_0 t$ pulses at $c_0={C0:.0f}$ m/s; "
        rf"{err_b*100:.1f}% shape error is FDTD numerical dispersion",
        fontsize=10,
    )
    axb.legend(fontsize=8, loc="upper right")
    axb.set_xlim(0, nxb * dxb * 1e3)
    axb.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Figure 1.1 — Wave-equation solver validation in a water column "
                 rf"($c_0={C0:.0f}$ m/s, $\rho_0={RHO0:.0f}$ kg/m³)", fontsize=12)
    fig.tight_layout()
    print(f"  Fig 1.1 (a) PSTD standing-wave max error: {max_err_a*100:.3f}%")
    print(f"  Fig 1.1 (b) FDTD travelling-pulse max error: {err_b*100:.3f}%")
    _save(fig, "fig01_standing_wave")


def fig_impedance_mismatch() -> None:
    """Figure 1.2 — normal-incidence intensity reflection at tissue interfaces."""
    interfaces = {
        "Air": 1.204 * 343.0,
        "Fat": 928 * 1440,
        "Blood": 1060 * 1575,
        "Soft tissue": 1050 * 1540,
        "Kidney": 1050 * 1560,
        "Liver": 1060 * 1555,
        "Tendon": 1100 * 1650,
        "Cartilage": 1100 * 1700,
        "Bone (cancellous)": 1800 * 2200,
        "Bone (cortical)": 1912 * 3500,
    }
    Z1 = 1050.0 * 1540.0
    names = list(interfaces.keys())
    Z2s = np.array(list(interfaces.values()))

    R_I = np.array([kw.reflection_pressure_coeff(Z1, z2) ** 2 for z2 in Z2s])
    T_I = 1.0 - R_I

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x_pos = np.arange(len(names))
    bars_r = ax.bar(x_pos, R_I * 100, color="#d62728", alpha=0.8, label="$R_I$ (reflected)")
    ax.bar(x_pos, T_I * 100, bottom=R_I * 100, color="#1f77b4",
           alpha=0.8, label="$T_I$ (transmitted)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intensity fraction [%]", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 1.2 — Normal-incidence power reflection at soft-tissue interfaces\n"
        r"$R_I = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2$ (reference $Z_1$ = soft tissue)",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    for rect, ri in zip(bars_r, R_I):
        if ri > 0.01:
            ax.text(rect.get_x() + rect.get_width() / 2, ri * 100 + 1.5,
                    f"{ri*100:.1f}%", ha="center", va="bottom", fontsize=8)
    _save(fig, "fig02_impedance_mismatch")


def fig_power_law_attenuation() -> None:
    """Figure 1.3 — power-law attenuation α(f) = α₀ f^y (Duck 1990)."""
    tissues = {
        "Water (viscothermal, y=2)": (0.0022, 2.0),
        "Blood": (0.18, 1.21),
        "Fat": (0.48, 1.0),
        "Soft tissue": (0.52, 1.0),
        "Muscle (along fibres)": (0.57, 1.0),
        "Liver": (0.50, 1.05),
        "Kidney": (1.0, 1.0),
    }
    f_MHz = np.linspace(0.5, 10.0, 500)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(tissues)))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for (name, (a0, y)), color in zip(tissues.items(), colors):
        alpha = kw.absorption_power_law_db_cm(f_MHz, a0, y)
        ax.plot(f_MHz, alpha, label=name, color=color, lw=1.8)

    ax.set_xlabel("Frequency $f$ [MHz]", fontsize=12)
    ax.set_ylabel(r"Attenuation $\alpha(f)$ [dB cm$^{-1}$]", fontsize=12)
    ax.set_title(r"Figure 1.3 — Power-law attenuation $\alpha(f) = \alpha_0 f^y$ (Duck 1990)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls=":", alpha=0.5)
    _save(fig, "fig03_power_law_attenuation")


def fig_harmonic_generation() -> None:
    """Figure 1.4 — exact Fubini harmonic growth and the Eq. (1.27) tangent."""
    sigma = np.linspace(0.0, 0.99, 400)
    n_max = 4
    # Exact Fubini Bessel amplitudes Bₙ(σ) = 2 Jₙ(nσ)/(nσ) from the Rust kernel.
    spectra = np.array([kw.fubini_harmonic_spectrum(n_max, s) for s in sigma])  # (Nσ, n_max)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    hcolors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for n in range(1, n_max + 1):
        ax.plot(sigma, np.abs(spectra[:, n - 1]) * 100,
                color=hcolors[n - 1], lw=1.9, label=f"$n={n}$ (exact Fubini)")

    # Quasi-linear tangent for the 2nd harmonic, Eq. (1.27): p₂/p₀ ≈ σ/2.
    ax.plot(sigma, sigma / 2.0 * 100, "k--", lw=1.4,
            label=r"$p_2/p_0 \approx \sigma/2$  (Eq. 1.27, small $\sigma$)")
    ax.axvline(1.0, color="grey", ls=":", lw=1.2)
    ax.text(0.92, 5, "shock\ndistance\n$\\sigma=1$", fontsize=8, color="grey", ha="right")

    ax.set_xlabel(r"Normalised distance $\sigma = z / z_{\mathrm{sh}}$", fontsize=12)
    ax.set_ylabel(r"Harmonic amplitude $|p_n| / p_0$  [%]", fontsize=12)
    ax.set_title(
        "Figure 1.4 — Exact Fubini harmonic generation and the quasi-linear tangent (Theorem 1.8)\n"
        r"lossless plane wave; $p_2$ grows as $\frac{1}{2}\sigma$ near the source, "
        r"then saturates as $\sigma \to 1$",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 105)
    ax.grid(True, ls=":", alpha=0.5)
    _save(fig, "fig04_harmonic_generation")


def fig_sound_speed_temperature() -> None:
    """Figure 1.5 — sound speed in water vs temperature (Del Grosso–Mader)."""
    T = np.linspace(0, 80, 500)
    c = kw.water_sound_speed_temperature(T)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(T, c, lw=2.0, color="#1f77b4")
    ax.axvline(20, color="k", ls=":", alpha=0.6)
    ax.axvline(37, color="#d62728", ls=":", alpha=0.6)
    ax.plot([37], [kw.water_sound_speed_temperature(np.array([37.0]))[0]],
            "o", color="#d62728", ms=6)
    ax.text(21, 1450, "20 °C", fontsize=9, color="k")
    ax.text(38, 1450, "37 °C\n(body temperature)", fontsize=9, color="#d62728")
    ax.set_xlabel("Temperature $T$ [°C]", fontsize=12)
    ax.set_ylabel(r"Sound speed $c_0$ [m s$^{-1}$]", fontsize=12)
    ax.set_title(
        "Figure 1.5 — Sound speed in water vs temperature\n"
        r"$c(T) = 1402.7 + 4.83T - 0.048T^2 + 1.47 \times 10^{-4} T^3$ (Del Grosso–Mader)",
        fontsize=11,
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(0, 80)
    _save(fig, "fig05_sound_speed_temperature")


def main() -> None:
    print(f"\nChapter 1 figures -> {OUT_DIR}\n")
    fig_standing_wave()
    fig_impedance_mismatch()
    fig_power_law_attenuation()
    fig_harmonic_generation()
    fig_sound_speed_temperature()
    print("\nDone.")


if __name__ == "__main__":
    main()
