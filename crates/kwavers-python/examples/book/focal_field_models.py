"""Focal-field models for the ch21e boiling-histotripsy lesion comparison.

Two ways to obtain the focal transverse pressure profile B(r) (normalised to 1
at the focus) that drives the boiling-onset lesion model:

  • ``analytic``        — Penttinen/O'Neil focused-bowl Gaussian (closed form).
  • ``pstd_nonlinear``  — a genuine PSTD simulation of the focused bowl WITH
                          nonlinearity (Westervelt shock steepening), giving the
                          diffraction- and shock-shaped focal field. Run once,
                          cached to ``data/kernels/`` (the run shows the focal
                          p_max/p_min asymmetry that marks shock formation).

Both are scaled to the same delivered focal peak `p_spot` in the lesion model, so
the comparison isolates the FIELD-SHAPE effect (the Gaussian misses the nonlinear
focal sharpening and the diffraction sidelobes the PSTD field has).
"""

from __future__ import annotations

import os

import numpy as np
import pykwavers as kw

import cavitation_kernel as ck

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    d = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "data", "lits17_sample")):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


def _cache_path(f0, roc, diameter, ppw):
    return os.path.join(
        _repo_root(), "data", "kernels",
        f"focal_nl_{f0/1e6:.2f}MHz_{roc*1e3:.0f}roc_{diameter*1e3:.0f}diam_ppw{ppw}.npz")


def gaussian_profile(sigma_lat_m):
    """Analytic normalised transverse focal profile B(r) = exp(−r²/2σ²)."""
    def B(r):
        return np.exp(-np.asarray(r, float) ** 2 / (2.0 * sigma_lat_m * sigma_lat_m))
    return B


def _run_pstd_nonlinear_bowl(f0, c0, rho0, b_over_a, roc=30e-3, diameter=30e-3,
                             ppw=6, pml=10, n_cycles=3, nt=None, source_pa=2.0e6):
    """Run a focused-bowl PSTD sim WITH nonlinearity; return the transverse focal
    radial profile of the peak (shock) pressure and the focal p_max/p_min."""
    grid, info = ck.build_grid(f0, c0, roc, diameter, ppw, pml)
    dx = info["dx"]
    dt = 1.0 / (12 * f0)
    if nt is None:
        # Enough steps for the pulse to propagate apex→focus and form the peak.
        nt = int(np.ceil(roc / c0 / dt)) + n_cycles * 12 + 6 * 12
    sig = ck.make_signal(nt, dt, f0, source_pa, n_cycles, 12)
    karray = ck.make_kwave_bowl(grid, info, roc, diameter)
    src = kw.Source.from_kwave_array(karray, sig.astype(np.float64), f0, mode="additive")
    sh = info["shape"]
    medium = kw.Medium(np.full(sh, c0), np.full(sh, rho0), None, None, np.full(sh, b_over_a))
    sensor = kw.Sensor.from_mask(np.ones(sh, dtype=bool))
    sensor.set_record(["p_max", "p_min"])
    sim = kw.Simulation(grid, medium, src, sensor, solver=kw.SolverType.PSTD, pml_size=info["pml"])
    sim.set_nonlinear(True)
    res = sim.run(time_steps=nt, dt=dt)
    pmax = np.asarray(res.p_max_field)
    pmin = np.asarray(res.p_min_field)
    fx, fy, fz = info["focus_idx"]
    # Transverse radial profile of the peak (shock) pressure at the focal slice.
    sl = pmax[fx]
    yy, zz = np.meshgrid(np.arange(sl.shape[0]) - fy, np.arange(sl.shape[1]) - fz, indexing="ij")
    rr = np.sqrt(yy.astype(float) ** 2 + zz.astype(float) ** 2) * dx
    edges = np.linspace(0.0, 8e-3, 61)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([
        sl[(rr >= edges[i]) & (rr < edges[i + 1])].mean()
        if np.any((rr >= edges[i]) & (rr < edges[i + 1])) else np.nan
        for i in range(len(rc))])
    prof = np.nan_to_num(prof, nan=0.0)
    prof = prof / max(prof[0], prof.max(), 1e-30)
    return rc, prof, float(pmax[fx, fy, fz]), float(-pmin[fx, fy, fz])


def pstd_nonlinear_profile(f0, c0, rho0, b_over_a, roc=30e-3, diameter=30e-3,
                           ppw=6, force=False):
    """Return (r [m], normalised B(r), {p_max_focus, p_min_focus}) for the PSTD
    nonlinear focal field of the (roc, diameter) bowl, loading the cache when
    present. Use the SAME (roc, diameter) as the analytic focal_fwhm for a fair
    comparison."""
    cache = _cache_path(f0, roc, diameter, ppw)
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return d["r"], d["prof"], {"p_max_focus": float(d["pmax"]), "p_min_focus": float(d["pmin"])}
    print(f"[focal] running PSTD nonlinear focused-bowl sim "
          f"(roc={roc*1e3:.0f}mm diam={diameter*1e3:.0f}mm ppw={ppw}) -> {os.path.basename(cache)} ...")
    rc, prof, pmax, pmin = _run_pstd_nonlinear_bowl(f0, c0, rho0, b_over_a, roc, diameter, ppw)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, r=rc, prof=prof, pmax=pmax, pmin=pmin)
    print(f"[focal] focal p_max=+{pmax/1e6:.1f} MPa, p_min=-{pmin/1e6:.1f} MPa "
          f"(asymmetry => nonlinear shock)")
    return rc, prof, {"p_max_focus": pmax, "p_min_focus": pmin}


def _lacuna_cache_path(f0, roc, diameter, ppw, beta):
    return os.path.join(
        _repo_root(), "data", "kernels",
        f"focal_lacuna_{f0/1e6:.2f}MHz_{roc*1e3:.0f}roc_{diameter*1e3:.0f}diam_"
        f"ppw{ppw}_b{beta:.0e}.npz")


def pstd_lacuna_focal_fields(f0, c0, rho0, beta_lacuna=2.0e-3, roc=30e-3,
                             diameter=30e-3, ppw=4, pml=10, lacuna_radius_m=3.0e-3,
                             lacuna_offset_m=10.0e-3, source_pa=2.0e6, force=False,
                             c_floor=300.0):
    """Full-3-D PSTD of the focused bowl with vs without a pre-focal LACUNA gas
    inclusion. The lacuna's Wood-collapsed sound speed (≈160 m/s at β=5e-3) is a
    near-pressure-release reflector: it casts an acoustic shadow over the focus and
    sets up a pre-lacuna standing wave — the resolved standing-wave/shielding field
    that the per-pulse path-integral model approximates. The β field and its Wood
    sound-speed coupling come from the Rust `ResidualGasField` (SSOT).

    Returns axial (beam-axis) and focal-plane peak-pressure fields for both cases.
    Cached to data/kernels."""
    cache = _lacuna_cache_path(f0, roc, diameter, ppw, beta_lacuna)
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    grid, info = ck.build_grid(f0, c0, roc, diameter, ppw, pml)
    dx = info["dx"]; sh = info["shape"]
    dt = 1.0 / (12 * f0)
    nt = int(np.ceil(roc / c0 / dt)) + 3 * 12 + 8 * 12
    sig = ck.make_signal(nt, dt, f0, source_pa, 3, 12)
    karray = ck.make_kwave_bowl(grid, info, roc, diameter)
    src = kw.Source.from_kwave_array(karray, sig.astype(np.float64), f0, mode="additive")
    fx, fy, fz = info["focus_idx"]
    # Lacuna gas sphere on the beam axis, pre-focal (between apex and focus).
    cx = max(fx - int(round(lacuna_offset_m / dx)), 1)
    xx, yy, zz = np.ogrid[: sh[0], : sh[1], : sh[2]]
    rr2 = (((xx - cx) * dx) ** 2 + ((yy - fy) * dx) ** 2 + ((zz - fz) * dx) ** 2)
    beta = np.where(rr2 <= lacuna_radius_m ** 2, beta_lacuna, 0.0)
    rgf = kw.ResidualGasField(sh[0], sh[1], sh[2], 30.0e-6)
    rgf.deposit(beta)
    # Wood collapse; floor the sound speed so the steep c-gradient stays PSTD-stable
    # while remaining a strong gas/tissue reflector (|R| large).
    c_lac = np.maximum(np.asarray(rgf.sound_speed_field(c0, rho0, 343.0, 1.2)), c_floor)

    def _run(c_field):
        medium = kw.Medium(np.ascontiguousarray(c_field), np.full(sh, rho0))
        sensor = kw.Sensor.from_mask(np.ones(sh, dtype=bool))
        sensor.set_record(["p_max"])
        sim = kw.Simulation(grid, medium, src, sensor, solver=kw.SolverType.PSTD,
                            pml_size=info["pml"])
        res = sim.run(time_steps=nt, dt=dt)
        return np.asarray(res.p_max_field)

    pmax_clean = _run(np.full(sh, c0))
    pmax_lac = _run(c_lac)
    out = {
        "dx": np.array(dx), "fx": np.array(fx), "fy": np.array(fy), "fz": np.array(fz),
        "cx": np.array(cx), "lacuna_radius_m": np.array(lacuna_radius_m),
        "clean_axial": pmax_clean[:, fy, fz], "lac_axial": pmax_lac[:, fy, fz],
        "clean_slice": pmax_clean[:, :, fz], "lac_slice": pmax_lac[:, :, fz],
        "beta_slice": beta[:, :, fz],
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


def pstd_profile_callable(rc, prof):
    """Wrap the cached PSTD radial profile as a B(r) callable (0 beyond range)."""
    def B(r):
        return np.interp(np.asarray(r, float), rc, prof, left=float(prof[0]), right=0.0)
    return B
