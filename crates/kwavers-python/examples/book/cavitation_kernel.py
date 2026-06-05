"""
Per-shot cavitation-kernel generator
====================================

Drives a single focused-bowl pulse through pykwavers' PSTD solver on a
homogeneous water-equivalent medium and dumps the per-voxel pressure
statistics that constitute the histotripsy "cavitation kernel":

    p_min[i, j, k]   peak rarefactional pressure   (drives intrinsic-threshold cavitation, Maxwell 2013)
    p_max[i, j, k]   peak compressional pressure   (shock-vapor heating proxy)
    p_rms[i, j, k]   RMS pressure                  (bulk-thermal proxy)

The kernel replaces the analytical Gaussian envelope used by the ch21d/e
treatment planners with a real PSTD-derived focal field — same Maxwell-
2013 erf-CDF post-processing applies, so the cavitation-probability
calculus in the planners is unchanged.

Output:
    data/kernels/kernel_<f0_MHz>_<pnp_MPa>_<roc_mm>roc_<diam_mm>diam.npz
        contains arrays p_min, p_max, p_rms, p_final and metadata
        (dx, f0, pnp, roc, aperture_diameter, bowl_apex_x_index).

Algorithmic discipline:
    * Grid is sized to encompass the focal envelope plus a CPML margin
      such that the focal voxel sits at info["focus_idx"] = (apex_x +
      roc/dx, ny//2, nz//2). Lateral half-width chosen as 4× FWHM_lat to
      capture sidelobes; axial extent runs from apex through 1.5× roc to
      capture pre-focal geometric divergence and post-focal diffraction.
    * Source is rasterised by KWaveArray.add_bowl_element; phase delays
      are computed by kwavers' bowl-element rasteriser, so the focal
      gain is an analytic O'Neil/Penttinen result, not an approximation.
    * Time steps run for `n_cycles` periods plus enough post-pulse time
      for the wave to clear the focal voxel (4× focal-radius / c0).
    * Pressure statistics are accumulated over the full simulation,
      including ring-up — this is the honest kernel because real
      treatment pulses see the same ring-up.

Usage:
    python pykwavers/examples/book/cavitation_kernel.py \
        --f0 1e6 --pnp 30e6 --n-cycles 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
KERNEL_DIR = os.path.join(REPO_ROOT, "data", "kernels")
os.makedirs(KERNEL_DIR, exist_ok=True)


def fwhm_focal_extents(f0: float, c0: float, roc: float, diameter: float) -> tuple[float, float]:
    """Penttinen 1976 focal-spot FWHM for a focused bowl.
    Returns (lateral_FWHM, axial_FWHM) in metres."""
    lam = c0 / f0
    fnum = roc / diameter
    return 1.41 * lam * fnum, 7.0 * lam * fnum * fnum


def build_grid(f0: float, c0: float, roc: float, diameter: float,
               ppw: int, pml: int) -> tuple[kw.Grid, dict]:
    """Build a grid that encompasses the focal envelope plus PML margin
    and returns the geometry metadata needed to interpret the kernel."""
    lam = c0 / f0
    dx = lam / ppw
    fwhm_lat, fwhm_ax = fwhm_focal_extents(f0, c0, roc, diameter)

    # Axial extent: bowl apex margin (5 vox) + roc + half-axial-FWHM ×
    # safety factor of 6 (post-focal). Lateral extent: 6× FWHM_lat each
    # side of the axis (covers main lobe + sidelobes for a typical
    # 50 mm/120 mm bowl). Round up to nearest 8 voxels for FFT
    # friendliness, and add 2 × pml margins on every face.
    apex_margin_vox = 5
    axial_post_focal_m = 3.0 * fwhm_ax
    nx_core = apex_margin_vox + int(np.ceil((roc + axial_post_focal_m) / dx))
    lateral_half_m = 6.0 * fwhm_lat
    ny_core = 2 * int(np.ceil(lateral_half_m / dx)) + 1
    nz_core = ny_core

    def round_up_8(n: int) -> int:
        return ((n + 7) // 8) * 8

    nx = round_up_8(nx_core + 2 * pml)
    ny = round_up_8(ny_core + 2 * pml)
    nz = round_up_8(nz_core + 2 * pml)
    grid = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx)

    apex_x_idx = pml + apex_margin_vox
    focus_x_idx = apex_x_idx + int(round(roc / dx))
    focus_y_idx = ny // 2
    focus_z_idx = nz // 2

    info = {
        "dx": dx,
        "shape": (nx, ny, nz),
        "apex_x_idx": apex_x_idx,
        "focus_idx": (focus_x_idx, focus_y_idx, focus_z_idx),
        "fwhm_lat_m": fwhm_lat,
        "fwhm_ax_m": fwhm_ax,
        "pml": pml,
    }
    return grid, info


def make_signal(nt: int, dt: float, f0: float, source_pa: float, n_cycles: int,
                ppp: int) -> np.ndarray:
    """Hann-tapered CW tone-burst signal, length `nt`. The tapered pulse
    suppresses spectral leakage so the realised focal pressure tracks
    the analytic O'Neil/Penttinen prediction."""
    t = np.arange(nt) * dt
    envelope = np.zeros(nt)
    pulse_steps = n_cycles * ppp
    envelope[:pulse_steps] = 1.0
    taper = max(int(0.5 * ppp), 1)
    envelope[:taper] = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper) / taper))
    envelope[pulse_steps - taper:pulse_steps] = envelope[:taper][::-1]
    return source_pa * envelope * np.sin(2.0 * np.pi * f0 * t)


def make_kwave_bowl(grid: kw.Grid, info: dict, roc: float, diameter: float) -> kw.KWaveArray:
    """Bowl with centre of curvature at `roc` past the apex along +x —
    KWaveArray rasterises the spherical-cap surface concave toward the
    focus and assigns phase delays so all bowl voxels arrive in-phase
    at the geometric focal point."""
    dx = info["dx"]
    apex_x = info["apex_x_idx"] * dx
    coc_x = apex_x + roc
    cy = info["focus_idx"][1] * dx
    cz = info["focus_idx"][2] * dx
    karray = kw.KWaveArray()
    karray.add_bowl_element((coc_x, cy, cz), roc, diameter)
    return karray


def build_focused_bowl_source(grid: kw.Grid, info: dict, f0: float, source_pa: float,
                              roc: float, diameter: float, n_cycles: int) -> tuple[kw.Source, int, float]:
    """Returns (source, n_time_steps, dt) for the full kernel run."""
    c0 = 1500.0
    ppp = 12
    dt = 1.0 / (ppp * f0)
    pulse_steps = n_cycles * ppp
    post_steps = int(np.ceil(4.0 * roc / c0 / dt))
    nt = pulse_steps + post_steps

    signal = make_signal(nt, dt, f0, source_pa, n_cycles, ppp)
    karray = make_kwave_bowl(grid, info, roc, diameter)
    source = kw.Source.from_kwave_array(karray, signal.astype(np.float64), f0,
                                         mode="additive")
    return source, nt, dt


def calibrate_source_pa(grid: kw.Grid, info: dict, medium: kw.Medium, f0: float,
                        roc: float, diameter: float, target_pnp: float,
                        c0: float, pml: int, n_cycles: int = 4) -> float:
    """Run a low-amplitude probe pulse, measure the realised peak
    rarefactional pressure at the focal voxel, and rescale the drive
    amplitude so the full sim hits `target_pnp`. Linear-regime
    assumption: focal pressure scales linearly with source amplitude
    (water-equivalent medium, B/A=0).

    Probe matches the full-sim `n_cycles` so the focal ring-up reaches
    the same steady-state peak — calibration agrees with the realised
    full-sim pnp to within a few %. The probe still skips the
    post-pulse decay (saving ~50 % of the full sim's steps).
    """
    ppp = 12
    dt = 1.0 / (ppp * f0)
    n_cycles_probe = n_cycles
    propagation_steps = int(np.ceil(roc / c0 / dt))
    nt = propagation_steps + n_cycles_probe * ppp + ppp  # +1 cycle margin

    probe_pa = 1.0e3  # 1 kPa drive — comfortably linear
    signal = make_signal(nt, dt, f0, probe_pa, n_cycles_probe, ppp)
    karray = make_kwave_bowl(grid, info, roc, diameter)
    source = kw.Source.from_kwave_array(karray, signal.astype(np.float64), f0,
                                         mode="additive")

    # Single-voxel sensor at the focal point — recording stats on the
    # full grid would defeat the purpose of a fast probe.
    nx, ny, nz = info["shape"]
    sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
    sensor_mask[info["focus_idx"]] = True
    sensor = kw.Sensor.from_mask(sensor_mask)
    sensor.set_record(["p_min"])

    sim = kw.Simulation(grid, medium, source, sensor,
                         solver=kw.SolverType.PSTD, pml_size=pml)
    print(f"[cal ] probe: {nt} steps @ probe_pa={probe_pa/1e3:.1f} kPa")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    print(f"[cal ] probe ran in {time.perf_counter()-t0:.1f} s")

    p_min_focal = float(np.asarray(result.p_min)[0])
    realised_probe_pnp = -p_min_focal
    if realised_probe_pnp <= 0:
        raise SystemExit(f"calibration probe saw no rarefaction at focus "
                         f"(p_min[focus]={p_min_focal:.3e} Pa). "
                         f"Check bowl geometry / focus index.")
    scale = target_pnp / realised_probe_pnp
    calibrated_pa = probe_pa * scale
    print(f"[cal ] focal probe peak rarefactional = {realised_probe_pnp/1e3:.2f} kPa; "
          f"scale = {scale:.2f}× -> source_pa = {calibrated_pa/1e6:.3f} MPa")
    return calibrated_pa


def run_kernel(args: argparse.Namespace) -> dict:
    c0 = args.c0
    rho0 = args.rho0
    f0 = args.f0
    pnp = args.pnp

    grid, info = build_grid(f0, c0, args.roc, args.diameter, args.ppw, args.pml)
    nx, ny, nz = info["shape"]
    print(f"[kernel] grid {nx}×{ny}×{nz}, dx={info['dx']*1e3:.3f} mm, "
          f"focus@{info['focus_idx']}")

    medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

    lam = c0 / f0
    fnum = args.roc / args.diameter
    if args.source_pa is not None:
        source_pa = args.source_pa
        print(f"[kernel] f# = {fnum:.2f}, source_pa override = {source_pa/1e6:.3f} MPa")
    elif args.no_calibrate:
        geometric_gain = (np.pi * args.diameter / 2.0) / (lam * fnum)
        source_pa = pnp / geometric_gain
        print(f"[kernel] f# = {fnum:.2f}, analytic gain {geometric_gain:.1f} "
              f"(uncalibrated); source_pa = {source_pa/1e6:.3f} MPa")
    else:
        source_pa = calibrate_source_pa(grid, info, medium, f0, args.roc,
                                         args.diameter, pnp, c0, info["pml"],
                                         n_cycles=args.n_cycles)

    source, nt, dt = build_focused_bowl_source(
        grid, info, f0, source_pa, args.roc, args.diameter, args.n_cycles)
    print(f"[kernel] n_cycles={args.n_cycles}, nt={nt}, dt={dt*1e9:.2f} ns, "
          f"total_t={nt*dt*1e6:.2f} us")

    # Whole-grid sensor with pressure statistics
    sensor_mask = np.ones((nx, ny, nz), dtype=bool)
    sensor = kw.Sensor.from_mask(sensor_mask)
    sensor.set_record(["p_max", "p_min", "p_rms", "p_final"])

    sim = kw.Simulation(grid, medium, source, sensor,
                         solver=kw.SolverType.PSTD, pml_size=info["pml"])

    print("[kernel] running PSTD ...")
    t_start = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    elapsed = time.perf_counter() - t_start
    print(f"[kernel] elapsed: {elapsed:.1f} s "
          f"({nt} steps × {nx*ny*nz/1e6:.1f}M voxels = {nt*nx*ny*nz/1e9*elapsed:.2f} ns/vox·step)")

    if result.p_min_field is None:
        raise SystemExit("No full-grid pressure statistics returned — check pyo3 binding.")

    p_min = np.asarray(result.p_min_field)
    p_max = np.asarray(result.p_max_field)
    p_rms = np.asarray(result.p_rms_field)
    p_final = np.asarray(result.p_final_field)

    # Sanity: peak rarefactional at focal voxel
    fx, fy, fz = info["focus_idx"]
    realised_pnp = -p_min[fx, fy, fz]
    realised_pmax = p_max[fx, fy, fz]
    print(f"[kernel] focal voxel: realised p_min = -{realised_pnp/1e6:.2f} MPa "
          f"(target {pnp/1e6:.2f} MPa), p_max = +{realised_pmax/1e6:.2f} MPa")
    print(f"[kernel] peak rarefactional anywhere: {-p_min.min()/1e6:.2f} MPa")

    return {
        "p_min": p_min,
        "p_max": p_max,
        "p_rms": p_rms,
        "p_final": p_final,
        "dx": info["dx"],
        "shape": info["shape"],
        "apex_x_idx": info["apex_x_idx"],
        "focus_idx": np.asarray(info["focus_idx"], dtype=np.int64),
        "fwhm_lat_m": info["fwhm_lat_m"],
        "fwhm_ax_m": info["fwhm_ax_m"],
        "f0": f0,
        "pnp_target": pnp,
        "pnp_realised": realised_pnp,
        "source_pa": source_pa,
        "roc": args.roc,
        "diameter": args.diameter,
        "c0": c0,
        "rho0": rho0,
        "n_cycles": args.n_cycles,
        "nt": nt,
        "dt": dt,
        "elapsed_s": elapsed,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f0", type=float, default=1.0e6, help="centre frequency (Hz)")
    p.add_argument("--pnp", type=float, default=30.0e6, help="target peak rarefactional pressure (Pa)")
    p.add_argument("--source-pa", type=float, default=None,
                   help="override drive amplitude (Pa); default = pnp / geometric_gain")
    p.add_argument("--roc", type=float, default=120.0e-3, help="bowl radius of curvature (m)")
    p.add_argument("--diameter", type=float, default=100.0e-3, help="bowl aperture diameter (m)")
    p.add_argument("--c0", type=float, default=1500.0, help="sound speed (m/s)")
    p.add_argument("--rho0", type=float, default=1000.0, help="density (kg/m^3)")
    p.add_argument("--ppw", type=int, default=4, help="grid points per wavelength")
    p.add_argument("--pml", type=int, default=10, help="PML thickness (voxels)")
    p.add_argument("--n-cycles", type=int, default=4, help="number of CW cycles in the pulse")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the probe-pass calibration (use analytic O'Neil gain)")
    p.add_argument("--out", type=str, default=None, help="output .npz path (default in data/kernels/)")
    p.add_argument("--sweep", action="store_true",
                   help="generate a parameter-space sweep instead of a single kernel "
                        "(uses --f0-list and --pnp-list)")
    p.add_argument("--f0-list", type=str, default="0.5e6,1e6",
                   help="comma-separated f0 values (Hz) for sweep mode")
    p.add_argument("--pnp-list", type=str, default="15e6,30e6",
                   help="comma-separated pnp values (Pa) for sweep mode")
    return p.parse_args(argv)


def kernel_filename(f0: float, pnp: float, roc: float, diameter: float) -> str:
    return (f"kernel_{f0/1e6:.2f}MHz_{pnp/1e6:.0f}MPa_"
            f"{roc*1e3:.0f}roc_{diameter*1e3:.0f}diam.npz")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.sweep:
        kernel = run_kernel(args)
        out = (args.out if args.out is not None
               else os.path.join(KERNEL_DIR,
                                  kernel_filename(args.f0, args.pnp, args.roc, args.diameter)))
        np.savez_compressed(out, **kernel)
        print(f"[kernel] saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
        return 0

    # Sweep mode: cross-product over (f0, pnp); fixed roc/diameter
    f0_values = [float(x) for x in args.f0_list.split(",")]
    pnp_values = [float(x) for x in args.pnp_list.split(",")]
    sweep_t0 = time.perf_counter()
    n_done = 0
    n_total = len(f0_values) * len(pnp_values)
    for f0 in f0_values:
        for pnp in pnp_values:
            n_done += 1
            print(f"\n[sweep] {n_done}/{n_total}: f0={f0/1e6:.2f} MHz, pnp={pnp/1e6:.1f} MPa")
            args.f0 = f0
            args.pnp = pnp
            args.source_pa = None  # always re-calibrate
            kernel = run_kernel(args)
            out = os.path.join(KERNEL_DIR, kernel_filename(f0, pnp, args.roc, args.diameter))
            np.savez_compressed(out, **kernel)
            print(f"[sweep] saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"\n[sweep] {n_total} kernels generated in {(time.perf_counter()-sweep_t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
