#!/usr/bin/env python3
"""Minimal small-grid reproducer isolating the pykwavers vs k-wave-python
amplitude mismatch observed in ``us_bmode_phased_array_compare.py``.

Strategy
--------
The full parity example runs on a 256x256x128 grid for 1658 steps, costing
~90 minutes per k-wave reference sweep on this host. That is too slow to
iterate on hypotheses. This script shrinks the problem to a 64x64x32 grid
driven by a single-voxel velocity source with identical source scaling
(2 * c0 * dt / dx) on both legs, and captures a pressure trace at a probe
point ~15dx downstream so we can compute peak- and rms-ratio directly.

If the ratio reproduces the observed 1.44 (harmonic ~= fund^2) signature,
the bug is in the elementary source injection / PSTD update. If the ratio
is ~1.0 here, the bug is specific to phased-array apodization / delay /
combine_sensor_data post-processing.

This script writes ``output/source_scaling_minimal_repro.png`` and an NPZ
dump to ``output/source_scaling_minimal_repro.npz``. It is diagnostic
code — not part of the parity gate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import DEFAULT_OUTPUT_DIR, bootstrap_example_paths, compute_trace_metrics

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--pml", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--freq", type=float, default=1.0e6)
    ap.add_argument("--source-strength-pa", type=float, default=1.0e6,
                    help="Tone-burst pressure amplitude [Pa]; same as us_bmode example.")
    ap.add_argument("--pykwavers-gpu", action="store_true",
                    help="Use GpuPstdSession instead of CPU PSTD.")
    ap.add_argument("--source-mode", type=str, default="additive",
                    choices=("additive", "additive_no_correction", "dirichlet"))
    ap.add_argument("--nonlinear", action="store_true",
                    help="Enable B/A nonlinearity (off by default for a clean amplitude test).")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Absorption coefficient dB/(MHz^y cm). Default 0 (lossless).")
    args = ap.parse_args()

    pml = Vector([args.pml, args.pml, args.pml])
    active = Vector([args.nx, args.ny, args.nz]) - 2 * pml
    dx = 50e-3 / 256  # match us_bmode example spacing
    spacing = Vector([dx, dx, dx])

    c0 = 1540.0
    rho0 = 1000.0

    kgrid = kWaveGrid(active, spacing)
    t_end = (active.x * dx) * 2.0 / c0
    kgrid.makeTime(c0, t_end=t_end)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    src_ix, src_iy, src_iz = 2, active.y // 2, active.z // 2  # left face, center
    probe_ix, probe_iy, probe_iz = src_ix + 15, active.y // 2, active.z // 2

    signal_amp = args.source_strength_pa / (c0 * rho0)
    tb = tone_burst(1.0 / dt, args.freq, args.cycles).squeeze()
    input_signal = signal_amp * tb.astype(np.float64)

    # ── k-wave-python leg ─────────────────────────────────────────────────
    medium = kWaveMedium(sound_speed=c0, density=rho0,
                         alpha_coeff=args.alpha, alpha_power=1.5,
                         BonA=6.0 if args.nonlinear else None)

    source = kSource()
    u_mask = np.zeros((active.x, active.y, active.z), dtype=bool)
    u_mask[src_ix, src_iy, src_iz] = True
    source.u_mask = u_mask
    source.ux = input_signal.reshape(1, -1).astype(np.float64)
    source.u_mode = args.source_mode

    sensor = kSensor()
    s_mask = np.zeros_like(u_mask)
    s_mask[probe_ix, probe_iy, probe_iz] = True
    sensor.mask = s_mask
    sensor.record = ["p"]

    print(f"[kwave] nx={active.x} ny={active.y} nz={active.z} nt={nt} dt={dt:.3e} dx={dx:.3e}")
    print(f"[kwave] src=({src_ix},{src_iy},{src_iz}) probe=({probe_ix},{probe_iy},{probe_iz})")
    print(f"[kwave] input_signal peak={np.max(np.abs(input_signal)):.4g} m/s")
    kw_out = kspaceFirstOrder3D(
        medium=deepcopy(medium),
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=SimulationOptions(pml_inside=False, pml_size=pml, data_cast="single", data_recast=True, save_to_disk=True, save_to_disk_exit=False),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
    )
    p_kw = np.asarray(kw_out["p"]).reshape(-1)

    # ── pykwavers leg ─────────────────────────────────────────────────────
    fnx, fny, fnz = args.nx, args.ny, args.nz
    px, py, pz = int(pml.x), int(pml.y), int(pml.z)
    grid = pkw.Grid(fnx, fny, fnz, dx, dx, dx)
    ss_full = np.full((fnx, fny, fnz), c0, dtype=np.float64)
    rho_full = np.full((fnx, fny, fnz), rho0, dtype=np.float64)
    alpha_full = np.full((fnx, fny, fnz), args.alpha, dtype=np.float64)
    bona_full = np.full((fnx, fny, fnz), 6.0 if args.nonlinear else 0.0, dtype=np.float64)

    mask_full = np.zeros((fnx, fny, fnz), dtype=np.float64)
    mask_full[src_ix + px, src_iy + py, src_iz + pz] = 1.0

    # Match the us_bmode script: pre-scale by 2*c*dt/dx (emulates
    # scale_transducer_source, which is NOT applied to kSource.ux — but
    # scale_velocity_source IS applied with the same factor).
    transducer_scale = 2.0 * c0 * dt / dx
    ux_signals = (input_signal * transducer_scale).astype(np.float64).reshape(1, -1)

    print(f"[pkw]  transducer_scale=2c dt/dx={transducer_scale:.4g}")
    print(f"[pkw]  ux_signals peak={np.max(np.abs(ux_signals)):.4g}")

    if args.pykwavers_gpu:
        session = pkw.GpuPstdSession(
            grid, ss_full, rho_full,
            dt=dt, time_steps=nt,
            absorption=alpha_full, nonlinearity=bona_full,
            pml_size_xyz=(px, py, pz), alpha_power=1.5,
        )
        session.set_source_sensor_mask(mask_full)
        session.set_velocity_signals(ux_signals)
        sensor_data = np.asarray(session.run_scan_line_cached())
    else:
        medium_pkw = pkw.Medium(sound_speed=ss_full, density=rho_full,
                                absorption=alpha_full, nonlinearity=bona_full)
        src_pkw = pkw.Source.from_velocity_mask_2d(mask_full, ux=ux_signals, mode=args.source_mode)
        probe_mask = np.zeros_like(mask_full, dtype=bool)
        probe_mask[probe_ix + px, probe_iy + py, probe_iz + pz] = True
        sensor_pkw = pkw.Sensor.from_mask(probe_mask)
        sim = pkw.Simulation(grid, medium_pkw, src_pkw, sensor_pkw, solver=pkw.SolverType.PSTD)
        sim.set_pml_size_xyz(px, py, pz)
        sim.set_nonlinear(args.nonlinear)
        result = sim.run(nt, dt=dt)
        sensor_data = np.asarray(result.sensor_data)

    # For the GPU path, the sensor mask was the (single-voxel) source+sensor mask;
    # extract a pressure trace at the probe explicitly if needed. For CPU we used a
    # dedicated probe mask. For simplicity we only support CPU's probe mask; for GPU
    # the single-voxel probe = source location so we read that voxel (not what we want).
    if args.pykwavers_gpu:
        print("[pkw] WARN: GPU probe coincides with source voxel — rerun with CPU for clean trace.")
        p_pkw = sensor_data.reshape(-1)
    else:
        p_pkw = sensor_data.reshape(-1)

    # Align lengths
    n = min(p_kw.size, p_pkw.size)
    p_kw = p_kw[:n].astype(np.float64)
    p_pkw = p_pkw[:n].astype(np.float64)
    t = np.arange(n) * dt

    m = compute_trace_metrics(p_kw, p_pkw)
    print(
        f"[probe] pearson_r={m['pearson_r']:.4f} rms_ratio={m['rms_ratio']:.4f} "
        f"peak_ratio={m['peak_ratio']:.4f} kw_peak={m['reference_peak']:.3g} "
        f"pkw_peak={m['candidate_peak']:.3g}"
    )

    npz = DEFAULT_OUTPUT_DIR / "source_scaling_minimal_repro.npz"
    np.savez(npz, t=t, p_kw=p_kw, p_pkw=p_pkw, dt=dt, dx=dx, c0=c0,
             transducer_scale=transducer_scale, source_mode=args.source_mode,
             pykwavers_gpu=args.pykwavers_gpu, nonlinear=args.nonlinear)
    print(f"[saved] {npz}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t * 1e6, p_kw, label=f"k-wave-python (peak={m['reference_peak']:.3g} Pa)", color="C0", lw=1.5)
    ax.plot(t * 1e6, p_pkw, label=f"pykwavers (peak={m['candidate_peak']:.3g} Pa)", color="C3", lw=1.1, ls="--")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_title(
        f"Minimal source-scale repro — probe 15dx downstream "
        f"(r={m['pearson_r']:.3f}, rms_ratio={m['rms_ratio']:.3f}, peak_ratio={m['peak_ratio']:.3f}, "
        f"mode={args.source_mode}, {'GPU' if args.pykwavers_gpu else 'CPU'})"
    )
    ax.legend(loc="best")
    ax.grid(True, ls=":")
    fig.tight_layout()
    png = DEFAULT_OUTPUT_DIR / "source_scaling_minimal_repro.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}")


if __name__ == "__main__":
    main()
