#!/usr/bin/env python3
"""
Print beamforming delays, appended_zeros, and beamforming_delays_offset
for 0 deg and 32 deg steering angles, to check if t0_offset is angle-dependent.

Also directly visualize kw vs pkw scan_line traces to see what's happening.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import numpy as np
from us_bmode_phased_array_compare import (
    STEERING_ANGLES_QUICK, build_reference_objects,
)

OUTPUT = Path(__file__).parent / "output"


def print_nt_info(not_transducer, label):
    print(f"\n--- {label} ---")
    print(f"  steering_angle = {not_transducer.steering_angle}")
    print(f"  appended_zeros = {not_transducer.appended_zeros}")
    print(f"  beamforming_delays_offset = {not_transducer.beamforming_delays_offset}")
    print(f"  input_signal length = {len(not_transducer.input_signal.squeeze())}")
    t0 = int(round(len(not_transducer.input_signal.squeeze()) / 2) +
             (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
    print(f"  t0_offset = {t0}")

    delays = not_transducer.beamforming_delays
    print(f"  beamforming_delays: shape={np.asarray(delays).shape}  "
          f"min={np.min(delays)}  max={np.max(delays)}  "
          f"mean={np.mean(delays):.1f}")
    # Which direction does -delays shift data?
    neg_delays = -np.asarray(delays)
    n_pos = int(np.sum(neg_delays > 0))
    n_neg = int(np.sum(neg_delays < 0))
    n_zero = int(np.sum(neg_delays == 0))
    print(f"  scan_line -delays: pos={n_pos}  neg={n_neg}  zero={n_zero}  "
          f"min={int(np.min(neg_delays))}  max={int(np.max(neg_delays))}")
    return t0


def main():
    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(20260401)

    # Check delays at 0 deg
    not_transducer.steering_angle = 0.0
    # Force recomputation of delay mask (calls underlying MATLAB code)
    _ = not_transducer.delay_mask()
    t0_0deg = print_nt_info(not_transducer, "0 deg")

    # Check delays at 32 deg (last angle in quick set)
    not_transducer.steering_angle = 32.0
    _ = not_transducer.delay_mask()
    t0_32deg = print_nt_info(not_transducer, "32 deg")

    print(f"\nt0_offset at 0 deg = {t0_0deg}")
    print(f"t0_offset at 32 deg = {t0_32deg}")
    print(f"Difference = {t0_32deg - t0_0deg} samples")

    # Now load cached scan_lines and compare trimming at 0deg vs 32deg offset
    kw_cache  = np.load(OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz")
    pkw_cache = np.load(OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz")
    kw_sl  = kw_cache['scan_lines']   # [9, nt]
    pkw_sl = pkw_cache['scan_lines']
    angles = list(STEERING_ANGLES_QUICK)
    ang0_idx = angles.index(0)

    kw_sl0  = kw_sl[ang0_idx]
    pkw_sl0 = pkw_sl[ang0_idx]

    for t0, label in [(t0_0deg, "t0 from 0deg"), (t0_32deg, "t0 from 32deg")]:
        kw_trim  = kw_sl0[t0:]
        pkw_trim = pkw_sl0[t0:]
        kw_rms  = float(np.sqrt(np.mean(kw_trim**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_trim**2)))
        r = float(np.corrcoef(kw_trim, pkw_trim)[0, 1])
        print(f"\n[0 deg scan_line trimmed at {label} = {t0}]  "
              f"rms_ratio={pkw_rms/kw_rms:.4f}  r={r:.4f}  kw_rms={kw_rms:.4g}  pkw_rms={pkw_rms:.4g}")

    # Also load probe data for same analysis
    dt_npz = np.load(OUTPUT / "us_bmode_phased_array_drive_trace.npz")
    kw_comb  = dt_npz["kw_combined"]   # [64, 1658]
    pkw_comb = dt_npz["pkw_combined"]  # [64, 1658]

    # Apply scan_line with 0 deg steering
    not_transducer.steering_angle = 0.0
    _ = not_transducer.delay_mask()
    kw_sl_probe  = np.asarray(not_transducer.scan_line(kw_comb.copy()),  dtype=float)
    pkw_sl_probe = np.asarray(not_transducer.scan_line(pkw_comb.copy()), dtype=float)
    print(f"\n--- Probe combined -> scan_line(0deg) ---")
    for t0, label in [(t0_0deg, "t0 from 0deg"), (t0_32deg, "t0 from 32deg")]:
        kw_trim  = kw_sl_probe[t0:]
        pkw_trim = pkw_sl_probe[t0:]
        kw_rms  = float(np.sqrt(np.mean(kw_trim**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_trim**2)))
        r = float(np.corrcoef(kw_trim, pkw_trim)[0, 1])
        print(f"  [{label} = {t0}]  rms_ratio={pkw_rms/kw_rms:.4f}  r={r:.4f}")

    # Print the actual scan_line values at both trim points to understand the signal
    print(f"\n--- kw probe scan_line vs pkw probe scan_line near t0={t0_0deg} (0deg trim) ---")
    for t in range(t0_0deg-10, t0_0deg+30, 2):
        if 0 <= t < len(kw_sl_probe):
            print(f"  t={t:4d}:  kw={kw_sl_probe[t]:+.4g}  pkw={pkw_sl_probe[t]:+.4g}")


if __name__ == "__main__":
    main()
