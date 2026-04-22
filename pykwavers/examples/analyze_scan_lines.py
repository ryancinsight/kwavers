#!/usr/bin/env python3
"""Diagnostic: analyze cached scan_lines to find where rms_ratio=2.07 originates."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "output"

# ----------- Load cached scan lines -----------
kw_path  = OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz"
pkw_path = OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz"

kw  = np.load(kw_path)
pkw = np.load(pkw_path)

kw_sl  = kw['scan_lines']    # [n_angles, nt]
pkw_sl = pkw['scan_lines']

print(f"kw  scan_lines shape: {kw_sl.shape}")
print(f"pkw scan_lines shape: {pkw_sl.shape}")
print()

angles = list(range(-32, 33, 8))  # quick set: 9 angles
for i in range(kw_sl.shape[0]):
    kw_rms  = float(np.sqrt(np.mean(kw_sl[i]**2)))
    pkw_rms = float(np.sqrt(np.mean(pkw_sl[i]**2)))
    ratio   = pkw_rms / kw_rms if kw_rms > 0 else float('nan')
    kw_pk   = float(np.max(np.abs(kw_sl[i])))
    pkw_pk  = float(np.max(np.abs(pkw_sl[i])))
    pk_ratio = pkw_pk / kw_pk if kw_pk > 0 else float('nan')
    print(f"  angle {angles[i]:+3d}:  rms_ratio={ratio:.4f}  peak_ratio={pk_ratio:.4f}  "
          f"kw_rms={kw_rms:.3g}  pkw_rms={pkw_rms:.3g}")

kw_all  = float(np.sqrt(np.mean(kw_sl**2)))
pkw_all = float(np.sqrt(np.mean(pkw_sl**2)))
print(f"\nOverall scan_lines rms_ratio: {pkw_all/kw_all:.4f}")

# ----------- Drive trace NPZ -----------
dt_path = OUTPUT / "us_bmode_phased_array_drive_trace.npz"
if dt_path.exists():
    dt = np.load(dt_path)
    print(f"\ndrive_trace.npz: angle={float(dt['angle'])}, gpu={bool(dt['pykwavers_gpu'])}")
    kw_comb  = dt['kw_combined']   # [n_elements, nt]
    pkw_comb = dt['pkw_combined']
    print(f"combined shape: kw={kw_comb.shape}, pkw={pkw_comb.shape}")

    rms_ratios = []
    for el in range(kw_comb.shape[0]):
        kw_rms  = float(np.sqrt(np.mean(kw_comb[el]**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_comb[el]**2)))
        ratio = pkw_rms / kw_rms if kw_rms > 1e-30 else float('nan')
        rms_ratios.append(ratio)
    rms_ratios = np.array(rms_ratios)
    print(f"All-element rms_ratio: mean={np.nanmean(rms_ratios):.4f}  "
          f"max={np.nanmax(rms_ratios):.4f}  min={np.nanmin(rms_ratios):.4f}  "
          f"std={np.nanstd(rms_ratios):.4f}")
    print(f"Elements > 1.5: {int(np.sum(rms_ratios > 1.5))}/{len(rms_ratios)}")
    print(f"Elements > 2.0: {int(np.sum(rms_ratios > 2.0))}/{len(rms_ratios)}")

    # Show worst 5 and best 5 elements
    sorted_idx = np.argsort(rms_ratios)[::-1]
    print("Top 10 elements by rms_ratio:")
    for idx in sorted_idx[:10]:
        print(f"  el {idx:3d}: rms_ratio={rms_ratios[idx]:.4f}")

    # Scan line from combined
    from kwave.ktransducer import NotATransducer
    # We can't re-run scan_line without not_transducer, but we can compare
    # raw voxel-level data
    kw_raw  = dt['kw_raw']   # [n_sensors, nt]
    pkw_raw = dt['pkw_raw']
    print(f"\nRaw voxel sensor data: kw={kw_raw.shape}, pkw={pkw_raw.shape}")
    kw_raw_rms  = float(np.sqrt(np.mean(kw_raw**2)))
    pkw_raw_rms = float(np.sqrt(np.mean(pkw_raw**2)))
    print(f"Raw voxel rms_ratio: {pkw_raw_rms/kw_raw_rms:.4f}")
else:
    print("(drive_trace.npz not found)")
