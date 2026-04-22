#!/usr/bin/env python3
"""Per-angle trace: find exactly which angles and which signal region has the 2x excess."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst
from kwave.data import Vector
from us_bmode_phased_array_compare import (
    PML_SIZE, GRID_SIZE_POINTS, GRID_SPACING_METERS,
    C0, RHO0, ALPHA_COEFF, ALPHA_POWER, BON_A, SOURCE_STRENGTH,
    TONE_BURST_FREQ, TONE_BURST_CYCLES, COMPRESSION_RATIO,
    STEERING_ANGLES_QUICK, build_reference_objects,
)

OUTPUT = Path(__file__).parent / "output"


def main():
    kw_path  = OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz"
    pkw_path = OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz"
    kw_cache  = np.load(kw_path)
    pkw_cache = np.load(pkw_path)
    kw_sl  = kw_cache['scan_lines']   # [n_angles, nt]
    pkw_sl = pkw_cache['scan_lines']
    steering_angles = list(STEERING_ANGLES_QUICK)

    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(20260401)
    not_transducer.steering_angle = float(steering_angles[-1])

    t0_offset = int(round(len(not_transducer.input_signal.squeeze()) / 2) +
                    (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
    print(f"t0_offset = {t0_offset} samples  (dt={float(kgrid.dt)*1e6:.4f} us  =>  t0={t0_offset*float(kgrid.dt)*1e6:.1f} us)")
    print(f"nt = {kw_sl.shape[1]}")
    print()

    print(f"{'Angle':>6s}  {'Full kw_rms':>12s}  {'Full pkw_rms':>13s}  {'Full ratio':>10s}  "
          f"{'Trim kw_rms':>12s}  {'Trim pkw_rms':>13s}  {'Trim ratio':>10s}")
    for i, ang in enumerate(steering_angles):
        kw_full  = kw_sl[i]
        pkw_full = pkw_sl[i]
        kw_rms_full  = float(np.sqrt(np.mean(kw_full**2)))
        pkw_rms_full = float(np.sqrt(np.mean(pkw_full**2)))

        kw_trim  = kw_full[t0_offset:]
        pkw_trim = pkw_full[t0_offset:]
        kw_rms_trim  = float(np.sqrt(np.mean(kw_trim**2)))
        pkw_rms_trim = float(np.sqrt(np.mean(pkw_trim**2)))

        ratio_full = pkw_rms_full / (kw_rms_full + 1e-30)
        ratio_trim = pkw_rms_trim / (kw_rms_trim + 1e-30)

        print(f"{ang:>+6d}  {kw_rms_full:>12.4g}  {pkw_rms_full:>13.4g}  {ratio_full:>10.4f}  "
              f"{kw_rms_trim:>12.4g}  {pkw_rms_trim:>13.4g}  {ratio_trim:>10.4f}")

    # Diagnose: where does kw energy go in t < t0_offset?
    print("\n--- Leading segment (t < t0_offset = first {} samples) ---".format(t0_offset))
    print(f"{'Angle':>6s}  {'Lead kw_rms':>12s}  {'Lead pkw_rms':>13s}  {'Lead ratio':>10s}")
    for i, ang in enumerate(steering_angles):
        kw_lead  = kw_sl[i, :t0_offset]
        pkw_lead = pkw_sl[i, :t0_offset]
        kw_rms  = float(np.sqrt(np.mean(kw_lead**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_lead**2)))
        ratio = pkw_rms / (kw_rms + 1e-30)
        print(f"{ang:>+6d}  {kw_rms:>12.4g}  {pkw_rms:>13.4g}  {ratio:>10.4f}")

    # Show the scan_lines shape around t0_offset
    print(f"\n--- kw  scan_line[4] (0 deg) around t0={t0_offset}: ---")
    sl0_kw  = kw_sl[4]   # 0 deg is index 4 in quick set (-32,-24,...,+32)
    sl0_pkw = pkw_sl[4]
    for t in [0, t0_offset//2, t0_offset-5, t0_offset, t0_offset+5, t0_offset+50, t0_offset+200]:
        if 0 <= t < len(sl0_kw):
            print(f"  t={t:4d}:  kw={sl0_kw[t]:+.4g}  pkw={sl0_pkw[t]:+.4g}  ratio={sl0_pkw[t]/(sl0_kw[t]+1e-30):.3g}")

    # Key diagnostic: what is the appended_zeros and beamforming_delays_offset?
    print(f"\nnot_transducer.appended_zeros = {not_transducer.appended_zeros}")
    print(f"not_transducer.beamforming_delays_offset = {not_transducer.beamforming_delays_offset}")
    print(f"len(not_transducer.input_signal) = {len(not_transducer.input_signal.squeeze())}")
    print(f"Reconstructed: t0_offset = round({len(not_transducer.input_signal.squeeze())/2}) + "
          f"({not_transducer.appended_zeros} - {not_transducer.beamforming_delays_offset}) "
          f"= {t0_offset}")


if __name__ == "__main__":
    main()
