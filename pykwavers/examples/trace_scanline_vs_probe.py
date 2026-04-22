#!/usr/bin/env python3
"""
Compare scan_line(combined) from the debug probe's saved combined arrays
with the cached parity scan_lines[4] (0 deg angle).

If they match, the debug probe and parity run have consistent physics;
the discrepancy is in the scan_line call itself.
If they don't match, the parity run used different raw data (different code).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths, compute_trace_metrics
bootstrap_example_paths()

import numpy as np
from us_bmode_phased_array_compare import (
    STEERING_ANGLES_QUICK, build_reference_objects,
)

OUTPUT = Path(__file__).parent / "output"


def main():
    # Load debug probe combined arrays (angle=0 deg)
    dt_npz = np.load(OUTPUT / "us_bmode_phased_array_drive_trace.npz")
    kw_comb  = dt_npz["kw_combined"]   # [64, nt]
    pkw_comb = dt_npz["pkw_combined"]  # [64, nt]
    angle    = float(dt_npz["angle"])
    print(f"Debug probe angle={angle}  kw_comb={kw_comb.shape}  pkw_comb={pkw_comb.shape}")

    # Load parity cached scan_lines
    kw_cache  = np.load(OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz")
    pkw_cache = np.load(OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz")
    kw_sl_all  = kw_cache["scan_lines"]   # [9, nt]
    pkw_sl_all = pkw_cache["scan_lines"]  # [9, nt]
    angles = list(STEERING_ANGLES_QUICK)
    ang0_idx = angles.index(0)
    kw_sl0  = kw_sl_all[ang0_idx]   # scan_line at 0 deg
    pkw_sl0 = pkw_sl_all[ang0_idx]
    print(f"Parity scan_line at 0 deg: kw_sl0={kw_sl0.shape}  pkw_sl0={pkw_sl0.shape}")

    # Rebuild not_transducer so we can call scan_line()
    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(20260401)
    not_transducer.steering_angle = 0.0  # debug probe angle

    # Apply scan_line to debug probe combined arrays
    kw_sl_probe  = np.asarray(not_transducer.scan_line(kw_comb),  dtype=float)
    pkw_sl_probe = np.asarray(not_transducer.scan_line(pkw_comb), dtype=float)
    print(f"\nScan_line from probe combined arrays: kw_sl_probe={kw_sl_probe.shape}")

    # Compare probe scan_line vs parity scan_line
    print("\n--- Debug probe combined -> scan_line vs parity cache ---")
    m_kw  = compute_trace_metrics(kw_sl0,  kw_sl_probe)
    m_pkw = compute_trace_metrics(pkw_sl0, pkw_sl_probe)
    print(f"  kw  parity vs kw  probe:  r={m_kw['pearson_r']:.4f}  rms_ratio={m_kw['rms_ratio']:.4f}  peak_ratio={m_kw['peak_ratio']:.4f}")
    print(f"  pkw parity vs pkw probe:  r={m_pkw['pearson_r']:.4f}  rms_ratio={m_pkw['rms_ratio']:.4f}  peak_ratio={m_pkw['peak_ratio']:.4f}")

    # Compare probe kw_sl vs probe pkw_sl (the 1.002 case)
    print("\n--- Debug probe scan_line: kw vs pkw ---")
    m_probe = compute_trace_metrics(kw_sl_probe, pkw_sl_probe)
    print(f"  kw probe vs pkw probe:  r={m_probe['pearson_r']:.4f}  rms_ratio={m_probe['rms_ratio']:.4f}  peak_ratio={m_probe['peak_ratio']:.4f}")

    # Compare parity cache kw vs pkw at 0 deg (FULL, before trim)
    print("\n--- Parity cache scan_line: kw vs pkw at 0 deg ---")
    m_parity = compute_trace_metrics(kw_sl0, pkw_sl0)
    print(f"  kw parity vs pkw parity:  r={m_parity['pearson_r']:.4f}  rms_ratio={m_parity['rms_ratio']:.4f}  peak_ratio={m_parity['peak_ratio']:.4f}")

    # t0_offset trimmed comparison
    not_transducer.steering_angle = float(STEERING_ANGLES_QUICK[-1])  # reset to last angle state
    t0 = int(round(len(not_transducer.input_signal.squeeze()) / 2) +
             (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
    print(f"\nt0_offset = {t0}")

    print("\n--- Parity cache trimmed scan_line: kw vs pkw at 0 deg ---")
    m_trim = compute_trace_metrics(kw_sl0[t0:], pkw_sl0[t0:])
    print(f"  kw vs pkw trimmed:  r={m_trim['pearson_r']:.4f}  rms_ratio={m_trim['rms_ratio']:.4f}  peak_ratio={m_trim['peak_ratio']:.4f}")

    print("\n--- Probe scan_line trimmed: kw vs pkw ---")
    m_probe_trim = compute_trace_metrics(kw_sl_probe[t0:], pkw_sl_probe[t0:])
    print(f"  kw vs pkw trimmed probe:  r={m_probe_trim['pearson_r']:.4f}  rms_ratio={m_probe_trim['rms_ratio']:.4f}  peak_ratio={m_probe_trim['peak_ratio']:.4f}")

    # Print point-by-point around t0 for PROBE scan_line
    print(f"\n--- Probe scan_line values around t0={t0} (0 deg) ---")
    for t in [t0-10, t0-5, t0, t0+5, t0+10, t0+50, t0+100]:
        if 0 <= t < len(kw_sl_probe):
            kw_v  = kw_sl_probe[t]
            pkw_v = pkw_sl_probe[t]
            print(f"  t={t:4d}:  kw={kw_v:+.4g}  pkw={pkw_v:+.4g}  ratio={pkw_v/(kw_v+1e-20):.3g}")


if __name__ == "__main__":
    main()
