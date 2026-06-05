#!/usr/bin/env python3
"""
Find the time offset between kw and pkw scan_lines using cross-correlation.
Also check per-element trace cross-correlation to see if the offset is in
the sensor data or introduced by scan_line.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import numpy as np
from us_bmode_phased_array_compare import STEERING_ANGLES_QUICK, build_reference_objects

OUTPUT = Path(__file__).parent / "output"


def xcorr_lag(a, b, max_lag=50):
    """Return the lag (in samples) of b relative to a that maximizes cross-correlation."""
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    n = len(a)
    # Compute cross-correlation using FFT
    corr = np.correlate(a, b, mode='full')
    lags = np.arange(-(n-1), n)
    # Restrict to max_lag
    mask = np.abs(lags) <= max_lag
    best_lag = int(lags[mask][np.argmax(corr[mask])])
    best_r = float(corr[mask][np.argmax(corr[mask])]) / (np.std(a) * np.std(b) * n)
    return best_lag, best_r


def main():
    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(20260401)

    # Load cached scan_lines from parity run
    kw_cache  = np.load(OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz")
    pkw_cache = np.load(OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz")
    kw_sl  = kw_cache['scan_lines']
    pkw_sl = pkw_cache['scan_lines']
    angles = list(STEERING_ANGLES_QUICK)

    # t0_offset
    not_transducer.steering_angle = 32.0
    _ = not_transducer.delay_mask()
    t0 = int(round(len(not_transducer.input_signal.squeeze()) / 2) +
             (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))

    print(f"t0_offset = {t0}")
    print()
    print(f"{'Angle':>6s}  {'Full xcorr lag':>14s}  {'Full r':>7s}  {'Trim xcorr lag':>14s}  {'Trim r':>7s}  {'Trim rms_ratio':>14s}")
    for i, ang in enumerate(angles):
        full_lag, full_r = xcorr_lag(kw_sl[i], pkw_sl[i])
        trim_lag, trim_r = xcorr_lag(kw_sl[i, t0:], pkw_sl[i, t0:])
        kw_rms  = float(np.sqrt(np.mean(kw_sl[i, t0:]**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_sl[i, t0:]**2)))
        ratio = pkw_rms / (kw_rms + 1e-30)
        print(f"{ang:>+6d}  {full_lag:>14d}  {full_r:>7.4f}  {trim_lag:>14d}  {trim_r:>7.4f}  {ratio:>14.4f}")

    # Load probe data and check per-element cross-correlation
    dt_npz = np.load(OUTPUT / "us_bmode_phased_array_drive_trace.npz")
    kw_comb  = dt_npz["kw_combined"]   # [64, 1658]
    pkw_comb = dt_npz["pkw_combined"]

    print(f"\n--- Per-element cross-correlation lags (probe, angle=0) ---")
    print(f"{'Elem':>5s}  {'Full lag':>8s}  {'Full r':>7s}  {'Trim lag':>8s}  {'Trim r':>7s}  {'Trim rms':>8s}")
    for el in [0, 16, 32, 48, 63]:
        fl, fr = xcorr_lag(kw_comb[el], pkw_comb[el])
        tl, tr = xcorr_lag(kw_comb[el, t0:], pkw_comb[el, t0:])
        kw_rms  = float(np.sqrt(np.mean(kw_comb[el, t0:]**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_comb[el, t0:]**2)))
        ratio = pkw_rms / (kw_rms + 1e-30)
        print(f"{el:>5d}  {fl:>8d}  {fr:>7.4f}  {tl:>8d}  {tr:>7.4f}  {ratio:>8.4f}")

    # Apply scan_line with 0deg and check lag in scan_line output (trimmed)
    not_transducer.steering_angle = 0.0
    _ = not_transducer.delay_mask()
    kw_sl_probe  = np.asarray(not_transducer.scan_line(kw_comb.copy()),  dtype=float)
    pkw_sl_probe = np.asarray(not_transducer.scan_line(pkw_comb.copy()), dtype=float)

    sl_full_lag, sl_full_r = xcorr_lag(kw_sl_probe, pkw_sl_probe)
    sl_trim_lag, sl_trim_r = xcorr_lag(kw_sl_probe[t0:], pkw_sl_probe[t0:])
    print(f"\nProbe scan_line(0deg) full:  lag={sl_full_lag}  r={sl_full_r:.4f}")
    print(f"Probe scan_line(0deg) trim:  lag={sl_trim_lag}  r={sl_trim_r:.4f}")

    # Compute rms_ratio after shift-correcting
    if sl_trim_lag != 0:
        lag = sl_trim_lag
        kw_t = kw_sl_probe[t0:]
        pkw_t = pkw_sl_probe[t0:]
        if lag > 0:
            # pkw arrives BEFORE kw by lag samples; shift pkw forward by lag to align
            pkw_shifted = pkw_t[lag:]
            kw_aligned  = kw_t[:len(pkw_shifted)]
        else:
            pkw_shifted = pkw_t[:len(pkw_t)+lag]
            kw_aligned  = kw_t[-lag:len(pkw_shifted)-lag]

        kw_rms  = float(np.sqrt(np.mean(kw_aligned**2)))
        pkw_rms = float(np.sqrt(np.mean(pkw_shifted**2)))
        r_shifted = float(np.corrcoef(kw_aligned, pkw_shifted)[0, 1])
        print(f"\nAfter lag-correction ({lag} samples):  rms_ratio={pkw_rms/kw_rms:.4f}  r={r_shifted:.4f}")

    # Check ALL per-element trim lags histogram
    print(f"\n--- Per-element trim lags histogram (probe, angle=0) ---")
    lags = []
    for el in range(64):
        tl, _ = xcorr_lag(kw_comb[el, t0:], pkw_comb[el, t0:])
        lags.append(tl)
    lags = np.array(lags)
    unique_lags, counts = np.unique(lags, return_counts=True)
    for ul, cnt in sorted(zip(unique_lags, counts)):
        print(f"  lag={int(ul):+4d}: {int(cnt)} elements")
    print(f"  mean lag = {np.mean(lags):.1f}  std = {np.std(lags):.1f}")


if __name__ == "__main__":
    main()
