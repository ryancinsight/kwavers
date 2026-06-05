#!/usr/bin/env python3
"""
ivp_timing_diagnostic.py
Diagnostic: load cached IVP waveforms, measure cross-correlation shift,
check if a rigid N-step shift gives r >= 0.97, and identify root cause of
pykwavers timing offset.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path
from scipy import signal as sp_signal

OUTPUT_DIR = Path(__file__).parent / "output"
KWAVE_CACHE  = OUTPUT_DIR / "ivp_photoacoustic_kwave_cache.npz"
PKWAV_CACHE  = OUTPUT_DIR / "ivp_photoacoustic_pykwavers_cache.npz"

DT = 2e-9   # s
NT = 150

def pearson_r(a, b):
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / denom) if denom > 1e-30 else 0.0

def main():
    if not KWAVE_CACHE.exists() or not PKWAV_CACHE.exists():
        print("Cache files not found. Run the comparison script first.")
        return

    kw  = np.load(KWAVE_CACHE)["trace"].flatten().astype(np.float64)
    pkw = np.load(PKWAV_CACHE)["trace"].flatten().astype(np.float64)

    assert len(kw) == NT and len(pkw) == NT, f"Expected {NT} samples, got kwave={len(kw)} pkw={len(pkw)}"

    t_ns = np.arange(NT) * DT * 1e9

    print(f"kwave  peak: {np.argmax(np.abs(kw))+1} steps  (t={t_ns[np.argmax(np.abs(kw))]:.1f} ns)  peak_val={kw[np.argmax(np.abs(kw))]:+.4e} Pa")
    print(f"pkwav  peak: {np.argmax(np.abs(pkw))+1} steps  (t={t_ns[np.argmax(np.abs(pkw))]:.1f} ns)  peak_val={pkw[np.argmax(np.abs(pkw))]:+.4e} Pa")

    # Zero crossing (sign change around the peak)
    def first_sign_change_after_peak(arr):
        pk = int(np.argmax(np.abs(arr)))
        for i in range(pk, len(arr)-1):
            if arr[i] * arr[i+1] < 0:
                return i  # step index (0-based)
        return -1

    kw_zc  = first_sign_change_after_peak(kw)
    pkw_zc = first_sign_change_after_peak(pkw)
    print(f"kwave  zero-cross after peak: step {kw_zc+1}  (t={t_ns[kw_zc]:.1f} ns)")
    print(f"pkwav  zero-cross after peak: step {pkw_zc+1}  (t={t_ns[pkw_zc]:.1f} ns)")
    print(f"Physical zero-cross expected: t = D/c0 = sqrt(123)*15.625e-6/1500 = {np.sqrt(123)*15.625e-6/1500*1e9:.1f} ns = step {np.sqrt(123)*15.625e-6/1500/DT:.1f}")

    print()
    # Raw Pearson r
    r_raw = pearson_r(kw, pkw)
    print(f"Pearson r (no shift):  {r_raw:.6f}")

    # Cross-correlation to find optimal shift
    corr = sp_signal.correlate(kw, pkw, mode="full")
    lags = sp_signal.correlation_lags(len(kw), len(pkw), mode="full")
    best_lag = int(lags[np.argmax(corr)])
    print(f"Cross-correlation best lag: {best_lag} steps (positive = pkwav leads kwave)")

    # Sweep shifts and report r
    print()
    print("Shift sweep (positive shift delays pykwavers to align with k-Wave):")
    for shift in range(-3, 15):
        if shift >= 0:
            # shift>0: pad start of pkw, trim end
            pkw_shifted = np.concatenate([np.zeros(shift), pkw[:-shift if shift else None]])
        else:
            # shift<0: trim start of pkw, pad end
            pkw_shifted = np.concatenate([pkw[-shift:], np.zeros(-shift)])
        r_s = pearson_r(kw, pkw_shifted)
        flag = " <-- best" if shift == best_lag else ("  <-- TARGET?" if abs(r_s) >= 0.97 else "")
        print(f"  shift={shift:+3d}: r={r_s:+.4f}  rms_ratio={np.sqrt(np.mean(pkw_shifted**2))/np.sqrt(np.mean(kw**2)):.4f}{flag}")

    print()
    # Detailed sensor trace around arrival zone (steps 35-70)
    print("Sensor trace (steps 35-70), kwave vs pykwavers:")
    print(f"{'Step':>5}  {'t_ns':>7}  {'kwave(Pa)':>12}  {'pkwav(Pa)':>12}  {'diff':>12}")
    for i in range(34, min(72, NT)):
        print(f"{i+1:5d}  {t_ns[i]:7.1f}  {kw[i]:12.4e}  {pkw[i]:12.4e}  {pkw[i]-kw[i]:12.4e}")

if __name__ == "__main__":
    main()
