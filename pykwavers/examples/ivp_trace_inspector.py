#!/usr/bin/env python3
"""
ivp_trace_inspector.py
Inspect the raw pykwavers and k-Wave traces in detail to understand
where pykwavers' pressure appears and when the peaks really are.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
KWAVE_CACHE = OUTPUT_DIR / "ivp_photoacoustic_kwave_cache.npz"
PKWAV_CACHE = OUTPUT_DIR / "ivp_photoacoustic_pykwavers_cache.npz"
PYREF_CACHE = OUTPUT_DIR / "ivp_pure_python_pstd_cache.npz"

DT = 2e-9
NT = 150

def main():
    kw  = np.load(KWAVE_CACHE)["trace"].flatten().astype(np.float64)
    pkw = np.load(PKWAV_CACHE)["trace"].flatten().astype(np.float64)
    py  = np.load(PYREF_CACHE)["trace"].flatten().astype(np.float64)

    t_ns = np.arange(NT) * DT * 1e9

    print(f"k-Wave   : peak abs at step {np.argmax(np.abs(kw))+1} (0-idx={np.argmax(np.abs(kw))}), val={kw[np.argmax(np.abs(kw))]:+.4e}")
    print(f"pykwavers: peak abs at step {np.argmax(np.abs(pkw))+1} (0-idx={np.argmax(np.abs(pkw))}), val={pkw[np.argmax(np.abs(pkw))]:+.4e}")
    print(f"py-PSTD  : peak abs at step {np.argmax(np.abs(py))+1} (0-idx={np.argmax(np.abs(py))}), val={py[np.argmax(np.abs(py))]:+.4e}")

    print(f"\nk-Wave   rms={np.sqrt(np.mean(kw**2)):.6e}  max={kw.max():.6e}  min={kw.min():.6e}")
    print(f"pykwavers rms={np.sqrt(np.mean(pkw**2)):.6e}  max={pkw.max():.6e}  min={pkw.min():.6e}")
    print(f"py-PSTD   rms={np.sqrt(np.mean(py**2)):.6e}  max={py.max():.6e}  min={py.min():.6e}")

    print("\nFirst 20 steps (should be ~0 before wave arrives):")
    print(f"{'Step':>5}  {'kwave':>12}  {'pkwav':>12}  {'py-ref':>12}")
    for i in range(20):
        print(f"{i+1:5d}  {kw[i]:12.4e}  {pkw[i]:12.4e}  {py[i]:12.4e}")

    print("\nSteps around pykwavers zero-cross (45-65):")
    print(f"{'Step':>5}  {'kwave':>12}  {'pkwav':>12}  {'py-ref':>12}")
    for i in range(44, 66):
        flag = " <-- pkw sign change" if i > 0 and pkw[i]*pkw[i-1] < 0 else ""
        flag2 = " <-- kw sign change" if i > 0 and kw[i]*kw[i-1] < 0 else ""
        print(f"{i+1:5d}  {kw[i]:12.4e}  {pkw[i]:12.4e}  {py[i]:12.4e}{flag}{flag2}")

    print("\nSteps 90-110 (py-PSTD zero-cross region):")
    print(f"{'Step':>5}  {'kwave':>12}  {'pkwav':>12}  {'py-ref':>12}")
    for i in range(85, min(110, NT)):
        flag = " <-- py sign change" if i > 0 and py[i]*py[i-1] < 0 else ""
        print(f"{i+1:5d}  {kw[i]:12.4e}  {pkw[i]:12.4e}  {py[i]:12.4e}{flag}")

    # Shift sweep: pykwavers vs k-Wave, full range
    from scipy import signal as sp_signal
    corr_kw_pkw = sp_signal.correlate(kw, pkw, mode="full")
    lags_kw_pkw = sp_signal.correlation_lags(len(kw), len(pkw), mode="full")
    best_kw_pkw = int(lags_kw_pkw[np.argmax(corr_kw_pkw)])
    print(f"\nCross-correlation k-Wave vs pykwavers: best_lag={best_kw_pkw}")

    corr_kw_py = sp_signal.correlate(kw, py, mode="full")
    lags_kw_py = sp_signal.correlation_lags(len(kw), len(py), mode="full")
    best_kw_py = int(lags_kw_py[np.argmax(corr_kw_py)])
    print(f"Cross-correlation k-Wave vs py-PSTD: best_lag={best_kw_py}")

    # Check: does pykwavers look like a SHIFTED version of k-Wave?
    def pearson_r(a, b):
        am = a - a.mean(); bm = b - b.mean()
        d = np.sqrt((am**2).sum() * (bm**2).sum())
        return float((am * bm).sum() / d) if d > 1e-30 else 0.0

    print(f"\nPearson r (no shift):")
    print(f"  k-Wave vs pykwavers: {pearson_r(kw, pkw):.6f}")
    print(f"  k-Wave vs py-PSTD:   {pearson_r(kw, py):.6f}")
    print(f"  py-PSTD vs pykwavers: {pearson_r(py, pkw):.6f}")

    print(f"\nShift sweep k-Wave vs pykwavers (positive = delay pykwavers):")
    for shift in range(30, 45):
        if shift >= 0:
            pkw_s = np.concatenate([np.zeros(shift), pkw[:-shift]])
        else:
            pkw_s = np.concatenate([pkw[-shift:], np.zeros(-shift)])
        r_s = pearson_r(kw, pkw_s)
        print(f"  shift={shift:+3d}: r={r_s:+.4f}")

if __name__ == "__main__":
    main()
