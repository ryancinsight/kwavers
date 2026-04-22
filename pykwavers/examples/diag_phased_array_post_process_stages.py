#!/usr/bin/env python3
"""Diagnostic: pinpoint at which post-processing stage the pykwavers↔k-wave
rms_ratio diverges for us_bmode_phased_array.

Single-angle face parity is ~perfect (r=0.9985, rms=1.002), but the 9-angle
B-mode image metrics fail (fund rms=1.55, harm rms=0.58). Something between
raw scan-lines and the final log-compressed image is inflating the fundamental
and deflating the harmonic.

Reads cached scan_lines_pkw and scan_lines_kw from the quick sweep, then
reproduces each post-process stage in turn and reports rms_ratio (pkw/kw) at
each stage. The stage where the ratio first departs from ~1 is the culprit.
"""

from __future__ import annotations

import numpy as np

from example_parity_utils import DEFAULT_OUTPUT_DIR, bootstrap_example_paths
bootstrap_example_paths()

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.reconstruction.beamform import envelope_detection, scan_conversion
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.filters import gaussian_filter
from kwave.utils.signals import get_win

# ─── Must match us_bmode_phased_array_compare.py ─────────────────────────────
PML_SIZE = Vector([15, 10, 10])
GRID_SIZE_POINTS = Vector([256, 256, 128]) - 2 * PML_SIZE
GRID_SIZE_METERS = 50e-3
GRID_SPACING_METERS = GRID_SIZE_METERS / Vector(
    [GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x]
)
C0 = 1540.0
TONE_BURST_FREQ = 1e6
COMPRESSION_RATIO = 3
STEERING_ANGLES_QUICK = np.arange(-32, 33, 8)


def rms_ratio(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=float).ravel()
    cand = np.asarray(cand, dtype=float).ravel()
    n = min(ref.size, cand.size)
    ref = ref[:n]; cand = cand[:n]
    r_rms = float(np.sqrt(np.mean(ref**2)))
    c_rms = float(np.sqrt(np.mean(cand**2)))
    return c_rms / (r_rms + 1e-30)


def main() -> None:
    out = DEFAULT_OUTPUT_DIR
    kw_npz = np.load(out / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz")
    pkw_npz = np.load(out / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz")
    kw_sl = np.asarray(kw_npz["scan_lines"], dtype=float)
    pkw_sl = np.asarray(pkw_npz["scan_lines"], dtype=float)
    print(f"[diag] shapes: kw={kw_sl.shape}  pkw={pkw_sl.shape}")
    print(f"[diag] raw scan_lines rms_ratio = {rms_ratio(kw_sl, pkw_sl):.4f}")

    # Recreate kgrid + t0_offset handling as in post_process
    kgrid = kWaveGrid(GRID_SIZE_POINTS, GRID_SPACING_METERS)
    t_end = (GRID_SIZE_POINTS.x * GRID_SPACING_METERS.x) * 2.2 / C0
    kgrid.makeTime(C0, t_end=t_end)

    # t0_offset is determined by NotATransducer; approximate via same values as
    # us_bmode_phased_array_compare.py hard-codes. The tone-burst length is
    # round(4 cycles * Fs/f0) + internal padding. We can recover nt from the
    # saved scan_lines since the trim happens inside post_process.
    # Easier: run the identical steps on both arrays.
    nt = kw_sl.shape[1]

    # Step 1: Tukey window
    tukey_win, _ = get_win(nt * 2, "Tukey", False, 0.05)
    # Note: t0_offset not known here; skip window-offset and apply raw Tukey
    # over the full nt — good enough to see where fund/harm diverge.
    win = tukey_win.T[:, :nt] if tukey_win.T.shape[1] >= nt else np.ones((1, nt))
    kw_win = kw_sl * win
    pkw_win = pkw_sl * win
    print(f"[diag] after Tukey window    rms_ratio = {rms_ratio(kw_win, pkw_win):.4f}")

    # Step 2: TGC (depth-dependent gain)
    r = C0 * np.arange(1, nt + 1) * kgrid.dt / 2
    alpha_db_cm = 0.75 * (TONE_BURST_FREQ * 1e-6) ** 1.5  # same as us_bmode_phased_array
    tgc_alpha_np_m = db2neper(alpha_db_cm) * 100
    tgc = np.exp(tgc_alpha_np_m * 2 * r)
    kw_tgc = kw_win * tgc
    pkw_tgc = pkw_win * tgc
    print(f"[diag] after TGC             rms_ratio = {rms_ratio(kw_tgc, pkw_tgc):.4f}")

    # Step 3: gaussian bandpass at fund (100% bw) and harm (30% bw)
    kw_fund = gaussian_filter(kw_tgc, 1 / kgrid.dt, TONE_BURST_FREQ, 100)
    pkw_fund = gaussian_filter(pkw_tgc, 1 / kgrid.dt, TONE_BURST_FREQ, 100)
    print(f"[diag] after gaussian fund   rms_ratio = {rms_ratio(kw_fund, pkw_fund):.4f}")

    kw_harm = gaussian_filter(kw_tgc, 1 / kgrid.dt, 2 * TONE_BURST_FREQ, 30)
    pkw_harm = gaussian_filter(pkw_tgc, 1 / kgrid.dt, 2 * TONE_BURST_FREQ, 30)
    print(f"[diag] after gaussian harm   rms_ratio = {rms_ratio(kw_harm, pkw_harm):.4f}")

    # Step 4: envelope detection
    kw_fund_e = envelope_detection(kw_fund)
    pkw_fund_e = envelope_detection(pkw_fund)
    print(f"[diag] after envelope fund   rms_ratio = {rms_ratio(kw_fund_e, pkw_fund_e):.4f}")

    kw_harm_e = envelope_detection(kw_harm)
    pkw_harm_e = envelope_detection(pkw_harm)
    print(f"[diag] after envelope harm   rms_ratio = {rms_ratio(kw_harm_e, pkw_harm_e):.4f}")

    # Step 5: log compression (with normalise=True)
    kw_fund_l = log_compression(kw_fund_e, COMPRESSION_RATIO, True)
    pkw_fund_l = log_compression(pkw_fund_e, COMPRESSION_RATIO, True)
    print(f"[diag] after log comp fund   rms_ratio = {rms_ratio(kw_fund_l, pkw_fund_l):.4f}")

    kw_harm_l = log_compression(kw_harm_e, COMPRESSION_RATIO, True)
    pkw_harm_l = log_compression(pkw_harm_e, COMPRESSION_RATIO, True)
    print(f"[diag] after log comp harm   rms_ratio = {rms_ratio(kw_harm_l, pkw_harm_l):.4f}")

    # Step 6: scan conversion
    image_size = [kgrid.Nx * kgrid.dx, kgrid.Ny * kgrid.dy]
    image_res = [256, 256]
    angles = STEERING_ANGLES_QUICK
    kw_bmode_fund = scan_conversion(kw_fund_l, angles, image_size, C0, kgrid.dt, image_res)
    pkw_bmode_fund = scan_conversion(pkw_fund_l, angles, image_size, C0, kgrid.dt, image_res)
    print(f"[diag] after scan_conv fund  rms_ratio = {rms_ratio(kw_bmode_fund, pkw_bmode_fund):.4f}")

    kw_bmode_harm = scan_conversion(kw_harm_l, angles, image_size, C0, kgrid.dt, image_res)
    pkw_bmode_harm = scan_conversion(pkw_harm_l, angles, image_size, C0, kgrid.dt, image_res)
    print(f"[diag] after scan_conv harm  rms_ratio = {rms_ratio(kw_bmode_harm, pkw_bmode_harm):.4f}")

    # Spectral probe on raw — look at fund and 2f0 energy directly
    print()
    print("-- Spectral probe on raw scan_lines --")
    # Use middle angle only to keep signal coherent
    mid = kw_sl.shape[0] // 2
    dt = float(kgrid.dt)
    sig_kw = kw_sl[mid]
    sig_pkw = pkw_sl[mid]
    n = sig_kw.size
    freqs = np.fft.rfftfreq(n, dt)
    KW = np.abs(np.fft.rfft(sig_kw))
    PKW = np.abs(np.fft.rfft(sig_pkw))
    # Integrate fund band [0.5, 1.5] MHz and harm band [1.7, 2.3] MHz
    def band_rms(freqs, spec, lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.sqrt(np.sum(spec[m] ** 2)))
    kw_fund_e_raw = band_rms(freqs, KW, 0.5e6, 1.5e6)
    pkw_fund_e_raw = band_rms(freqs, PKW, 0.5e6, 1.5e6)
    kw_harm_e_raw = band_rms(freqs, KW, 1.7e6, 2.3e6)
    pkw_harm_e_raw = band_rms(freqs, PKW, 1.7e6, 2.3e6)
    print(f"[diag] center-angle spectral fund band [0.5,1.5] MHz:")
    print(f"         kw={kw_fund_e_raw:.4g}  pkw={pkw_fund_e_raw:.4g}  ratio={pkw_fund_e_raw/(kw_fund_e_raw+1e-30):.4f}")
    print(f"[diag] center-angle spectral harm band [1.7,2.3] MHz:")
    print(f"         kw={kw_harm_e_raw:.4g}  pkw={pkw_harm_e_raw:.4g}  ratio={pkw_harm_e_raw/(kw_harm_e_raw+1e-30):.4f}")

    report = out / "phased_array_post_process_stages_report.txt"
    with open(report, "w") as f:
        f.write(f"raw rms_ratio = {rms_ratio(kw_sl, pkw_sl):.4f}\n")
        f.write(f"after Tukey   = {rms_ratio(kw_win, pkw_win):.4f}\n")
        f.write(f"after TGC     = {rms_ratio(kw_tgc, pkw_tgc):.4f}\n")
        f.write(f"after gauss fund = {rms_ratio(kw_fund, pkw_fund):.4f}\n")
        f.write(f"after gauss harm = {rms_ratio(kw_harm, pkw_harm):.4f}\n")
        f.write(f"after env   fund = {rms_ratio(kw_fund_e, pkw_fund_e):.4f}\n")
        f.write(f"after env   harm = {rms_ratio(kw_harm_e, pkw_harm_e):.4f}\n")
        f.write(f"after logc  fund = {rms_ratio(kw_fund_l, pkw_fund_l):.4f}\n")
        f.write(f"after logc  harm = {rms_ratio(kw_harm_l, pkw_harm_l):.4f}\n")
        f.write(f"after scanc fund = {rms_ratio(kw_bmode_fund, pkw_bmode_fund):.4f}\n")
        f.write(f"after scanc harm = {rms_ratio(kw_bmode_harm, pkw_bmode_harm):.4f}\n")
        f.write(f"center-angle spectral fund [0.5,1.5]MHz pkw/kw = {pkw_fund_e_raw/(kw_fund_e_raw+1e-30):.4f}\n")
        f.write(f"center-angle spectral harm [1.7,2.3]MHz pkw/kw = {pkw_harm_e_raw/(kw_harm_e_raw+1e-30):.4f}\n")
    print(f"\n[diag] wrote {report}")


if __name__ == "__main__":
    main()
