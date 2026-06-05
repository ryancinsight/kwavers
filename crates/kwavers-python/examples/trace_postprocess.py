#!/usr/bin/env python3
"""Trace exactly where the 2x rms_ratio emerges in post_process.

Load cached scan_lines, run post_process step by step, and print rms_ratio
at each stage to find where the 2x appears.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.reconstruction.beamform import envelope_detection, scan_conversion
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import gaussian_filter
from kwave.utils.signals import get_win, tone_burst
from kwave.data import Vector
from us_bmode_phased_array_compare import (
    PML_SIZE, GRID_SIZE_POINTS, GRID_SPACING_METERS,
    C0, RHO0, ALPHA_COEFF, ALPHA_POWER, BON_A, SOURCE_STRENGTH,
    TONE_BURST_FREQ, TONE_BURST_CYCLES, COMPRESSION_RATIO,
    STEERING_ANGLES_QUICK, build_reference_objects,
)

OUTPUT = Path(__file__).parent / "output"


def rms_ratio_str(kw, pkw, label):
    kw_a = np.asarray(kw, dtype=float)
    pkw_a = np.asarray(pkw, dtype=float)
    kw_rms  = float(np.sqrt(np.mean(kw_a**2)))
    pkw_rms = float(np.sqrt(np.mean(pkw_a**2)))
    ratio = pkw_rms / (kw_rms + 1e-30)
    kw_pk  = float(np.max(np.abs(kw_a)))
    pkw_pk = float(np.max(np.abs(pkw_a)))
    pk_ratio = pkw_pk / (kw_pk + 1e-30)
    print(f"  [{label:35s}] rms_ratio={ratio:.4f}  peak_ratio={pk_ratio:.4f}  "
          f"kw_rms={kw_rms:.4g}  pkw_rms={pkw_rms:.4g}  kw_pk={kw_pk:.4g}  pkw_pk={pkw_pk:.4g}")
    return ratio


def main():
    steering_angles = STEERING_ANGLES_QUICK  # 9 angles

    # Load cached scan lines
    kw_path  = OUTPUT / "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu.npz"
    pkw_path = OUTPUT / "us_bmode_phased_array_pykwavers_quick_seed20260401_kwgpu_pkwgpu.npz"
    kw_cache  = np.load(kw_path)
    pkw_cache = np.load(pkw_path)

    kw_sl_orig  = kw_cache['scan_lines'].copy()   # [n_angles, nt]
    pkw_sl_orig = pkw_cache['scan_lines'].copy()
    print(f"kw  scan_lines: {kw_sl_orig.shape}")
    print(f"pkw scan_lines: {pkw_sl_orig.shape}")

    rms_ratio_str(kw_sl_orig, pkw_sl_orig, "raw scan_lines (full)")
    print()

    # Rebuild kgrid + not_transducer so we can call post_process correctly
    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(20260401)
    # After the simulation loops, not_transducer.steering_angle is set to the
    # last steering angle. Replicate that state:
    not_transducer.steering_angle = float(steering_angles[-1])

    # ── Replicate post_process step by step ──────────────────────────────────
    def step_by_step(kw_sl, pkw_sl, tag):
        print(f"\n=== {tag} ===")
        # 1. t0 trim
        t0_offset = int(round(len(not_transducer.input_signal.squeeze()) / 2) +
                        (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
        print(f"  t0_offset = {t0_offset}")
        kw  = kw_sl[:, t0_offset:].copy()
        pkw = pkw_sl[:, t0_offset:].copy()
        rms_ratio_str(kw, pkw, "after t0_trim")
        nt = kw.shape[1]

        # 2. Tukey window
        tukey_win, _ = get_win(nt * 2, "Tukey", False, 0.05)
        scan_line_win = np.concatenate(
            (np.zeros([1, t0_offset * 2]),
             tukey_win.T[:, : int(len(tukey_win) / 2) - t0_offset * 2]),
            axis=1,
        )
        kw  = kw  * scan_line_win
        pkw = pkw * scan_line_win
        rms_ratio_str(kw, pkw, "after Tukey window")

        # 3. TGC
        r = C0 * np.arange(1, nt + 1) * kgrid.dt / 2
        tgc_alpha_db_cm = medium.alpha_coeff * (TONE_BURST_FREQ * 1e-6) ** medium.alpha_power
        tgc_alpha_np_m  = db2neper(tgc_alpha_db_cm) * 100
        tgc = np.exp(tgc_alpha_np_m * 2 * r)
        kw  = kw  * tgc
        pkw = pkw * tgc
        rms_ratio_str(kw, pkw, "after TGC")

        # 4. Gaussian bandpass
        kw_fund  = gaussian_filter(kw,  1 / kgrid.dt, TONE_BURST_FREQ,     100)
        kw_harm  = gaussian_filter(kw,  1 / kgrid.dt, 2 * TONE_BURST_FREQ,  30)
        pkw_fund = gaussian_filter(pkw, 1 / kgrid.dt, TONE_BURST_FREQ,     100)
        pkw_harm = gaussian_filter(pkw, 1 / kgrid.dt, 2 * TONE_BURST_FREQ,  30)
        rms_ratio_str(kw_fund, pkw_fund, "after BPF (fund)")
        rms_ratio_str(kw_harm, pkw_harm, "after BPF (harm)")

        # 5. Envelope detection
        kw_fund  = envelope_detection(kw_fund)
        kw_harm  = envelope_detection(kw_harm)
        pkw_fund = envelope_detection(pkw_fund)
        pkw_harm = envelope_detection(pkw_harm)
        rms_ratio_str(kw_fund, pkw_fund, "after envelope (fund)")
        rms_ratio_str(kw_harm, pkw_harm, "after envelope (harm)")

        # 6. Log compression
        kw_fund  = log_compression(kw_fund,  COMPRESSION_RATIO, True)
        kw_harm  = log_compression(kw_harm,  COMPRESSION_RATIO, True)
        pkw_fund = log_compression(pkw_fund, COMPRESSION_RATIO, True)
        pkw_harm = log_compression(pkw_harm, COMPRESSION_RATIO, True)
        rms_ratio_str(kw_fund, pkw_fund, "after log_compress (fund)")
        rms_ratio_str(kw_harm, pkw_harm, "after log_compress (harm)")

        # 7. Scan conversion
        image_size = [kgrid.Nx * kgrid.dx, kgrid.Ny * kgrid.dy]
        image_res  = [256, 256]
        kw_bmode_fund  = scan_conversion(kw_fund,  steering_angles, image_size, C0, kgrid.dt, image_res)
        kw_bmode_harm  = scan_conversion(kw_harm,  steering_angles, image_size, C0, kgrid.dt, image_res)
        pkw_bmode_fund = scan_conversion(pkw_fund, steering_angles, image_size, C0, kgrid.dt, image_res)
        pkw_bmode_harm = scan_conversion(pkw_harm, steering_angles, image_size, C0, kgrid.dt, image_res)
        rms_ratio_str(kw_bmode_fund, pkw_bmode_fund, "B-mode image (fund)")
        rms_ratio_str(kw_bmode_harm, pkw_bmode_harm, "B-mode image (harm)")
        return kw_bmode_fund, kw_bmode_harm, pkw_bmode_fund, pkw_bmode_harm

    step_by_step(kw_sl_orig, pkw_sl_orig, "Cached scan_lines -> post_process")


if __name__ == "__main__":
    main()
