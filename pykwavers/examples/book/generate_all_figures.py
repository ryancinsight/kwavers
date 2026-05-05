"""
Master Figure Generation Script — kwavers Ultrasound Physics Book
=================================================================

Runs all chapter figure generation scripts in order. Each chapter
script is independent and can also be run standalone.

Usage::

    python generate_all_figures.py [--chapter N]

With no arguments, generates all chapters. With --chapter N, generates
only Chapter N (1-indexed, N=1..20).

Output: docs/book/figures/chNN/ for each chapter NN = 01..21.

Requires: numpy, matplotlib, scipy
Optional: pykwavers (Chapter 1 fig06 and Chapter 3 fig06 only)
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BOOK_SCRIPTS_DIR = os.path.dirname(__file__)

CHAPTER_SCRIPTS = [
    (1,  "ch01_wave_physics_fundamentals.py",    "Wave Physics Fundamentals"),
    (2,  "ch02_numerical_methods.py",            "Numerical Methods: FDTD and PSTD"),
    (3,  "ch03_nonlinear_acoustics.py",           "Nonlinear Acoustics"),
    (4,  "ch04_transducer_arrays_beamforming.py", "Transducer Arrays and Beamforming"),
    (5,  "ch05_ultrasound_imaging.py",            "Ultrasound Imaging"),
    (6,  "ch06_therapeutic_ultrasound.py",        "Therapeutic Ultrasound"),
    (7,  "ch07_theranostics.py",                  "Theranostics"),
    (8,  "ch08_acoustic_propagation.py",          "Acoustic Propagation"),
    (9,  "ch09_cavitation_and_bubbles.py",        "Cavitation and Bubble Dynamics"),
    (10, "ch10_elastography.py",                  "Elastography"),
    (11, "ch11_sources_and_transducers.py",       "Sources and Transducers"),
    (12, "ch12_media_and_tissue_models.py",       "Media and Tissue Models"),
    (13, "ch13_photoacoustics.py",                "Photoacoustics"),
    (14, "ch14_sensors_and_measurements.py",      "Sensors and Measurements"),
    (15, "ch15_safety_and_dosimetry.py",          "Safety and Dosimetry"),
    (16, "ch16_transcranial_ultrasound.py",       "Transcranial Ultrasound"),
    (17, "ch17_inverse_problems_and_pinns.py",    "Inverse Problems and PINNs"),
    (18, "ch18_sonogenetics.py",                  "Sonogenetics"),
    (19, "ch19_performance_and_memory.py",        "Performance and Memory"),
    (20, "ch20_validation_and_benchmarking.py",   "Validation and Benchmarking"),
    (21, "ch21_histotripsy_comparison.py",         "Histotripsy: Classical vs ms-Pulse"),
]


def run_chapter(ch_num: int, script_name: str, ch_title: str) -> bool:
    """Run a chapter figure script and return success status."""
    script_path = os.path.join(BOOK_SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  [SKIP] Chapter {ch_num}: {script_name} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Chapter {ch_num}: {ch_title}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location(f"ch{ch_num:02d}", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        elapsed = time.perf_counter() - t0
        print(f"  [OK] Chapter {ch_num} completed in {elapsed:.1f}s")
        return True
    except Exception:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] Chapter {ch_num} failed after {elapsed:.1f}s:")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all book figures for the kwavers ultrasound physics book."
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Generate only this chapter number (1-21). Default: all chapters."
    )
    args = parser.parse_args()

    sys.path.insert(0, BOOK_SCRIPTS_DIR)

    chapters_to_run = CHAPTER_SCRIPTS
    if args.chapter is not None:
        chapters_to_run = [(n, s, t) for n, s, t in CHAPTER_SCRIPTS if n == args.chapter]
        if not chapters_to_run:
            print(f"ERROR: Chapter {args.chapter} not found. Valid range: 1–21.")
            sys.exit(1)

    t_total_start = time.perf_counter()
    results = []
    for ch_num, script_name, ch_title in chapters_to_run:
        ok = run_chapter(ch_num, script_name, ch_title)
        results.append((ch_num, ch_title, ok))

    t_total = time.perf_counter() - t_total_start

    # Summary
    print(f"\n{'='*60}")
    print("FIGURE GENERATION SUMMARY")
    print(f"{'='*60}")
    n_ok = sum(1 for _, _, ok in results if ok)
    n_fail = len(results) - n_ok
    for ch_num, ch_title, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] Chapter {ch_num}: {ch_title}")
    print(f"\nTotal: {n_ok}/{len(results)} chapters completed in {t_total:.1f}s")
    if n_fail > 0:
        print(f"WARNING: {n_fail} chapter(s) failed — check error output above.")
        sys.exit(1)

    # List output directories
    print("\nOutput directories:")
    for ch_num, _, ok in results:
        if ok:
            out_dir = os.path.join(REPO_ROOT, "docs", "book", "figures", f"ch{ch_num:02d}")
            if os.path.isdir(out_dir):
                n_files = len([f for f in os.listdir(out_dir) if f.endswith(".pdf")])
                print(f"  docs/book/figures/ch{ch_num:02d}/  ({n_files} PDF files)")


if __name__ == "__main__":
    main()
