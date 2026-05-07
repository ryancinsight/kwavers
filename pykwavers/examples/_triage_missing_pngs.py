#!/usr/bin/env python3
"""Run a list of scripts that historically don't produce comparison PNGs and
classify the failure mode (script bug, missing dep, real parity issue, etc.).

Captures the last 6 lines of stderr+stdout per script.
"""
import os, subprocess, sys, time

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "na_filtering_part_1_compare.py",
    "na_filtering_part_2_compare.py",
    "na_filtering_part_3_compare.py",
    "sd_directional_array_elements_compare.py",
    "tvsp_doppler_effect_compare.py",
]

for s in SCRIPTS:
    print(f"\n=== {s} ===")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(EXAMPLES, s), "--allow-failure"],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>180s)")
        continue
    elapsed = time.perf_counter() - t0
    stdout = (r.stdout or b"").decode("utf-8", errors="replace")
    stderr = (r.stderr or b"").decode("utf-8", errors="replace")
    last = (stdout + "\n" + stderr).strip().splitlines()[-8:]
    print(f"  rc={r.returncode}  elapsed={elapsed:.1f}s")
    for line in last:
        print(f"  | {line}")
