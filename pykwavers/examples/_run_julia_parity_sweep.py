#!/usr/bin/env python3
"""Run all KWave.jl parity scripts and tabulate metrics.

Each script runs end-to-end against ``external/k-wave-julia/KWave.jl`` for a
physics target that has no equivalent example in
``external/k-wave-python/examples``. Honours each script's failure-tolerance
flag so a single FAIL doesn't abort the sweep.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent

# (script, failure-tolerance arg)
SCRIPTS = [
    ("diff_bioheat_1d_jl_compare.py",      "--allow-failure"),
    ("ewp_elastic_2d_jl_compare.py",       None),         # never strict by
                                                           # default — see
                                                           # script docstring
    ("pr_time_reversal_2d_jl_compare.py",  "--allow-failure"),
    ("us_phased_array_3d_jl_compare.py",   "--allow-failure"),
    ("us_beamforming_2d_jl_compare.py",    "--allow-failure"),
]


def run_one(script: str, fail_arg: str | None) -> dict:
    cmd = [sys.executable, str(EXAMPLES / script)]
    if fail_arg is not None:
        cmd.append(fail_arg)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - t0
    out = proc.stdout
    result = "PASS" if "RESULT" in out and re.search(r"RESULT\s*[:=]\s*PASS", out) else "FAIL"
    if proc.returncode != 0 and result != "PASS":
        result = f"FAIL (exit {proc.returncode})"
    pearson = "-"
    m = re.search(r"pearson_r\s*[:=]\s*([\-\d\.]+)", out)
    if m:
        pearson = m.group(1)
    return {
        "script": script,
        "result": result,
        "pearson": pearson,
        "elapsed_s": f"{elapsed:.1f}",
    }


def main() -> int:
    print(f"{'script':<45} {'result':<8} {'pearson':<10} {'time':<8}")
    print("-" * 75)
    all_ok = True
    for script, fail_arg in SCRIPTS:
        r = run_one(script, fail_arg)
        if not r["result"].startswith("PASS") and script != "ewp_elastic_2d_jl_compare.py":
            all_ok = False
        print(f"{r['script']:<45} {r['result']:<8} {r['pearson']:<10} {r['elapsed_s']:<8} s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
