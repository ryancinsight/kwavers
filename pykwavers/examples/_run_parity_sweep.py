#!/usr/bin/env python3
"""Run a strategic subset of parity scripts and tabulate metrics.

Captures Status / Pearson / PSNR from each script's stdout. Honours each
script's `--allow-failure` so failing parity doesn't abort the sweep.
"""
import os
import re
import subprocess
import sys
import time

EXAMPLES = os.path.dirname(os.path.abspath(__file__))

# Time-domain sensor-trace scripts most likely to benefit from the time-axis
# alignment fix. Spatial-aggregate scripts (p_max / p_rms / p_final) and
# scan-line / frequency-domain comparisons are excluded — those are unaffected
# by a 1-sample time shift.
SCRIPTS = [
    "ivp_1D_simulation_compare.py",
    "ivp_3D_simulation_compare.py",
    "ivp_homogeneous_medium_compare.py",
    "ivp_loading_external_image_compare.py",
    "ivp_heterogeneous_medium_compare.py",
    "ivp_photoacoustic_waveforms_compare.py",
    "ivp_recording_particle_velocity_compare.py",
    "ivp_binary_sensor_mask_compare.py",
    "ivp_opposing_corners_sensor_mask_compare.py",
    "na_modelling_absorption_compare.py",
    "na_source_smoothing_compare.py",
    "na_controlling_the_pml_compare.py",
    "sd_focussed_detector_2D_compare.py",
    "sd_focussed_detector_3D_compare.py",
    "tvsp_homogeneous_medium_monopole_compare.py",
    "tvsp_homogeneous_medium_dipole_compare.py",
    "tvsp_transducer_field_patterns_compare.py",
    "tvsp_snells_law_compare.py",
    "ewp_plane_wave_absorption_compare.py",
    "ewp_3D_simulation_compare.py",
    "ewp_layered_medium_compare.py",
]

re_status = re.compile(r"^\s*[Ss]tatus\s*:\s*(\S+)", re.M)
re_pearson = re.compile(r"^\s*pearson_r(?:_min)?\s*:\s*([-\d.eE+nan]+)", re.M)
re_psnr = re.compile(r"^\s*psnr_db(?:_min)?\s*:\s*([-\d.eE+nan]+)", re.M)
re_overall = re.compile(r"^\s*(?:Overall.*|RESULT)\s*:\s*(\S+)", re.M)


def run_one(script: str) -> dict:
    cmd = [sys.executable, os.path.join(EXAMPLES, script), "--allow-failure"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.perf_counter()
    try:
        out = subprocess.run(
            cmd, env=env, capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"script": script, "status": "TIMEOUT", "pearson": "-",
                "psnr": "-", "elapsed": 300.0, "rc": -1}
    elapsed = time.perf_counter() - t0

    stdout = (out.stdout or b"").decode("utf-8", errors="replace")
    stderr = (out.stderr or b"").decode("utf-8", errors="replace")
    text = stdout + "\n" + stderr
    status = None
    overall = re_overall.search(text)
    if overall:
        status = overall.group(1)
    else:
        s = re_status.search(text)
        status = s.group(1) if s else "?"
    pearson = re_pearson.search(text)
    psnr = re_psnr.search(text)

    return {
        "script": script,
        "status": status,
        "pearson": pearson.group(1) if pearson else "-",
        "psnr": psnr.group(1) if psnr else "-",
        "elapsed": elapsed,
        "rc": out.returncode,
    }


print(f"{'script':<55s} {'status':>6s} {'pearson':>10s} {'psnr':>9s} {'sec':>7s}")
print("-" * 92)
for s in SCRIPTS:
    if not os.path.exists(os.path.join(EXAMPLES, s)):
        print(f"{s:<55s} {'MISSING':>6s}")
        continue
    r = run_one(s)
    print(f"{r['script']:<55s} {r['status']:>6s} {r['pearson']:>10s} {r['psnr']:>9s} {r['elapsed']:>7.1f}")
