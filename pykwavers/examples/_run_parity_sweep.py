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
    # Thermal parity scripts
    "diff_homogeneous_medium_diffusion_compare.py",
    "diff_homogeneous_medium_source_compare.py",
    "diff_binary_sensor_mask_compare.py",
    "diff_focused_ultrasound_heating_compare.py",
    # Elastic wave parity scripts
    "ewp_shear_wave_snells_law_compare.py",
    # Axisymmetric validation
    "ivp_axisymmetric_simulation_compare.py",
    # TVSP — propagator / diffraction
    "tvsp_slit_diffraction_compare.py",
    # NA — numerical analysis characterisation
    "na_optimising_time_step_compare.py",
    "na_optimising_grid_parameters_compare.py",
    # PR — photoacoustic reconstruction
    "pr_2D_FFT_reconstruction_compare.py",
    "pr_3D_FFT_reconstruction_compare.py",
    # AT — transducer array scripts (confirmed PASS)
    "at_circular_piston_AS_compare.py",
    "at_focused_bowl_AS_compare.py",
    "at_focused_bowl_3D_compare.py",
    "at_circular_piston_3D_compare.py",
    "at_focused_annular_array_3D_compare.py",
    # TVSP 3D
    "tvsp_3D_simulation_compare.py",
    # NA — filtering and nonlinearity
    "na_filtering_part_1_compare.py",
    "na_filtering_part_2_compare.py",
    "na_filtering_part_3_compare.py",
    "na_modelling_nonlinearity_compare.py",
    # Misc — checkpointing
    "checkpointing_compare.py",
    # US — ultrasound beam patterns and phased array (velocity source scaling fix)
    "us_beam_patterns_compare.py",
    "us_bmode_phased_array_tiny_compare.py",
    # Additional AT scripts confirmed PASS
    "at_array_as_sensor_compare.py",
    "at_array_as_source_compare.py",
    "at_linear_array_transducer_compare.py",
    "at_linear_array_transducer_mask_compare.py",
    "at_focused_annular_array_3D_full_compare.py",
    "at_focused_annular_array_3D_mask_compare.py",
    "at_focused_annular_array_3D_weights_compare.py",
    # SD — directivity modelling confirmed PASS
    "sd_directional_array_elements_compare.py",
    "sd_directivity_modelling_2D_compare.py",
    "sd_directivity_modelling_3D_compare.py",
    # TVSP — doppler and steering confirmed PASS
    "tvsp_doppler_effect_compare.py",
    "tvsp_steering_linear_array_compare.py",
    # US — defining transducer and bmode (velocity source scaling fixed)
    "us_defining_transducer_compare.py",
    # us_bmode_linear_transducer: requires GPU pykwavers build (GpuPstdSession); 16 lines×27s/line
    # exceeds 300s sweep timeout. Run manually once with miniforge GPU pykwavers to seed NPZ
    # cache, then uncomment: ("us_bmode_linear_transducer_compare.py", ["--quick"]),
    # TVSP — angular spectrum propagator (pure-NumPy parity vs k-wave-python)
    "tvsp_angular_spectrum_method_compare.py",
    # TVSP — acoustic field propagator (pykwavers ASM vs RS-2 numerical integral)
    "tvsp_acoustic_field_propagator_compare.py",
    # TVSP — equivalent-source holography (backward ASM conjugate propagator)
    "tvsp_equivalent_source_holography_compare.py",
    # PR — CW angular spectrum: attenuation compensation and directional sensors
    "pr_2D_attenuation_compensation_compare.py",
    "pr_2D_TR_directional_sensors_compare.py",
    "pr_3D_TR_directional_sensors_compare.py",
    ("us_bmode_phased_array_compare.py", ["--quick", "--pykwavers-gpu"]),
]

re_status = re.compile(r"^\s*(?:parity_)?[Ss]tatus\s*:\s*(\S+)", re.M)
re_pearson = re.compile(r"^\s*pearson_r(?:_min)?\s*:\s*([-\d.eE+nan]+)", re.M)
re_psnr = re.compile(r"^\s*psnr_db(?:_min)?\s*:\s*([-\d.eE+nan]+)", re.M)
re_overall = re.compile(r"^\s*(?:Overall.*|RESULT)\s*:\s*(\S+)", re.M)


def run_one(entry) -> dict:
    if isinstance(entry, tuple):
        script, extra_args = entry
    else:
        script, extra_args = entry, []
    cmd = [sys.executable, os.path.join(EXAMPLES, script), "--allow-failure"] + extra_args
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
    script_name = s[0] if isinstance(s, tuple) else s
    if not os.path.exists(os.path.join(EXAMPLES, script_name)):
        print(f"{script_name:<55s} {'MISSING':>6s}")
        continue
    r = run_one(s)
    print(f"{r['script']:<55s} {r['status']:>6s} {r['pearson']:>10s} {r['psnr']:>9s} {r['elapsed']:>7.1f}")
