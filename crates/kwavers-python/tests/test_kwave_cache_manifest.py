"""Fast manifest checks for cached k-Wave and KWave.jl parity artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pykwavers as kw
import pytest

from parity_test_utils import (
    assert_decodable_nonblank_png,
    has_nonzero_payload,
    load_example_module,
    load_numeric_cache,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUTPUT = EXAMPLES / "output"
BOOK_EXAMPLES = EXAMPLES / "book"
DOCS_BOOK = ROOT.parents[1] / "docs" / "book"
TESTS = ROOT / "tests"
EXTERNAL_KWAVE_PYTHON_EXAMPLES = ROOT.parents[1] / "external" / "k-wave-python" / "examples"

KWAVE_ONLY_CACHE_STEMS = {
    "ivp_axisymmetric",
    "pr_2D_FFT_line_sensor",
    "pr_3D_FFT_planar_sensor",
    "pr_3D_TR_planar_sensor",
    "tvsp_slit",
    "us_bmode_phased_array_kwave_quick_seed20260401_kwgpu_pkwgpu",
    "us_defining_transducer_kwave_cache_cpu",
}

EXPLICIT_PARITY_PAIRS = {
    "sd_focussed_3D_src1": (
        "sd_focussed_3D_kwave_src1.npz",
        "sd_focussed_3D_pykwavers_src1.npz",
    ),
    "sd_focussed_3D_src2": (
        "sd_focussed_3D_kwave_src2.npz",
        "sd_focussed_3D_pykwavers_src2.npz",
    ),
    "sd_focussed_detector_2D_on_axis": (
        "sd_focussed_detector_2D_kwave_on_axis.npz",
        "sd_focussed_detector_2D_pykwavers_on_axis.npz",
    ),
    "sd_focussed_detector_2D_off_axis": (
        "sd_focussed_detector_2D_kwave_off_axis.npz",
        "sd_focussed_detector_2D_pykwavers_off_axis.npz",
    ),
    "us_bmode_phased_array_quick": (
        "us_bmode_phased_array_kwave_quick_seed20260401_kwcpu_pkwgpu.npz",
        "us_bmode_phased_array_pykwavers_quick_seed20260401_kwcpu_pkwgpu.npz",
    ),
}

DIRECTLY_TESTED_COMPARE_SCRIPTS = {
    "at_array_as_sensor_compare.py",
    "at_array_as_source_compare.py",
    "at_circular_piston_3D_compare.py",
    "at_circular_piston_AS_compare.py",
    "at_focused_bowl_3D_compare.py",
    "at_focused_bowl_AS_compare.py",
    "at_focused_annular_array_3D_compare.py",
    "at_focused_annular_array_3D_full_compare.py",
    "at_linear_array_transducer_compare.py",
    "checkpointing_compare.py",
    "compare_pr_2D_FFT_line_sensor.py",
    "compare_pr_2D_TR_line_sensor.py",
    "compare_pr_3D_FFT_planar_sensor.py",
    "compare_pr_3D_TR_planar_sensor.py",
    "diff_bioheat_1d_jl_compare.py",
    "diff_homogeneous_medium_source_2d_jl_compare.py",
    "ewp_elastic_2d_jl_compare.py",
    "ivp_1D_simulation_compare.py",
    "ivp_3D_simulation_compare.py",
    "ivp_binary_sensor_mask_compare.py",
    "ivp_heterogeneous_medium_compare.py",
    "ivp_homogeneous_medium_compare.py",
    "ivp_loading_external_image_compare.py",
    "ivp_photoacoustic_waveforms_compare.py",
    "ivp_recording_particle_velocity_compare.py",
    "ivp_saving_movie_files_compare.py",
    "na_controlling_the_pml_compare.py",
    "na_filtering_part_1_compare.py",
    "na_filtering_part_2_compare.py",
    "na_filtering_part_3_compare.py",
    "na_modelling_absorption_compare.py",
    "na_modelling_nonlinearity_compare.py",
    "na_optimising_performance_compare.py",
    "na_source_smoothing_compare.py",
    "sd_directivity_modelling_2D_compare.py",
    "sd_directivity_modelling_3D_compare.py",
    "sd_directional_array_elements_compare.py",
    "sd_focussed_detector_2D_compare.py",
    "sd_focussed_detector_3D_compare.py",
    "tvsp_3D_simulation_compare.py",
    "tvsp_doppler_effect_compare.py",
    "tvsp_homogeneous_medium_dipole_compare.py",
    "tvsp_homogeneous_medium_monopole_compare.py",
    "tvsp_snells_law_compare.py",
    "tvsp_steering_linear_array_compare.py",
    "pr_time_reversal_2d_jl_compare.py",
    "us_beam_patterns_compare.py",
    "us_beamforming_2d_jl_compare.py",
    "us_bmode_linear_transducer_compare.py",
    "us_bmode_phased_array_compare.py",
    "us_bmode_phased_array_tiny_compare.py",
    "us_defining_transducer_compare.py",
    "us_phased_array_3d_jl_compare.py",
}

KWAVE_JULIA_COMPARE_SCRIPTS = {
    "diff_bioheat_1d_jl_compare.py",
    "diff_homogeneous_medium_source_2d_jl_compare.py",
    "ewp_elastic_2d_jl_compare.py",
    "pr_time_reversal_2d_jl_compare.py",
    "us_beamforming_2d_jl_compare.py",
    "us_phased_array_3d_jl_compare.py",
}

MANIFEST_VALIDATED_COMPARE_SCRIPTS = KWAVE_JULIA_COMPARE_SCRIPTS

REFERENCE_DIAGNOSTIC_THRESHOLD_SCRIPTS = {
    "diff_homogeneous_medium_diffusion_compare.py",
    "diff_homogeneous_medium_source_compare.py",
    "ivp_opposing_corners_sensor_mask_compare.py",
    "tvsp_acoustic_field_propagator_compare.py",
    "tvsp_angular_spectrum_method_compare.py",
    "tvsp_equivalent_source_holography_compare.py",
    "tvsp_transducer_field_patterns_compare.py",
}

DASHBOARD_ONLY_ARTIFACT_SCRIPTS = {
    "cavitation_bubble_validation.py",
    "hifu_procedure_simulation.py",
    "phase_compare_minimal.py",
}

CHAPTER20_FORBIDDEN_SYNTHETIC_PARITY_TOKENS = {
    "np.random.default_rng",
    "standard_normal",
    "Stand-in kwavers",
    "pykwavers is required",
    "circular_piston_onaxis",
    "kw.pearson",
    "kw.rmse",
    "kw.psnr",
    "r = np.cos(phi_rad)",
    "np.degrees(np.arccos(",
    "PSNR_dB = -20 * np.log10(eps)",
}

CHAPTER20_REQUIRED_RUST_BINDING_CALLS = {
    "kw.phase_shift_correlation_curve",
    "kw.phase_error_degrees_for_correlation",
    "kw.validation_psnr_from_relative_rmse",
}

CHAPTER20_REQUIRED_PYKWAVERS_EXPORTS = {
    "phase_shift_correlation_curve",
    "phase_error_degrees_for_correlation",
    "validation_psnr_from_relative_rmse",
}

CHAPTER05_FORBIDDEN_PYTHON_PHYSICS_TOKENS = {
    "_HAS_PYKWAVERS",
    "from scipy.signal import hilbert",
    "fallback (no pykwavers)",
    "np.random.default_rng",
    "standard_normal",
    "_fast_sweep_eikonal",
    "_cw_spectrum",
    "def ricker",
    "np.fft.fft",
    "np.hanning",
    "np.sin(2 * np.pi * F0 * t_ax",
    "pulse[idx_in]",
    "np.linalg.solve",
    "np.floor(idx",
    "np.gradient(",
    "np.sqrt(mu_lo * 1e3 / RHO)",
    "np.sqrt(mu_hi * 1e3 / RHO)",
}

CHAPTER05_REQUIRED_RUST_BINDING_CALLS = {
    "kw.centered_hann_tone_burst_waveform",
    "kw.bmode_envelope",
    "kw.lateral_psf_sinc2",
    "kw.contrast_agent_doppler_spectrum",
    "kw.continuous_wave_vector_flow_fixture",
    "kw.gaussian_absorber_photoacoustic_profile",
    "kw.solve_rayleigh_plesset",
    "kw.shear_wave_speed",
}

CHAPTER05_REQUIRED_PYKWAVERS_EXPORTS = {
    "centered_hann_tone_burst_waveform",
    "lateral_psf_sinc2",
    "axial_psf_rect",
    "doppler_frequency_shift",
    "contrast_agent_doppler_spectrum",
    "continuous_wave_vector_flow_fixture",
    "gaussian_absorber_photoacoustic_profile",
    "pw_compounding_lateral_psf",
    "lateral_resolution_m",
    "shear_wave_speed",
}

CHAPTER05_FIGURE_STEMS = {
    "fig01_psf_profiles",
    "fig02_plane_wave_compounding",
    "fig03_doppler_spectrum",
    "fig04_photoacoustic_signal",
    "fig05_hemoglobin_spectra",
    "fig06_shear_wave_elastography",
    "fig11_cw_vector_doppler",
    "fig12_bmode_pipeline",
}

CHAPTER10_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "cylindrical stiff inclusion",
    "np.random.default_rng",
    "rng.uniform",
    "for line in range(n_lines)",
}

CHAPTER10_REQUIRED_RUST_BINDING_CALLS = {
    "kw.shear_wave_speed",
    "kw.pwave_to_swave_velocity_ratio",
    "kw.voigt_complex_modulus",
    "kw.voigt_shear_wave_dispersion",
    "kw.mre_displacement_field",
    "kw.mre_displacement_envelope",
    "kw.thermal_strain_rf_fixture",
    "kw.thermal_strain_reconstruct",
}

CHAPTER10_REQUIRED_PYKWAVERS_EXPORTS = {
    "shear_wave_speed",
    "pwave_to_swave_velocity_ratio",
    "voigt_complex_modulus",
    "voigt_shear_wave_dispersion",
    "mre_displacement_field",
    "mre_displacement_envelope",
    "thermal_strain_combined_coefficient",
    "thermal_strain_rf_fixture",
    "thermal_strain_reconstruct",
}

CHAPTER10_FIGURE_STEMS = {
    "fig01_shear_wave_speed",
    "fig02_wave_velocity_ratio",
    "fig03_voigt_viscoelastic",
    "fig04_shear_dispersion",
    "fig05_mre_displacement",
    "fig06_thermal_strain",
}

CHAPTER11_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "Whittaker-Shannon interpolation",
    "aliasing error ~",
}

CHAPTER11_REQUIRED_RUST_BINDING_CALLS = {
    "kw.circular_piston_directivity",
    "kw.focused_bowl_onaxis",
    "kw.linear_array_factor",
    "kw.delay_law_focus_2d",
    "kw.bli_stencil_weights",
    "kw.acoustic_lens_delay_profile",
    "kw.fresnel_zone_radii",
    "kw.isoplanatic_steering_curve",
}

CHAPTER11_REQUIRED_PYKWAVERS_EXPORTS = {
    "circular_piston_directivity",
    "focused_bowl_onaxis",
    "linear_array_factor",
    "delay_law_focus_2d",
    "bli_stencil_weights",
    "acoustic_lens_delay_profile",
    "fresnel_zone_radii",
    "isoplanatic_steering_curve",
}

CHAPTER11_FIGURE_STEMS = {
    "fig01_piston_directivity",
    "fig02_focused_bowl_onaxis",
    "fig03_array_beam_pattern",
    "fig04_delay_law",
    "fig05_bli_accuracy",
    "fig06_acoustic_lens",
    "fig07_lens_steering",
}

CHAPTER12_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "alpha_perf",
    "dT_max =",
}

CHAPTER12_REQUIRED_RUST_BINDING_CALLS = {
    "kw.water_sound_speed_temperature",
    "kw.tissue_properties",
    "kw.ba_parameter",
    "kw.power_law_attenuation_np_m",
    "kw.pennes_steady_state_temperature_profile",
}

CHAPTER12_REQUIRED_PYKWAVERS_EXPORTS = {
    "water_sound_speed_temperature",
    "tissue_properties",
    "ba_parameter",
    "power_law_attenuation_np_m",
    "pennes_steady_state_temperature_profile",
}

CHAPTER12_FIGURE_STEMS = {
    "fig01_sound_speed_temperature",
    "fig02_impedance_bar",
    "fig03_ba_parameter",
    "fig04_fractional_absorption",
    "fig05_bioheat",
}

CHAPTER13_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "np.random.default_rng",
    "standard_normal",
}

CHAPTER13_REQUIRED_RUST_BINDING_CALLS = {
    "kw.hbo2_molar_absorption",
    "kw.hb_molar_absorption",
    "kw.gruneisen_parameter_water",
    "kw.pa_sphere_pressure_signal",
    "kw.pa_axial_resolution",
    "kw.spectroscopic_unmixing_so2_sweep",
}

CHAPTER13_REQUIRED_PYKWAVERS_EXPORTS = {
    "hbo2_molar_absorption",
    "hb_molar_absorption",
    "gruneisen_parameter_water",
    "pa_sphere_pressure_signal",
    "pa_axial_resolution",
    "spectroscopic_unmixing_lstsq",
    "spectroscopic_unmixing_so2_sweep",
}

CHAPTER13_FIGURE_STEMS = {
    "fig01_absorption_spectra",
    "fig02_gruneisen_temperature",
    "fig03_pa_sphere_signal",
    "fig04_bandwidth_vs_radius",
    "fig05_spectroscopic_unmixing",
}

CHAPTER14_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "np.random.default_rng",
    "standard_normal",
    "sinc pattern",
}

CHAPTER14_REQUIRED_RUST_BINDING_CALLS = {
    "kw.circular_piston_directivity",
    "kw.linear_array_factor",
    "kw.plane_wave_pressure_velocity_1d",
    "kw.tissue_properties",
    "kw.pa_sphere_pressure_signal",
    "kw.add_noise",
}

CHAPTER14_REQUIRED_PYKWAVERS_EXPORTS = {
    "circular_piston_directivity",
    "linear_array_factor",
    "plane_wave_pressure_velocity_1d",
    "tissue_properties",
    "pa_sphere_pressure_signal",
    "add_noise",
}

CHAPTER14_FIGURE_STEMS = {
    "fig01_hydrophone_directivity",
    "fig02_grating_lobes",
    "fig03_pressure_velocity",
    "fig04_time_reversal",
    "fig05_signal_comparison",
}

CHAPTER17_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_PYKWAVERS",
    "except ImportError",
    "pykwavers is required",
    "[skip] fig04 requires pykwavers",
    "[skip] fig05 requires pykwavers",
    "np.random.default_rng",
    "standard_normal",
}

CHAPTER17_REQUIRED_RUST_BINDING_CALLS = {
    "kw.helmholtz_1d_fd_matrix",
    "kw.matrix_singular_values",
    "kw.tikhonov_lcurve",
    "kw.gaussian_deconvolution_fixture",
    "kw.l_curve_corner",
    "kw.exponential_convergence_curve",
    "kw.FrequencyDomainFwiConfig",
    "kw.simulate_breast_fwi_frequency_observation",
    "kw.invert_breast_fwi",
    "kw.eikonal_traveltime_2d",
    "kw.kirchhoff_point_scatterer_image_2d",
}

CHAPTER17_REQUIRED_PYKWAVERS_EXPORTS = {
    "helmholtz_1d_fd_matrix",
    "matrix_singular_values",
    "tikhonov_lcurve",
    "gaussian_deconvolution_fixture",
    "l_curve_corner",
    "exponential_convergence_curve",
    "FrequencyDomainFwiConfig",
    "simulate_breast_fwi_frequency_observation",
    "invert_breast_fwi",
    "eikonal_traveltime_2d",
    "kirchhoff_point_scatterer_image_2d",
}

CHAPTER17_FIGURE_STEMS = {
    "fig01_svd_spectrum",
    "fig02_lcurve",
    "fig03_pinn_loss",
    "fig04_convergence_comparison",
    "fig05_sound_speed_reconstruction",
    "fig06_eikonal_kirchhoff",
}

CHAPTER18_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
    "[skip fig07]",
    "Optogenetics (ChR2)",
    "energy_nJ",
}

CHAPTER18_REQUIRED_RUST_BINDING_CALLS = {
    "kw.boltzmann_open_probability_py",
    "kw.compute_acoustic_membrane_tension_py",
    "kw.pressure_threshold_open_probability_py",
    "kw.acoustic_monopole_contrast",
    "kw.acoustic_dipole_contrast",
    "kw.gorkov_radiation_force_1d",
    "kw.acoustic_streaming_velocity",
    "kw.cem43_at_temperatures",
    "kw.simulate_lif_neuron_py",
}

CHAPTER18_REQUIRED_PYKWAVERS_EXPORTS = {
    "boltzmann_open_probability_py",
    "compute_acoustic_membrane_tension_py",
    "pressure_threshold_open_probability_py",
    "acoustic_monopole_contrast",
    "acoustic_dipole_contrast",
    "gorkov_radiation_force_1d",
    "acoustic_streaming_velocity",
    "cem43_at_temperatures",
    "simulate_lif_neuron_py",
}

CHAPTER18_FIGURE_STEMS = {
    "fig01_channel_gating",
    "fig02_radiation_force",
    "fig03_streaming_shear",
    "fig04_safety_budget",
    "fig05_activation_comparison",
    "fig06_pipeline_schematic",
    "fig07_lif_raster_vs_duty",
}

CHAPTER21_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
}

CHAPTER21_REQUIRED_RUST_BINDING_CALLS = {
    "kw.solve_rayleigh_plesset",
    "kw.solve_keller_miksis",
    "kw.solve_gilmore",
}

CHAPTER21_REQUIRED_PYKWAVERS_EXPORTS = {
    "solve_rayleigh_plesset",
    "solve_keller_miksis",
    "solve_gilmore",
}

CHAPTER21_FIGURE_STEMS = {
    "fig01_bubble_ode_comparison",
}

CHAPTER34_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS = {
    "_HAS_PYKWAVERS",
    "except ImportError",
    "kw = None",
    "pykwavers is required",
}

CHAPTER34_REQUIRED_RUST_BINDING_CALLS = {
    "kw.acoustic_resolution_lateral",
    "kw.soap_focal_gain",
    "kw.numerical_aperture_from_geometry",
    "kw.f_number_from_na",
}

CHAPTER34_REQUIRED_PYKWAVERS_EXPORTS = {
    "acoustic_resolution_lateral",
    "soap_focal_gain",
    "numerical_aperture_from_geometry",
    "f_number_from_na",
}

CHAPTER34_FIGURE_STEMS = {
    "fig01_soap_resolution_gain",
}

CHAPTER20_CLOSED_GAPS_WITH_PSNR = {
    "PSTD 1-D plane wave",
    "PSTD focused bowl 3-D",
    "PSTD annular array 3-D",
    "Phased array 2-D (GPU, fund.)",
    "Phased array 2-D (GPU, harm.)",
    "B-mode scan lines (raw)",
    "Focused detector 3-D (CPU PSTD)",
    "PSTD absorption (power law)",
    "AS circular piston",
    "AS focused bowl",
}

CHAPTER20_FIGURE_STEMS = {
    "fig01_pearson_phase_sensitivity",
    "fig02_psnr_amplitude",
    "fig03_pstd_convergence",
    "fig04_side_by_side_parity",
    "fig05_validation_scatter",
}

UPSTREAM_KWAVE_PYTHON_EXAMPLE_COVERAGE = {
    "ivp_1D_simulation.py": "ivp_1D_simulation_compare.py",
    "ivp_3D_simulation.py": "ivp_3D_simulation_compare.py",
    "ivp_binary_sensor_mask.py": "ivp_binary_sensor_mask_compare.py",
    "ivp_heterogeneous_medium.py": "ivp_heterogeneous_medium_compare.py",
    "ivp_homogeneous_medium.py": "ivp_homogeneous_medium_compare.py",
    "ivp_loading_external_image.py": "ivp_loading_external_image_compare.py",
    "ivp_photoacoustic_waveforms.py": "ivp_photoacoustic_waveforms_compare.py",
    "ivp_recording_particle_velocity.py": "ivp_recording_particle_velocity_compare.py",
    "ivp_saving_movie_files.py": "ivp_saving_movie_files_compare.py",
    "legacy/at_array_as_sensor/at_array_as_sensor.py": "at_array_as_sensor_compare.py",
    "legacy/at_array_as_source/at_array_as_source.py": "at_array_as_source_compare.py",
    "legacy/at_circular_piston_3D/at_circular_piston_3D.py": "at_circular_piston_3D_compare.py",
    "legacy/at_circular_piston_AS/at_circular_piston_AS.py": "at_circular_piston_AS_compare.py",
    "legacy/at_focused_annular_array_3D/at_focused_annular_array_3D.py": "at_focused_annular_array_3D_compare.py",
    "legacy/at_focused_bowl_3D/at_focused_bowl_3D.py": "at_focused_bowl_3D_compare.py",
    "legacy/at_focused_bowl_AS/at_focused_bowl_AS.py": "at_focused_bowl_AS_compare.py",
    "legacy/at_linear_array_transducer/at_linear_array_transducer.py": "at_linear_array_transducer_compare.py",
    "legacy/checkpointing/checkpoint.py": "checkpointing_compare.py",
    "legacy/ivp_photoacoustic_waveforms/ivp_photoacoustic_waveforms.py": "ivp_photoacoustic_waveforms_compare.py",
    "legacy/pr_2D_FFT_line_sensor/pr_2D_FFT_line_sensor.py": "compare_pr_2D_FFT_line_sensor.py",
    "legacy/pr_2D_TR_line_sensor/pr_2D_TR_line_sensor.py": "compare_pr_2D_TR_line_sensor.py",
    "legacy/pr_3D_FFT_planar_sensor/pr_3D_FFT_planar_sensor.py": "compare_pr_3D_FFT_planar_sensor.py",
    "legacy/pr_3D_TR_planar_sensor/pr_3D_TR_planar_sensor.py": "compare_pr_3D_TR_planar_sensor.py",
    "legacy/sd_directivity_modelling_2D/sd_directivity_modelling_2D.py": "sd_directivity_modelling_2D_compare.py",
    "legacy/sd_focussed_detector_2D/sd_focussed_detector_2D.py": "sd_focussed_detector_2D_compare.py",
    "legacy/sd_focussed_detector_3D/sd_focussed_detector_3D.py": "sd_focussed_detector_3D_compare.py",
    "legacy/us_beam_patterns/us_beam_patterns.py": "us_beam_patterns_compare.py",
    "legacy/us_bmode_linear_transducer/us_bmode_linear_transducer.py": "us_bmode_linear_transducer_compare.py",
    "legacy/us_bmode_phased_array/us_bmode_phased_array.py": "us_bmode_phased_array_compare.py",
    "legacy/us_defining_transducer/us_defining_transducer.py": "us_defining_transducer_compare.py",
    "na_controlling_the_PML.py": "na_controlling_the_pml_compare.py",
    "na_filtering_part_1.py": "na_filtering_part_1_compare.py",
    "na_filtering_part_2.py": "na_filtering_part_2_compare.py",
    "na_filtering_part_3.py": "na_filtering_part_3_compare.py",
    "na_modelling_nonlinearity.py": "na_modelling_nonlinearity_compare.py",
    "na_optimising_performance.py": "na_optimising_performance_compare.py",
    "na_source_smoothing.py": "na_source_smoothing_compare.py",
    "pr_2D_FFT_line_sensor.py": "compare_pr_2D_FFT_line_sensor.py",
    "pr_3D_FFT_planar_sensor.py": "compare_pr_3D_FFT_planar_sensor.py",
    "sd_directional_array_elements.py": "sd_directional_array_elements_compare.py",
    "sd_directivity_modelling_2D.py": "sd_directivity_modelling_2D_compare.py",
    "sd_directivity_modelling_3D.py": "sd_directivity_modelling_3D_compare.py",
    "sd_focussed_detector_2D.py": "sd_focussed_detector_2D_compare.py",
    "sd_focussed_detector_3D.py": "sd_focussed_detector_3D_compare.py",
    "tvsp_3D_simulation.py": "tvsp_3D_simulation_compare.py",
    "tvsp_doppler_effect.py": "tvsp_doppler_effect_compare.py",
    "tvsp_homogeneous_medium_dipole.py": "tvsp_homogeneous_medium_dipole_compare.py",
    "tvsp_homogeneous_medium_monopole.py": "tvsp_homogeneous_medium_monopole_compare.py",
    "tvsp_snells_law.py": "tvsp_snells_law_compare.py",
    "tvsp_steering_linear_array.py": "tvsp_steering_linear_array_compare.py",
}

UPSTREAM_NON_STANDALONE_EXAMPLES = {
    "legacy/us_bmode_linear_transducer/example_utils.py",
}

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NONFINITE_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z])"
)

REFERENCE_OR_DIAGNOSTIC_COMPARE_SCRIPTS = {
    "at_focused_annular_array_3D_mask_compare.py",
    "at_focused_annular_array_3D_weights_compare.py",
    "at_linear_array_transducer_mask_compare.py",
    "compare_2d.py",
    "compare_all_simulators.py",
    "compare_initial_pressure.py",
    "compare_plane_wave.py",
    "compare_point_source.py",
    "compare_us_transducer.py",
    "diff_binary_sensor_mask_compare.py",
    "diff_focused_ultrasound_heating_compare.py",
    "diff_homogeneous_medium_diffusion_compare.py",
    "diff_homogeneous_medium_source_compare.py",
    "ewp_3D_simulation_compare.py",
    "ewp_fluid_and_elastic_comparison_compare.py",
    "ewp_layered_medium_compare.py",
    "ewp_plane_wave_absorption_compare.py",
    "ewp_shear_wave_snells_law_compare.py",
    "histotripsy_cavitation_compare.py",
    "ivp_axisymmetric_simulation_compare.py",
    "ivp_opposing_corners_sensor_mask_compare.py",
    "na_optimising_grid_parameters_compare.py",
    "na_optimising_time_step_compare.py",
    "passive_acoustic_mapping_compare.py",
    "pr_2D_FFT_reconstruction_compare.py",
    "pr_2D_TR_directional_sensors_compare.py",
    "pr_2D_attenuation_compensation_compare.py",
    "pr_3D_FFT_reconstruction_compare.py",
    "pr_3D_TR_directional_sensors_compare.py",
    "tvsp_acoustic_field_propagator_compare.py",
    "tvsp_angular_spectrum_method_compare.py",
    "tvsp_equivalent_source_holography_compare.py",
    "tvsp_slit_diffraction_compare.py",
    "tvsp_transducer_field_patterns_compare.py",
}


def _standard_pairs() -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    unclassified: list[str] = []

    for kwave_path in sorted(OUTPUT.glob("*_kwave_cache.npz")):
        stem = kwave_path.name.removesuffix("_kwave_cache.npz")
        pykwavers_path = OUTPUT / f"{stem}_pykwavers_cache.npz"
        if pykwavers_path.exists():
            pairs[stem] = (kwave_path, pykwavers_path)
        elif stem not in KWAVE_ONLY_CACHE_STEMS:
            unclassified.append(kwave_path.name)

    assert unclassified == [], (
        "Every k-Wave cache must either have a pykwavers peer or be explicitly "
        f"classified as reference-only; unclassified={unclassified}"
    )
    return pairs


def _compare_scripts() -> set[str]:
    scripts = {path.name for path in EXAMPLES.glob("*_compare.py")}
    scripts.update(path.name for path in EXAMPLES.glob("compare_*.py"))
    return scripts


def _scripts_with_parity_thresholds() -> set[str]:
    return {
        path.name
        for path in EXAMPLES.glob("*.py")
        if "PARITY_THRESHOLDS" in path.read_text(
            encoding="utf-8", errors="ignore"
        )
    }


def _vendored_kwave_python_examples() -> set[str]:
    return {
        path.relative_to(EXTERNAL_KWAVE_PYTHON_EXAMPLES).as_posix()
        for path in EXTERNAL_KWAVE_PYTHON_EXAMPLES.rglob("*.py")
    }


def _standard_cache_stem_for_script(script: str) -> str:
    return script.removesuffix("_compare.py").removesuffix(".py")


def _kwave_test_sources(*, include_manifest: bool) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(TESTS.glob("test_kwave*.py"))
        if include_manifest or path.name != Path(__file__).name
    )


def _chapter20_figure_index_stems() -> set[str]:
    text = (DOCS_BOOK / "validation_and_benchmarking.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    stems: set[str] = set()
    in_index = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "## 19.12 Figure Index":
            in_index = True
            continue
        if in_index and stripped.startswith("## "):
            break
        if not in_index or not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("19."):
            stems.add(cells[3])

    return stems


def _coverage_reference_tokens(script: str) -> tuple[str, str]:
    compare_stem = script.removesuffix(".py")
    example_stem = script.removesuffix("_compare.py").removesuffix(".py")
    return compare_stem, example_stem


def _finite_json_numbers(value) -> list[float]:
    if isinstance(value, dict):
        numbers: list[float] = []
        for child in value.values():
            numbers.extend(_finite_json_numbers(child))
        return numbers
    if isinstance(value, list):
        numbers = []
        for child in value:
            numbers.extend(_finite_json_numbers(child))
        return numbers
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def _load_csv_numeric(path: Path) -> np.ndarray:
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    skip_header = 1 if any(char.isalpha() for char in first_line) else 0
    return np.atleast_1d(np.loadtxt(path, delimiter=",", skiprows=skip_header)).astype(np.float64)


def _metric_value(metrics_text: str, label: str) -> float:
    match = re.search(
        rf"(?m)^\s*{re.escape(label)}\s*[:=]\s*({_FLOAT_PATTERN})",
        metrics_text,
    )
    assert match is not None, label
    return float(match.group(1))


def _metric_value_anywhere(metrics_text: str, label: str) -> float:
    match = re.search(
        rf"(?m)\b{re.escape(label)}\s*[:=]\s*({_FLOAT_PATTERN})",
        metrics_text,
    )
    assert match is not None, label
    return float(match.group(1))


def _inline_metric_value(metrics_text: str, label: str, metric: str) -> float:
    match = re.search(
        rf"(?m)^\s*{re.escape(label)}\s*:\s*.*?\b{re.escape(metric)}=({_FLOAT_PATTERN})",
        metrics_text,
    )
    assert match is not None, (label, metric)
    return float(match.group(1))


def _metric_range(metrics_text: str, label: str) -> tuple[float, float]:
    match = re.search(
        rf"(?m)^\s*{re.escape(label)}\s*:\s*\[({_FLOAT_PATTERN}),\s*({_FLOAT_PATTERN})\]",
        metrics_text,
    )
    assert match is not None, label
    return float(match.group(1)), float(match.group(2))


def _assert_result_pass(metrics_text: str, script: str):
    match = re.search(r"(?m)^\s*RESULT\s*:\s*(\S+)", metrics_text)
    assert match is not None, script
    assert match.group(1) == "PASS", script


def _assert_kwave_julia_metric_thresholds(script: str, metrics_text: str):
    thresholds = load_example_module(script).PARITY_THRESHOLDS

    if script == "diff_bioheat_1d_jl_compare.py":
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["pearson_r"]
        assert thresholds["rms_ratio_min"] <= _metric_value(metrics_text, "rms_ratio")
        assert _metric_value(metrics_text, "rms_ratio") <= thresholds["rms_ratio_max"]
        assert _metric_value(metrics_text, "psnr_db") >= thresholds["psnr_db"]
        assert _metric_value(metrics_text, "linf_C") <= thresholds["linf_max_C"]
    elif script == "diff_homogeneous_medium_source_2d_jl_compare.py":
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["pearson_r"]
        assert thresholds["rms_ratio_min"] <= _metric_value(metrics_text, "rms_ratio")
        assert _metric_value(metrics_text, "rms_ratio") <= thresholds["rms_ratio_max"]
        assert _metric_value(metrics_text, "psnr_db") >= thresholds["psnr_db"]
    elif script == "ewp_elastic_2d_jl_compare.py":
        peak_min, peak_max = _metric_range(metrics_text, "peak ratios")
        assert thresholds["peak_ratio_min"] <= peak_min
        assert peak_max <= thresholds["peak_ratio_max"]
    elif script == "pr_time_reversal_2d_jl_compare.py":
        peak_min, peak_max = thresholds["peak_recon_vs_recon"]
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["r_recon_vs_recon"]
        assert peak_min <= _metric_value(metrics_text, "peak_ratio")
        assert _metric_value(metrics_text, "peak_ratio") <= peak_max
        assert _inline_metric_value(metrics_text, "KWave.jl", "r") >= thresholds["r_recon_vs_p0"]
        assert _inline_metric_value(metrics_text, "pykwavers", "r") >= thresholds["r_recon_vs_p0"]
    elif script == "us_beamforming_2d_jl_compare.py":
        assert _metric_value(metrics_text, "offset (cells, max)") <= thresholds["peak_loc_offset_cells"]
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["pearson_r_min"]
    elif script == "us_phased_array_3d_jl_compare.py":
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["pearson_r"]
        assert thresholds["peak_ratio_min"] <= _metric_value(metrics_text, "peak_ratio")
        assert _metric_value(metrics_text, "peak_ratio") <= thresholds["peak_ratio_max"]
    else:
        raise AssertionError(f"unhandled KWave.jl metric contract: {script}")


def _assert_reference_diagnostic_metric_thresholds(script: str, metrics_text: str):
    thresholds = load_example_module(script).PARITY_THRESHOLDS

    if script == "tvsp_angular_spectrum_method_compare.py":
        assert (
            _metric_value_anywhere(metrics_text, "pearson_r_min")
            >= thresholds["pearson_r_min"]
        )
        assert (
            _metric_value_anywhere(metrics_text, "psnr_db_min")
            >= thresholds["psnr_db_min"]
        )
    elif script in {
        "diff_homogeneous_medium_diffusion_compare.py",
        "diff_homogeneous_medium_source_compare.py",
        "ivp_opposing_corners_sensor_mask_compare.py",
        "tvsp_acoustic_field_propagator_compare.py",
        "tvsp_equivalent_source_holography_compare.py",
    }:
        assert _metric_value(metrics_text, "pearson_r") >= thresholds["pearson_r"]
        if "rms_ratio_min" in thresholds:
            assert thresholds["rms_ratio_min"] <= _metric_value(metrics_text, "rms_ratio")
            assert _metric_value(metrics_text, "rms_ratio") <= thresholds["rms_ratio_max"]
        assert _metric_value(metrics_text, "psnr_db") >= thresholds["psnr_db"]
    elif script == "tvsp_transducer_field_patterns_compare.py":
        for field in ("p_max", "p_rms", "p_final"):
            assert _inline_metric_value(metrics_text, field, "r") >= thresholds["pearson_r"]
            rms_ratio = _inline_metric_value(metrics_text, field, "rms_ratio")
            assert thresholds["rms_ratio_min"] <= rms_ratio <= thresholds["rms_ratio_max"]
            assert _inline_metric_value(metrics_text, field, "PSNR") >= thresholds["psnr_db"]
    else:
        raise AssertionError(f"unhandled reference diagnostic contract: {script}")


def test_compare_script_manifest_classifies_all_drivers():
    scripts = _compare_scripts()
    classified = DIRECTLY_TESTED_COMPARE_SCRIPTS | REFERENCE_OR_DIAGNOSTIC_COMPARE_SCRIPTS

    assert DIRECTLY_TESTED_COMPARE_SCRIPTS.isdisjoint(REFERENCE_OR_DIAGNOSTIC_COMPARE_SCRIPTS)
    assert scripts - classified == set(), f"unclassified compare scripts: {sorted(scripts - classified)}"
    assert classified - scripts == set(), f"stale manifest entries: {sorted(classified - scripts)}"
    assert len(DIRECTLY_TESTED_COMPARE_SCRIPTS) >= 20
    assert "diff_homogeneous_medium_source_2d_jl_compare.py" in classified
    assert "us_phased_array_3d_jl_compare.py" in classified


def test_directly_tested_compare_scripts_have_pytest_references():
    test_sources = _kwave_test_sources(include_manifest=False)

    unreferenced = [
        script
        for script in sorted(DIRECTLY_TESTED_COMPARE_SCRIPTS)
        if script not in MANIFEST_VALIDATED_COMPARE_SCRIPTS
        and not any(token in test_sources for token in _coverage_reference_tokens(script))
    ]
    assert unreferenced == []


def test_reference_diagnostic_scripts_do_not_hide_standard_cache_pairs():
    hidden_pairs = [
        script
        for script in sorted(REFERENCE_OR_DIAGNOSTIC_COMPARE_SCRIPTS)
        if _standard_cache_stem_for_script(script) in _standard_pairs()
    ]
    assert hidden_pairs == []


def test_reference_diagnostic_metric_reports_match_driver_thresholds():
    reference_threshold_scripts = (
        REFERENCE_OR_DIAGNOSTIC_COMPARE_SCRIPTS & _scripts_with_parity_thresholds()
    )
    assert REFERENCE_DIAGNOSTIC_THRESHOLD_SCRIPTS == reference_threshold_scripts

    for script in sorted(REFERENCE_DIAGNOSTIC_THRESHOLD_SCRIPTS):
        stem = script.removesuffix("_compare.py")
        metrics_path = OUTPUT / f"{stem}_metrics.txt"
        figure_path = OUTPUT / f"{stem}_compare.png"

        assert metrics_path.exists(), script
        assert figure_path.exists(), script

        metrics_text = metrics_path.read_text(encoding="utf-8", errors="ignore")
        assert (
            "parity_status: PASS" in metrics_text
            or re.search(r"(?m)^\s*Status\s*:\s*PASS\b", metrics_text)
        ), script
        _assert_reference_diagnostic_metric_thresholds(script, metrics_text)
        assert_decodable_nonblank_png(figure_path)


def test_chapter20_validation_figures_use_cached_parity_artifacts():
    module = load_example_module("book/ch20_validation_and_benchmarking.py")
    source = (BOOK_EXAMPLES / "ch20_validation_and_benchmarking.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER20_FORBIDDEN_SYNTHETIC_PARITY_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER20_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER20_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "at_focused_bowl_AS_compare.png" in source
    assert "plt.imread(source_path)" in source

    parsed_gaps = {
        name
        for name, _pearson, _psnr in module.closed_validation_gaps_with_psnr()
    }
    assert parsed_gaps == CHAPTER20_CLOSED_GAPS_WITH_PSNR

    cached_figure = OUTPUT / "at_focused_bowl_AS_compare.png"
    chapter_figure = DOCS_BOOK / "figures" / "ch20" / "fig04_side_by_side_parity.png"
    assert_decodable_nonblank_png(cached_figure)
    assert_decodable_nonblank_png(chapter_figure)


def test_chapter20_phase_sensitivity_bindings_match_theorem():
    phase = np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=np.float64)

    observed = np.asarray(kw.phase_shift_correlation_curve(phase))

    np.testing.assert_allclose(observed, np.cos(phase), rtol=0.0, atol=1.0e-15)
    assert kw.phase_error_degrees_for_correlation(0.99) == pytest.approx(
        np.degrees(np.arccos(0.99)),
        abs=1.0e-15,
    )
    with pytest.raises(ValueError, match="correlation must be in"):
        kw.phase_error_degrees_for_correlation(1.01)


def test_chapter20_relative_rmse_psnr_binding_matches_definition():
    relative_rmse = np.array([1.0, 0.1, 0.01, 0.001], dtype=np.float64)

    observed = np.asarray(kw.validation_psnr_from_relative_rmse(relative_rmse))

    np.testing.assert_allclose(observed, [0.0, 20.0, 40.0, 60.0], rtol=0.0, atol=1.0e-12)
    with pytest.raises(ValueError, match="relative_rmse\\[0\\]"):
        kw.validation_psnr_from_relative_rmse(np.array([0.0], dtype=np.float64))


def test_chapter05_diagnostic_figures_use_rust_imaging_bindings():
    source = (BOOK_EXAMPLES / "ch05_ultrasound_imaging.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER05_FORBIDDEN_PYTHON_PHYSICS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER05_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER05_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []


def test_chapter05_tone_burst_waveform_matches_centered_hann_pulse_contract():
    sample_rate_hz = 40.0e6
    frequency_hz = 5.0e6
    n_cycles = 2
    pulse_duration_s = n_cycles / frequency_hz
    time_s = np.arange(-3.0e-6, 3.0e-6, 1.0 / sample_rate_hz)

    pulse = np.asarray(kw.centered_hann_tone_burst_waveform(time_s, 1.0, frequency_hz, n_cycles))

    idx_in = np.abs(time_s) < pulse_duration_s / 2.0
    expected = np.zeros_like(time_s)
    expected[idx_in] = (
        np.sin(2.0 * np.pi * frequency_hz * time_s[idx_in])
        * np.hanning(np.sum(idx_in))
    )

    np.testing.assert_allclose(pulse, expected, rtol=0.0, atol=1.0e-14)


def test_chapter05_contrast_agent_doppler_binding_matches_kasai_velocity():
    spectrum = kw.contrast_agent_doppler_spectrum(
        128,
        4,
        10_000.0,
        0.3,
        np.deg2rad(60.0),
        5.0e6,
        1540.0,
        0.02,
    )

    assert np.asarray(spectrum["slow_time_s"]).shape == (128,)
    assert np.asarray(spectrum["iq_real"]).shape == (128,)
    assert np.asarray(spectrum["iq_imag"]).shape == (128,)
    assert np.asarray(spectrum["velocity_m_s"]).shape == (512,)
    assert np.asarray(spectrum["power"]).shape == (512,)
    assert np.isclose(float(spectrum["estimated_velocity_m_s"]), 0.3, rtol=0.0, atol=1.0e-12)
    assert np.isclose(float(spectrum["nyquist_velocity_m_s"]), 1.54, rtol=0.0, atol=1.0e-12)


def test_chapter05_photoacoustic_profile_binding_matches_closed_form():
    depth = np.array([0.019, 0.020, 0.021], dtype=np.float64)
    sound_speed = 1540.0
    time = depth / sound_speed
    fixture = kw.gaussian_absorber_photoacoustic_profile(
        depth,
        time,
        0.18,
        100.0,
        0.02,
        0.020,
        0.001,
        sound_speed,
    )

    pressure = np.asarray(fixture["initial_pressure_pa"])
    signal = np.asarray(fixture["surface_signal_pa_per_m"])
    p0 = 0.18 * 100.0 * 0.02
    expected_pressure = p0 * np.exp(-0.5 * ((depth - 0.020) / 0.001) ** 2)
    expected_signal = -((depth - 0.020) / (0.001**2)) * expected_pressure

    assert np.allclose(pressure, expected_pressure, rtol=1.0e-14, atol=1.0e-15)
    assert np.allclose(signal, expected_signal, rtol=1.0e-14, atol=1.0e-9)
    assert signal[0] > 0.0
    assert signal[1] == pytest.approx(0.0, abs=1.0e-10)
    assert signal[2] < 0.0


def test_chapter05_shear_wave_speed_binding_matches_closed_form():
    density_kg_m3 = 1060.0

    for shear_modulus_kpa in (0.5, 0.8, 2.0, 5.0, 15.0, 80.0):
        shear_modulus_pa = shear_modulus_kpa * 1e3
        expected_speed_m_s = np.sqrt(shear_modulus_pa / density_kg_m3)

        observed_speed_m_s = kw.shear_wave_speed(shear_modulus_pa, density_kg_m3)

        assert observed_speed_m_s == pytest.approx(
            expected_speed_m_s,
            rel=1.0e-15,
            abs=0.0,
        )


def test_chapter05_cw_vector_flow_fixture_uses_rust_kernels():
    fixture = kw.continuous_wave_vector_flow_fixture()

    velocity = np.asarray(fixture["cw_velocity_m_s"])
    power = np.asarray(fixture["cw_power"])
    peak_velocity = float(velocity[np.argmax(power)])
    bin_width = float(abs(velocity[1] - velocity[0]))

    assert velocity.shape == (2048,)
    assert power.shape == (2048,)
    assert np.asarray(fixture["beam_directions"]).shape == (2, 2)
    assert np.asarray(fixture["projected_velocity_m_s"]).shape == (2,)
    assert abs(peak_velocity - 2.0) < 3.0 * bin_width
    assert 2.0 > float(fixture["pulsed_wave_nyquist_velocity_m_s"])
    assert np.allclose(
        np.asarray(fixture["recovered_velocity_m_s"]),
        np.asarray(fixture["true_velocity_m_s"]),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert float(fixture["vector_error_m_s"]) < 1.0e-12


def test_chapter05_diagnostic_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER05_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch05" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch05" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter10_elastography_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch10_elastography.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER10_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER10_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER10_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "kw.mre_displacement_field" in source
    assert "np.exp(-z / penetration_depth_m)" not in source
    assert "stiff cylindrical inclusion" not in (
        DOCS_BOOK / "elastography.md"
    ).read_text(encoding="utf-8", errors="ignore")


def test_chapter10_mre_displacement_envelope_binding_matches_decay():
    import pykwavers as kw

    z = np.array([0.0, 0.035, 0.070], dtype=np.float64)
    envelope = np.asarray(kw.mre_displacement_envelope(z, 25.0e-6, 0.035))
    expected = 25.0e-6 * np.exp(-z / 0.035)

    np.testing.assert_allclose(envelope, expected, rtol=0.0, atol=1.0e-20)
    assert envelope[0] == 25.0e-6
    assert envelope[2] < envelope[1]


def test_chapter10_thermal_strain_rf_fixture_binding_is_seeded_and_warped():
    import pykwavers as kw

    reference_a, tracked_a = kw.thermal_strain_rf_fixture(
        3, 32, -1.0e-3, 6.0, 5.0, 2024
    )
    reference_b, tracked_b = kw.thermal_strain_rf_fixture(
        3, 32, -1.0e-3, 6.0, 5.0, 2024
    )
    reference_c, _tracked_c = kw.thermal_strain_rf_fixture(
        3, 32, -1.0e-3, 6.0, 5.0, 2025
    )

    ref_a = np.asarray(reference_a)
    trk_a = np.asarray(tracked_a)
    assert ref_a.shape == (3, 1, 32)
    assert trk_a.shape == (3, 1, 32)
    assert np.array_equal(ref_a, np.asarray(reference_b))
    assert np.array_equal(trk_a, np.asarray(tracked_b))
    assert not np.array_equal(ref_a, np.asarray(reference_c))
    assert not np.array_equal(ref_a, trk_a)


def test_chapter10_thermal_strain_rf_fixture_zero_shift_preserves_reference():
    import pykwavers as kw

    reference, tracked = kw.thermal_strain_rf_fixture(
        2, 24, -1.0e-3, 0.0, 5.0, 7
    )

    assert np.array_equal(np.asarray(reference), np.asarray(tracked))


def test_chapter10_elastography_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER10_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch10" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch10" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter11_sources_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch11_sources_and_transducers.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER11_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER11_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER11_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "kw.bli_stencil_weights" in source
    assert "Rust `kw.bli_stencil_weights`" in (
        DOCS_BOOK / "sources_and_transducers.md"
    ).read_text(encoding="utf-8", errors="ignore")


def test_chapter11_sources_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER11_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch11" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch11" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter12_media_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch12_media_and_tissue_models.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER12_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER12_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER12_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "kw.pennes_steady_state_temperature_profile" in source
    assert "Rust `kw.pennes_steady_state_temperature_profile`" in (
        DOCS_BOOK / "media_and_tissue_models.md"
    ).read_text(encoding="utf-8", errors="ignore")


def test_chapter12_media_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER12_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch12" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch12" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter13_photoacoustic_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch13_photoacoustics.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER13_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER13_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER13_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "deterministic measurement perturbations" in source
    assert "deterministic measurement perturbations" in (
        DOCS_BOOK / "photoacoustics.md"
    ).read_text(encoding="utf-8", errors="ignore")


def test_chapter13_spectroscopic_unmixing_sweep_preserves_unperturbed_so2():
    sweep = kw.spectroscopic_unmixing_so2_sweep(
        np.array([760.0, 850.0]),
        0.0,
        1.0,
        11,
        np.array([0.0, 0.02]),
    )

    truth = np.asarray(sweep["true_so2"])
    estimates = np.asarray(sweep["estimated_so2_by_perturbation"])
    assert truth.shape == (11,)
    assert estimates.shape == (2, 11)
    np.testing.assert_allclose(estimates[0], truth, rtol=0.0, atol=1.0e-10)


def test_chapter13_photoacoustic_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER13_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch13" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch13" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter14_sensor_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch14_sensors_and_measurements.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER14_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER14_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER14_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "kw.add_noise" in source
    assert "Rust `kw.add_noise`" in (
        DOCS_BOOK / "sensors_and_measurements.md"
    ).read_text(encoding="utf-8", errors="ignore")


def test_chapter14_sensor_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER14_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch14" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch14" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter17_inverse_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch17_inverse_problems_and_pinns.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    book_text = (DOCS_BOOK / "inverse_problems_and_pinns.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER17_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER17_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER17_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "np.exp(-0.5 * ((t[:, None] - t[None, :]) ** 2)" not in source
    assert "np.sin(2.0 * np.pi * 7.0 * t)" not in source
    assert "return L0 * np.exp(-eps / tau) + floor" not in source
    assert "Rust `kw.helmholtz_1d_fd_matrix`" in book_text
    assert "Rust `kw.eikonal_traveltime_2d`" in book_text
    assert "Rust `kw.kirchhoff_point_scatterer_image_2d`" in book_text
    assert "deterministic measurement perturbation" in book_text


def test_chapter17_gaussian_deconvolution_fixture_binding_matches_formula():
    import pykwavers as kw

    matrix, truth, measurement = kw.gaussian_deconvolution_fixture(5, 0.05, 0.01)
    matrix = np.asarray(matrix)
    truth = np.asarray(truth)
    measurement = np.asarray(measurement)

    t = np.linspace(0.0, 1.0, 5)
    expected_matrix = np.exp(-0.5 * ((t[:, None] - t[None, :]) ** 2) / 0.05**2) / (
        0.05 * np.sqrt(2.0 * np.pi)
    )
    expected_matrix /= 5.0
    expected_truth = np.exp(-0.5 * ((t - 0.3) ** 2) / 0.01) + 0.7 * np.exp(
        -0.5 * ((t - 0.7) ** 2) / 0.01
    )
    expected_measurement = expected_matrix @ expected_truth + 0.01 * (
        np.sin(2.0 * np.pi * 7.0 * t) + 0.5 * np.cos(2.0 * np.pi * 11.0 * t)
    )

    np.testing.assert_allclose(matrix, expected_matrix, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(truth, expected_truth, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(measurement, expected_measurement, rtol=0.0, atol=1.0e-15)


def test_chapter17_exponential_convergence_binding_matches_formula():
    import pykwavers as kw

    epochs = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    curve = np.asarray(kw.exponential_convergence_curve(epochs, 0.5, 2.0, 1.0e-4))
    expected = 0.5 * np.exp(-epochs / 2.0) + 1.0e-4

    np.testing.assert_allclose(curve, expected, rtol=0.0, atol=1.0e-15)
    assert curve[0] == 0.5001
    assert curve[2] < curve[1]


def test_chapter17_inverse_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER17_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch17" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch17" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter18_sonogenetics_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch18_sonogenetics.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    book_text = (DOCS_BOOK / "sonogenetics.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER18_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER18_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER18_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "Rust `kw.pressure_threshold_open_probability_py`" in book_text
    assert "Gorkov/Yosioka-Kawasima cell force" in book_text


def test_chapter18_sonogenetics_figure_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER18_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch18" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch18" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter21_simulation_orchestration_uses_rust_bindings():
    source = (BOOK_EXAMPLES / "ch21_simulation_orchestration.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    book_text = (DOCS_BOOK / "simulation_orchestration.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER21_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER21_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER21_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "No ODE is reimplemented in Python" in book_text
    assert "kw.solve_rayleigh_plesset" in book_text
    assert "kw.solve_keller_miksis" in book_text
    assert "kw.solve_gilmore" in book_text


def test_chapter21_simulation_orchestration_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER21_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch21sim" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch21sim" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_chapter34_optoacoustic_figures_use_rust_bindings():
    source = (BOOK_EXAMPLES / "ch34_optoacoustic_focused_ultrasound.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    package_source = (ROOT / "python" / "pykwavers" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    book_text = (DOCS_BOOK / "optoacoustic_focused_ultrasound.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    forbidden = [
        token
        for token in sorted(CHAPTER34_FORBIDDEN_OPTIONAL_PYKWAVERS_TOKENS)
        if token in source
    ]
    missing = [
        token
        for token in sorted(CHAPTER34_REQUIRED_RUST_BINDING_CALLS)
        if token not in source
    ]
    missing_exports = [
        token
        for token in sorted(CHAPTER34_REQUIRED_PYKWAVERS_EXPORTS)
        if token not in package_source
    ]

    assert forbidden == []
    assert missing == []
    assert missing_exports == []
    assert "kw.acoustic_resolution_lateral" in book_text
    assert "kw.soap_focal_gain" in book_text
    assert "single source of truth for Eqs. 34.4" in book_text
    assert "34.5" in book_text


def test_chapter34_optoacoustic_artifacts_exist_and_decode():
    for stem in sorted(CHAPTER34_FIGURE_STEMS):
        png_path = DOCS_BOOK / "figures" / "ch34" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch34" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_validation_chapter_uses_current_python_parity_commands():
    text = (DOCS_BOOK / "validation_and_benchmarking.md").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "cd pykwavers" not in text
    assert "pytest examples/" not in text
    assert "crates/kwavers-python/tests/test_kwave_cache_manifest.py" in text
    assert 'pytest crates/kwavers-python/tests -q -k "kwave_example"' in text


def test_chapter20_figure_index_artifacts_exist_and_decode():
    figure_stems = _chapter20_figure_index_stems()

    assert figure_stems == CHAPTER20_FIGURE_STEMS
    for stem in sorted(figure_stems):
        png_path = DOCS_BOOK / "figures" / "ch20" / f"{stem}.png"
        pdf_path = DOCS_BOOK / "figures" / "ch20" / f"{stem}.pdf"

        assert_decodable_nonblank_png(png_path)
        assert pdf_path.exists(), stem
        assert pdf_path.stat().st_size > 1024, stem


def test_dashboard_source_manifest_classifies_non_compare_artifacts():
    dashboard = load_example_module("parity_dashboard.py")
    records = dashboard.collect_records()
    dashboard_scripts = {record.script for record in records}
    compare_scripts = _compare_scripts()

    assert DASHBOARD_ONLY_ARTIFACT_SCRIPTS <= dashboard_scripts
    assert DASHBOARD_ONLY_ARTIFACT_SCRIPTS.isdisjoint(compare_scripts)
    assert dashboard_scripts - compare_scripts == DASHBOARD_ONLY_ARTIFACT_SCRIPTS


def test_vendored_kwave_python_example_manifest_classifies_all_sources():
    upstream_examples = _vendored_kwave_python_examples()
    classified = (
        set(UPSTREAM_KWAVE_PYTHON_EXAMPLE_COVERAGE)
        | UPSTREAM_NON_STANDALONE_EXAMPLES
    )
    dashboard = load_example_module("parity_dashboard.py")
    covered_scripts = _compare_scripts() | {
        record.script for record in dashboard.collect_records()
    }

    assert upstream_examples
    assert set(UPSTREAM_KWAVE_PYTHON_EXAMPLE_COVERAGE).isdisjoint(
        UPSTREAM_NON_STANDALONE_EXAMPLES
    )
    assert upstream_examples - classified == set(), (
        f"unclassified vendored k-wave-python examples: "
        f"{sorted(upstream_examples - classified)}"
    )
    assert classified - upstream_examples == set(), (
        f"stale vendored k-wave-python manifest entries: "
        f"{sorted(classified - upstream_examples)}"
    )

    for upstream_script, local_script in sorted(
        UPSTREAM_KWAVE_PYTHON_EXAMPLE_COVERAGE.items()
    ):
        assert local_script in covered_scripts, (upstream_script, local_script)
        assert (EXAMPLES / local_script).exists(), (upstream_script, local_script)


def test_kwave_julia_artifacts_are_classified_finite_and_passing():
    scripts = {path.name for path in EXAMPLES.glob("*_jl_compare.py")}

    assert scripts == KWAVE_JULIA_COMPARE_SCRIPTS
    assert KWAVE_JULIA_COMPARE_SCRIPTS <= DIRECTLY_TESTED_COMPARE_SCRIPTS

    for script in sorted(KWAVE_JULIA_COMPARE_SCRIPTS):
        stem = script.removesuffix("_compare.py")
        metrics_path = OUTPUT / f"{stem}_metrics.txt"
        meta_path = OUTPUT / f"{stem}_meta.json"
        figure_path = OUTPUT / f"{stem}_compare.png"

        assert metrics_path.exists(), script
        assert meta_path.exists(), script
        assert figure_path.exists(), script

        metrics_text = metrics_path.read_text(encoding="utf-8", errors="ignore")
        _assert_result_pass(metrics_text, script)
        _assert_kwave_julia_metric_thresholds(script, metrics_text)
        assert_decodable_nonblank_png(figure_path)

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata_numbers = _finite_json_numbers(metadata)
        assert metadata_numbers, script
        assert np.all(np.isfinite(metadata_numbers)), script

        numeric_payloads = []
        for csv_path in sorted(OUTPUT.glob(f"{stem}_*.csv")):
            array = _load_csv_numeric(csv_path)
            numeric_payloads.append(array)
            assert array.size > 0, csv_path.name
            assert np.all(np.isfinite(array)), csv_path.name

        npy_path = OUTPUT / f"{stem}_kwave_cache.npy"
        if npy_path.exists():
            array = np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.float64)
            numeric_payloads.append(array)
            assert array.size > 0, npy_path.name
            assert np.all(np.isfinite(array)), npy_path.name

        assert numeric_payloads, script
        assert any(float(np.max(np.abs(array))) > 0.0 for array in numeric_payloads), script


def test_kwave_cache_manifest_classifies_all_reference_outputs():
    pairs = _standard_pairs()
    pair_names = set(pairs)
    pair_names.update(EXPLICIT_PARITY_PAIRS)

    assert len(pair_names) >= 40
    assert "ivp_1D" in pair_names
    assert "na_optimising_performance" in pair_names
    assert "at_focused_bowl_3D" in pair_names
    assert "us_bmode_phased_array_quick" in pair_names


def test_paired_kwave_pykwavers_caches_have_finite_numeric_payloads():
    pairs = _standard_pairs()
    pairs.update(
        {
            name: (OUTPUT / kwave_name, OUTPUT / pykwavers_name)
            for name, (kwave_name, pykwavers_name) in EXPLICIT_PARITY_PAIRS.items()
        }
    )

    for name, (kwave_path, pykwavers_path) in pairs.items():
        assert kwave_path.exists(), name
        assert pykwavers_path.exists(), name

        kwave_arrays = load_numeric_cache(kwave_path)
        pykwavers_arrays = load_numeric_cache(pykwavers_path)

        assert kwave_arrays, name
        assert pykwavers_arrays, name
        assert has_nonzero_payload(kwave_arrays), name
        assert has_nonzero_payload(pykwavers_arrays), name

        common_keys = sorted(set(kwave_arrays) & set(pykwavers_arrays))
        assert common_keys, name
        for key in common_keys:
            if key == "runtime_s":
                continue
            kwave_array = kwave_arrays[key]
            pykwavers_array = pykwavers_arrays[key]
            assert np.all(np.isfinite(kwave_array)), (name, key, "kwave")
            assert np.all(np.isfinite(pykwavers_array)), (name, key, "pykwavers")


def test_kwave_julia_diffusion_cache_is_classified_and_finite():
    cache_path = OUTPUT / "diff_homogeneous_medium_source_2d_jl_kwave_cache.npy"
    meta_path = OUTPUT / "diff_homogeneous_medium_source_2d_jl_kwave_cache_meta.json"

    assert cache_path.exists()
    assert meta_path.exists()

    field = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float64)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["engine"] == "KWave.jl/kwave_diffusion_2D"
    assert field.shape == (metadata["nx"], metadata["ny"])
    assert np.all(np.isfinite(field))
    assert float(field.max()) > float(metadata["initial_temperature"])
    assert float(field.max()) == metadata["T_max_final_C"]


def test_parity_dashboard_records_match_current_example_sources():
    dashboard = load_example_module("parity_dashboard.py")
    records = dashboard.collect_records()
    orphan_metrics = dashboard.orphan_metric_files()
    metric_only_scripts = dashboard.scripts_without_comparison_png(records)
    missing_declared_pngs = dashboard.missing_declared_pngs(records)
    scripts = {record.script for record in records}
    metric_files = sorted(path.name for path in OUTPUT.glob("*_metrics.txt"))

    assert records
    assert all((EXAMPLES / script).exists() for script in scripts)
    assert all(record.status == "PASS" for record in records)
    assert sorted(orphan_metrics) == [
        "canonical_ivp_metrics.txt",
        "canonical_point_source_metrics.txt",
    ]
    assert metric_only_scripts == []
    assert missing_declared_pngs == []
    assert len(records) + len(orphan_metrics) == len(metric_files)
    assert "canonical_ivp_compare.py" not in scripts
    assert "canonical_point_source_compare.py" not in scripts
    assert "compare_initial_pressure.py" in scripts
    assert "ivp_axisymmetric_simulation_compare.py" in scripts
    assert "hifu_procedure_simulation.py" in scripts
    assert "cavitation_bubble_validation.py" in scripts

    dashboard_md = OUTPUT / "parity_dashboard.md"
    dashboard_png = OUTPUT / "parity_dashboard.png"
    assert dashboard_md.exists()
    assert dashboard_png.exists()
    markdown = dashboard_md.read_text(encoding="utf-8")
    assert f"**Total scripts:** {len(records)}" in markdown
    assert "## Skipped orphan metric files" in markdown
    for name in orphan_metrics:
        assert f"`{name}`" in markdown
    assert "## Current rows without comparison PNG artifacts" not in markdown
    assert "canonical_ivp_compare.py" not in markdown
    assert "canonical_point_source_compare.py" not in markdown
    assert_decodable_nonblank_png(dashboard_png)

    pngs_by_script = {
        record.script: {
            OUTPUT / png_name
            for stem in dashboard.metric_stems_for_script(record.script)
            for png_name in dashboard.comparison_pngs_for_stem(stem)
        }
        for record in records
    }
    assert all(pngs_by_script.values())
    comparison_pngs = set().union(*pngs_by_script.values())
    assert len(comparison_pngs) >= len(records)
    for png_path in sorted(comparison_pngs):
        assert_decodable_nonblank_png(png_path)


def test_current_dashboard_metric_reports_are_passing_and_finite():
    dashboard = load_example_module("parity_dashboard.py")
    records = dashboard.collect_records()

    assert records
    for record in records:
        stems = dashboard.metric_stems_for_script(record.script)
        assert stems, record.script
        for stem in stems:
            metrics_path = OUTPUT / f"{stem}_metrics.txt"
            metrics_text = metrics_path.read_text(encoding="utf-8", errors="ignore")

            assert metrics_text.strip(), metrics_path.name
            assert "PASS" in metrics_text, metrics_path.name
            assert _NONFINITE_TOKEN_PATTERN.search(metrics_text) is None, metrics_path.name
