//! PyO3 bindings for `kwavers::physics::analytical::*`.
//!
//! Each sub-module wraps one analytical physics domain and exposes its
//! functions as `pub` items so they can be registered into the Python module.
//!
//! Organisation mirrors the Rust analytical sub-module layout:
//!   wave, transducer, cavitation, tissue, safety, skull, photoacoustics,
//!   elastography, imaging, thermal, bbb, inverse, sonogenetics, rtm

pub mod bbb;
pub mod cavitation;
pub mod elastography;
pub mod imaging;
pub mod inverse;
pub mod photoacoustics;
pub mod rtm;
pub mod safety;
pub mod skull;
pub mod sonogenetics;
pub mod thermal;
pub mod tissue;
pub mod transducer;
pub mod wave;

use pyo3::prelude::*;

/// Register all book-physics functions into the given Python sub-module.
pub fn register_book(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // wave
    m.add_function(wrap_pyfunction!(wave::standing_wave_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::plane_wave_pressure_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::spherical_wave_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(wave::reflection_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(wave::transmission_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(wave::power_law_attenuation_np_m, m)?)?;
    m.add_function(wrap_pyfunction!(wave::absorption_power_law_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fdtd_phase_error_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::pstd_phase_error, m)?)?;
    m.add_function(wrap_pyfunction!(wave::kspace_correction_error, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fdtd_cfl_limit, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fubini_harmonic_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fubini_harmonic_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_formation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wave::westervelt_harmonic_evolution, m)?)?;
    // transducer
    m.add_function(wrap_pyfunction!(transducer::circular_piston_directivity, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::linear_array_factor, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::grating_lobe_angles, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::apodization_weights, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::delay_law_focus_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::beam_pattern_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::circular_piston_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::focused_bowl_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::bli_stencil_weights, m)?)?;
    // cavitation
    m.add_function(wrap_pyfunction!(cavitation::minnaert_resonance_hz, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::blake_threshold_pa, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::rayleigh_collapse_time_s, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::rayleigh_plesset_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::keller_miksis_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::bubble_power_spectrum, m)?)?;
    // tissue
    m.add_function(wrap_pyfunction!(tissue::water_sound_speed_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::water_density_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::ba_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::tissue_absorption_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::kramers_kronig_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::tissue_properties, m)?)?;
    // safety
    m.add_function(wrap_pyfunction!(safety::mechanical_index, m)?)?;
    m.add_function(wrap_pyfunction!(safety::thermal_index_soft_tissue, m)?)?;
    m.add_function(wrap_pyfunction!(safety::thermal_index_bone, m)?)?;
    m.add_function(wrap_pyfunction!(safety::cem43_cumulative, m)?)?;
    m.add_function(wrap_pyfunction!(safety::arrhenius_damage_integral, m)?)?;
    m.add_function(wrap_pyfunction!(safety::fda_ispta_limit_mw_cm2, m)?)?;
    m.add_function(wrap_pyfunction!(safety::fda_isppa_limit_w_cm2, m)?)?;
    // skull
    m.add_function(wrap_pyfunction!(skull::skull_insertion_loss_two_way_db, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_phase_screen, m)?)?;
    m.add_function(wrap_pyfunction!(skull::hu_to_sound_speed_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(skull::hu_to_density_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(skull::strehl_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_surface_temperature_rise, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_transfer_matrix_transmission, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_transmission_spectrum, m)?)?;
    // photoacoustics
    m.add_function(wrap_pyfunction!(photoacoustics::hbo2_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::hb_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::gruneisen_parameter_water, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::pa_sphere_pressure_signal, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::pa_axial_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::spectroscopic_unmixing_lstsq, m)?)?;
    // elastography
    m.add_function(wrap_pyfunction!(elastography::shear_wave_speed, m)?)?;
    m.add_function(wrap_pyfunction!(elastography::voigt_complex_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(elastography::springpot_complex_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(elastography::voigt_shear_wave_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(elastography::mre_displacement_field, m)?)?;
    // imaging
    m.add_function(wrap_pyfunction!(imaging::lateral_psf_sinc2, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::axial_psf_rect, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::doppler_frequency_shift, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::pw_compounding_lateral_psf, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::lateral_resolution_m, m)?)?;
    // thermal
    m.add_function(wrap_pyfunction!(thermal::bioheat_focal_temperature_rise, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::hifu_focal_pressure_gain, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::gaussian_power_deposition_2d, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::acoustic_intensity_depth_profile, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::acoustic_power_deposition_depth_profile, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::acoustic_heat_source_density, m)?)?;
    // bbb
    m.add_function(wrap_pyfunction!(bbb::bbb_permeability_hill, m)?)?;
    m.add_function(wrap_pyfunction!(bbb::bbb_closure_kinetics, m)?)?;
    m.add_function(wrap_pyfunction!(bbb::ceus_backscatter_signal, m)?)?;
    // inverse
    m.add_function(wrap_pyfunction!(inverse::helmholtz_1d_fd_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(inverse::matrix_singular_values, m)?)?;
    m.add_function(wrap_pyfunction!(inverse::tikhonov_lcurve, m)?)?;
    m.add_function(wrap_pyfunction!(inverse::born_inversion_regularized, m)?)?;
    m.add_function(wrap_pyfunction!(inverse::adjoint_gradient_convergence, m)?)?;
    // sonogenetics
    m.add_function(wrap_pyfunction!(sonogenetics::hill_activation_probability, m)?)?;
    m.add_function(wrap_pyfunction!(sonogenetics::radiation_force_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sonogenetics::acoustic_streaming_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(sonogenetics::ispta_w_cm2, m)?)?;
    // rtm
    m.add_function(wrap_pyfunction!(rtm::focused_gaussian_beam_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::backprop_green_function_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::rtm_imaging_condition, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::rtm_multi_frequency_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::temporal_modulation_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::standing_wave_suppression_gain, m)?)?;
    Ok(())
}
