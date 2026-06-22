//! PyO3 bindings for `kwavers_physics::analytical::*`.
//!
//! Each sub-module wraps one analytical physics domain and exposes its
//! functions as `pub` items so they can be registered into the Python module.
//!
//! Organisation mirrors the Rust analytical sub-module layout:
//!   wave, transducer, cavitation, tissue, safety, skull, photoacoustics,
//!   elastography, imaging, thermal, bbb, inverse, sonogenetics, rtm
//!
//! FFI-boundary lint allowances (`type_complexity`, `too_many_arguments`) are set
//! crate-wide in `lib.rs`; see the justification there.

pub mod acousto_optics;
pub mod bbb;
pub mod cavitation;
pub mod elastography;
pub mod imaging;
pub mod inverse;
pub mod mems;
pub mod neuromodulation;
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
    // acousto-optic diffraction (Raman–Nath / Bragg / Klein–Cook)
    m.add_function(wrap_pyfunction!(acousto_optics::klein_cook_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(acousto_optics::raman_nath_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(
        acousto_optics::raman_nath_order_intensities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        acousto_optics::bragg_diffraction_efficiency,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(acousto_optics::diffraction_angle_rad, m)?)?;
    m.add_function(wrap_pyfunction!(
        acousto_optics::diffraction_frequency_shift_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(acousto_optics::bragg_angle_rad, m)?)?;
    m.add_function(wrap_pyfunction!(acousto_optics::solve_coupled_orders, m)?)?;

    // mems (CMUT / PMUT / IVUS — Chapter 33)
    m.add_function(wrap_pyfunction!(mems::mems_clamped_plate_resonance, m)?)?;
    m.add_function(wrap_pyfunction!(mems::mems_immersion_resonance, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_resonance_immersion, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_collapse_voltage, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_coupling_k2, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_self_heating, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_fractional_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(mems::pmut_resonance_immersion, m)?)?;
    m.add_function(wrap_pyfunction!(mems::pmut_coupling_k2, m)?)?;
    m.add_function(wrap_pyfunction!(mems::pmut_self_heating, m)?)?;
    m.add_function(wrap_pyfunction!(mems::pmut_fractional_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_max_output_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(mems::cmut_flex_gap_derating, m)?)?;
    m.add_function(wrap_pyfunction!(mems::pmut_max_output_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(mems::ivus_figure_of_merit, m)?)?;
    m.add_function(wrap_pyfunction!(mems::therapy_figure_of_merit, m)?)?;

    // wave
    m.add_function(wrap_pyfunction!(wave::tone_burst_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(wave::pulse_train_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fubini_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_vapor_pulse_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(wave::goldberg_shock_parameter_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_enhanced_absorption_gain, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_waveform_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_heat_source_density, m)?)?;
    m.add_function(wrap_pyfunction!(wave::standing_wave_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::plane_wave_pressure_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::spherical_wave_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(wave::reflection_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(wave::transmission_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(wave::power_law_attenuation_np_m, m)?)?;
    m.add_function(wrap_pyfunction!(wave::absorption_power_law_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(wave::stokes_kirchhoff_absorption_np_m, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fdtd_phase_error_1d, m)?)?;
    m.add_function(wrap_pyfunction!(wave::pstd_phase_error, m)?)?;
    m.add_function(wrap_pyfunction!(wave::kspace_correction_error, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fdtd_cfl_limit, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fubini_harmonic_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(wave::fubini_harmonic_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(wave::shock_formation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wave::westervelt_harmonic_evolution, m)?)?;
    // transducer
    m.add_function(wrap_pyfunction!(
        transducer::circular_piston_directivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::linear_array_factor, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::grating_lobe_angles, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::apodization_weights, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::delay_law_focus_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::beam_pattern_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::circular_piston_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::focused_bowl_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::acoustic_lens_delay_profile, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::fresnel_zone_radii, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::isoplanatic_steering_curve, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::corrective_lens_thickness, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::bli_stencil_weights, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::linear_array_positions, m)?)?;
    m.add_function(wrap_pyfunction!(
        transducer::focused_bowl_element_positions_3d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::delay_law_focus_3d, m)?)?;
    m.add_function(wrap_pyfunction!(
        transducer::steered_aperture_pressure_3d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        transducer::focused_bowl_steered_pressure_profile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::near_field_distance, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::steering_focus_point, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::delay_law_steer_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::beam_pattern_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::beam_pattern_2d_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::multi_focus_delay_laws_2d, m)?)?;
    m.add_function(wrap_pyfunction!(
        transducer::multi_focus_field_magnitude_2d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        transducer::linear_array_aperiodic_positions,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::steered_beam_pattern_1d, m)?)?;
    m.add_function(wrap_pyfunction!(
        transducer::steering_grating_lobe_ratio_1d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::safe_steering_halfangle, m)?)?;
    m.add_function(wrap_pyfunction!(
        transducer::electronic_steering_efficiency,
        m
    )?)?;
    // optoacoustic / SOAP (Ch34)
    m.add_function(wrap_pyfunction!(
        transducer::numerical_aperture_from_geometry,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(transducer::f_number_from_na, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::soap_focal_gain, m)?)?;
    m.add_function(wrap_pyfunction!(transducer::acoustic_resolution_lateral, m)?)?;
    // cavitation
    m.add_function(wrap_pyfunction!(
        cavitation::intrinsic_threshold_cavitation_probability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::frequency_dependent_intrinsic_threshold_pa,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::cumulative_cavitation_probability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::prf_efficacy_factor, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::minnaert_resonance_hz, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::minnaert_resonance_corrected_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::blake_threshold_pa, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::rayleigh_collapse_time_s, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::bubble_power_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::bubble_acoustic_emission_pressure,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::hann_windowed_power_spectrum,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::ensemble_emission_superposition,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::simulate_bubble_emission, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::simulate_coated_bubble_emission,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::epstein_plesset_dissolution_time,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::shelled_dissolution_time, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::wood_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::bubbly_cloud_attenuation, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::bubbly_cloud_phase_velocity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::cavitation_emission_bands, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::cumulative_cavitation_dose, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::cavitation_controller_pressure,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::integrate_receiver_array_psd,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::emission_energy_in_volume, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::sonication_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::forward_delivery_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::received_signal_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::pressure_transmission_coefficient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::interface_pressure_enhancement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::lacuna_cavitation_susceptibility,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::lacuna_void_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::histotripsy_kill_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::histotripsy_lethal_dose, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::clipped_lateral_radius_for_clearance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::ellipsoid_respects_allowed_mask,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::scale_measured_emission_spectrum,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::delivered_histotripsy_progress,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::fractionation_backscatter_coefficient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::fractionation_acoustic_impedance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::boiling_lesion_from_pressure_profile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::boiling_time_profile_from_pressure,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::receiver_channel_psd_from_source,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::integrate_channel_psd, m)?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::histotripsy_pulses_for_lesion_radius,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::histotripsy_lesion_radius_m,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::inertial_cavitation_dose, m)?)?;
    // swept-frequency (chirp) cavitation control
    m.add_function(wrap_pyfunction!(
        cavitation::swept_vs_monochromatic_engagement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::chirped_peak_expansion_ratio,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::inter_pulse_residual_clearance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::residual_dissolution_time_s,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cavitation::cavitation_optimal_frequency,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cavitation::staged_sonication_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::simulate_shielding_trace, m)?)?;
    m.add_function(wrap_pyfunction!(cavitation::compare_shielding_control, m)?)?;
    // tissue
    m.add_function(wrap_pyfunction!(tissue::water_sound_speed_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::water_density_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::ba_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::tissue_absorption_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::kramers_kronig_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::tissue_properties, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::histotripsy_tissue_properties, m)?)?;
    m.add_function(wrap_pyfunction!(tissue::tissue_thermal_properties, m)?)?;
    // safety
    m.add_function(wrap_pyfunction!(safety::mechanical_index, m)?)?;
    m.add_function(wrap_pyfunction!(safety::mechanical_index_field, m)?)?;
    m.add_function(wrap_pyfunction!(safety::thermal_index_soft_tissue, m)?)?;
    m.add_function(wrap_pyfunction!(safety::thermal_index_bone, m)?)?;
    m.add_function(wrap_pyfunction!(safety::thermal_index_cranial, m)?)?;
    m.add_function(wrap_pyfunction!(safety::cem43_cumulative, m)?)?;
    m.add_function(wrap_pyfunction!(safety::arrhenius_damage_integral, m)?)?;
    m.add_function(wrap_pyfunction!(safety::arrhenius_cumulative, m)?)?;
    m.add_function(wrap_pyfunction!(safety::arrhenius_kill_probability, m)?)?;
    m.add_function(wrap_pyfunction!(
        safety::arrhenius_steady_kill_probability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(safety::combined_kill_probability, m)?)?;
    m.add_function(wrap_pyfunction!(safety::fda_ispta_limit_mw_cm2, m)?)?;
    m.add_function(wrap_pyfunction!(safety::fda_isppa_limit_w_cm2, m)?)?;
    // skull
    m.add_function(wrap_pyfunction!(skull::skull_insertion_loss_two_way_db, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_phase_screen, m)?)?;
    m.add_function(wrap_pyfunction!(skull::hu_to_sound_speed_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(skull::hu_to_density_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(skull::strehl_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(skull::skull_surface_temperature_rise, m)?)?;
    m.add_function(wrap_pyfunction!(
        skull::skull_transfer_matrix_transmission,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(skull::skull_transmission_spectrum, m)?)?;
    // photoacoustics
    m.add_function(wrap_pyfunction!(photoacoustics::hbo2_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::hb_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(
        photoacoustics::gruneisen_parameter_water,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        photoacoustics::gruneisen_parameter_soft_tissue,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        photoacoustics::pa_sphere_pressure_signal,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(photoacoustics::pa_axial_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(
        photoacoustics::spectroscopic_unmixing_lstsq,
        m
    )?)?;
    // elastography
    m.add_function(wrap_pyfunction!(elastography::shear_wave_speed, m)?)?;
    m.add_function(wrap_pyfunction!(
        elastography::pwave_to_swave_velocity_ratio,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(elastography::voigt_complex_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(
        elastography::springpot_complex_modulus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        elastography::voigt_shear_wave_dispersion,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(elastography::mre_displacement_field, m)?)?;
    m.add_function(wrap_pyfunction!(
        elastography::thermal_strain_combined_coefficient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        elastography::thermal_strain_reconstruct,
        m
    )?)?;
    // imaging
    m.add_function(wrap_pyfunction!(imaging::lateral_psf_sinc2, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::axial_psf_rect, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::doppler_frequency_shift, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::pw_compounding_lateral_psf, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::lateral_resolution_m, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::simulate_receive_rf, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::bmode_db_fixed_reference, m)?)?;
    m.add_function(wrap_pyfunction!(imaging::delta_bmode_db, m)?)?;
    // thermal
    m.add_function(wrap_pyfunction!(
        thermal::adiabatic_temperature_rise_kelvin,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        thermal::bioheat_focal_temperature_rise,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(thermal::hifu_focal_pressure_gain, m)?)?;
    m.add_function(wrap_pyfunction!(thermal::gaussian_power_deposition_2d, m)?)?;
    m.add_function(wrap_pyfunction!(
        thermal::acoustic_intensity_depth_profile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        thermal::acoustic_power_deposition_depth_profile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        thermal::acoustic_intensity_from_amplitude,
        m
    )?)?;
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
    m.add_function(wrap_pyfunction!(
        sonogenetics::hill_activation_probability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sonogenetics::radiation_force_1d, m)?)?;
    m.add_function(wrap_pyfunction!(
        sonogenetics::acoustic_streaming_velocity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sonogenetics::ispta_w_cm2, m)?)?;
    // neuromodulation (Hodgkin–Huxley + NICE intramembrane cavitation)
    m.add_function(wrap_pyfunction!(
        neuromodulation::hodgkin_huxley_response,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        neuromodulation::nice_bilayer_sonophore_response,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(neuromodulation::nice_sonic_response, m)?)?;
    m.add_function(wrap_pyfunction!(
        neuromodulation::nice_quasistatic_response,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(neuromodulation::nice_dynamic_response, m)?)?;
    m.add_function(wrap_pyfunction!(neuromodulation::bls_deflection_curve, m)?)?;
    m.add_function(wrap_pyfunction!(
        neuromodulation::cortical_sonic_response,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        neuromodulation::bilayer_capacitance_curve,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(neuromodulation::pulse_train_dosimetry, m)?)?;
    m.add_function(wrap_pyfunction!(neuromodulation::itrusst_safety, m)?)?;
    m.add_function(wrap_pyfunction!(
        neuromodulation::neuromod_threshold_pressure_pa,
        m
    )?)?;
    // rtm
    m.add_function(wrap_pyfunction!(rtm::focused_gaussian_beam_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::backprop_green_function_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::rtm_imaging_condition, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::rtm_multi_frequency_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::temporal_modulation_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(rtm::standing_wave_suppression_gain, m)?)?;
    m.add_function(wrap_pyfunction!(
        rtm::standing_wave_spatial_frequency_cycles_m,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rtm::standing_wave_modulation_period_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rtm::standing_wave_field_1d, m)?)?;
    m.add_function(wrap_pyfunction!(
        rtm::standing_wave_intensity_statistics,
        m
    )?)?;
    Ok(())
}
