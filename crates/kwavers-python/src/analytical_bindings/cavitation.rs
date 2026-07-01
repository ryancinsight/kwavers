//! PyO3 bindings for `kwavers_physics::analytical::cavitation`.

mod bubble;
mod chirp;
mod emission;
mod lesion;
mod medium;
mod monitor;
mod passive_map;
mod passive_receive;
mod probability;
mod spectrum;
mod therapy;

pub use bubble::{
    blake_threshold_pa, minnaert_radius_for_frequency_m, minnaert_resonance_corrected_hz,
    minnaert_resonance_hz, rayleigh_collapse_time_s,
};
pub use chirp::{
    cavitation_optimal_frequency, chirped_peak_expansion_ratio, compare_shielding_control,
    inter_pulse_residual_clearance, residual_dissolution_time_s, simulate_shielding_trace,
    staged_sonication_sweep, swept_vs_monochromatic_engagement,
};
pub use emission::{
    population_emission_sweep, simulate_bubble_emission, simulate_coated_bubble_emission,
    simulate_population_emission, volume_emission_spectrum, volume_emission_sweep,
};
pub use lesion::{
    boiling_lesion_from_pressure_profile, boiling_time_profile_from_pressure,
    fractionation_acoustic_impedance, fractionation_backscatter_coefficient,
    histotripsy_lesion_radius_m, histotripsy_pulses_for_lesion_radius, inertial_cavitation_dose,
    lacuna_void_fraction,
};
pub use medium::{
    bubbly_cloud_attenuation, bubbly_cloud_phase_velocity, epstein_plesset_dissolution_time,
    shelled_dissolution_time, wood_sound_speed,
};
pub use monitor::{
    cavitation_controller_pressure, cavitation_inertial_fraction_onset_index,
    cavitation_monitor_timeseries, cavitation_therapeutic_window_indices,
    closed_loop_cavitation_sonication, per_spot_cavitation_dose_grid, raster_cavitation_pulsing,
    simulated_population_monitor_timeseries,
};
pub use passive_map::{emission_energy_in_volume, integrate_receiver_array_psd};
pub use passive_receive::{
    integrate_channel_psd, passive_cavitation_point_source_rf, receiver_channel_psd_from_source,
    van_cittert_zernike_coherence,
};
pub use probability::{
    cumulative_cavitation_probability, frequency_dependent_intrinsic_threshold_pa,
    intrinsic_threshold_cavitation_probability, prf_efficacy_factor,
};
pub use spectrum::{
    bubble_acoustic_emission_pressure, bubble_power_spectrum, cavitation_emission_bands,
    cumulative_cavitation_dose, ensemble_emission_superposition, hann_windowed_power_spectrum,
    keller_miksis_pcd_controller_trace, keller_miksis_pcd_spectrum,
    normalized_cavitation_emission_spectrum, passive_cavitation_dose_fixture,
};
pub use therapy::{
    clipped_lateral_radius_for_clearance, cloud_erosion_validation_metrics,
    delivered_histotripsy_progress, ellipsoid_respects_allowed_mask, forward_delivery_fraction,
    histotripsy_kill_fraction, histotripsy_lethal_dose, interface_pressure_enhancement,
    lacuna_cavitation_susceptibility, pressure_transmission_coefficient, received_signal_fraction,
    scale_measured_emission_spectrum, sonication_schedule,
};
