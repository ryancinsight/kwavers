//! Bubble dynamics and cavitation physics for book chapters ch07, ch09.
//!
//! Covers: Minnaert resonance, Blake threshold, Rayleigh collapse time,
//! Rayleigh–Plesset and Keller–Miksis ODE integrators (RK4), power spectrum,
//! mechanical index, inertial cavitation dose, and histotripsy lesion radius.

mod dynamics;
mod histotripsy;
mod passive_dose;
mod power_spectrum;
mod shielding;
mod sonication;
mod swept_frequency;
#[cfg(test)]
mod tests;
mod therapy_delivery;

pub use dynamics::{
    blake_threshold_pa, keller_miksis_rk4, keller_miksis_shelled_rk4,
    minnaert_radius_for_frequency_m, minnaert_resonance_corrected_hz, minnaert_resonance_hz,
    rayleigh_collapse_time_s, rayleigh_plesset_rk4,
};
pub use histotripsy::{
    cumulative_cavitation_probability, frequency_dependent_intrinsic_threshold_pa,
    histotripsy_lesion_radius_m, inertial_cavitation_dose,
    intrinsic_threshold_cavitation_probability, intrinsic_threshold_thermal_correction_pa,
    mechanical_index, prf_efficacy_factor,
};
pub use passive_dose::{
    bubble_acoustic_emission_pressure, cavitation_controller_pressure,
    cavitation_inertial_fraction_onset_index, cavitation_therapeutic_window_indices,
    cumulative_cavitation_dose, decompose_emission_spectrum, emission_energy_in_volume,
    ensemble_emission_superposition, hann_windowed_power_spectrum, integrate_receiver_array_psd,
    keller_miksis_pcd_controller_trace, keller_miksis_pcd_spectrum,
    normalized_cavitation_emission_spectrum, passive_cavitation_dose_fixture,
    passive_cavitation_point_source_rf, pcd_band_signals, population_emission_sweep,
    simulate_bubble_emission, simulate_coated_bubble_emission, simulate_population_emission,
    van_cittert_zernike_coherence, volume_emission_spectrum, volume_emission_sweep,
    BubbleDriveConfig, BubbleEmissionTrace, CavitationBandEnergies, CavitationEmissionRegime,
    CavitationTherapeuticWindow, KellerMiksisPcdControllerTrace, KellerMiksisPcdSpectrum,
    PassiveCavitationDoseFixture, PcdBandSignals, PopulationEmission, PopulationEmissionInput,
    PopulationEmissionSweep, PopulationEmissionSweepInput, PopulationMedium, PopulationShell,
    ShellDriveConfig, VolumeEmissionSpectrum, VolumeEmissionSpectrumInput, VolumeEmissionSweep,
    VolumeEmissionSweepInput, VolumeSpectrumMedium,
};
pub use power_spectrum::{bubble_power_spectrum, period_doubling_ratio};
pub use shielding::{
    compare_shielding_control, simulate_shielding, CavitationProduction, DriveFrequency,
    PulseProtocol, ShieldingComparison, ShieldingConfig, ShieldingMedium, ShieldingSummary,
    ShieldingTrace,
};
pub use sonication::{
    build_sonication_schedule, forward_delivery_fraction, histotripsy_kill_fraction,
    histotripsy_lethal_dose, histotripsy_pulses_for_lesion_radius, interface_pressure_enhancement,
    lacuna_cavitation_susceptibility, lacuna_void_fraction, pressure_transmission_coefficient,
    received_signal_fraction, SonicationOrder, SonicationSchedule,
};
pub use swept_frequency::{
    cavitation_optimal_frequency, chirped_keller_miksis_rk4, chirped_peak_expansion_ratio,
    inter_pulse_residual_clearance, monochromatic_engaged_fraction, residual_dissolution_time_s,
    staged_sonication_sweep, swept_engaged_fraction, swept_vs_monochromatic_engagement,
    tissue_gas_diffusion, CavitationMedium, EngagementConfig, EngagementResult, FrequencySweep,
    InterPulseClearance, NucleiSizeDistribution, StagedSonication, SweepProfile,
};
pub use therapy_delivery::{
    boiling_lesion_from_pressure_profile, boiling_time_profile_from_pressure,
    cavitation_monitor_trace, clipped_lateral_radius_for_clearance,
    closed_loop_cavitation_sonication, cloud_erosion_validation_metrics,
    delivered_histotripsy_progress, ellipsoid_respects_allowed_mask,
    fractionation_acoustic_impedance, fractionation_backscatter_coefficient, integrate_channel_psd,
    per_spot_cavitation_dose_grid, raster_cavitation_pulsing, receiver_channel_psd_from_source,
    scale_measured_emission_spectrum, simulated_population_monitor_trace, BoilingLesionPlan,
    CavitationMonitorTrace, CavitationMonitorTraceInput, ClosedLoopCavitationSonicationInput,
    ClosedLoopCavitationSonicationTrace, CloudErosionValidation, PerSpotCavitationDoseGrid,
    PerSpotCavitationDoseInput, RasterPulsingInput, RasterPulsingSchedule, RasterPulsingTrace,
    SimulatedPopulationMonitorInput, SimulatedPopulationMonitorTrace,
};
