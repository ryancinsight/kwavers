//! Core cavitation mechanics functionality
//!
//! This module provides fundamental cavitation detection and modeling
//! based on acoustic pressure thresholds and bubble dynamics.

pub mod bader_influx;
pub mod growth;
pub mod kreider_convolution;
pub mod louisnard_gomez;
pub mod model;
pub mod state;
pub mod thresholds;

pub use bader_influx::{
    bader_expansion_influx, bader_expansion_influx_checked, bader_holland_dmax,
    bader_signed_expansion_influx, compression_phase_outflux, equilibrium_gas_content,
    equilibrium_gas_content_derivative, equilibrium_radius_for_pressure,
    henry_surface_concentration, residual_radius_as_printed, residual_radius_equilibrium,
    tensile_phase_duration, BaderHollandFlags, CompressionSamples, OutfluxMedium,
    BADER_INFLUX_MAX_CYCLES, BADER_INFLUX_MAX_PRESSURE_PA, BADER_INFLUX_MIN_PRESSURE_PA,
    CONTINUAL_GROWTH_THRESHOLD_PA, MOLECULES_PER_MOLE,
};
pub use growth::{
    oscillation_amplitude_fraction, rectified_diffusion_rate, rectified_diffusion_rate_checked,
    rectified_diffusion_rate_with_concentrations, rectified_diffusion_terms,
    RectifiedDiffusionTerms, RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO,
};
pub use kreider_convolution::{
    latent_heat_molar, saturation_vapour_pressure, vapour_rate_hertz_knudsen, wall_temperature_scl,
    EllerFlynnHistory, KreiderCycleDiag, KreiderGainModel, KreiderGainParams,
    KREIDER_ALPHA_ACCOMMODATION, KREIDER_A_H, KREIDER_A_M_FITTED, KREIDER_A_T, KREIDER_F_M_FITTED,
};
pub use louisnard_gomez::{
    cycle_average_terms, louisnard_gomez_growth_rate, louisnard_gomez_growth_rate_checked,
    louisnard_gomez_per_cycle_influx, louisnard_gomez_regime, van_der_waals_gas_pressure,
    CycleAverageTerms, LouisnardGomezParams, LouisnardGomezRegime,
    LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ, LOUISNARD_GOMEZ_MAX_PRESSURE_PA,
    LOUISNARD_GOMEZ_MAX_RADIUS_M, LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ,
    LOUISNARD_GOMEZ_MIN_PRESSURE_PA,
};
pub use model::{CavitationCore, CavitationModel};
pub use state::{CavitationDose, CavitationMechanicsState};
pub use thresholds::{
    blake_critical_radius, blake_threshold, flynn_criterion, flynn_threshold, neppiras_threshold,
    ThresholdModel,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::analysis::calculate_mechanical_index;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
    use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
    use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
    use leto::Array3;

    #[test]
    fn test_threshold_calculations() {
        let p0 = ATMOSPHERIC_PRESSURE;
        let pv = VAPOR_PRESSURE_WATER;
        let sigma = SURFACE_TENSION_WATER;
        let r0 = 1e-6;

        let blake = blake_threshold(sigma, r0, p0, pv);
        // Surface tension always adds to the threshold: blake > P0 − Pv.
        // For small R0 (1 µm) the threshold can substantially exceed P0 + Pv,
        // so the old bound `blake < p0 + pv` was wrong for stiff nuclei.
        assert!(
            blake > p0 - pv,
            "Blake threshold must exceed P0 − Pv (got {blake:.1})"
        );
        assert!(blake > 0.0, "Blake threshold must be positive");

        let neppiras = neppiras_threshold(p0, pv, sigma, r0);
        assert!(neppiras > 0.0);

        let mi = calculate_mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
        assert_relative_eq!(mi, 1.0);
    }

    #[test]
    fn test_cavitation_model_update() {
        let mut model = CavitationModel::new([5, 5, 5]);
        model.threshold_model = ThresholdModel::MechanicalIndex;

        let mut pressure = Array3::from_elem((5, 5, 5), ATMOSPHERIC_PRESSURE);
        pressure[[2, 2, 2]] = -MPA_TO_PA; // Beyond typical threshold

        model.update_states(&pressure, MHZ_TO_HZ, 1e-6, 1e-6);

        assert!(model.states[[2, 2, 2]].is_cavitating);
        assert!(!model.states[[0, 0, 0]].is_cavitating);
        assert_relative_eq!(model.states[[2, 2, 2]].mechanical_index, 1.0);
    }
}
