//! `AcousticSimulator` DIP trait + in-crate fallback + (feature-gated) kwavers backend (Phase 6).
//!
//! The experiment orchestrator depends on [`AcousticSimulator`] only; the concrete simulator
//! is injected at the call site. The default simulator is [`InCrateAcousticSim`], which runs the
//! same physics as [`crate::validate::validate_against_budget`] but operates directly on the
//! already-computed [`crate::validate::KwaversBeamStep`] + [`crate::manifest::EnergyBudgetReport`]
//! pair — no manifest required at simulation time.
//!
//! When the `kwavers` Cargo feature is on, `KwaversSim` is in scope. It calls
//! `kwavers_pressure_map_from_step` to synthesize the exact element geometry and propagate the
//! focused pressure envelope through `kwavers-transducer`.

use crate::error::Experiment as ExperimentError;
use crate::manifest::EnergyBudgetReport;
use crate::physics::acoustic::{
    acoustic_intensity_w_per_m2, focal_pressure_gain, max_grating_free_steer_deg, mechanical_index,
    near_field_distance_m,
};
use crate::ssot::*;
use crate::validate::KwaversBeamStep;

/// Acoustic pressure-field output — every scalar the in-crate model and kwavers backend emit.
///
/// `PartialEq` is derived for test assertions; `f64` fields preclude `Eq` (NaN ≠ NaN).
#[derive(Debug, Clone, PartialEq)]
pub struct PressureMap {
    /// Coherent focal pressure (Pa) — `N`-fold sum × per-element current × article sensitivity.
    pub focal_pressure_pa: f64,
    /// Mechanical Index at the focus = `p_mpa / √f_mhz` (FDA Track-3 dimensionless).
    pub mechanical_index: f64,
    /// Spatial-peak pulse-average intensity (W/cm²) at the focus.
    pub isppa_w_cm2: f64,
    /// 6 dB axial intensity half-width proxy (mm) = `2 · f# · λ`.
    pub axial_extent_mm: f64,
    /// 6 dB lateral intensity half-width proxy (mm) = `λ · f#`.
    pub lateral_extent_mm: f64,
    /// True iff the element pitch is grating-lobe-free over the full ±90° steering range.
    pub grating_lobe_free: bool,
    /// True iff the focus lies beyond the near-field (Fraunhofer) distance. Information-only:
    /// focused-beam operation is typically in the near-field and that is not a defect.
    pub in_far_field: bool,
}

/// DIP trait — the experiment orchestrator depends on this interface only.
/// Implementors supply a [`PressureMap`] from `(step, budget)`; the orchestrator
/// does not know whether the simulation uses in-crate physics or a kwavers call.
pub trait AcousticSimulator {
    /// Simulate the acoustic field for the given pre-step and energy budget.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the simulation produces a non-finite value or reaches a DIP seam.
    fn simulate(
        &self,
        step: &KwaversBeamStep,
        budget: &EnergyBudgetReport,
    ) -> Result<PressureMap, crate::Error>;
}

/// In-crate acoustic simulator — uses the physics in [`crate::physics::acoustic`].
///
/// Same computation as the [`crate::validate::validate_against_budget`] physics block,
/// re-derived from `(step, budget)` so no manifest is required at simulation time.
/// Active when the `kwavers` Cargo feature is off (the default at Phase 5).
pub struct InCrateAcousticSim;

impl AcousticSimulator for InCrateAcousticSim {
    fn simulate(
        &self,
        step: &KwaversBeamStep,
        budget: &EnergyBudgetReport,
    ) -> Result<PressureMap, crate::Error> {
        // Per-element peak current — budget.peak_i_a is the per-tile sum across all channels.
        let per_element_i_a = budget.peak_i_a / (CHANNELS_PER_TILE_V2 as f64);
        // Coherent N-fold focal pressure (Pa); same formula as `estimate_focal_pressure_pa`
        // in `validate::kwavers_beam` but derived from the already-computed step.lanes.
        let focal_pressure_pa = focal_pressure_gain(step.lanes)
            * per_element_i_a
            * KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA;
        if !focal_pressure_pa.is_finite() {
            return Err(ExperimentError::NonFiniteTransient { step: 0, t_s: 0.0 }.into());
        }
        let isppa_w_cm2 = acoustic_intensity_w_per_m2(focal_pressure_pa, PHYSICS_WATER_Z0_RAYL)
            / UNIT_W_CM2_PER_W_M2;
        let mi = mechanical_index(
            focal_pressure_pa / UNIT_PA_PER_MPA,
            step.frequency_hz / UNIT_MHZ_PER_HZ,
        );
        let grating_lobe_free = max_grating_free_steer_deg(step.pitch_m, step.wavelength_m)
            >= KWVERS_MIN_GRATING_FREE_STEER_DEG;
        let n_far = near_field_distance_m(step.aperture_m, step.wavelength_m);
        let in_far_field = step.focal_m >= n_far;
        // 6 dB half-widths from uniform-illumination analytical model; kwavers refines these.
        let axial_extent_mm = 2.0 * step.f_number * step.wavelength_m * UNIT_MM_PER_M;
        let lateral_extent_mm = step.wavelength_m * step.f_number * UNIT_MM_PER_M;
        Ok(PressureMap {
            focal_pressure_pa,
            mechanical_index: mi,
            isppa_w_cm2,
            axial_extent_mm,
            lateral_extent_mm,
            grating_lobe_free,
            in_far_field,
        })
    }
}

/// kwavers-transducer acoustic simulator. Gated on the `kwavers` Cargo feature.
///
/// Uses `kwavers_pressure_map_from_step` to synthesize the exact element geometry from the
/// step parameters and propagate a focused complex pressure envelope from the realized channel
/// coordinates.
///
/// Differences vs [`InCrateAcousticSim`]:
/// * `grating_lobe_free` — from `ArrayDesign.grating_lobe_free` (steered-axis pitch/wavelength
///   test on the realized pitch after quantization) rather than the driver-side
///   `max_grating_free_steer_deg` approximation.
/// * `in_far_field` — from `design.aperture_y_m()` (realized aperture after integer element
///   count) rather than `step.aperture_m` (requested aperture, pre-quantization).
/// * `focal_pressure_pa`, `mechanical_index`, `isppa_w_cm2`, and beam extents — from
///   `kwavers-transducer` propagation, not rederived in this crate.
#[cfg(feature = "kwavers")]
pub struct KwaversSim;

/// Synthesize the kwavers-transducer array geometry for a driver pre-step.
///
/// [`KwaversBeamStep::aperture_m`] is the first-to-last channel-centre span. In
/// `kwavers-transducer`, [`kwavers_transducer::ApertureDesignSpec::aperture_y_m`] is the
/// pitch-cell span. The adapter therefore adds one channel pitch before calling
/// [`kwavers_transducer::design_array`], so the realized channel count remains equal to the
/// manifest lane count and the returned channel-centre positions still span the manifest
/// aperture.
///
/// # Errors
///
/// Returns [`crate::error::Validate::KwaversBeamStepContract`] when kwavers-transducer rejects
/// the synthesized aperture specification.
#[cfg(feature = "kwavers")]
pub(crate) fn array_design_from_step(
    step: &KwaversBeamStep,
) -> Result<kwavers_transducer::ArrayDesign, crate::Error> {
    use crate::error::Validate;
    use kwavers_transducer::{
        design_array, ApertureDesignSpec, ChannelWiring, DEFAULT_KERF_FRACTION,
    };

    // 1-D linear array: elevation axis collapsed (aperture_x = 0 -> nx = 1 element),
    // steering along aperture_y. ColumnsAsChannels wires the single elevation row
    // into n_channels = ny independently-driven linear channels.
    let pitch_fraction = (step.pitch_m / step.wavelength_m).clamp(1e-9, 2.0);
    let spec = ApertureDesignSpec {
        aperture_x_m: 0.0,
        aperture_y_m: step.aperture_m + step.pitch_m,
        frequency_hz: step.frequency_hz,
        sound_speed_m_s: step.sound_speed_m_s,
        max_pitch_fraction: pitch_fraction,
        kerf_fraction: DEFAULT_KERF_FRACTION,
        wiring: ChannelWiring::ColumnsAsChannels,
    };
    design_array(&spec)
        .map_err(|e| Validate::KwaversBeamStepContract(format!("design_array: {e}")).into())
}

/// Run focused propagation through `kwavers-transducer` and convert its map to the driver
/// experiment shape.
///
/// # Errors
///
/// Returns [`crate::error::Validate::KwaversBeamStepContract`] when `kwavers-transducer`
/// rejects the synthesized propagation contract.
#[cfg(feature = "kwavers")]
pub(crate) fn kwavers_pressure_map_from_step(
    step: &KwaversBeamStep,
    budget: &EnergyBudgetReport,
) -> Result<PressureMap, crate::Error> {
    use crate::error::Validate;
    use kwavers_transducer::{propagate_focused_linear_array, FocusedLinearArrayPropagationSpec};

    let design = array_design_from_step(step)?;
    let per_element_i_a = budget.peak_i_a / (CHANNELS_PER_TILE_V2 as f64);
    let map = propagate_focused_linear_array(&FocusedLinearArrayPropagationSpec {
        design,
        center_m: [0.0, 0.0, 0.0],
        focus_m: [0.0, 0.0, step.focal_m],
        frequency_hz: step.frequency_hz,
        sound_speed_m_s: step.sound_speed_m_s,
        per_channel_peak_current_a: per_element_i_a,
        pressure_per_amp_pa: KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA,
        acoustic_impedance_rayl: PHYSICS_WATER_Z0_RAYL,
    })
    .map_err(|e| {
        Validate::KwaversBeamStepContract(format!("propagate_focused_linear_array: {e}"))
    })?;

    Ok(PressureMap {
        focal_pressure_pa: map.focal_pressure_pa,
        mechanical_index: map.mechanical_index,
        isppa_w_cm2: map.isppa_w_cm2,
        axial_extent_mm: map.axial_extent_mm,
        lateral_extent_mm: map.lateral_extent_mm,
        grating_lobe_free: map.grating_lobe_free,
        in_far_field: map.in_far_field,
    })
}

#[cfg(feature = "kwavers")]
impl AcousticSimulator for KwaversSim {
    fn simulate(
        &self,
        step: &KwaversBeamStep,
        budget: &EnergyBudgetReport,
    ) -> Result<PressureMap, crate::Error> {
        kwavers_pressure_map_from_step(step, budget)
    }
}
