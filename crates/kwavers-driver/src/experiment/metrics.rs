//! Aggregated experiment metrics — acoustic + thermal in one view (Phase 5).
//!
//! [`ExperimentMetrics`] collects every scalar a downstream consumer (UI dashboard, report
//! template, acceptance test) needs without having to walk two separate structs. All fields are
//! mirrored verbatim from [`PressureMap`] / [`ThermalState`]; no new computation lives here.
//!
//! [`build_beam_report`] assembles the 4-check kwavers-beam [`PhysicsReport`] from the
//! already-computed [`PressureMap`] scalars + the per-tile resistor margin minimum. Separating
//! the report construction from `validate_against_budget` means the runner can source it from
//! whatever acoustic model ran (in-crate or kwavers) rather than from the analytical pre-step
//! estimate.

use crate::ssot::*;
use crate::validate::{Check, KwaversBeamStep, PhysicsReport};

use super::acoustic::PressureMap;
use super::thermal::ThermalState;

/// Aggregated acoustic + thermal experiment output — one struct for dashboards and tests.
///
/// `PartialEq` is derived for test assertions; the `f64` fields preclude `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentMetrics {
    /// Coherent focal pressure (Pa).
    pub focal_pressure_pa: f64,
    /// Mechanical Index at the focus (dimensionless).
    pub mechanical_index: f64,
    /// Spatial-peak pulse-average intensity (W/cm²).
    pub isppa_w_cm2: f64,
    /// 6 dB axial extent proxy (mm).
    pub axial_extent_mm: f64,
    /// 6 dB lateral extent proxy (mm).
    pub lateral_extent_mm: f64,
    /// True iff grating-lobe-free over ±90°.
    pub grating_lobe_free: bool,
    /// True iff focus is in the far field (information-only).
    pub in_far_field: bool,
    /// Peak tile junction temperature rise (K).
    pub peak_thermal_rise_k: f64,
    /// Thermal headroom (K) = `dt_max_k − peak_thermal_rise_k`.
    pub thermal_headroom_k: f64,
}

impl ExperimentMetrics {
    /// Assemble from already-computed [`PressureMap`] and [`ThermalState`].
    #[must_use]
    pub fn from_parts(pressure: &PressureMap, thermal: &ThermalState) -> Self {
        Self {
            focal_pressure_pa: pressure.focal_pressure_pa,
            mechanical_index: pressure.mechanical_index,
            isppa_w_cm2: pressure.isppa_w_cm2,
            axial_extent_mm: pressure.axial_extent_mm,
            lateral_extent_mm: pressure.lateral_extent_mm,
            grating_lobe_free: pressure.grating_lobe_free,
            in_far_field: pressure.in_far_field,
            peak_thermal_rise_k: thermal.peak_rise_k,
            thermal_headroom_k: thermal.headroom_k,
        }
    }
}

/// Build the 4-check kwavers-beam [`PhysicsReport`] from an already-computed [`PressureMap`],
/// the beam geometry in `step`, and the per-tile minimum resistor margin (W).
///
/// This is the SSOT for the 4-check aggregation in the *experiment* layer; the same check
/// contract lives at [`crate::validate::validate_against_budget`] for the *validate* layer.
/// The constants (`CHECK_*`, `KWVERS_*`) are sourced from [`crate::ssot`] in both places.
#[must_use]
pub fn build_beam_report(
    map: &PressureMap,
    _step: &KwaversBeamStep,
    min_resistor_margin_w: f64,
) -> PhysicsReport {
    PhysicsReport::new(vec![
        Check::lower(
            CHECK_FOCAL_PRESSURE_NAME,
            map.focal_pressure_pa,
            KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA,
            "Pa",
        ),
        Check::upper(
            CHECK_MI_NAME,
            map.mechanical_index,
            KWVERS_MI_CAVITATION_CEILING,
            "",
        ),
        Check::lower(
            CHECK_GRATING_LOBE_NAME,
            if map.grating_lobe_free { 1.0 } else { 0.0 },
            1.0,
            "bool",
        ),
        Check::lower(
            CHECK_RESISTOR_MARGIN_NAME,
            min_resistor_margin_w,
            KWVERS_MIN_RESISTOR_MARGIN_W,
            "W",
        ),
    ])
}
