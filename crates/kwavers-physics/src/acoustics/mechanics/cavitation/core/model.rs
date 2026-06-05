//! Core cavitation modeling struct and traits
//!
//! # Mathematical Specification
//!
//! The cavitation detection pipeline applies a threshold model to the acoustic
//! pressure field $p(x,t)$ at each grid point. A point is deemed cavitating when
//! $p < -P_{threshold}$, where $P_{threshold}$ is computed from the selected
//! physical model (Blake, Neppiras, Flynn, or Mechanical Index).
//!
//! The accumulated cavitation dose follows:
//! $$ D(t) = \int_0^t I(\tau)\, d\tau $$
//! where $I$ is the local cavitation intensity.

use super::state::{CavitationDose, CavitationMechanicsState};
use super::thresholds::{blake_threshold, flynn_threshold, neppiras_threshold, ThresholdModel};
use crate::acoustics::analysis::calculate_mechanical_index;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use kwavers_core::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Mechanical Index (MI) threshold for the onset of inertial cavitation in water
/// at 1 MHz, based on the Apfel-Holland theoretical framework.
///
/// Note: This is a physical onset threshold, distinct from the FDA regulatory
/// limit of MI ≤ 1.9 for diagnostic ultrasound.
///
/// # References
/// - Apfel, R. E., & Holland, C. K. (1991). "Gauging the likelihood of cavitation
///   from short-pulse, low-duty cycle diagnostic ultrasound." Ultrasound in Med. & Biol., 17(2), 179-185.
pub const APFEL_HOLLAND_CAVITATION_THRESHOLD_1MHZ_PA: f64 = 0.7e6;

/// Core cavitation detection and modeling
pub trait CavitationCore: Send + Sync {
    /// Detect cavitation based on pressure threshold
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn detect_cavitation(&self, pressure: f64, threshold: f64) -> bool;

    /// Calculate cavitation index
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn cavitation_index(&self, pressure: f64, vapor_pressure: f64, ambient_pressure: f64) -> f64;

    /// Update cavitation state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update(&mut self, pressure_field: &Array3<f64>, dt: f64) -> KwaversResult<()>;
}

/// Main cavitation model implementation
///
/// Uses `BubbleParameters` as the Single Source of Truth for all physical
/// constants (surface tension, radius, pressures). No field duplication.
#[derive(Debug, Clone)]
pub struct CavitationModel {
    /// Threshold model to use
    pub threshold_model: ThresholdModel,
    /// Bubble parameters — SSOT for surface_tension, initial_radius, ambient_pressure, vapor_pressure
    pub params: BubbleParameters,
    /// Current cavitation states
    pub states: Array3<CavitationMechanicsState>,
    /// Cavitation dose accumulator
    pub dose: CavitationDose,
}

impl CavitationModel {
    /// Create new cavitation model with default water/air bubble parameters
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize)) -> Self {
        Self {
            threshold_model: ThresholdModel::MechanicalIndex,
            params: BubbleParameters::default(),
            states: Array3::default(grid_shape),
            dose: CavitationDose::new(),
        }
    }

    /// Compute the cavitation threshold pressure from the selected model.
    ///
    /// **SSOT**: This is the single authoritative threshold computation.
    /// Both `update_states` and `CavitationCore::update` delegate here.
    #[must_use]
    fn compute_threshold(&self) -> f64 {
        match self.threshold_model {
            ThresholdModel::Blake => blake_threshold(
                self.params.sigma,
                self.params.r0,
                self.params.p0,
                self.params.pv,
            ),
            ThresholdModel::Neppiras => neppiras_threshold(
                self.params.p0,
                self.params.pv,
                self.params.sigma,
                self.params.r0,
            ),
            ThresholdModel::Flynn => flynn_threshold(
                self.params.p0,
                self.params.pv,
                self.params.sigma,
                self.params.r0,
            ),
            ThresholdModel::MechanicalIndex => {
                // MI-based threshold: onset of inertial cavitation
                APFEL_HOLLAND_CAVITATION_THRESHOLD_1MHZ_PA
            }
        }
    }

    /// Compute cavitation intensity from pressure and ambient conditions.
    ///
    /// **SSOT**: This is the single authoritative intensity formula.
    /// Both `update_states` and `CavitationCore::update` delegate here.
    ///
    /// # Definition
    /// $$ I = \min\!\left(\frac{|p_{\text{neg}}|}{P_0},\, 1\right) $$
    ///
    /// This normalises the peak negative pressure against the ambient
    /// pressure, yielding a dimensionless intensity in [0, 1]. The
    /// clamping ensures saturation at extreme rarefaction.
    #[must_use]
    #[inline]
    fn compute_intensity(peak_negative_pressure: f64, ambient_pressure: f64) -> f64 {
        (peak_negative_pressure.abs() / ambient_pressure).min(1.0)
    }

    /// Update cavitation state based on pressure field (vectorized)
    ///
    /// Uses `ndarray::Zip` for cache-friendly vectorized iteration instead of
    /// scalar `indexed_iter`.
    pub fn update_states(
        &mut self,
        pressure_field: &Array3<f64>,
        frequency: f64,
        dt: f64,
        time: f64,
    ) {
        let threshold = self.compute_threshold();
        let ambient_pressure = self.params.p0;

        // Accumulate dose updates in a local variable to avoid borrow conflict
        let mut dose_intensity_sum = 0.0;
        let mut dose_count = 0u64;

        Zip::from(&mut self.states)
            .and(pressure_field)
            .for_each(|state, &p| {
                // Check for cavitation
                let was_cavitating = state.is_cavitating;
                state.is_cavitating = p < -threshold;

                if state.is_cavitating {
                    state.duration += dt;
                    state.peak_negative_pressure = state.peak_negative_pressure.min(p);
                    state.mechanical_index = calculate_mechanical_index(p, frequency);
                    state.intensity =
                        Self::compute_intensity(state.peak_negative_pressure, ambient_pressure);

                    dose_intensity_sum += state.intensity;
                    dose_count += 1;
                } else if was_cavitating {
                    // Just stopped cavitating
                    state.duration = 0.0;
                    state.intensity = 0.0;
                }
            });

        // Update dose with average intensity across cavitating points
        if dose_count > 0 {
            let avg_intensity = dose_intensity_sum / dose_count as f64;
            self.dose.update(avg_intensity, dt, time);
        }
    }
}

impl CavitationCore for CavitationModel {
    fn detect_cavitation(&self, pressure: f64, threshold: f64) -> bool {
        pressure < -threshold
    }

    /// Cavitation index: (P₀ + p − Pᵥ) / (P₀ − Pᵥ).
    ///
    /// - p = 0 → index = 1.0 (ambient, no cavitation).
    /// - p = Pᵥ − P₀ → index = 0.0 (onset of cavitation).
    fn cavitation_index(&self, pressure: f64, vapor_pressure: f64, ambient_pressure: f64) -> f64 {
        (ambient_pressure + pressure - vapor_pressure) / (ambient_pressure - vapor_pressure)
    }

    fn update(&mut self, pressure_field: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        let threshold = self.compute_threshold();
        let ambient_pressure = self.params.p0;

        // Accumulate dose in local variables to avoid borrow conflict
        let mut dose_intensity_sum = 0.0;
        let mut dose_count = 0u64;

        Zip::from(&mut self.states)
            .and(pressure_field)
            .for_each(|state, &pressure| {
                if pressure < -threshold {
                    if !state.is_cavitating {
                        state.is_cavitating = true;
                        state.duration = 0.0;
                    }
                    state.duration += dt;
                    state.peak_negative_pressure = state.peak_negative_pressure.min(pressure);
                    // SSOT: delegate to the single authoritative intensity formula
                    state.intensity =
                        Self::compute_intensity(state.peak_negative_pressure, ambient_pressure);

                    dose_intensity_sum += state.intensity;
                    dose_count += 1;
                } else {
                    state.is_cavitating = false;
                    state.intensity = 0.0;
                }
            });

        // Fix: divide by cavitating count, not total grid size
        if dose_count > 0 {
            let avg_intensity = dose_intensity_sum / dose_count as f64;
            self.dose.update(avg_intensity, dt, dt);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MPA_TO_PA;
    use ndarray::Array3;

    /// `CavitationModel::new` defaults to the MechanicalIndex threshold model.
    #[test]
    fn new_defaults_to_mechanical_index_model() {
        let m = CavitationModel::new((4, 4, 4));
        assert_eq!(m.threshold_model, ThresholdModel::MechanicalIndex);
        assert_eq!(m.states.dim(), (4, 4, 4));
    }

    /// `detect_cavitation` returns true when pressure is more negative than -threshold.
    #[test]
    fn detect_cavitation_true_when_pressure_below_negative_threshold() {
        let m = CavitationModel::new((2, 2, 2));
        assert!(
            m.detect_cavitation(-200.0, 100.0),
            "p=-200 < -100 → cavitation"
        );
        assert!(
            !m.detect_cavitation(-50.0, 100.0),
            "p=-50 > -100 → no cavitation"
        );
        assert!(
            !m.detect_cavitation(0.0, 100.0),
            "p=0 > -100 → no cavitation"
        );
    }

    /// `cavitation_index` at p=0 (ambient pressure, no rarefaction) equals 1.0.
    ///
    /// Analytical: (P₀ + 0 − Pᵥ) / (P₀ − Pᵥ) = 1.0.
    #[test]
    fn cavitation_index_unity_at_zero_pressure_perturbation() {
        let m = CavitationModel::new((2, 2, 2));
        let p0 = m.params.p0;
        let pv = m.params.pv;
        let ci = m.cavitation_index(0.0, pv, p0);
        assert!(
            (ci - 1.0).abs() < 1e-12,
            "cavitation_index at p=0 must be 1.0 (got {ci:.6})"
        );
    }

    /// `cavitation_index` at p = Pᵥ − P₀ (cavitation onset) equals 0.0.
    #[test]
    fn cavitation_index_zero_at_cavitation_onset() {
        let m = CavitationModel::new((2, 2, 2));
        let p0 = m.params.p0;
        let pv = m.params.pv;
        let p_onset = pv - p0; // (P₀ + (Pᵥ-P₀) - Pᵥ) = 0
        let ci = m.cavitation_index(p_onset, pv, p0);
        assert!(
            (ci).abs() < 1e-12,
            "cavitation_index at onset must be 0.0 (got {ci:.6})"
        );
    }

    /// `CavitationCore::update` marks cells with pressure below -threshold as cavitating.
    #[test]
    fn update_marks_cavitating_cells_and_leaves_others_clear() {
        let mut m = CavitationModel::new((2, 2, 2));
        // MechanicalIndex threshold = APFEL_HOLLAND_CAVITATION_THRESHOLD_1MHZ_PA = 0.7e6 Pa
        // Any pressure < -0.7e6 triggers cavitation.
        let mut field = Array3::<f64>::zeros((2, 2, 2));
        field[[0, 0, 0]] = -MPA_TO_PA; // below -0.7MPa → cavitating
        field[[1, 1, 1]] = 0.0; // not below threshold

        m.update(&field, 1e-6).unwrap();

        assert!(
            m.states[[0, 0, 0]].is_cavitating,
            "cell [0,0,0] must be cavitating"
        );
        assert!(
            !m.states[[1, 1, 1]].is_cavitating,
            "cell [1,1,1] must not be cavitating"
        );
    }
}
