//! Correction validation: field simulation and performance metrics
//!
//! ## Theory: Transcranial Ultrasound Focusing Metrics
//!
//! ### Focal Intensity
//! For an acoustic pressure field p(r), the time-averaged acoustic intensity is
//! (O'Neil 1949):
//! ```text
//!   I(r) = |p(r)|² / (2 ρ₀ c₀)   [W/m²]
//! ```
//! The field array stores `|p|²` (norm_sqr of complex pressure). Target point values
//! are recovered by trilinear interpolation to avoid discretization artifacts when
//! the physical target does not coincide with a grid node.
//!
//! ### Sidelobe Level (Peak Sidelobe Level, PSL)
//! The main lobe is defined as the −6 dB region (cells where `I ≥ I_peak / 4`).
//! The bounding box of this region delineates the main lobe extent.
//! The peak sidelobe level is:
//! ```text
//!   PSL = 10 · log10(I_sidelobe_peak / I_main_peak)   (dB)
//! ```
//! Reference: Zhu & Steinberg (1993), IEEE Trans. UFFC 40(6):726–737.
//!
//! ### Focal Spot Size — FWHM per Axis
//! The 3D Full Width at Half Maximum (FWHM) characterizes spatial resolution.
//! For each axis α ∈ {x, y, z}:
//! 1. Extract the 1D profile through the peak voxel along α.
//! 2. Find the leftmost and rightmost indices where `I ≥ 0.5 · I_peak`.
//! 3. `FWHM_α = (right_index − left_index) · Δα`.
//!
//! The scalar focal spot size stored in `CorrectionValidation` is the geometric
//! mean of the three FWHM values: `(FWHM_x · FWHM_y · FWHM_z)^(1/3)`, providing
//! a single isotropic focal extent measure.
//!
//! ## References
//! - O'Neil (1949). J. Acoust. Soc. Am. 21(5):516–526. (acoustic intensity)
//! - Zhu & Steinberg (1993). IEEE Trans. UFFC 40(6):726–737. (PSL definition)
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2):021314. (FWHM focal metrics)
//! - Press et al. (2007). Numerical Recipes, §3.6. (trilinear interpolation)

mod field;
mod metrics;
mod types;

#[cfg(test)]
mod tests;

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use kwavers_core::error::KwaversResult;
pub use types::CorrectionValidation;

impl TranscranialAberrationCorrection {
    /// Validate correction performance against three metrics.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn validate_correction(
        &self,
        correction: &PhaseCorrection,
        skull_model: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<CorrectionValidation> {
        let corrected_field = self.simulate_corrected_field(
            correction,
            skull_model,
            transducer_positions,
            target_point,
        )?;

        let focal_intensity = self.calculate_focal_intensity(&corrected_field, target_point);
        let sidelobe_ratio = self.calculate_sidelobe_level(&corrected_field, target_point);
        let sidelobe_level_db = if sidelobe_ratio > 0.0 {
            10.0 * sidelobe_ratio.log10()
        } else {
            f64::NEG_INFINITY
        };

        Ok(CorrectionValidation {
            focal_intensity,
            sidelobe_level_db,
            focal_spot_size: self.calculate_focal_spot_size(&corrected_field, target_point),
        })
    }
}
