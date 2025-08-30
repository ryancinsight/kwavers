//! Amplitude Correction Module
//!
//! Implements amplitude correction for geometric spreading and absorption compensation.

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use log::debug;
use std::sync::Arc;

/// Physical constants for amplitude correction
const DEFAULT_MAX_AMPLIFICATION: f64 = 10.0;

/// Amplitude corrector for time-reversal signals
#[derive(Debug))]
pub struct AmplitudeCorrector {
    max_amplification: f64,
}

impl AmplitudeCorrector {
    /// Create a new amplitude corrector
    pub fn new(max_amplification: f64) -> Self {
        Self { max_amplification }
    }

    /// Apply amplitude correction including geometric spreading and absorption
    pub fn apply_correction(
        &self,
        signal: Vec<f64>,
        dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
        frequency: f64,
    ) -> KwaversResult<Vec<f64>> {
        // Get medium properties at the center of the grid
        let cx = grid.nx as f64 / 2.0 * grid.dx;
        let cy = grid.ny as f64 / 2.0 * grid.dy;
        let cz = grid.nz as f64 / 2.0 * grid.dz;

        let c0 = medium.sound_speed(cx, cy, cz, grid);
        let alpha = crate::medium::core::CoreMedium::absorption_coefficient(
            medium.as_ref(),
            cx,
            cy,
            cz,
            grid,
            frequency,
        );

        let corrected: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                // Time from the beginning of the recording
                let t = i as f64 * dt;

                // Estimate propagation distance (assuming spherical spreading)
                let distance = c0 * t;

                // Geometric spreading correction (1/r for 3D spherical waves)
                let geometric_correction = if distance > 0.0 { distance } else { 1.0 };

                // Absorption compensation: exp(alpha * distance)
                let absorption_correction = (alpha * distance).exp();

                // Apply both corrections
                let corrected_val = val * geometric_correction * absorption_correction;

                // Prevent excessive amplification
                if corrected_val.abs() > val.abs() * self.max_amplification {
                    val * self.max_amplification * corrected_val.signum()
                } else {
                    corrected_val
                }
            })
            .collect();

        debug!("Applied amplitude correction with geometric spreading and absorption compensation");
        Ok(corrected)
    }

    /// Apply dispersion correction for frequency-dependent sound speed
    pub fn apply_dispersion_correction(
        &self,
        signal: Vec<f64>,
        dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
        reference_speed: f64,
    ) -> KwaversResult<Vec<f64>> {
        // Get medium properties at the center
        let cx = grid.nx as f64 / 2.0 * grid.dx;
        let cy = grid.ny as f64 / 2.0 * grid.dy;
        let cz = grid.nz as f64 / 2.0 * grid.dz;

        let actual_speed = medium.sound_speed(cx, cy, cz, grid);

        // Calculate phase correction factor
        let phase_factor = reference_speed / actual_speed;

        // Apply phase correction via time stretching/compression
        let n_original = signal.len();
        let n_corrected = (n_original as f64 * phase_factor) as usize;

        let mut corrected = vec![0.0; n_corrected];

        // Linear interpolation for time-axis resampling
        for i in 0..n_corrected {
            let original_index = i as f64 / phase_factor;
            let idx_low = original_index.floor() as usize;
            let idx_high = (idx_low + 1).min(n_original - 1);
            let fraction = original_index - idx_low as f64;

            if idx_low < n_original {
                corrected[i] = signal[idx_low] * (1.0 - fraction) + signal[idx_high] * fraction;
            }
        }

        debug!(
            "Applied dispersion correction with phase factor {:.3}",
            phase_factor
        );
        Ok(corrected)
    }
}

impl Default for AmplitudeCorrector {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_AMPLIFICATION)
    }
}
