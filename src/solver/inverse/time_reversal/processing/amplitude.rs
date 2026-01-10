//! Amplitude Correction Module
//!
//! Implements amplitude correction for geometric spreading and absorption compensation.

use crate::{
    domain::core::error::KwaversResult,
    domain::{grid::Grid, medium::Medium},
};
use log::debug;
use std::sync::Arc;

/// Physical constants for amplitude correction
const DEFAULT_MAX_AMPLIFICATION: f64 = 10.0;

/// Amplitude corrector for time-reversal signals
#[derive(Debug)]
pub struct AmplitudeCorrector {
    max_amplification: f64,
}

impl AmplitudeCorrector {
    /// Create a new amplitude corrector
    #[must_use]
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
        sensor_position_meters: [f64; 3],
        absorption_compensation: bool,
    ) -> KwaversResult<Vec<f64>> {
        let [sx, sy, sz] = sensor_position_meters;
        let c0 = crate::domain::medium::sound_speed_at(medium.as_ref(), sx, sy, sz, grid);
        let alpha = crate::domain::medium::AcousticProperties::absorption_coefficient(
            medium.as_ref(),
            sx,
            sy,
            sz,
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

                let absorption_correction =
                    if absorption_compensation && alpha > 0.0 && distance > 0.0 {
                        let max_factor = (self.max_amplification / geometric_correction).max(1.0);
                        (alpha * distance).min(max_factor.ln()).exp()
                    } else {
                        1.0
                    };

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
        _dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
        reference_speed: f64,
    ) -> KwaversResult<Vec<f64>> {
        // Get medium properties at the center
        let cx = grid.nx as f64 / 2.0 * grid.dx;
        let cy = grid.ny as f64 / 2.0 * grid.dy;
        let cz = grid.nz as f64 / 2.0 * grid.dz;

        let actual_speed = crate::domain::medium::sound_speed_at(medium.as_ref(), cx, cy, cz, grid);

        // Calculate phase correction factor
        let phase_factor = reference_speed / actual_speed;

        // Apply phase correction via time stretching/compression
        let n_original = signal.len();
        let n_resampled = (n_original as f64 * phase_factor) as usize;

        let mut resampled_signal = vec![0.0; n_resampled];

        // Linear interpolation for time-axis resampling
        for (i, resampled_val) in resampled_signal.iter_mut().enumerate() {
            let original_index = i as f64 / phase_factor;
            let idx_low = original_index.floor() as usize;
            let idx_high = (idx_low + 1).min(n_original - 1);
            let fraction = original_index - idx_low as f64;

            if idx_low < n_original {
                *resampled_val = signal[idx_low] * (1.0 - fraction) + signal[idx_high] * fraction;
            }
        }

        debug!(
            "Applied dispersion correction with phase factor {:.3}",
            phase_factor
        );
        Ok(resampled_signal)
    }
}

impl Default for AmplitudeCorrector {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_AMPLIFICATION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::homogeneous::HomogeneousMedium;
    use std::sync::Arc;

    #[test]
    fn absorption_compensation_toggle_changes_gain() -> KwaversResult<()> {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0)?;
        let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        medium.set_acoustic_properties(0.5, 0.0, 5.0)?;
        let medium: Arc<dyn Medium> = Arc::new(medium);

        let corrector = AmplitudeCorrector::new(1e12);
        let dt = 1e-3;
        let frequency = 1e6;
        let signal = vec![1.0, 1.0];
        let sensor_pos = [0.0, 0.0, 0.0];

        let with_abs = corrector.apply_correction(
            signal.clone(),
            dt,
            &medium,
            &grid,
            frequency,
            sensor_pos,
            true,
        )?;
        let without_abs =
            corrector.apply_correction(signal, dt, &medium, &grid, frequency, sensor_pos, false)?;

        let ratio = with_abs[1] / without_abs[1];
        let expected = (0.5 * 1500.0 * dt).exp();
        assert!((ratio - expected).abs() < 1e-12);
        Ok(())
    }
}
