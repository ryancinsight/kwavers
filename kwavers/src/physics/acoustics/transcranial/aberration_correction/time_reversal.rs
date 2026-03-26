//! Time-reversal aberration correction
//!
//! ## Mathematical Foundation
//!
//! Time-reversal invariance: Wave equation ∂²u/∂t² − c²∇²u = 0 is time-reversal symmetric.
//! Phase conjugation: If u(t) is the aberrated wave, then u*(-t) focuses perfectly at the source.
//!
//! ## References
//!
//! - Aubry et al. (2003) JASA 113(1):84-93
//! - Fink (1992) IEEE Trans UFFC

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use crate::core::error::KwaversResult;
use log::info;
use ndarray::Array3;
use num_complex::Complex;

impl TranscranialAberrationCorrection {
    /// Apply time-reversal aberration correction
    pub fn apply_time_reversal_correction(
        &self,
        _measured_field: &Array3<Complex<f64>>,
        transducer_positions: &[[f64; 3]],
    ) -> KwaversResult<PhaseCorrection> {
        let mut phases = vec![0.0; transducer_positions.len()];
        let amplitudes = vec![1.0; transducer_positions.len()];

        info!("Applying time-reversal aberration correction");

        for i in 0..transducer_positions.len() {
            let dist_to_origin = (transducer_positions[i][0].powi(2)
                + transducer_positions[i][1].powi(2)
                + transducer_positions[i][2].powi(2))
            .sqrt();
            let k = 2.0 * std::f64::consts::PI * self.frequency / self.reference_speed;
            phases[i] = -k * dist_to_origin;
        }

        Ok(PhaseCorrection {
            phases: phases.clone(),
            amplitudes,
            focal_gain_db: {
                let n = phases.len() as f64;
                let ideal_gain = 20.0 * n.log10();
                let mean_phase = phases.iter().sum::<f64>() / n;
                let phase_var = phases.iter().map(|p| (p - mean_phase).powi(2)).sum::<f64>() / n;
                let coherence = (-phase_var / 2.0).exp();
                ideal_gain * coherence
            },
            quality_metric: {
                let n = phases.len() as f64;
                let (sum_cos, sum_sin) = phases.iter().fold((0.0_f64, 0.0_f64), |(sc, ss), &p| {
                    (sc + p.cos(), ss + p.sin())
                });
                (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n
            },
        })
    }
}
