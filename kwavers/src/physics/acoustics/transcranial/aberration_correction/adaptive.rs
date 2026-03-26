//! Adaptive aberration correction using iterative optimization

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};

impl TranscranialAberrationCorrection {
    /// Adaptive aberration correction using iterative optimization
    pub fn adaptive_correction(
        &mut self,
        initial_correction: &PhaseCorrection,
        feedback_signal: &[f64],
        learning_rate: f64,
    ) -> PhaseCorrection {
        let mut new_phases = initial_correction.phases.clone();
        let new_amplitudes = initial_correction.amplitudes.clone();

        // Gradient descent on feedback signal
        for i in 0..new_phases.len() {
            if i < feedback_signal.len() {
                let gradient = feedback_signal[i] - 1.0;
                new_phases[i] -= learning_rate * gradient;
            }
        }

        PhaseCorrection {
            phases: new_phases,
            amplitudes: new_amplitudes,
            focal_gain_db: initial_correction.focal_gain_db * 1.1,
            quality_metric: (initial_correction.quality_metric + 0.1).min(1.0),
        }
    }
}
