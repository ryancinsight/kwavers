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
            phases: new_phases.clone(),
            amplitudes: new_amplitudes,
            focal_gain_db: Self::focal_gain_improvement_db(&new_phases),
            quality_metric: Self::circular_coherence(&new_phases),
        }
    }
}
