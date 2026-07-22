//! Adaptive aberration correction using iterative phase-conjugation descent.
//!
//! ## Mathematical Foundation
//!
//! ### Feedback Signal Interpretation
//!
//! `feedback_signal`i`` is the unwrapped phase angle φ_received`i` (radians)
//! of the received wavefront at element `i`, measured from back-propagation of
//! the target-focus return signal (time-reversal or cross-correlation of the
//! transmitted and received waveforms).
//!
//! ### Objective
//!
//! Minimise the circular phase-mismatch cost:
//! ```text
//! L(φ) = Σ_i [1 − cos(φ_i + φ_received`i`)]
//! ```
//! At the minimum, `φ_i = −φ_received`i`` (phase conjugation), which exactly
//! cancels the skull-induced aberration and maximises focal coherence.
//!
//! ### Gradient
//!
//! ```text
//! ∂L/∂φ_i = sin(φ_i + φ_received`i`)
//! ```
//!
//! The update rule (gradient descent, step size η = `learning_rate`):
//! ```text
//! φ_i ← φ_i − η · sin(φ_i + φ_received`i`)
//! ```
//!
//! Convergence: `sin(φ_i + φ_received`i`) = 0` at `φ_i = −φ_received`i``,
//! i.e., the correction converges to the time-reversal phase conjugate.
//!
//! ## References
//! - Fink M (1992). "Time reversal of ultrasonic fields."
//!   *IEEE Trans. UFFC* **39**(5):555–566.
//! - Vignon F et al. (2006). "Adaptive focusing using the monopole operator."
//!   *J. Acoust. Soc. Am.* **120**(5):2737–2745.

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};

impl TranscranialAberrationCorrection {
    /// Iterative adaptive aberration correction via circular phase-mismatch descent.
    ///
    /// Each call performs one gradient-descent step minimising
    /// `L = Σ_i [1 − cos(φ_i + φ_received`i`)]`.
    ///
    /// The gradient `∂L/∂φ_i = sin(φ_i + φ_received`i`)` is bounded to
    /// `[−1, 1]`, providing unconditional stability for any step size
    /// `learning_rate ≤ 1`.  At convergence `φ_i → −feedback_signal`i``
    /// (time-reversal phase conjugate).
    ///
    /// # Arguments
    ///
    /// * `initial_correction` – current phase correction estimate.
    /// * `feedback_signal`    – `feedback_signal`i`` = received wavefront phase
    ///   at element `i` \[radians\], from back-propagation
    ///   or cross-correlation with the target return.
    /// * `learning_rate`      – gradient-descent step size η > 0 (dimensionless).
    ///   Stable for η ≤ 1; typical value 0.1–0.5.
    pub fn adaptive_correction(
        &mut self,
        initial_correction: &PhaseCorrection,
        feedback_signal: &[f64],
        learning_rate: f64,
    ) -> PhaseCorrection {
        let mut new_phases = initial_correction.phases.clone();
        let new_amplitudes = initial_correction.amplitudes.clone();

        // ∂L/∂φ_i = sin(φ_i + φ_received[i]);  update: φ_i ← φ_i − η·∂L/∂φ_i
        for (i, phase) in new_phases.iter_mut().enumerate() {
            if let Some(&received) = feedback_signal.get(i) {
                let gradient = (*phase + received).sin();
                *phase -= learning_rate * gradient;
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
