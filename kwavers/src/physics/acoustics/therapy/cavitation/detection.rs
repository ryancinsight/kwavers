use super::constants::{BUBBLE_Q_FACTOR, DEFAULT_NUCLEUS_RADIUS};
use super::{CavitationDetectionMethod, TherapyCavitationDetector};
use ndarray::Array3;

impl TherapyCavitationDetector {
    /// Detect cavitation in a snapshot pressure field.
    ///
    /// Returns a boolean array with the same shape as `pressure`.
    /// `true` indicates cavitation is predicted at that voxel.
    #[must_use]
    pub fn detect(&self, pressure: &Array3<f64>) -> Array3<bool> {
        let mut cavitation = Array3::from_elem(pressure.dim(), false);
        match self.method {
            CavitationDetectionMethod::PressureThreshold => {
                self.detect_by_threshold(pressure, &mut cavitation);
            }
            CavitationDetectionMethod::Spectral | CavitationDetectionMethod::Combined => {
                self.detect_by_spectral(pressure, &mut cavitation);
            }
        }
        cavitation
    }

    /// Per-voxel Blake threshold detection: cavitates iff `−p > P_Blake`.
    fn detect_by_threshold(&self, pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        cavitation
            .iter_mut()
            .zip(pressure.iter())
            .for_each(|(cav, &p)| {
                *cav = -p > self.blake_threshold;
            });
    }

    /// Per-voxel resonance-enhanced threshold detection.
    ///
    /// ## Algorithm
    /// 1. Compute Minnaert resonance frequency f₀ for a 1 µm reference nucleus.
    /// 2. Compute resonance enhancement E(f) via the linearised bubble oscillator:
    ///    ```text
    ///      E(f) = max(1, 1 / sqrt((1−(f/f₀)²)² + (f/(Q·f₀))²))
    ///    ```
    /// 3. Effective threshold: P_eff = P_Blake / E(f).
    /// 4. Voxel cavitates iff `−p > P_eff`.
    ///
    /// Away from resonance E = 1 and this reduces to Blake threshold detection.
    fn detect_by_spectral(&self, pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        let effective_threshold = self.resonance_effective_threshold(DEFAULT_NUCLEUS_RADIUS);

        cavitation
            .iter_mut()
            .zip(pressure.iter())
            .for_each(|(cav, &p)| {
                *cav = -p > effective_threshold;
            });
    }

    pub(super) fn resonance_effective_threshold(&self, r0: f64) -> f64 {
        let f0 = self.minnaert_frequency(r0);
        let f_over_f0 = self.frequency / f0;
        let detuning = 1.0 - f_over_f0 * f_over_f0;
        let dissipation = f_over_f0 / BUBBLE_Q_FACTOR;
        let denominator = (detuning * detuning + dissipation * dissipation).sqrt();
        let enhancement = if denominator < 1.0 {
            1.0 / denominator
        } else {
            1.0
        };
        self.blake_threshold / enhancement
    }
}
