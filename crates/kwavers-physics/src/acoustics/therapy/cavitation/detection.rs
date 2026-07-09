use super::constants::{BUBBLE_Q_FACTOR, DEFAULT_NUCLEUS_RADIUS};
use super::{CavitationDetectionMethod, TherapyCavitationDetector};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::Array3;

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
        mark_cavitation(pressure, cavitation, self.blake_threshold);
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

        mark_cavitation(pressure, cavitation, effective_threshold);
    }

    pub(super) fn resonance_effective_threshold(&self, r0: f64) -> f64 {
        let f0 = self.minnaert_frequency(r0);
        let f_over_f0 = self.frequency / f0;
        let detuning = f_over_f0.mul_add(-f_over_f0, 1.0);
        let dissipation = f_over_f0 / BUBBLE_Q_FACTOR;
        let denominator = detuning.hypot(dissipation);
        let enhancement = if denominator < 1.0 {
            1.0 / denominator
        } else {
            1.0
        };
        self.blake_threshold / enhancement
    }
}

fn mark_cavitation(pressure: &Array3<f64>, cavitation: &mut Array3<bool>, threshold: f64) {
    assert_eq!(
        pressure.dim(),
        cavitation.dim(),
        "invariant: cavitation detection pressure and output shapes must match"
    );

    let cavitation = cavitation
        .as_slice_memory_order_mut()
        .expect("invariant: cavitation detection output must be contiguous");
    let pressure = pressure
        .as_slice_memory_order()
        .expect("invariant: cavitation pressure field must be contiguous");

    enumerate_mut_with::<Adaptive, _, _>(cavitation, |idx, cav| {
        *cav = -pressure[idx] > threshold;
    });
}
