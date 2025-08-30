//! Element Coupling Module
//!
//! Models acoustic and electrical coupling between array elements.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Element coupling characteristics
///
/// Based on Turnbull & Foster (1991): "Beam steering with pulsed two-dimensional transducer arrays"
#[derive(Debug, Clone))]
pub struct ElementCoupling {
    /// Acoustic crosstalk matrix
    pub acoustic_coupling: Array2<f64>,
    /// Electrical crosstalk matrix
    pub electrical_coupling: Array2<f64>,
    /// Mutual impedance matrix
    pub mutual_impedance: Array2<f64>,
}

impl ElementCoupling {
    /// Calculate coupling for linear array
    pub fn linear_array(num_elements: usize, pitch: f64, frequency: f64) -> Self {
        let wavelength = 1540.0 / frequency;
        let k = 2.0 * PI / wavelength;

        let mut acoustic_coupling = Array2::eye(num_elements);
        let mut electrical_coupling = Array2::eye(num_elements);
        let mut mutual_impedance = Array2::zeros((num_elements, num_elements));

        for i in 0..num_elements {
            for j in 0..num_elements {
                if i != j {
                    let distance = ((i as f64 - j as f64) * pitch).abs();

                    // Acoustic coupling decreases with distance
                    let acoustic_factor = (-k * distance).exp() / (k * distance + 1.0);
                    acoustic_coupling[[i, j] = acoustic_factor * 0.1; // Typical 10% max coupling

                    // Electrical coupling (capacitive)
                    let electrical_factor = 1.0 / (1.0 + distance / pitch);
                    electrical_coupling[[i, j] = electrical_factor * 0.05; // Typical 5% max

                    // Mutual impedance
                    mutual_impedance[[i, j] = 50.0 * acoustic_factor; // Scaled by nominal impedance
                }
            }
        }

        Self {
            acoustic_coupling,
            electrical_coupling,
            mutual_impedance,
        }
    }

    /// Calculate effective element pattern with coupling
    pub fn effective_pattern(&self, element_idx: usize, excitation: &Array1<f64>) -> Array1<f64> {
        let num_elements = self.acoustic_coupling.nrows();
        let mut pattern = Array1::zeros(excitation.len());

        for i in 0..num_elements {
            pattern[i] = excitation[i] * self.acoustic_coupling[[element_idx, i];
        }

        pattern
    }

    /// Calculate crosstalk level in dB
    pub fn crosstalk_level(&self) -> f64 {
        let mut max_crosstalk = 0.0_f64;
        let n = self.acoustic_coupling.nrows();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    max_crosstalk = max_crosstalk.max(self.acoustic_coupling[[i, j]);
                }
            }
        }

        20.0_f64 * max_crosstalk.log10()
    }

    /// Check if coupling is within acceptable limits
    pub fn validate_coupling(&self, max_crosstalk_db: f64) -> bool {
        self.crosstalk_level() < max_crosstalk_db
    }
}
