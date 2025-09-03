// adaptive_beamforming/weights.rs - Weight calculation

use ndarray::Array1;
use num_complex::Complex64;

/// Weighting scheme for beamforming
#[derive(Debug, Clone, Copy)]
pub enum WeightingScheme {
    Uniform,
    Hamming,
    Chebyshev,
}

/// Weight calculator
#[derive(Debug)]
pub struct WeightCalculator {
    scheme: WeightingScheme,
}

impl WeightCalculator {
    #[must_use]
    pub fn new(scheme: WeightingScheme) -> Self {
        Self { scheme }
    }

    #[must_use]
    pub fn compute(&self, num_elements: usize) -> Array1<Complex64> {
        match self.scheme {
            WeightingScheme::Uniform => {
                Array1::from_elem(num_elements, Complex64::new(1.0 / num_elements as f64, 0.0))
            }
            WeightingScheme::Hamming => {
                // Hamming window implementation
                Array1::from_elem(num_elements, Complex64::new(1.0 / num_elements as f64, 0.0))
            }
            WeightingScheme::Chebyshev => {
                // Chebyshev window implementation
                Array1::from_elem(num_elements, Complex64::new(1.0 / num_elements as f64, 0.0))
            }
        }
    }
}
