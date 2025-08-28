// interpolation/coefficients.rs - Interpolation coefficients

use ndarray::Array1;

/// Interpolation coefficients for various schemes
pub struct InterpolationCoefficients {
    pub weights: Array1<f64>,
    pub order: usize,
}

impl InterpolationCoefficients {
    pub fn linear() -> Self {
        Self {
            weights: Array1::from_vec(vec![0.5, 0.5]),
            order: 1,
        }
    }

    pub fn cubic() -> Self {
        Self {
            weights: Array1::from_vec(vec![-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0]),
            order: 3,
        }
    }
}
