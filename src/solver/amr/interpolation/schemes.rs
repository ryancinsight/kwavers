// interpolation/schemes.rs - Interpolation schemes for AMR

use ndarray::Array1;

/// Type of interpolation scheme
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationType {
    Linear,
    Quadratic,
    Cubic,
    Spectral,
}

/// Interpolation scheme configuration
#[derive(Debug, Clone)]
pub struct InterpolationScheme {
    pub interpolation_type: InterpolationType,
    pub order: usize,
    pub preserve_conservation: bool,
}

impl InterpolationScheme {
    /// Create a new interpolation scheme
    pub fn new(interpolation_type: InterpolationType) -> Self {
        let order = match interpolation_type {
            InterpolationType::Linear => 1,
            InterpolationType::Quadratic => 2,
            InterpolationType::Cubic => 3,
            InterpolationType::Spectral => 8,
        };

        Self {
            interpolation_type,
            order,
            preserve_conservation: true,
        }
    }

    /// Get stencil size for this scheme
    pub fn stencil_size(&self) -> usize {
        self.order + 1
    }

    /// Compute interpolation weights
    pub fn compute_weights(&self, x: f64) -> Array1<f64> {
        match self.interpolation_type {
            InterpolationType::Linear => self.linear_weights(x),
            InterpolationType::Quadratic => self.quadratic_weights(x),
            InterpolationType::Cubic => self.cubic_weights(x),
            InterpolationType::Spectral => self.spectral_weights(x),
        }
    }

    fn linear_weights(&self, x: f64) -> Array1<f64> {
        Array1::from_vec(vec![1.0 - x, x])
    }

    fn quadratic_weights(&self, x: f64) -> Array1<f64> {
        Array1::from_vec(vec![0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)])
    }

    fn cubic_weights(&self, x: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            -x * (x - 1.0) * (x - 2.0) / 6.0,
            (x + 1.0) * (x - 1.0) * (x - 2.0) / 2.0,
            -x * (x + 1.0) * (x - 2.0) / 2.0,
            x * (x + 1.0) * (x - 1.0) / 6.0,
        ])
    }

    fn spectral_weights(&self, _x: f64) -> Array1<f64> {
        // Simplified - would use FFT-based interpolation
        Array1::zeros(self.order + 1)
    }
}
