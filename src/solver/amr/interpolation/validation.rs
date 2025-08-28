// interpolation/validation.rs - Interpolation validation

use ndarray::Array3;

/// Validator for interpolation operations
pub struct InterpolationValidator;

impl InterpolationValidator {
    /// Check conservation properties
    pub fn check_conservation(coarse: &Array3<f64>, fine: &Array3<f64>, ratio: usize) -> bool {
        let coarse_sum = coarse.sum();
        let fine_sum = fine.sum();
        let scale = ratio.pow(3) as f64;

        (coarse_sum * scale - fine_sum).abs() / coarse_sum.abs().max(1e-10) < 1e-10
    }

    /// Check smoothness
    pub fn check_smoothness(field: &Array3<f64>) -> f64 {
        field.std(0.0)
    }
}
