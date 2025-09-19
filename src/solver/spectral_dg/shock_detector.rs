//! Shock detection for Discontinuous Galerkin methods
//!
//! Implements shock detectors to identify regions requiring limiting.

use crate::KwaversResult;
use ndarray::Array3;

/// Shock detection threshold constants
pub const SHOCK_DETECTOR_THRESHOLD: f64 = 0.5;
pub const MODAL_DECAY_THRESHOLD: f64 = 1e-3;

/// Shock detector for identifying discontinuities
#[derive(Debug)]
pub struct ShockDetector {
    threshold: f64,
    #[allow(dead_code)]
    polynomial_order: usize,
}

impl ShockDetector {
    /// Create a new shock detector
    #[must_use]
    pub fn new(polynomial_order: usize) -> Self {
        Self {
            threshold: SHOCK_DETECTOR_THRESHOLD,
            polynomial_order,
        }
    }

    /// Apply shock detection to identify cells needing limiting
    ///
    /// Uses modal decay indicator: if high-order modes contain significant
    /// energy, the solution likely contains a discontinuity.
    #[must_use]
    pub fn detect(&self, field: &Array3<f64>) -> Array3<bool> {
        let shape = field.dim();
        let mut shock_cells = Array3::from_elem(shape, false);

        // For each element
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    // Extract modal coefficients for this element
                    let value = field[[i, j, k]];

                    // Simple gradient-based detector
                    let mut max_gradient: f64 = 0.0;

                    // Check x-direction gradient
                    if i > 0 && i < shape.0 - 1 {
                        let grad_x = (field[[i + 1, j, k]] - field[[i - 1, j, k]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_x);
                    }

                    // Check y-direction gradient
                    if j > 0 && j < shape.1 - 1 {
                        let grad_y = (field[[i, j + 1, k]] - field[[i, j - 1, k]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_y);
                    }

                    // Check z-direction gradient
                    if k > 0 && k < shape.2 - 1 {
                        let grad_z = (field[[i, j, k + 1]] - field[[i, j, k - 1]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_z);
                    }

                    // Mark as shock if gradient exceeds threshold relative to value
                    if value.abs() > 1e-10 {
                        shock_cells[[i, j, k]] = max_gradient / value.abs() > self.threshold;
                    }
                }
            }
        }

        shock_cells
    }

    /// Apply modal shock detector using coefficient decay
    pub fn detect_modal(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<bool>> {
        let shape = coefficients.dim();
        let mut shock_cells = Array3::from_elem(shape, false);

        // Analyze modal content
        // High-order modes should decay exponentially for smooth solutions
        // If they don't, we have a discontinuity

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    // For DG, we would analyze modal coefficients
                    // Here we use a simple gradient check as placeholder
                    let value = coefficients[[i, j, k]];

                    // Check if this cell has high-frequency content
                    if i > 0 && i < shape.0 - 1 {
                        let second_deriv =
                            coefficients[[i + 1, j, k]] - 2.0 * value + coefficients[[i - 1, j, k]];

                        if value.abs() > 1e-10 {
                            shock_cells[[i, j, k]] =
                                (second_deriv.abs() / value.abs()) > MODAL_DECAY_THRESHOLD;
                        }
                    }
                }
            }
        }

        Ok(shock_cells)
    }
}
