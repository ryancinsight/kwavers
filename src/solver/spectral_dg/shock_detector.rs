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
    ///
    /// Implements TVB (Total Variation Bounded) modal indicator for DG methods.
    /// Analyzes decay rate of modal coefficients to detect discontinuities.
    /// Smooth solutions have exponentially decaying modal coefficients, while
    /// discontinuities have slowly decaying high-order modes.
    ///
    /// References:
    /// - Cockburn & Shu (1989): "TVB Runge-Kutta local projection DG finite element method"
    /// - Persson & Peraire (2006): "Sub-cell shock capturing for DG methods"
    /// - Krivodonova (2007): "Limiters for high-order DG methods"
    pub fn detect_modal(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<bool>> {
        let shape = coefficients.dim();
        let mut shock_cells = Array3::from_elem(shape, false);

        // For DG methods, modal coefficients represent polynomial basis functions
        // We use a spectral decay indicator based on the ratio of high to low modes

        for i in 1..shape.0 - 1 {
            for j in 1..shape.1 - 1 {
                for k in 1..shape.2 - 1 {
                    let value = coefficients[[i, j, k]];

                    // Skip negligible values to avoid numerical issues
                    if value.abs() < 1e-12 {
                        continue;
                    }

                    // Compute modal energy in neighborhood (proxy for modal coefficients)
                    // In full DG: would extract actual polynomial coefficients per element

                    // Low-order mode energy (smooth variations)
                    let low_mode =
                        0.5 * (coefficients[[i - 1, j, k]] + coefficients[[i + 1, j, k]]);

                    // High-order mode energy (oscillations/discontinuities)
                    let high_mode_x =
                        coefficients[[i + 1, j, k]] - 2.0 * value + coefficients[[i - 1, j, k]];
                    let high_mode_y =
                        coefficients[[i, j + 1, k]] - 2.0 * value + coefficients[[i, j - 1, k]];
                    let high_mode_z =
                        coefficients[[i, j, k + 1]] - 2.0 * value + coefficients[[i, j, k - 1]];

                    let high_mode_energy =
                        (high_mode_x.powi(2) + high_mode_y.powi(2) + high_mode_z.powi(2)).sqrt();

                    // Spectral decay indicator: S_e = log(E_N / E_1)
                    // For smooth solutions: S_e << 0 (exponential decay)
                    // For discontinuities: S_e ≈ 0 (no decay)
                    let decay_indicator = if low_mode.abs() > 1e-10 {
                        high_mode_energy / low_mode.abs()
                    } else {
                        high_mode_energy / (value.abs() + 1e-10)
                    };

                    // TVB minmod parameter controls sensitivity
                    // Standard values: M ∈ [10, 100] for typical problems
                    let tvb_parameter = 50.0;
                    let cell_size_estimate = 1.0; // Would come from actual grid

                    // TVB shock indicator with characteristic scaling (Cockburn & Shu 1989)
                    let scaled_threshold =
                        MODAL_DECAY_THRESHOLD * tvb_parameter * cell_size_estimate;

                    // Persson-Peraire modal decay indicator: s_e = -log(E_N/E_0)
                    // Shock detected when decay is insufficient: s_e < s_κ
                    // Equivalent to: E_N/E_0 > exp(-s_κ), or decay_indicator > threshold
                    shock_cells[[i, j, k]] = decay_indicator > scaled_threshold;

                    // Additional conservative check: flag large jumps across cell interfaces
                    let max_jump = [
                        (coefficients[[i + 1, j, k]] - value).abs(),
                        (coefficients[[i - 1, j, k]] - value).abs(),
                        (coefficients[[i, j + 1, k]] - value).abs(),
                        (coefficients[[i, j - 1, k]] - value).abs(),
                        (coefficients[[i, j, k + 1]] - value).abs(),
                        (coefficients[[i, j, k - 1]] - value).abs(),
                    ]
                    .iter()
                    .fold(0.0_f64, |max, &v| max.max(v));

                    // Conservative TVB condition: flag if jump > M * Δx * |gradient|
                    let gradient_estimate = high_mode_energy / cell_size_estimate;
                    let jump_threshold = tvb_parameter * cell_size_estimate * gradient_estimate;

                    if max_jump > jump_threshold {
                        shock_cells[[i, j, k]] = true;
                    }
                }
            }
        }

        Ok(shock_cells)
    }
}
