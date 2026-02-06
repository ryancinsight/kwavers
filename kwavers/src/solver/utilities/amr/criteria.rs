//! Refinement criteria and error estimation

use crate::core::error::KwaversResult;
use ndarray::{Array2, Array3, Axis};

/// Refinement criterion types
#[derive(Debug, Clone, Copy)]
pub enum RefinementCriterion {
    /// Gradient-based criterion
    Gradient,
    /// Curvature-based criterion
    Curvature,
    /// Richardson extrapolation
    Richardson,
    /// Wavelet-based criterion
    Wavelet,
    /// Physics-based (e.g., shock detection)
    Physics,
}

/// Error estimator for adaptive refinement
#[derive(Debug)]
pub struct ErrorEstimator {
    criterion: RefinementCriterion,
    /// Smoothing parameter for noise reduction
    smoothing: f64,
}

impl Default for ErrorEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorEstimator {
    /// Create a new error estimator
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: RefinementCriterion::Gradient,
            smoothing: 0.1,
        }
    }

    /// Estimate error in the field
    pub fn estimate_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        match self.criterion {
            RefinementCriterion::Gradient => self.gradient_error(field),
            RefinementCriterion::Curvature => self.curvature_error(field),
            RefinementCriterion::Richardson => self.richardson_error(field),
            RefinementCriterion::Wavelet => self.wavelet_error(field),
            RefinementCriterion::Physics => self.physics_error(field),
        }
    }

    /// Gradient-based error estimation
    fn gradient_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dx = field[[i + 1, j, k]] - field[[i - 1, j, k]];
                    let dy = field[[i, j + 1, k]] - field[[i, j - 1, k]];
                    let dz = field[[i, j, k + 1]] - field[[i, j, k - 1]];

                    error[[i, j, k]] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
        }

        // Apply smoothing
        if self.smoothing > 0.0 {
            self.smooth_field(&mut error)?;
        }

        Ok(error)
    }

    /// Curvature-based error estimation
    fn curvature_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        // Compute Laplacian (curvature measure)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let d2x = field[[i + 1, j, k]] - 2.0 * field[[i, j, k]] + field[[i - 1, j, k]];
                    let d2y = field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]];
                    let d2z = field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]];

                    error[[i, j, k]] = (d2x.abs() + d2y.abs() + d2z.abs()) / 3.0;
                }
            }
        }

        Ok(error)
    }

    /// Richardson extrapolation error estimation
    ///
    /// Estimates truncation error by comparing solutions on different grid resolutions.
    /// The difference between fine and coarse solutions provides an error estimate.
    ///
    /// For a pth-order method: error ≈ (u_h - u_2h) / (2^p - 1)
    ///
    /// References:
    /// - Richardson (1911): "The approximate arithmetical solution by finite differences"
    /// - Berger & Oliger (1984): "Adaptive mesh refinement for hyperbolic PDEs"
    fn richardson_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        use super::interpolation::ConservativeInterpolator;

        // Create interpolator for grid transfers
        let interpolator = ConservativeInterpolator::new();

        // Restrict to coarse grid
        let coarse = interpolator.restrict(field);

        // Prolongate back to fine grid
        let prolonged = interpolator.prolongate(&coarse);

        // Error estimate from difference between fine and prolonged solutions
        // For 2nd order methods: error ≈ (u_h - u_2h) / 3
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        let order = 2.0; // Assume 2nd order spatial discretization
        let scaling = 1.0 / (2.0_f64.powf(order) - 1.0);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Richardson error estimate
                    let diff = (field[[i, j, k]] - prolonged[[i, j, k]]).abs();
                    error[[i, j, k]] = diff * scaling;
                }
            }
        }

        // Apply smoothing to reduce noise in error estimate
        if self.smoothing > 0.0 {
            self.smooth_field(&mut error)?;
        }

        Ok(error)
    }

    /// Wavelet-based error estimation
    ///
    /// Uses multiresolution wavelet analysis to detect discontinuities and sharp features.
    /// High-frequency wavelet coefficients indicate regions requiring refinement.
    ///
    /// Based on:
    /// - Harten (1995): "Multiresolution algorithms for the numerical solution of hyperbolic conservation laws"
    /// - Cohen et al. (2003): "Wavelet methods in numerical analysis"
    fn wavelet_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        use super::wavelet::{WaveletBasis, WaveletTransform};

        // Use Daubechies-4 wavelets for good localization and smoothness
        let wavelet = WaveletTransform::new(WaveletBasis::Daubechies(4), 2);

        // Forward wavelet transform
        let coeffs = wavelet.forward(field)?;

        // Error estimate from high-frequency wavelet coefficients
        // Detail coefficients in second half of each dimension indicate local irregularity
        let (nx, ny, nz) = coeffs.dim();
        let mut error = Array3::zeros(field.dim());

        // Compute error from wavelet detail coefficients
        // Higher detail coefficients indicate regions needing refinement
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Aggregate detail energy across all wavelet subbands
                    let mut detail_energy = 0.0;

                    // High-frequency components in x, y, z directions
                    if i >= nx / 2 {
                        detail_energy += coeffs[[i, j, k]].abs();
                    }
                    if j >= ny / 2 {
                        detail_energy += coeffs[[i, j, k]].abs();
                    }
                    if k >= nz / 2 {
                        detail_energy += coeffs[[i, j, k]].abs();
                    }

                    error[[i, j, k]] = detail_energy;
                }
            }
        }

        // Normalize error by maximum for stability
        let max_error = error.iter().fold(0.0_f64, |max, &val| max.max(val));
        if max_error > 1e-10 {
            error.mapv_inplace(|e| e / max_error);
        }

        // Apply smoothing to reduce noise in error estimate
        if self.smoothing > 0.0 {
            self.smooth_field(&mut error)?;
        }

        Ok(error)
    }

    /// Physics-based error estimation
    ///
    /// Detects physical features requiring refinement:
    /// - Shocks and discontinuities via gradient jumps
    /// - High curvature regions via Laplacian
    /// - Vortical structures via Q-criterion (when applicable)
    ///
    /// References:
    /// - Lohner (1987): "An adaptive finite element scheme for transient problems"
    /// - Berger & Colella (1989): "Local adaptive mesh refinement for shock hydrodynamics"
    fn physics_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        // Compute both gradient and curvature for comprehensive detection
        let grad = self.gradient_error(field)?;
        let curv = self.curvature_error(field)?;

        // Combine using weighted sum with shock detection
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let gradient = grad[[i, j, k]];
                    let curvature = curv[[i, j, k]];
                    let value = field[[i, j, k]];

                    // Shock indicator: ratio of second to first derivative
                    // Large values indicate discontinuities
                    let shock_indicator = if gradient > 1e-10 {
                        curvature / (gradient + 1e-10)
                    } else {
                        0.0
                    };

                    // Normalized field variation for scale-invariance
                    let max_neighbor = [
                        field[[i + 1, j, k]],
                        field[[i - 1, j, k]],
                        field[[i, j + 1, k]],
                        field[[i, j - 1, k]],
                        field[[i, j, k + 1]],
                        field[[i, j, k - 1]],
                    ]
                    .iter()
                    .fold(value.abs(), |max, &v| max.max(v.abs()));

                    let normalized_variation = if max_neighbor > 1e-10 {
                        gradient / max_neighbor
                    } else {
                        0.0
                    };

                    // Combine indicators with physics-based weighting
                    // Emphasize shock regions (high curvature relative to gradient)
                    let shock_weight = 2.0 * shock_indicator.abs().min(1.0);
                    let gradient_weight = 1.0 + normalized_variation;

                    error[[i, j, k]] = shock_weight * curvature + gradient_weight * gradient;
                }
            }
        }

        // Normalize for stability
        let max_error = error.iter().fold(0.0_f64, |max, &val| max.max(val));
        if max_error > 1e-10 {
            error.mapv_inplace(|e| e / max_error);
        }

        Ok(error)
    }

    /// Apply smoothing to reduce noise
    fn smooth_field(&self, field: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 3 || ny < 3 || nz < 3 {
            return Ok(());
        }

        // Use a sliding window approach to avoid full 3D clone
        // buffers store the (i-1)-th, i-th, and (i+1)-th slices
        let mut prev_plane = field.index_axis(Axis(0), 0).to_owned();
        let mut curr_plane = field.index_axis(Axis(0), 1).to_owned();
        let mut next_plane = Array2::<f64>::zeros((ny, nz));

        // Simple 3x3x3 averaging filter
        for i in 1..nx - 1 {
            // Load next plane
            next_plane.assign(&field.index_axis(Axis(0), i + 1));

            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let mut sum = 0.0;

                    // Sum 3x3 window from prev_plane (slice i-1)
                    sum += prev_plane[[j - 1, k - 1]]
                        + prev_plane[[j - 1, k]]
                        + prev_plane[[j - 1, k + 1]]
                        + prev_plane[[j, k - 1]]
                        + prev_plane[[j, k]]
                        + prev_plane[[j, k + 1]]
                        + prev_plane[[j + 1, k - 1]]
                        + prev_plane[[j + 1, k]]
                        + prev_plane[[j + 1, k + 1]];

                    // Sum 3x3 window from curr_plane (slice i)
                    sum += curr_plane[[j - 1, k - 1]]
                        + curr_plane[[j - 1, k]]
                        + curr_plane[[j - 1, k + 1]]
                        + curr_plane[[j, k - 1]]
                        + curr_plane[[j, k]]
                        + curr_plane[[j, k + 1]]
                        + curr_plane[[j + 1, k - 1]]
                        + curr_plane[[j + 1, k]]
                        + curr_plane[[j + 1, k + 1]];

                    // Sum 3x3 window from next_plane (slice i+1)
                    sum += next_plane[[j - 1, k - 1]]
                        + next_plane[[j - 1, k]]
                        + next_plane[[j - 1, k + 1]]
                        + next_plane[[j, k - 1]]
                        + next_plane[[j, k]]
                        + next_plane[[j, k + 1]]
                        + next_plane[[j + 1, k - 1]]
                        + next_plane[[j + 1, k]]
                        + next_plane[[j + 1, k + 1]];

                    let count = 27.0;

                    let old_val = curr_plane[[j, k]];
                    field[[i, j, k]] =
                        (1.0 - self.smoothing) * old_val + self.smoothing * (sum / count);
                }
            }

            // Cycle buffers: prev -> next (recycle), curr -> prev, next -> curr
            let temp = prev_plane;
            prev_plane = curr_plane;
            curr_plane = next_plane;
            next_plane = temp;
        }

        Ok(())
    }
}
