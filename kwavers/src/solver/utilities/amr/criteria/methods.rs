//! Error estimation algorithms for [`ErrorEstimator`].
//!
//! All methods are `pub(super)` — dispatched exclusively through
//! `ErrorEstimator::estimate_error` in `mod.rs`.

use super::ErrorEstimator;
use crate::core::error::KwaversResult;
use ndarray::{Array2, Array3, Axis};

impl ErrorEstimator {
    /// Gradient-based error estimation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn gradient_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dx = field[[i + 1, j, k]] - field[[i - 1, j, k]];
                    let dy = field[[i, j + 1, k]] - field[[i, j - 1, k]];
                    let dz = field[[i, j, k + 1]] - field[[i, j, k - 1]];

                    error[[i, j, k]] = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn curvature_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        // Compute Laplacian (curvature measure)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let d2x = 2.0f64.mul_add(-field[[i, j, k]], field[[i + 1, j, k]])
                        + field[[i - 1, j, k]];
                    let d2y = 2.0f64.mul_add(-field[[i, j, k]], field[[i, j + 1, k]])
                        + field[[i, j - 1, k]];
                    let d2z = 2.0f64.mul_add(-field[[i, j, k]], field[[i, j, k + 1]])
                        + field[[i, j, k - 1]];

                    error[[i, j, k]] = (d2x.abs() + d2y.abs() + d2z.abs()) / 3.0;
                }
            }
        }

        Ok(error)
    }

    /// Richardson extrapolation error estimation.
    ///
    /// Estimates truncation error by comparing solutions on different grid resolutions.
    /// For a pth-order method: error ≈ (u_h - u_2h) / (2^p - 1).
    ///
    /// References:
    /// - Richardson (1911): "The approximate arithmetical solution by finite differences"
    /// - Berger & Oliger (1984): "Adaptive mesh refinement for hyperbolic PDEs"
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn richardson_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        use crate::solver::utilities::amr::interpolation::ConservativeInterpolator;

        let interpolator = ConservativeInterpolator::new();
        let coarse = interpolator.restrict(field);
        let prolonged = interpolator.prolongate(&coarse);

        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        // 2nd order spatial discretization: scaling = 1 / (2^2 - 1) = 1/3
        let order: f64 = 2.0;
        let scaling = 1.0 / (order.exp2() - 1.0);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let diff = (field[[i, j, k]] - prolonged[[i, j, k]]).abs();
                    error[[i, j, k]] = diff * scaling;
                }
            }
        }

        if self.smoothing > 0.0 {
            self.smooth_field(&mut error)?;
        }

        Ok(error)
    }

    /// Wavelet-based error estimation.
    ///
    /// Uses multiresolution wavelet analysis to detect discontinuities.
    /// High-frequency wavelet coefficients indicate regions requiring refinement.
    ///
    /// References:
    /// - Harten (1995): "Multiresolution algorithms for the numerical solution of hyperbolic conservation laws"
    /// - Cohen et al. (2003): "Wavelet methods in numerical analysis"
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn wavelet_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        use crate::solver::utilities::amr::wavelet::{WaveletBasis, WaveletTransform};

        let wavelet = WaveletTransform::new(WaveletBasis::Daubechies(4), 2);
        let coeffs = wavelet.forward(field)?;

        let (nx, ny, nz) = coeffs.dim();
        let mut error = Array3::zeros(field.dim());

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut detail_energy = 0.0;
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

        let max_error = error.iter().fold(0.0_f64, |max, &val| max.max(val));
        if max_error > 1e-10 {
            error.par_mapv_inplace(|e| e / max_error);
        }

        if self.smoothing > 0.0 {
            self.smooth_field(&mut error)?;
        }

        Ok(error)
    }

    /// Physics-based error estimation.
    ///
    /// Detects shocks via gradient/curvature ratio; combines gradient and
    /// curvature indicators with scale-invariant normalization.
    ///
    /// References:
    /// - Lohner (1987): "An adaptive finite element scheme for transient problems"
    /// - Berger & Colella (1989): "Local adaptive mesh refinement for shock hydrodynamics"
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn physics_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut error = Array3::zeros(field.dim());

        let grad = self.gradient_error(field)?;
        let curv = self.curvature_error(field)?;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let gradient = grad[[i, j, k]];
                    let curvature = curv[[i, j, k]];
                    let value = field[[i, j, k]];

                    let shock_indicator = if gradient > 1e-10 {
                        curvature / (gradient + 1e-10)
                    } else {
                        0.0
                    };

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

                    let shock_weight = 2.0 * shock_indicator.abs().min(1.0);
                    let gradient_weight = 1.0 + normalized_variation;

                    error[[i, j, k]] = shock_weight * curvature + gradient_weight * gradient;
                }
            }
        }

        let max_error = error.iter().fold(0.0_f64, |max, &val| max.max(val));
        if max_error > 1e-10 {
            error.par_mapv_inplace(|e| e / max_error);
        }

        Ok(error)
    }

    /// Apply smoothing to reduce noise.
    ///
    /// Sliding 3×3×3 averaging filter; `self.smoothing` controls the blend
    /// weight between the original and smoothed values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn smooth_field(&self, field: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 3 || ny < 3 || nz < 3 {
            return Ok(());
        }

        let mut prev_plane = field.index_axis(Axis(0), 0).to_owned();
        let mut curr_plane = field.index_axis(Axis(0), 1).to_owned();
        let mut next_plane = Array2::<f64>::zeros((ny, nz));

        for i in 1..nx - 1 {
            next_plane.assign(&field.index_axis(Axis(0), i + 1));

            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let mut sum = 0.0;

                    sum += prev_plane[[j - 1, k - 1]]
                        + prev_plane[[j - 1, k]]
                        + prev_plane[[j - 1, k + 1]]
                        + prev_plane[[j, k - 1]]
                        + prev_plane[[j, k]]
                        + prev_plane[[j, k + 1]]
                        + prev_plane[[j + 1, k - 1]]
                        + prev_plane[[j + 1, k]]
                        + prev_plane[[j + 1, k + 1]];

                    sum += curr_plane[[j - 1, k - 1]]
                        + curr_plane[[j - 1, k]]
                        + curr_plane[[j - 1, k + 1]]
                        + curr_plane[[j, k - 1]]
                        + curr_plane[[j, k]]
                        + curr_plane[[j, k + 1]]
                        + curr_plane[[j + 1, k - 1]]
                        + curr_plane[[j + 1, k]]
                        + curr_plane[[j + 1, k + 1]];

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
                        (1.0 - self.smoothing).mul_add(old_val, self.smoothing * (sum / count));
                }
            }

            // Cycle plane buffers to avoid allocations
            let temp = prev_plane;
            prev_plane = curr_plane;
            curr_plane = next_plane;
            next_plane = temp;
        }

        Ok(())
    }
}
