//! Gradient processing: smoothing, regularization, near-source mute, TV/Laplacian helpers.

use super::FwiProcessor;
use crate::core::error::KwaversResult;
use ndarray::{s, Array3, Zip};

/// Zero the gradient within `radius` voxels (L2 norm) of every active source voxel.
///
/// ## Theorem (near-source artefact suppression)
///
/// At voxels within `λ/2` of a source, `∂²p/∂t²` is dominated by the second
/// derivative of the source wavelet, not by scattered wave physics.  The
/// cross-correlation produces a gradient 10–100× larger than the physical
/// sensitivity, masking the useful signal and causing the normalized gradient to
/// point in the wrong direction.
///
/// ## Reference
///
/// Virieux & Operto (2009), *Geophysics* 74(6), WCC1–WCC26, §Gradient preconditioner.
pub(super) fn mute_gradient_near_sources(
    gradient: &mut Array3<f64>,
    source_p_mask: &Array3<f64>,
    radius: usize,
) {
    let r_sq = (radius * radius) as f64;
    let (nx, ny, nz) = gradient.dim();
    for ((si, sj, sk), &m) in source_p_mask.indexed_iter() {
        if m > 0.5 {
            let imin = si.saturating_sub(radius);
            let imax = (si + radius + 1).min(nx);
            let jmin = sj.saturating_sub(radius);
            let jmax = (sj + radius + 1).min(ny);
            let kmin = sk.saturating_sub(radius);
            let kmax = (sk + radius + 1).min(nz);
            for i in imin..imax {
                for j in jmin..jmax {
                    for k in kmin..kmax {
                        let dr_sq = ((i as isize - si as isize).pow(2)
                            + (j as isize - sj as isize).pow(2)
                            + (k as isize - sk as isize).pow(2))
                            as f64;
                        if dr_sq <= r_sq {
                            gradient[[i, j, k]] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

impl FwiProcessor {
    /// Calculate interaction between two fields (used for testing gradient kernel).
    #[must_use]
    pub fn calculate_interaction(
        &self,
        forward_field: &Array3<f64>,
        adjoint_field: &Array3<f64>,
    ) -> Array3<f64> {
        let mut gradient = Array3::zeros(forward_field.dim());
        Zip::from(&mut gradient)
            .and(forward_field)
            .and(adjoint_field)
            .for_each(|g, &fwd, &adj| {
                *g = -fwd * adj;
            });
        self.smooth_gradient(&gradient)
    }

    /// Apply smoothing to gradient to reduce high-frequency artifacts.
    ///
    /// # Algorithm
    ///
    /// **ny ≤ 2 (quasi-2-D):** 3×3 box filter in the x–z plane applied to every
    /// y-slice independently.
    ///
    /// **ny > 2 (full-3-D):** 6-connected stencil with centre weighted 3/9.
    /// Each of the six face-connected neighbours contributes 1/9; centre 3/9.
    ///
    /// # Allocation strategy
    ///
    /// Allocates one `Array3::zeros` instead of `gradient.clone()`. Only the
    /// boundary faces (O(N²) elements) are copied from the input.
    #[must_use]
    pub(super) fn smooth_gradient(&self, gradient: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = gradient.dim();
        let mut smoothed = Array3::<f64>::zeros((nx, ny, nz));

        smoothed
            .slice_mut(s![0, .., ..])
            .assign(&gradient.slice(s![0, .., ..]));
        smoothed
            .slice_mut(s![nx - 1, .., ..])
            .assign(&gradient.slice(s![nx - 1, .., ..]));
        smoothed
            .slice_mut(s![.., 0, ..])
            .assign(&gradient.slice(s![.., 0, ..]));
        smoothed
            .slice_mut(s![.., ny - 1, ..])
            .assign(&gradient.slice(s![.., ny - 1, ..]));
        smoothed
            .slice_mut(s![.., .., 0])
            .assign(&gradient.slice(s![.., .., 0]));
        smoothed
            .slice_mut(s![.., .., nz - 1])
            .assign(&gradient.slice(s![.., .., nz - 1]));

        if ny <= 2 {
            for i in 1..nx - 1 {
                for j in 0..ny {
                    for k in 1..nz - 1 {
                        smoothed[[i, j, k]] = (gradient[[i - 1, j, k - 1]]
                            + gradient[[i, j, k - 1]]
                            + gradient[[i + 1, j, k - 1]]
                            + gradient[[i - 1, j, k]]
                            + gradient[[i, j, k]]
                            + gradient[[i + 1, j, k]]
                            + gradient[[i - 1, j, k + 1]]
                            + gradient[[i, j, k + 1]]
                            + gradient[[i + 1, j, k + 1]])
                            / 9.0;
                    }
                }
            }
        } else {
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        smoothed[[i, j, k]] = (gradient[[i - 1, j, k]]
                            + gradient[[i + 1, j, k]]
                            + gradient[[i, j - 1, k]]
                            + gradient[[i, j + 1, k]]
                            + gradient[[i, j, k - 1]]
                            + gradient[[i, j, k + 1]]
                            + 3.0 * gradient[[i, j, k]])
                            / 9.0;
                    }
                }
            }
        }

        smoothed
    }

    /// Apply regularization to gradient.
    pub(super) fn apply_regularization(
        &self,
        gradient: &Array3<f64>,
        model: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut regularized = gradient.clone();
        let reg_params = &self.parameters.regularization;

        if reg_params.tikhonov_weight > 0.0 {
            let w = reg_params.tikhonov_weight;
            Zip::from(&mut regularized)
                .and(model)
                .par_for_each(|r, &m| *r += m * w);
        }

        if reg_params.tv_weight > 0.0 {
            let tv_term = self.compute_total_variation_gradient(model);
            let w = reg_params.tv_weight;
            Zip::from(&mut regularized)
                .and(&tv_term)
                .par_for_each(|r, &t| *r += t * w);
        }

        if reg_params.smoothness_weight > 0.0 {
            let smoothness_term = self.compute_smoothness_gradient(model);
            let w = reg_params.smoothness_weight;
            Zip::from(&mut regularized)
                .and(&smoothness_term)
                .par_for_each(|r, &s| *r += s * w);
        }

        Ok(regularized)
    }

    /// Compute total variation gradient for regularization.
    ///
    /// # Loop ordering
    ///
    /// `i`-outer, `k`-inner: inner loop reads `model[[i,j,k±1]]` at stride 1
    /// (C-order last-index varies fastest), minimising cache misses.
    #[must_use]
    pub(super) fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut tv_gradient = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dx = model[[i + 1, j, k]] - model[[i - 1, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j - 1, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k - 1]];
                    let grad_mag = (dx * dx + dy * dy + dz * dz).sqrt();
                    if grad_mag > f64::EPSILON {
                        tv_gradient[[i, j, k]] = grad_mag;
                    }
                }
            }
        }

        tv_gradient
    }

    /// Compute smoothness gradient (Laplacian) for regularization.
    ///
    /// # Loop ordering
    ///
    /// `i`-outer, `k`-inner: inner-loop accesses `model[[i,j,k±1]]` at stride 1.
    #[must_use]
    pub(super) fn compute_smoothness_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    laplacian[[i, j, k]] = model[[i + 1, j, k]]
                        + model[[i - 1, j, k]]
                        + model[[i, j + 1, k]]
                        + model[[i, j - 1, k]]
                        + model[[i, j, k + 1]]
                        + model[[i, j, k - 1]]
                        - 6.0 * model[[i, j, k]];
                }
            }
        }

        laplacian
    }
}
