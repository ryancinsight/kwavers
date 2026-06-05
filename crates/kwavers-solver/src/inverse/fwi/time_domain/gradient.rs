//! Gradient processing: smoothing, regularization, near-source mute, TV/Laplacian helpers.

use super::FwiProcessor;
use kwavers_core::error::KwaversResult;
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
            .par_for_each(|g, &fwd, &adj| {
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
                        smoothed[[i, j, k]] = 3.0f64.mul_add(
                            gradient[[i, j, k]],
                            gradient[[i - 1, j, k]]
                                + gradient[[i + 1, j, k]]
                                + gradient[[i, j - 1, k]]
                                + gradient[[i, j + 1, k]]
                                + gradient[[i, j, k - 1]]
                                + gradient[[i, j, k + 1]],
                        ) / 9.0;
                    }
                }
            }
        }

        smoothed
    }

    /// Apply regularization to gradient.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// Returns `∂TV/∂m` using the Huber-smoothed isotropic TV functional:
    ///   TV(m) = Σ_ijk √(fx² + fy² + fz² + ε²)
    /// where fx[i,j,k] = m[i+1,j,k]−m[i,j,k] are forward differences.
    ///
    /// The functional derivative at (i,j,k) is (via discrete chain rule):
    ///   ∂TV/∂m[i,j,k] = −(fx + fy + fz)/W[i,j,k]
    ///                   + fx[i−1,j,k]/W[i−1,j,k]  (if i > 0)
    ///                   + fy[i,j−1,k]/W[i,j−1,k]  (if j > 0)
    ///                   + fz[i,j,k−1]/W[i,j,k−1]  (if k > 0)
    /// where W[i,j,k] = √(fx² + fy² + fz² + ε²) is the Huber weight.
    ///
    /// ## Reference
    ///
    /// Rudin, Osher & Fatemi (1992). Physica D 60, 259–268, Eq. (11).
    #[must_use]
    pub(super) fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let eps2 = 1e-16_f64; // Huber ε² prevents division by zero

        // Pre-compute forward differences.
        let mut fx = Array3::<f64>::zeros((nx, ny, nz)); // fx[i] = m[i+1] − m[i]
        let mut fy = Array3::<f64>::zeros((nx, ny, nz));
        let mut fz = Array3::<f64>::zeros((nx, ny, nz));

        for i in 0..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    fx[[i, j, k]] = model[[i + 1, j, k]] - model[[i, j, k]];
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny - 1 {
                for k in 0..nz {
                    fy[[i, j, k]] = model[[i, j + 1, k]] - model[[i, j, k]];
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz - 1 {
                    fz[[i, j, k]] = model[[i, j, k + 1]] - model[[i, j, k]];
                }
            }
        }

        // Huber weights W[i,j,k] = √(fx² + fy² + fz² + ε²).
        let mut w = Array3::<f64>::zeros((nx, ny, nz));
        Zip::from(&mut w)
            .and(&fx)
            .and(&fy)
            .and(&fz)
            .par_for_each(|w_val, &dx, &dy, &dz| {
                *w_val = dz.mul_add(dz, dx.mul_add(dx, dy * dy) + eps2).sqrt();
            });

        // Functional derivative: divergence of the normalised gradient field.
        let mut tv_gradient = Array3::<f64>::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let w_c = w[[i, j, k]];
                    // Negative self-contribution from the forward differences leaving (i,j,k).
                    let mut g = -(fx[[i, j, k]] + fy[[i, j, k]] + fz[[i, j, k]]) / w_c;
                    // Positive back-contributions from forward differences ending at (i,j,k).
                    if i > 0 {
                        g += fx[[i - 1, j, k]] / w[[i - 1, j, k]];
                    }
                    if j > 0 {
                        g += fy[[i, j - 1, k]] / w[[i, j - 1, k]];
                    }
                    if k > 0 {
                        g += fz[[i, j, k - 1]] / w[[i, j, k - 1]];
                    }
                    tv_gradient[[i, j, k]] = g;
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
                    laplacian[[i, j, k]] = 6.0f64.mul_add(
                        -model[[i, j, k]],
                        model[[i + 1, j, k]]
                            + model[[i - 1, j, k]]
                            + model[[i, j + 1, k]]
                            + model[[i, j - 1, k]]
                            + model[[i, j, k + 1]]
                            + model[[i, j, k - 1]],
                    );
                }
            }
        }

        laplacian
    }
}
