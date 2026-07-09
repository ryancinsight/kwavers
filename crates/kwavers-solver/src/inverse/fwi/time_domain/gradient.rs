//! Gradient processing: smoothing, regularization, near-source mute, TV/Laplacian helpers.

use super::FwiProcessor;
use crate::inverse::fwi::time_domain::field_ops::{add_scaled_field, write_negative_product};
use kwavers_core::error::KwaversResult;
use leto::Array3;

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
    let [nx, ny, nz] = gradient.shape();
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
        let mut gradient = Array3::zeros(forward_field.shape());
        write_negative_product(&mut gradient, forward_field, adjoint_field);
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
        let [nx, ny, nz] = gradient.shape();
        let mut smoothed = Array3::<f64>::zeros((nx, ny, nz));

        smoothed
            .slice_mut(s![0, .., ..]).unwrap().unwrap().assign(&gradient.slice(s![0, .., ..]));
        smoothed
            .slice_mut(s![nx - 1, .., ..]).unwrap().unwrap().assign(&gradient.slice(s![nx - 1, .., ..]));
        smoothed
            .slice_mut(s![.., 0, ..]).unwrap().unwrap().assign(&gradient.slice(s![.., 0, ..]));
        smoothed
            .slice_mut(s![.., ny - 1, ..]).unwrap().unwrap().assign(&gradient.slice(s![.., ny - 1, ..]));
        smoothed
            .slice_mut(s![.., .., 0]).unwrap().unwrap().assign(&gradient.slice(s![.., .., 0]));
        smoothed
            .slice_mut(s![.., .., nz - 1]).unwrap().unwrap().assign(&gradient.slice(s![.., .., nz - 1]));

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
    ///
    /// `dtv_scale` multiplies the four-direction-TV (FDTV) weight only; it is the
    /// adaptive schedule factor (see [`adaptive_dtv_scale`]). Pass `1.0` for the
    /// constant-weight (non-adaptive) behavior used by every driver except the
    /// steepest-descent [`FwiProcessor::invert`] loop.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_regularization(
        &self,
        gradient: &Array3<f64>,
        model: &Array3<f64>,
        dtv_scale: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut regularized = gradient.clone();
        let reg_params = &self.parameters.regularization;

        if reg_params.tikhonov_weight > 0.0 {
            let w = reg_params.tikhonov_weight;
            add_scaled_field(&mut regularized, model, w);
        }

        if reg_params.tv_weight > 0.0 {
            let tv_term = self.compute_total_variation_gradient(model);
            let w = reg_params.tv_weight;
            add_scaled_field(&mut regularized, &tv_term, w);
        }

        if reg_params.directional_tv_weight > 0.0 {
            let fdtv_term = directional_tv_gradient(model, &FDTV_DIRECTIONS);
            let w = reg_params.directional_tv_weight * dtv_scale;
            add_scaled_field(&mut regularized, &fdtv_term, w);
        }

        if reg_params.smoothness_weight > 0.0 {
            let smoothness_term = compute_smoothness_gradient(model);
            let w = reg_params.smoothness_weight;
            add_scaled_field(&mut regularized, &smoothness_term, w);
        }

        Ok(regularized)
    }

    /// Compute the axis-aligned (isotropic ROF) total-variation gradient.
    ///
    /// Thin wrapper over the generic [`directional_tv_gradient`] operator with
    /// the three Cartesian-axis difference directions. See that function for the
    /// functional and its analytically derived gradient.
    ///
    /// ## Reference
    ///
    /// Rudin, Osher & Fatemi (1992). Physica D 60, 259–268, Eq. (11).
    #[must_use]
    pub(super) fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        directional_tv_gradient(model, &AXIS_TV_DIRECTIONS)
    }
}

/// Floor for the adaptive directional-TV scale.
///
/// The FDTV prior is never fully switched off (retains 10 % of its weight at
/// convergence): a residual edge-preserving regularization keeps the
/// ill-conditioned null-space components of the sparse-aperture problem damped
/// even after the data-fit has stabilized.
pub(in crate::inverse::fwi::time_domain) const DIRECTIONAL_TV_MIN_SCALE: f64 = 0.1;

/// Adaptive directional-TV (FDTV) weight scale for the current iteration.
///
/// Returns `clamp(rel_change / max_change, min_scale, 1.0)`, the FWI analog of
/// the adaptive parameter control of the FDTV+POCS scheme (PMC10745410):
/// `rel_change` is the relative model change observed on the previous step and
/// `max_change` the largest such change seen so far, so the factor is `1.0`
/// while the model is moving fastest (early iterations) and decays toward
/// `min_scale` as the inversion converges — preserving recovered detail near
/// the solution. Returns `1.0` when no change has yet been recorded
/// (`max_change <= 0`), i.e. on the first iteration.
#[must_use]
pub(in crate::inverse::fwi::time_domain) fn adaptive_dtv_scale(
    rel_change: f64,
    max_change: f64,
    min_scale: f64,
) -> f64 {
    if max_change <= 0.0 || !rel_change.is_finite() {
        return 1.0;
    }
    (rel_change / max_change).clamp(min_scale, 1.0)
}

/// Huber smoothing constant ε² shared by the TV functional and its gradient.
///
/// Identical value in both so the analytic gradient is the exact derivative of
/// the discrete functional (verified by the finite-difference differential test).
const TV_EPSILON_SQUARED: f64 = 1e-16;

/// Axis-aligned difference directions (classic isotropic ROF TV).
///
/// Each entry is `(offset, a_d)` where `a_d = 1/|offset|²` distance-normalizes
/// the squared difference so every direction contributes a per-unit-length
/// directional derivative.
pub(in crate::inverse::fwi::time_domain) const AXIS_TV_DIRECTIONS: [([isize; 3], f64); 3] =
    [([1, 0, 0], 1.0), ([0, 1, 0], 1.0), ([0, 0, 1], 1.0)];

/// Four-direction TV (FDTV) stencil generalized to 3-D: the three Cartesian
/// axes plus the six in-plane face diagonals (step length √2, `a_d = 1/2`).
///
/// In any coordinate plane this reduces to the horizontal, vertical, and two
/// diagonal differences of the FDTV operator (PMC10745410), giving a more
/// rotation-invariant discretization than the axis-only stencil.
pub(in crate::inverse::fwi::time_domain) const FDTV_DIRECTIONS: [([isize; 3], f64); 9] = [
    ([1, 0, 0], 1.0),
    ([0, 1, 0], 1.0),
    ([0, 0, 1], 1.0),
    ([1, 1, 0], 0.5),
    ([1, -1, 0], 0.5),
    ([1, 0, 1], 0.5),
    ([1, 0, -1], 0.5),
    ([0, 1, 1], 0.5),
    ([0, 1, -1], 0.5),
];

/// In-bounds neighbor index `p + offset`, or `None` if outside the grid.
#[inline]
fn offset_index(p: [usize; 3], offset: [isize; 3], dims: [usize; 3]) -> Option<[usize; 3]> {
    let mut out = [0usize; 3];
    for ax in 0..3 {
        let v = p[ax] as isize + offset[ax];
        if v < 0 || v >= dims[ax] as isize {
            return None;
        }
        out[ax] = v as usize;
    }
    Some(out)
}

/// Per-voxel Huber-smoothed directional-TV weight
/// `W[p] = √(ε² + Σ_d a_d (m[p+d] − m[p])²)` over in-bounds forward neighbors.
fn directional_tv_weights(model: &Array3<f64>, directions: &[([isize; 3], f64)]) -> Array3<f64> {
    let [nx, ny, nz] = model.shape();
    let dims = [nx, ny, nz];
    let mut w = Array3::<f64>::zeros((nx, ny, nz));
    for ((i, j, k), w_val) in w.indexed_iter_mut() {
        let p = [i, j, k];
        let center = model[[i, j, k]];
        let mut acc = TV_EPSILON_SQUARED;
        for &(offset, a) in directions {
            if let Some(q) = offset_index(p, offset, dims) {
                let d = model[q] - center;
                acc += a * d * d;
            }
        }
        *w_val = acc.sqrt();
    }
    w
}

/// Discrete Huber-smoothed directional-TV functional value
/// `J(m) = Σ_p √(ε² + Σ_d a_d (m[p+d] − m[p])²)`.
///
/// Exposed for the finite-difference differential test that proves
/// [`directional_tv_gradient`] is the exact functional derivative.
#[cfg(test)]
#[must_use]
pub(in crate::inverse::fwi::time_domain) fn directional_tv_functional(
    model: &Array3<f64>,
    directions: &[([isize; 3], f64)],
) -> f64 {
    directional_tv_weights(model, directions).sum()
}

/// Analytically derived gradient `∂J/∂m` of the directional-TV functional
/// [`directional_tv_functional`].
///
/// ## Derivation
///
/// With `W[p] = √(ε² + Σ_d a_d (m[p+d]−m[p])²)`, the chain rule gives
/// `∂J/∂m[q] = −(1/W[q]) Σ_{d: q+d∈Ω} a_d (m[q+d]−m[q])`
/// `           + Σ_{d: q−d∈Ω} a_d (m[q]−m[q−d]) / W[q−d]`,
/// i.e. the divergence of the W-normalized directional-difference field. The
/// axis-only special case ([`AXIS_TV_DIRECTIONS`]) is the standard ROF gradient;
/// the FDTV stencil ([`FDTV_DIRECTIONS`]) adds the diagonal directions.
#[must_use]
pub(in crate::inverse::fwi::time_domain) fn directional_tv_gradient(
    model: &Array3<f64>,
    directions: &[([isize; 3], f64)],
) -> Array3<f64> {
    let [nx, ny, nz] = model.shape();
    let dims = [nx, ny, nz];
    let w = directional_tv_weights(model, directions);

    let mut grad = Array3::<f64>::zeros((nx, ny, nz));
    for ((i, j, k), g_val) in grad.indexed_iter_mut() {
        let p = [i, j, k];
        let center = model[[i, j, k]];
        let w_c = w[[i, j, k]];
        let mut g = 0.0;
        for &(offset, a) in directions {
            // Self-contribution: forward difference leaving p.
            if let Some(q) = offset_index(p, offset, dims) {
                g -= a * (model[q] - center) / w_c;
            }
            // Back-contribution: forward difference ending at p (from p−d).
            let neg = [-offset[0], -offset[1], -offset[2]];
            if let Some(r) = offset_index(p, neg, dims) {
                g += a * (center - model[r]) / w[r];
            }
        }
        *g_val = g;
    }
    grad
}

/// Compute smoothness gradient (Laplacian) for regularization.
///
/// # Loop ordering
///
/// `i`-outer, `k`-inner: inner-loop accesses `model[[i,j,k±1]]` at stride 1.
#[must_use]
pub(in crate::inverse::fwi::time_domain) fn compute_smoothness_gradient(
    model: &Array3<f64>,
) -> Array3<f64> {
    let [nx, ny, nz] = model.shape();
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
