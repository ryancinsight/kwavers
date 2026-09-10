//! The adjoint sweep and the gradient it accumulates.
//!
//! The adjoint is the same scheme run backward in time, which is why the
//! finite-difference check returns a scale factor of one.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array3, ArrayView2, ArrayView3, ArrayView4};

use super::operators::{
    apply_helmholtz, array3_from_view, coeffs, dims3, leapfrog_combine, w_inverse,
};
use super::types::{Acquisition, SelfAdjointConfig, Spacing};

/// Compute the **exact** reduced gradient `g = ∂J/∂c` (ADR 016).
///
/// `residual` is `r^m = ∂J/∂d^m` in `receiver_voxels` order — for the L2 misfit
/// `J = (dt/2)Σ‖d−d_obs‖²` this is the un-reversed data residual `d_syn − d_obs`.
/// `history` is the forward `p`-history from [`forward`]. `source_mute`, if
/// supplied, zeros the gradient at source voxels (`> 0.5`).
// The adjoint-gradient kernel needs the residual, model, density, grid, config,
// acquisition, forward history, source mute, and damping as independent inputs;
// bundling these physically-distinct arrays would not aid clarity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient(
    residual: ArrayView2<'_, f64>,
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    history: ArrayView4<'_, f64>,
    source_mute: Option<ArrayView3<'_, f64>>,
    damping: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<Array3<f64>> {
    let dims = grid.dimensions();
    if residual.shape() != [acq.receiver_voxels.len(), cfg.nt] {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint gradient: residual {:?} must be (n_receivers {}, nt {})",
                    residual.shape(),
                    acq.receiver_voxels.len(),
                    cfg.nt
                ),
            },
        ));
    }
    if history.shape() != [cfg.nt, dims.0, dims.1, dims.2] {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint gradient: history {:?} must be (nt {}, {:?})",
                    history.shape(),
                    cfg.nt,
                    dims
                ),
            },
        ));
    }

    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, damping, cfg.dt);
    let dt = cfg.dt;
    let dt2 = dt * dt;
    // coeff = (∂W/∂c)/dt² = −2/(ρc³ dt²): the damped multipliers carry the dt²
    // scaling absorbed here, so the result equals the lossless gradient when b=0.
    let mut coeff = Array3::<f64>::zeros(dims);
    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let c = model_c[[i, j, k]];
                let rho = density[[i, j, k]];
                coeff[[i, j, k]] = -2.0 / (rho * c * c * c * dt2);
            }
        }
    }

    let mut xi_next = Array3::<f64>::zeros(dims); // ξ^{m+1}
    let mut xi_curr = Array3::<f64>::zeros(dims); // ξ^{m}   (ξ^{N-1} = 0)
    let mut xi_prev = Array3::<f64>::zeros(dims); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros(dims);
    let mut gradient = Array3::<f64>::zeros(dims);

    // Backward sweep m = N-1 … 1, producing ξ^{m-1} (i.e. ξ^n for n = m-1):
    // ξ^{m-1} = (1/a⁺)[ (m_diag + D) ξ^m − a⁻ ξ^{m+1} − dt Rᵀ r^m ].
    for m in (1..cfg.nt).rev() {
        apply_helmholtz(xi_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut xi_prev, &dlap, &co, &xi_curr, &xi_next);
        // Adjoint source −dt Rᵀ r^m injected at receiver voxels (through 1/a⁺).
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            xi_prev[[i, j, k]] -= co.inv_a_plus[[i, j, k]] * dt * residual[[r, m]];
        }

        // Gradient term for n = m-1: g += coeff · ξ^n · (p^{m} − 2p^{m-1} + p^{m-2}).
        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    let pm = history[[m, i, j, k]];
                    let pm1 = history[[m - 1, i, j, k]];
                    let pm2 = if m >= 2 {
                        history[[m - 2, i, j, k]]
                    } else {
                        0.0
                    };
                    gradient[[i, j, k]] +=
                        coeff[[i, j, k]] * xi_prev[[i, j, k]] * (pm - 2.0 * pm1 + pm2);
                }
            }
        }

        // Rotate (xi_next, xi_curr, xi_prev) ← (xi_curr, xi_prev, old xi_next):
        // old xi_next becomes next step's scratch; no per-step allocation.
        std::mem::swap(&mut xi_next, &mut xi_curr);
        std::mem::swap(&mut xi_curr, &mut xi_prev);
    }

    if let Some(mute) = source_mute {
        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    if mute[[i, j, k]] > 0.5 {
                        gradient[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    Ok(gradient)
}

/// Memory-efficient exact gradient (lossless only): identical result to
/// [`gradient`] but reconstructs the forward field backward in lockstep with the
/// adjoint sweep instead of consuming a stored `O(nt·N)` history — peak memory
/// drops to `O(N)` (a handful of 3-D arrays).
///
/// Seeded by the final two forward states `(p_last = p^{N−1}, p_second_last =
/// p^{N−2})` from [`forward_tail`]. The lossless leapfrog is exactly reversible
/// (`c_prev = 1`):
/// ```text
/// p^{n−1} = inv_a_plus·(D p^n) + c_curr·p^n + inv_a_plus·s^n − p^{n+1},
/// ```
/// so the reconstructed `{p^m, p^{m−1}, p^{m−2}}` window matches the stored
/// history to round-off (energy conservation keeps the reverse sweep stable). A
/// sponge would anti-amplify the reverse step, so this path is lossless-only;
/// the damped engine keeps the stored-history [`gradient`].
///
/// The compute cost is one extra Helmholtz apply per backward step (forward
/// reconstruction alongside the adjoint), the standard FWI memory↔recompute
/// trade.
// Same independent-array signature rationale as `gradient`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_reconstructed(
    residual: ArrayView2<'_, f64>,
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    p_last: ArrayView3<'_, f64>,
    p_second_last: ArrayView3<'_, f64>,
    source_mute: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<Array3<f64>> {
    let dims = grid.dimensions();
    if residual.shape() != [acq.receiver_voxels.len(), cfg.nt] {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint reconstructed gradient: residual {:?} must be (n_receivers {}, nt {})",
                    residual.shape(),
                    acq.receiver_voxels.len(),
                    cfg.nt
                ),
            },
        ));
    }
    let dims_shape = dims3(dims);
    if p_last.shape() != dims_shape || p_second_last.shape() != dims_shape {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint reconstructed gradient: seed states must be {dims:?}"
                ),
            },
        ));
    }

    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, None, cfg.dt); // lossless
    let dt = cfg.dt;
    let dt2 = dt * dt;
    let scalar_src = acq.source_signal.shape()[0] == 1;
    let mut coeff = Array3::<f64>::zeros(dims);
    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let c = model_c[[i, j, k]];
                let rho = density[[i, j, k]];
                coeff[[i, j, k]] = -2.0 / (rho * c * c * c * dt2);
            }
        }
    }

    let mut xi_next = Array3::<f64>::zeros(dims); // ξ^{m+1}
    let mut xi_curr = Array3::<f64>::zeros(dims); // ξ^{m}
    let mut xi_prev = Array3::<f64>::zeros(dims); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros(dims); // adjoint Laplacian
    let mut dlap_fwd = Array3::<f64>::zeros(dims); // forward-reconstruction Laplacian
    let mut gradient = Array3::<f64>::zeros(dims);

    // Forward window during the backward sweep: pf_m = p^m, pf_m1 = p^{m-1}.
    let mut pf_m = array3_from_view(p_last); // p^{N-1}
    let mut pf_m1 = array3_from_view(p_second_last); // p^{N-2}
    let mut pf_m2 = Array3::<f64>::zeros(dims); // reused scratch (overwritten, or zeroed at m=1)

    for m in (1..cfg.nt).rev() {
        // Adjoint step: ξ^{m-1} from ξ^m, ξ^{m+1}, and the receiver residual.
        apply_helmholtz(xi_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut xi_prev, &dlap, &co, &xi_curr, &xi_next);
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            xi_prev[[i, j, k]] -= co.inv_a_plus[[i, j, k]] * dt * residual[[r, m]];
        }

        // Reconstruct p^{m-2} (only needed for m ≥ 2; reverse leapfrog, n = m-1):
        // p^{m-2} = inv_a_plus·(D p^{m-1}) + c_curr·p^{m-1} + inv_a_plus·s^{m-1} − p^m.
        // The lossless coeffs give c_prev = 1 exactly, so `leapfrog_combine` with
        // (curr = pf_m1, prev = pf_m) reproduces `iap·dl + (c_curr·pf_m1 − pf_m)`
        // bitwise. At m = 1 the window has no p^{m-2}, so it is zero.
        if m >= 2 {
            apply_helmholtz(pf_m1.view(), inv_rho.view(), &sp, &mut dlap_fwd);
            leapfrog_combine(&mut pf_m2, &dlap_fwd, &co, &pf_m1, &pf_m);
            for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
                let s = if scalar_src {
                    acq.source_signal[[0, m - 1]]
                } else {
                    acq.source_signal[[idx, m - 1]]
                };
                pf_m2[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
            }
        } else {
            pf_m2.fill(0.0);
        }

        // Imaging condition for n = m-1: g += coeff · ξ^{m-1} · (p^m − 2p^{m-1} + p^{m-2}).
        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    gradient[[i, j, k]] += coeff[[i, j, k]]
                        * xi_prev[[i, j, k]]
                        * (pf_m[[i, j, k]] - 2.0 * pf_m1[[i, j, k]] + pf_m2[[i, j, k]]);
                }
            }
        }

        // Slide both windows for the next (m-1) iteration via pointer swaps
        // (old head buffer becomes the next step's reused scratch; no allocation).
        std::mem::swap(&mut xi_next, &mut xi_curr);
        std::mem::swap(&mut xi_curr, &mut xi_prev);
        std::mem::swap(&mut pf_m, &mut pf_m1);
        std::mem::swap(&mut pf_m1, &mut pf_m2);
    }

    if let Some(mute) = source_mute {
        for i in 0..dims.0 {
            for j in 0..dims.1 {
                for k in 0..dims.2 {
                    if mute[[i, j, k]] > 0.5 {
                        gradient[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    Ok(gradient)
}
