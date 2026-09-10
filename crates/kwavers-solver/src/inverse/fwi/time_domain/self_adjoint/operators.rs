//! The symmetric operators the scheme is built from, and the validation
//! that their operands agree in shape.
//!
//! `D` being its own transpose is what makes the discrete adjoint exact,
//! so every operator here is symmetric by construction.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array3, ArrayView3};

use super::types::{Acquisition, SelfAdjointConfig, Spacing};

pub(super) fn dims3(dims: (usize, usize, usize)) -> [usize; 3] {
    [dims.0, dims.1, dims.2]
}

pub(super) fn array3_from_view(view: ArrayView3<'_, f64>) -> Array3<f64> {
    Array3::from_vec(view.shape(), view.iter().copied().collect())
        .expect("self-adjoint view shape must match data length")
}

pub(super) fn validate(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
) -> KwaversResult<()> {
    let dims = grid.dimensions();
    let dims_shape = dims3(dims);
    if model_c.shape() != dims_shape || density.shape() != dims_shape {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint engine: model {:?} / density {:?} must match grid {:?}",
                    model_c.shape(),
                    density.shape(),
                    dims
                ),
            },
        ));
    }
    if cfg.nt < 3 || cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: "self-adjoint engine: require nt ≥ 3 and a positive finite dt".to_owned(),
            },
        ));
    }
    if model_c.iter().any(|&c| !c.is_finite() || c <= 0.0)
        || density.iter().any(|&r| !r.is_finite() || r <= 0.0)
    {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: "self-adjoint engine: c and ρ must be finite and strictly positive"
                    .to_owned(),
            },
        ));
    }
    let rows = acq.source_signal.shape()[0];
    let source_count = acq.source_voxels.len();
    if rows != 1 && rows != source_count {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint engine: source_signal rows {} must be 1 or n_sources {}",
                    rows, source_count
                ),
            },
        ));
    }
    if acq.source_signal.shape()[1] < cfg.nt {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "self-adjoint engine: source_signal has {} samples, need nt = {}",
                    acq.source_signal.shape()[1],
                    cfg.nt
                ),
            },
        ));
    }
    Ok(())
}

/// Apply the symmetric heterogeneous Dirichlet Laplacian `D = ∇·(1/ρ ∇)`.
///
/// Face coefficient = arithmetic mean of `1/ρ` across the face; pressure outside
/// the domain is treated as zero (Dirichlet halo). The resulting matrix is
/// symmetric (`D[i,j] = D[j,i] = b_face/dα²`), which is what makes the discrete
/// adjoint identical to the forward operator.
pub(super) fn apply_helmholtz(
    p: ArrayView3<'_, f64>,
    inv_rho: ArrayView3<'_, f64>,
    sp: &Spacing,
    out: &mut Array3<f64>,
) {
    let [nx, ny, nz] = p.shape();
    // Flat contiguous traversal: the inputs are always full standard-layout
    // (C-order) arrays here, so the strided neighbour offsets are computed once
    // (`stride_i = ny·nz`, `stride_j = nz`, `stride_k = 1`) and the linear index
    // `off` is advanced by 1 per voxel instead of recomputing a 3-D offset (with
    // bounds checks) for each of the ~13 indexed accesses. The arithmetic is
    // unchanged from the indexed form, so the result is bitwise-identical (the
    // exact discrete self-adjoint operator, ADR 016, is preserved).
    const INV: &str = "invariant: self-adjoint Helmholtz operands are full standard-layout arrays";
    let ps = p.as_slice().expect(INV);
    let irs = inv_rho.as_slice().expect(INV);
    let os = out.as_slice_mut().expect(INV);
    let stride_i = ny * nz;
    let stride_j = nz;
    let (inv_dx2, inv_dy2, inv_dz2) = (sp.inv_dx2, sp.inv_dy2, sp.inv_dz2);

    let mut off = 0usize;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // off == (i·ny + j)·nz + k by construction.
                let pc = ps[off];
                let irc = irs[off];
                let mut acc = 0.0;

                if inv_dx2 != 0.0 {
                    let (pn, irn) = if i + 1 < nx {
                        (ps[off + stride_i], irs[off + stride_i])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irn) * (pn - pc) * inv_dx2;
                    let (pp, irp) = if i > 0 {
                        (ps[off - stride_i], irs[off - stride_i])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irp) * (pp - pc) * inv_dx2;
                }
                if inv_dy2 != 0.0 {
                    let (pn, irn) = if j + 1 < ny {
                        (ps[off + stride_j], irs[off + stride_j])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irn) * (pn - pc) * inv_dy2;
                    let (pp, irp) = if j > 0 {
                        (ps[off - stride_j], irs[off - stride_j])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irp) * (pp - pc) * inv_dy2;
                }
                if inv_dz2 != 0.0 {
                    let (pn, irn) = if k + 1 < nz {
                        (ps[off + 1], irs[off + 1])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irn) * (pn - pc) * inv_dz2;
                    let (pp, irp) = if k > 0 {
                        (ps[off - 1], irs[off - 1])
                    } else {
                        (0.0, irc)
                    };
                    acc += 0.5 * (irc + irp) * (pp - pc) * inv_dz2;
                }
                os[off] = acc;
                off += 1;
            }
        }
    }
}

/// `W⁻¹ = ρc²` (the inverse of the energy weight `W = 1/(ρc²)`).
pub(super) fn w_inverse(model_c: ArrayView3<'_, f64>, density: ArrayView3<'_, f64>) -> Array3<f64> {
    let shape = model_c.shape();
    let mut wm1 = Array3::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let c = model_c[[i, j, k]];
                let rho = density[[i, j, k]];
                wm1[[i, j, k]] = rho * c * c;
            }
        }
    }
    wm1
}

/// Per-voxel diagonal coefficients of the damped leapfrog (ADR 016 absorbing
/// extension). For the damped wave equation `W p̈ + B ṗ = D p + s` with
/// `W = 1/(ρc²)` and a symmetric diagonal sponge `B = diag(b) ≥ 0`, centred
/// differences give
/// ```text
/// a⁺ = W/dt² + b/(2dt),  a⁻ = W/dt² − b/(2dt),  m = 2W/dt²
/// p^{n+1} = (1/a⁺)[ m p^n − a⁻ p^{n−1} + D p^n + s^n ]
/// ```
/// `b = 0` recovers the lossless scheme exactly (`a⁺ = a⁻ = W/dt²`, `m = 2W/dt²`).
/// Precombined update coefficients (kept to ≤ 3 arrays per `Zip`, ndarray's
/// 6-producer limit): `p^{n+1} = inv_a_plus·D p^n + c_curr·p^n − c_prev·p^{n−1}`
/// with `c_curr = m/a⁺`, `c_prev = a⁻/a⁺`. `b = 0` ⇒ `inv_a_plus = dt²/W`,
/// `c_curr = 2`, `c_prev = 1` (the lossless leapfrog).
pub(super) struct Coeffs {
    pub(super) inv_a_plus: Array3<f64>,
    pub(super) c_curr: Array3<f64>,
    pub(super) c_prev: Array3<f64>,
}

pub(super) fn coeffs(wm1: &Array3<f64>, damping: Option<ArrayView3<'_, f64>>, dt: f64) -> Coeffs {
    let dt2 = dt * dt;
    let b = damping.map_or_else(|| Array3::zeros(wm1.shape()), array3_from_view);
    let mut inv_a_plus = Array3::zeros(wm1.shape());
    let mut c_curr = Array3::zeros(wm1.shape());
    let mut c_prev = Array3::zeros(wm1.shape());
    let [nx, ny, nz] = wm1.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let w_inv = wm1[[i, j, k]];
                let b_val = b[[i, j, k]];
                let w = 1.0 / w_inv; // W = 1/(ρc²).
                let a_plus = w / dt2 + b_val / (2.0 * dt);
                let a_minus = w / dt2 - b_val / (2.0 * dt);
                let m_diag = 2.0 * w / dt2;
                inv_a_plus[[i, j, k]] = 1.0 / a_plus;
                c_curr[[i, j, k]] = m_diag / a_plus;
                c_prev[[i, j, k]] = a_minus / a_plus;
            }
        }
    }
    Coeffs {
        inv_a_plus,
        c_curr,
        c_prev,
    }
}

/// Invariant message for the flat-slice leapfrog/stencil kernels: every operand
/// is a full standard-layout (C-order) array, so `as_slice` is always `Some`.
pub(super) const FLAT_INV: &str =
    "invariant: self-adjoint leapfrog operands are full standard-layout arrays";

/// Single-pass fused leapfrog update over flat contiguous slices:
/// `next = inv_a_plus·dlap + (c_curr·curr − c_prev·prev)`.
///
/// `next` is fully overwritten (assignment, not accumulation), so it may be a
/// reused, un-zeroed buffer — no per-step allocation is needed. The
/// parenthesisation reproduces the previous two-pass form (`next = iap·dl; next
/// += c_curr·curr − c_prev·prev`) bitwise, so the exact discrete operator
/// (ADR 016) is unchanged; this is verified by the gradient-vs-FD and
/// reconstructed==stored differential tests.
pub(super) fn leapfrog_combine(
    next: &mut Array3<f64>,
    dlap: &Array3<f64>,
    co: &Coeffs,
    curr: &Array3<f64>,
    prev: &Array3<f64>,
) {
    let n = next.as_slice_mut().expect(FLAT_INV);
    let len = n.len();
    // Resliced to `len` up front so the per-element indexing carries no bounds
    // check (all operands share the grid shape), which lets the loop vectorize.
    let dl = &dlap.as_slice().expect(FLAT_INV)[..len];
    let iap = &co.inv_a_plus.as_slice().expect(FLAT_INV)[..len];
    let cc = &co.c_curr.as_slice().expect(FLAT_INV)[..len];
    let cp = &co.c_prev.as_slice().expect(FLAT_INV)[..len];
    let pc = &curr.as_slice().expect(FLAT_INV)[..len];
    let pp = &prev.as_slice().expect(FLAT_INV)[..len];
    for idx in 0..len {
        n[idx] = iap[idx] * dl[idx] + (cc[idx] * pc[idx] - cp[idx] * pp[idx]);
    }
}

/// Build a self-adjoint edge sponge: a symmetric diagonal damping `b(x) ≥ 0`
/// rising quadratically from 0 in the interior to `b_max` at the domain faces,
/// over `thickness` cells. Being a diagonal (hence symmetric) operator, it keeps
/// the discrete adjoint exact (`κ ≈ 1`) while absorbing outgoing waves.
///
/// `b_max` has units of `W·(1/time)`; a physically reasonable scale is
/// `b_max ≈ 1/(ρ c · thickness · dx)` (decay over one sponge traversal).
#[cfg(test)]
pub(crate) fn build_edge_sponge(grid: &Grid, thickness: usize, b_max: f64) -> Array3<f64> {
    let (nx, ny, nz) = grid.dimensions();
    let t = thickness.max(1) as f64;
    let profile = |i: usize, n: usize| -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let from_edge = i.min(n - 1 - i);
        if (from_edge as f64) >= t {
            0.0
        } else {
            let d = (t - from_edge as f64) / t; // 1 at face → 0 at sponge inner edge.
            d * d
        }
    };
    let mut b = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = profile(i, nx).max(profile(j, ny)).max(profile(k, nz));
                b[[i, j, k]] = b_max * p;
            }
        }
    }
    b
}
