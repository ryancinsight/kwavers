//! Self-adjoint second-order acoustic engine for **exact-gradient** FWI.
//!
//! See ADR 016. The existing FDTD/PSTD-driven FWI path produces only an
//! *approximate* adjoint (correct descent direction, wrong absolute magnitude
//! and ~20% direction-dependent shape error). This engine is a self-contained,
//! provably self-adjoint discretisation whose discrete adjoint is the same
//! scheme run backward in time, so the finite-difference gradient test returns
//! `κ = (g·δm)/(dJ/ds) ≈ 1` for every direction.
//!
//! # Forward scheme
//! Energy-form variable-density acoustic wave equation, `W = diag(1/(ρc²))`,
//! `D = ∇·(1/ρ ∇)` a **symmetric** heterogeneous Dirichlet Laplacian:
//! ```text
//! p^{n+1} = 2p^n − p^{n−1} + dt² W⁻¹ (D p^n + s^n),   p^{-1}=p^0=0
//! d^n = R p^n,   J = (dt/2) Σ_n ‖R p^n − d_obs^n‖²
//! ```
//!
//! # Exact adjoint and gradient (ADR 016)
//! ```text
//! ξ^{m−1} = 2ξ^m − ξ^{m+1} + dt² W⁻¹ D ξ^m − dt W⁻¹ Rᵀ r^m,  ξ^{N−1}=ξ^N=0
//! g_x = (−2/(ρ_x c_x³)) Σ_{n=0}^{N−2} ξ_x^n (p^{n+1} − 2p^n + p^{n−1})_x
//! ```
//! Because `D = Dᵀ`, the 3-point time operator is self-adjoint under reversal,
//! and the adjoint source `−dt W⁻¹ Rᵀ r^m` is the exact transpose of receiver
//! sampling injected through the same `W⁻¹` path as the forward source, `g` is
//! the literal algebraic gradient of the discrete `J`.

#[cfg(test)]
mod tests;

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};

/// Time-stepping parameters for the self-adjoint engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelfAdjointConfig {
    /// Number of time samples (`N`).
    pub nt: usize,
    /// Time step \[s\].
    pub dt: f64,
}

/// Acquisition geometry: source voxels + per-source signal, receiver voxels.
///
/// `source_signal` is `(n_rows, nt)` with `n_rows == (source_voxels.len())` for a
/// per-voxel signal, or `n_rows == 1` to broadcast one signal to every source
/// voxel. Receiver traces are returned/consumed in `receiver_voxels` order.
#[derive(Clone, Copy)]
pub(crate) struct Acquisition<'a> {
    pub source_voxels: &'a [(usize, usize, usize)],
    pub source_signal: ArrayView2<'a, f64>,
    pub receiver_voxels: &'a [(usize, usize, usize)],
}

/// Per-axis inverse-square spacings; a degenerate axis (`n == 1`) contributes 0.
struct Spacing {
    inv_dx2: f64,
    inv_dy2: f64,
    inv_dz2: f64,
}

impl Spacing {
    fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            inv_dx2: if nx > 1 {
                1.0 / (grid.dx * grid.dx)
            } else {
                0.0
            },
            inv_dy2: if ny > 1 {
                1.0 / (grid.dy * grid.dy)
            } else {
                0.0
            },
            inv_dz2: if nz > 1 {
                1.0 / (grid.dz * grid.dz)
            } else {
                0.0
            },
        }
    }
}

fn dims3(dims: (usize, usize, usize)) -> [usize; 3] {
    [dims.0, dims.1, dims.2]
}

fn array3_from_view(view: ArrayView3<'_, f64>) -> Array3<f64> {
    Array3::from_vec(view.shape(), view.iter().copied().collect())
        .expect("self-adjoint view shape must match data length")
}

fn validate(
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
fn apply_helmholtz(
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
fn w_inverse(model_c: ArrayView3<'_, f64>, density: ArrayView3<'_, f64>) -> Array3<f64> {
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
struct Coeffs {
    inv_a_plus: Array3<f64>,
    c_curr: Array3<f64>,
    c_prev: Array3<f64>,
}

fn coeffs(wm1: &Array3<f64>, damping: Option<ArrayView3<'_, f64>>, dt: f64) -> Coeffs {
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
const FLAT_INV: &str = "invariant: self-adjoint leapfrog operands are full standard-layout arrays";

/// Single-pass fused leapfrog update over flat contiguous slices:
/// `next = inv_a_plus·dlap + (c_curr·curr − c_prev·prev)`.
///
/// `next` is fully overwritten (assignment, not accumulation), so it may be a
/// reused, un-zeroed buffer — no per-step allocation is needed. The
/// parenthesisation reproduces the previous two-pass form (`next = iap·dl; next
/// += c_curr·curr − c_prev·prev`) bitwise, so the exact discrete operator
/// (ADR 016) is unchanged; this is verified by the gradient-vs-FD and
/// reconstructed==stored differential tests.
fn leapfrog_combine(
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

/// Run the self-adjoint forward model.
///
/// Returns `(synthetic, history)` where `synthetic` is `(n_receivers, nt)` and
/// `history` is `(nt, nx, ny, nz)` holding `p^0 … p^{nt−1}` (`p^0 = 0`).
/// `damping` is the optional self-adjoint sponge `b(x) ≥ 0` (`None` ⇒ lossless,
/// reflecting boundaries).
pub(crate) fn forward(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    damping: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<(Array2<f64>, Array4<f64>)> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, damping, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut history = Array4::<f64>::zeros((cfg.nt, nx, ny, nz));
    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz)); // p^{n-1}
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz)); // p^{n}
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));

    // history[0] = p^0 = 0 (already zero).
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        history.index_axis_mut(0, n + 1).unwrap().assign(&p_next);
        // Rotate buffers (p_prev, p_curr, p_next) ← (p_curr, p_next, old p_prev):
        // two pointer swaps, no allocation; old p_prev becomes next step's scratch.
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }

    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
        for n in 0..cfg.nt {
            synthetic[[r, n]] = history[[n, i, j, k]];
        }
    }
    Ok((synthetic, history))
}

/// Sensor-only forward (no history retained); used for the line search / FD.
pub(crate) fn forward_sensor_only(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    damping: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<Array2<f64>> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, damping, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));
    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    // n = 0 trace is p^0 = 0 (already zero).
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            synthetic[[r, n + 1]] = p_next[[i, j, k]];
        }
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }
    Ok(synthetic)
}

/// Lossless forward keeping only the **final two** states `(p^{N−1}, p^{N−2})`
/// plus the receiver traces — `O(N)` memory instead of the `O(nt·N)` full
/// history. Used to seed the reverse-reconstruction gradient
/// ([`gradient_reconstructed`]), which re-derives the forward field backward in
/// lockstep with the adjoint sweep.
///
/// Returns `(synthetic, p_last, p_second_last)` with `p_last = p^{N−1}` and
/// `p_second_last = p^{N−2}`. Requires the lossless scheme (no sponge): the
/// energy-conserving leapfrog is exactly reversible, whereas a damped step would
/// anti-amplify round-off when reconstructed backward.
pub(crate) fn forward_tail(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
) -> KwaversResult<(Array2<f64>, Array3<f64>, Array3<f64>)> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, None, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));
    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            synthetic[[r, n + 1]] = p_next[[i, j, k]];
        }
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }
    // After the loop: p_curr = p^{N-1}, p_prev = p^{N-2}.
    Ok((synthetic, p_curr, p_prev))
}

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
