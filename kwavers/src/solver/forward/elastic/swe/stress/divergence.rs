//! Two-pass elastic stress tensor divergence computation.
//!
//! ## Theorem (race-freedom under parallel execution)
//!
//! **Pass 1** reads `{ux,uy,uz,λ,μ}` (immutable views) and writes to six
//! separate output arrays `{σxx,σyy,σzz,σxy,σxz,σyz}`.  Each output element
//! `σ[i,j,k]` is written exactly once and is never read by another iteration,
//! so parallel execution across `(i,j,k)` is race-free.
//!
//! **Pass 2** reads `{σxx,…,σyz}` (immutable after Pass 1 completes) and
//! writes to `{div_x,div_y,div_z}`.  Same argument applies.
//!
//! **Reference**: LeVeque (2002), "Finite Volume Methods for Hyperbolic
//! Problems", §2.13 (stress-velocity formulation for elastic waves).
//!
//! ## Memory layout
//!
//! [`stress_divergence_into`] writes into pre-allocated fields of
//! [`ElasticStepScratch`], eliminating all per-call heap allocations.
//! [`stress_divergence`] is a convenience wrapper that allocates its own
//! scratch internally; use it only in test code or non-hot paths.

use super::super::scratch::ElasticStepScratch;
use super::super::types::ElasticWaveField;
use super::fd_stencils::{fd1_x, fd1_y, fd1_z};
use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};

/// Fill `scratch.{sxx,…,syz,div_x,div_y,div_z}` with the elastic stress
/// tensor divergence ∇·σ, reusing the caller's pre-allocated workspace.
///
/// ## Theorem (operator isolation)
///
/// `stress_divergence_into` is split into three independent Zip passes:
/// - Pass 1a writes `{sxx,syy,szz}` from displacement views (read-only).
/// - Pass 1b writes `{sxy,sxz,syz}` from displacement views (read-only).
/// - Pass 2 reads the six stress fields (immutable views taken after Pass 1
///   releases all mutable borrows) and writes `{div_x,div_y,div_z}`.
///
/// Rust's NLL field-split borrow rules guarantee that taking immutable views
/// of `{sxx,…,syz}` while holding mutable views of `{div_x,div_y,div_z}`
/// is safe because all twelve struct fields reside in distinct memory
/// regions.
///
/// ## Parameters
///
/// - `scratch`: pre-allocated workspace; all fields are overwritten before
///   use (no reads of stale data).
pub fn stress_divergence_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
) {
    let (nx, ny, nz) = field.ux.dim();
    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    let ux = field.ux.view();
    let uy = field.uy.view();
    let uz = field.uz.view();

    // --- Pass 1a: diagonal stress components {σxx, σyy, σzz} ---
    //
    // Theorem (race-freedom): each output element σ[i,j,k] is written exactly
    // once; ux/uy/uz/λ/μ are read-only views captured by the closure.
    // ndarray 0.16 Zip::indexed supports ≤ 5 arrays (6 tuple elements total
    // including the index).  Three outputs fit within the limit.
    Zip::indexed(scratch.sxx.view_mut())
        .and(scratch.syy.view_mut())
        .and(scratch.szz.view_mut())
        .par_for_each(|(i, j, k), o_sxx, o_syy, o_szz| {
            let exx = fd1_x(ux, i, j, k, nx, dx);
            let eyy = fd1_y(uy, i, j, k, ny, dy);
            let ezz = fd1_z(uz, i, j, k, nz, dz);
            let la = lambda[[i, j, k]];
            let mv = mu[[i, j, k]];
            let la2mu = 2.0f64.mul_add(mv, la);
            *o_sxx = la2mu.mul_add(exx, la * (eyy + ezz));
            *o_syy = la2mu.mul_add(eyy, la * (exx + ezz));
            *o_szz = la2mu.mul_add(ezz, la * (exx + eyy));
        });

    // --- Pass 1b: off-diagonal stress components {σxy, σxz, σyz} ---
    Zip::indexed(scratch.sxy.view_mut())
        .and(scratch.sxz.view_mut())
        .and(scratch.syz.view_mut())
        .par_for_each(|(i, j, k), o_sxy, o_sxz, o_syz| {
            let exy_2 = fd1_y(ux, i, j, k, ny, dy) + fd1_x(uy, i, j, k, nx, dx);
            let exz_2 = fd1_z(ux, i, j, k, nz, dz) + fd1_x(uz, i, j, k, nx, dx);
            let eyz_2 = fd1_z(uy, i, j, k, nz, dz) + fd1_y(uz, i, j, k, ny, dy);
            let mv = mu[[i, j, k]];
            *o_sxy = mv * exy_2;
            *o_sxz = mv * exz_2;
            *o_syz = mv * eyz_2;
        });

    // --- Pass 2: ∇·σ (parallelised over i-j-k) ---
    //
    // σ arrays are fully computed above (mutable borrows released); taking
    // read-only views here is safe.  div_{x,y,z} elements are written exactly
    // once and are independent across iterations → race-free.
    //
    // NLL field-split borrow: sxx_v … syz_v borrow distinct fields of
    // `scratch` immutably; div_{x,y,z}.view_mut() borrow three other distinct
    // fields mutably.  These coexist without conflict.
    let sxx_v = scratch.sxx.view();
    let syy_v = scratch.syy.view();
    let szz_v = scratch.szz.view();
    let sxy_v = scratch.sxy.view();
    let sxz_v = scratch.sxz.view();
    let syz_v = scratch.syz.view();

    Zip::indexed(scratch.div_x.view_mut())
        .and(scratch.div_y.view_mut())
        .and(scratch.div_z.view_mut())
        .par_for_each(|(i, j, k), o_dx, o_dy, o_dz| {
            *o_dx = fd1_x(sxx_v, i, j, k, nx, dx)
                + fd1_y(sxy_v, i, j, k, ny, dy)
                + fd1_z(sxz_v, i, j, k, nz, dz);
            *o_dy = fd1_x(sxy_v, i, j, k, nx, dx)
                + fd1_y(syy_v, i, j, k, ny, dy)
                + fd1_z(syz_v, i, j, k, nz, dz);
            *o_dz = fd1_x(sxz_v, i, j, k, nx, dx)
                + fd1_y(syz_v, i, j, k, ny, dy)
                + fd1_z(szz_v, i, j, k, nz, dz);
        });
}

/// Compute the elastic stress tensor divergence ∇·σ, returning owned arrays.
///
/// Allocates an `ElasticStepScratch` internally and calls
/// [`stress_divergence_into`].  Use this function only in non-hot-path code
/// (tests, one-off analyses).  In the time loop, pre-allocate
/// [`ElasticStepScratch`] and call [`stress_divergence_into`] directly.
///
/// Returns `(div_x, div_y, div_z)` where each element satisfies:
/// ```text
/// (∇·σ)_α = ∂σαx/∂x + ∂σαy/∂y + ∂σαz/∂z
/// ```
pub fn stress_divergence(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = field.ux.dim();
    let mut scratch = ElasticStepScratch::new(nx, ny, nz);
    stress_divergence_into(grid, lambda, mu, field, &mut scratch);
    (scratch.div_x, scratch.div_y, scratch.div_z)
}
