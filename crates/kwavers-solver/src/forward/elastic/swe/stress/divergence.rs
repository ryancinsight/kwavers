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
use kwavers_grid::Grid;
use leto::Array3;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

// Three output chunks plus the captured displacement/material inputs remain
// within a 32 KiB L1 working set at 256 f64 elements per chunk.
const STRESS_CHUNK: usize = 256;

/// Fill `scratch.{sxx,…,syz,div_x,div_y,div_z}` with the elastic stress
/// tensor divergence ∇·σ, reusing the caller's pre-allocated workspace.
///
/// ## Theorem (operator isolation)
///
/// `stress_divergence_into` is split into three independent chunk passes:
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
    let [nx, ny, nz] = field.ux.shape();
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
    //
    {
        let sxx_slice = scratch
            .sxx
            .as_slice_mut()
            .expect("sxx: standard-layout asserted just above; layout matched");
        let syy_slice = scratch
            .syy
            .as_slice_mut()
            .expect("syy: standard-layout asserted just above; layout matched");
        let szz_slice = scratch
            .szz
            .as_slice_mut()
            .expect("szz: standard-layout asserted just above; layout matched");
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            sxx_slice,
            syy_slice,
            szz_slice,
            STRESS_CHUNK,
            |chunk_idx, sxx_chunk, syy_chunk, szz_chunk| {
                let start = chunk_idx * STRESS_CHUNK;
                for offset in 0..sxx_chunk.len() {
                    let idx = start + offset;
                    let i = idx / (ny * nz);
                    let j = (idx / nz) % ny;
                    let k = idx % nz;
                    let exx = fd1_x(ux, i, j, k, nx, dx);
                    let eyy = fd1_y(uy, i, j, k, ny, dy);
                    let ezz = fd1_z(uz, i, j, k, nz, dz);
                    let la = lambda[[i, j, k]];
                    let mv = mu[[i, j, k]];
                    let la2mu = 2.0f64.mul_add(mv, la);
                    sxx_chunk[offset] = la2mu.mul_add(exx, la * (eyy + ezz));
                    syy_chunk[offset] = la2mu.mul_add(eyy, la * (exx + ezz));
                    szz_chunk[offset] = la2mu.mul_add(ezz, la * (exx + eyy));
                }
            },
        );
    }

    // --- Pass 1b: off-diagonal stress components {σxy, σxz, σyz} ---
    //
    {
        let sxy_slice = scratch
            .sxy
            .as_slice_mut()
            .expect("sxy: standard-layout asserted just above; layout matched");
        let sxz_slice = scratch
            .sxz
            .as_slice_mut()
            .expect("sxz: standard-layout asserted just above; layout matched");
        let syz_slice = scratch
            .syz
            .as_slice_mut()
            .expect("syz: standard-layout asserted just above; layout matched");
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            sxy_slice,
            sxz_slice,
            syz_slice,
            STRESS_CHUNK,
            |chunk_idx, sxy_chunk, sxz_chunk, syz_chunk| {
                let start = chunk_idx * STRESS_CHUNK;
                for offset in 0..sxy_chunk.len() {
                    let idx = start + offset;
                    let i = idx / (ny * nz);
                    let j = (idx / nz) % ny;
                    let k = idx % nz;
                    let exy_2 = fd1_y(ux, i, j, k, ny, dy) + fd1_x(uy, i, j, k, nx, dx);
                    let exz_2 = fd1_z(ux, i, j, k, nz, dz) + fd1_x(uz, i, j, k, nx, dx);
                    let eyz_2 = fd1_z(uy, i, j, k, nz, dz) + fd1_y(uz, i, j, k, ny, dy);
                    let mv = mu[[i, j, k]];
                    sxy_chunk[offset] = mv * exy_2;
                    sxz_chunk[offset] = mv * exz_2;
                    syz_chunk[offset] = mv * eyz_2;
                }
            },
        );
    }

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

    {
        let div_x_slice = scratch
            .div_x
            .as_slice_mut()
            .expect("div_x: standard-layout asserted just above; layout matched");
        let div_y_slice = scratch
            .div_y
            .as_slice_mut()
            .expect("div_y: standard-layout asserted just above; layout matched");
        let div_z_slice = scratch
            .div_z
            .as_slice_mut()
            .expect("div_z: standard-layout asserted just above; layout matched");
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            div_x_slice,
            div_y_slice,
            div_z_slice,
            STRESS_CHUNK,
            |chunk_idx, div_x_chunk, div_y_chunk, div_z_chunk| {
                let start = chunk_idx * STRESS_CHUNK;
                for offset in 0..div_x_chunk.len() {
                    let idx = start + offset;
                    let i = idx / (ny * nz);
                    let j = (idx / nz) % ny;
                    let k = idx % nz;
                    div_x_chunk[offset] = fd1_x(sxx_v, i, j, k, nx, dx)
                        + fd1_y(sxy_v, i, j, k, ny, dy)
                        + fd1_z(sxz_v, i, j, k, nz, dz);
                    div_y_chunk[offset] = fd1_x(sxy_v, i, j, k, nx, dx)
                        + fd1_y(syy_v, i, j, k, ny, dy)
                        + fd1_z(syz_v, i, j, k, nz, dz);
                    div_z_chunk[offset] = fd1_x(sxz_v, i, j, k, nx, dx)
                        + fd1_y(syz_v, i, j, k, ny, dy)
                        + fd1_z(szz_v, i, j, k, nz, dz);
                }
            },
        );
    }
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
    let [nx, ny, nz] = field.ux.shape();
    let mut scratch = ElasticStepScratch::new(nx, ny, nz);
    stress_divergence_into(grid, lambda, mu, field, &mut scratch);
    (scratch.div_x, scratch.div_y, scratch.div_z)
}
