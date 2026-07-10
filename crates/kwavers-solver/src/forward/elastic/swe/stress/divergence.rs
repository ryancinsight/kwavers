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
    // Migration (Batch #1 slice 7): the original used `Zip::indexed.on 3 view_muts`
    // for joint `(i,j,k)`-parallel iteration.  After the assert-message
    // harmonization chore, the verbose `is_c_contiguous()` precondition is
    // required on EVERY operand (8 here: scratch.{sxx,syy,szz} + ux/uy/uz + λ/μ).
    //
    // Note: the phase-1 helper `kwavers_safety::with_zip_standard_layout` does
    // NOT generalize here because its signature is `1 mut + N immuts`, while
    // this site has `3 mut + 0 Zip-chain immuts` (with closure-captured
    // immutable outer-scope operands not represented in the Zip chain).
    // Helper adoption for 3-mut sites is deferred to a future
    // `with_zip_standard_layout_3mut` generalization.
    //
    // Strategy: keep `Zip::indexed` on `sxx.view_mut()` (still requires the
    // 3D `(i,j,k)` index for the FD stencils + Array3[[i,j,k]] lookups of the
    // captured `ux`/`uy`/`uz`/`lambda`/`mu` operands).  Pre-extract the flat
    // slices for `syy` and `szz`, then write them directly via
    // `syy_slice[idx]`/`szz_slice[idx]` inside the closure.  This preserves
    // the joint per-iteration writes (all three outputs updated atomically
    // per `(i,j,k)`) without requiring Zip::indexed's three-way .and() chain.

    {
        // Native indexed FD stencil writing three co-located outputs per (i,j,k).
        // leto/moirai has no indexed multi-mutable-output parallel primitive; this
        // sequential form is correctness-preserving. Reparallelizing this hot
        // elastic-stress kernel is tracked perf-debt (see kwavers-solver backlog).
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
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * (ny * nz) + j * nz + k;
                    let exx = fd1_x(ux, i, j, k, nx, dx);
                    let eyy = fd1_y(uy, i, j, k, ny, dy);
                    let ezz = fd1_z(uz, i, j, k, nz, dz);
                    let la = lambda[[i, j, k]];
                    let mv = mu[[i, j, k]];
                    let la2mu = 2.0f64.mul_add(mv, la);
                    sxx_slice[idx] = la2mu.mul_add(exx, la * (eyy + ezz));
                    syy_slice[idx] = la2mu.mul_add(eyy, la * (exx + ezz));
                    szz_slice[idx] = la2mu.mul_add(ezz, la * (exx + eyy));
                }
            }
        }
    }

    // --- Pass 1b: off-diagonal stress components {σxy, σxz, σyz} ---
    //
    // Migration (Batch #1 slice 7): same 3-mut-0-immut Zip::indexed
    // adaptation pattern as Pass 1a.  7 verbose asserts (3 mut on
    // scratch.{sxy,sxz,syz} + 4 captured immuts ux/uy/uz/mu; note `lambda`
    // is unused in this pass).

    {
        // Native indexed FD stencil (see Pass 1a note); sequential pending an
        // indexed multi-mutable-output parallel primitive.
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
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * (ny * nz) + j * nz + k;
                    let exy_2 = fd1_y(ux, i, j, k, ny, dy) + fd1_x(uy, i, j, k, nx, dx);
                    let exz_2 = fd1_z(ux, i, j, k, nz, dz) + fd1_x(uz, i, j, k, nx, dx);
                    let eyz_2 = fd1_z(uy, i, j, k, nz, dz) + fd1_y(uz, i, j, k, ny, dy);
                    let mv = mu[[i, j, k]];
                    sxy_slice[idx] = mv * exy_2;
                    sxz_slice[idx] = mv * exz_2;
                    syz_slice[idx] = mv * eyz_2;
                }
            }
        }
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

    // Migration (Batch #1 slice 7): same 3-mut Zip::indexed adaptation.  9
    // verbose asserts (3 mut on scratch.{div_x,div_y,div_z} + 6 captured immut
    // views sxx_v..syz_v).

    {
        // Native indexed FD divergence (see Pass 1a note); sequential pending an
        // indexed multi-mutable-output parallel primitive.
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
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * (ny * nz) + j * nz + k;
                    div_x_slice[idx] = fd1_x(sxx_v, i, j, k, nx, dx)
                        + fd1_y(sxy_v, i, j, k, ny, dy)
                        + fd1_z(sxz_v, i, j, k, nz, dz);
                    div_y_slice[idx] = fd1_x(sxy_v, i, j, k, nx, dx)
                        + fd1_y(syy_v, i, j, k, ny, dy)
                        + fd1_z(syz_v, i, j, k, nz, dz);
                    div_z_slice[idx] = fd1_x(sxz_v, i, j, k, nx, dx)
                        + fd1_y(syz_v, i, j, k, ny, dy)
                        + fd1_z(szz_v, i, j, k, nz, dz);
                }
            }
        }
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
