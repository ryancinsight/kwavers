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
use super::fd_stencils::{fd1_x, fd1_y};
use super::kernel::{
    divergence_components, stress_components, FlatField, StencilPoint, StridedField,
};
use kwavers_grid::Grid;
use leto::Array3;
use moirai_parallel::{
    for_each_chunk_buffers_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with, Adaptive,
};

// At 16³, 1,024-element chunks produce four independent tasks, matching the
// hosted four-core runner while amortizing scheduler bookkeeping. Larger grids
// retain broad task fanout; paired operational runs rejected 512 elements.
const STRESS_CHUNK: usize = 1024;

/// Standard-layout coordinates advanced without per-voxel division.
struct GridPosition {
    i: usize,
    j: usize,
    k: usize,
    ny: usize,
    nz: usize,
}

impl GridPosition {
    fn from_flat(index: usize, ny: usize, nz: usize) -> Self {
        let yz_len = ny * nz;
        let i = index / yz_len;
        let remainder = index % yz_len;
        Self {
            i,
            j: remainder / nz,
            k: remainder % nz,
            ny,
            nz,
        }
    }

    fn advance(&mut self) {
        self.k += 1;
        if self.k == self.nz {
            self.k = 0;
            self.j += 1;
            if self.j == self.ny {
                self.j = 0;
                self.i += 1;
            }
        }
    }

    fn coordinates(&self) -> [usize; 3] {
        [self.i, self.j, self.k]
    }
}

fn validate_stress_divergence_shapes(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &ElasticStepScratch,
) {
    let expected = [grid.nx, grid.ny, grid.nz];
    for (name, actual) in [
        ("lambda", lambda.shape()),
        ("mu", mu.shape()),
        ("field.ux", field.ux.shape()),
        ("field.uy", field.uy.shape()),
        ("field.uz", field.uz.shape()),
        ("scratch.sxx", scratch.sxx.shape()),
        ("scratch.syy", scratch.syy.shape()),
        ("scratch.szz", scratch.szz.shape()),
        ("scratch.sxy", scratch.sxy.shape()),
        ("scratch.sxz", scratch.sxz.shape()),
        ("scratch.syz", scratch.syz.shape()),
        ("scratch.div_x", scratch.div_x.shape()),
        ("scratch.div_y", scratch.div_y.shape()),
        ("scratch.div_z", scratch.div_z.shape()),
    ] {
        assert!(
            actual == expected,
            "invariant: {name} shape {actual:?} must match grid shape {expected:?}"
        );
    }
}

fn try_stress_standard_layout(
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
    shape: [usize; 3],
    spacing: [f64; 3],
) -> bool {
    let (Some(ux), Some(uy), Some(uz), Some(lambda), Some(mu)) = (
        field.ux.as_slice(),
        field.uy.as_slice(),
        field.uz.as_slice(),
        lambda.as_slice(),
        mu.as_slice(),
    ) else {
        return false;
    };
    let (Some(sxx), Some(sxy), Some(sxz), Some(syy), Some(syz), Some(szz)) = (
        scratch.sxx.as_slice_mut(),
        scratch.sxy.as_slice_mut(),
        scratch.sxz.as_slice_mut(),
        scratch.syy.as_slice_mut(),
        scratch.syz.as_slice_mut(),
        scratch.szz.as_slice_mut(),
    ) else {
        return false;
    };
    let fields = [
        FlatField(ux),
        FlatField(uy),
        FlatField(uz),
        FlatField(lambda),
        FlatField(mu),
    ];
    let [_, ny, nz] = shape;
    for_each_chunk_buffers_mut_enumerated_with::<Adaptive, _, _, 6>(
        [sxx, sxy, sxz, syy, syz, szz],
        STRESS_CHUNK,
        |chunk_idx, [sxx, sxy, sxz, syy, syz, szz]| {
            let start = chunk_idx * STRESS_CHUNK;
            let mut position = GridPosition::from_flat(start, ny, nz);
            for offset in 0..sxx.len() {
                let [xx, xy, xz, yy, yz, zz] = stress_components(
                    fields,
                    StencilPoint {
                        index: start + offset,
                        position: position.coordinates(),
                        shape,
                        spacing,
                    },
                );
                sxx[offset] = xx;
                sxy[offset] = xy;
                sxz[offset] = xz;
                syy[offset] = yy;
                syz[offset] = yz;
                szz[offset] = zz;
                position.advance();
            }
        },
    )
    .expect("invariant: validated stress fields have equal lengths");
    true
}

fn stress_strided_layout(
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
    shape: [usize; 3],
    spacing: [f64; 3],
) {
    let fields = [
        StridedField(field.ux.view()),
        StridedField(field.uy.view()),
        StridedField(field.uz.view()),
        StridedField(lambda.view()),
        StridedField(mu.view()),
    ];
    let mut sxx = scratch.sxx.view_mut();
    let mut sxy = scratch.sxy.view_mut();
    let mut sxz = scratch.sxz.view_mut();
    let mut syy = scratch.syy.view_mut();
    let mut syz = scratch.syz.view_mut();
    let mut szz = scratch.szz.view_mut();
    let mut index = 0;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let point = StencilPoint {
                    index,
                    position: [i, j, k],
                    shape,
                    spacing,
                };
                let [xx, xy, xz, yy, yz, zz] = stress_components(fields, point);
                sxx[[i, j, k]] = xx;
                sxy[[i, j, k]] = xy;
                sxz[[i, j, k]] = xz;
                syy[[i, j, k]] = yy;
                syz[[i, j, k]] = yz;
                szz[[i, j, k]] = zz;
                index += 1;
            }
        }
    }
}

fn try_divergence_standard_layout(
    scratch: &mut ElasticStepScratch,
    shape: [usize; 3],
    spacing: [f64; 3],
) -> bool {
    let (Some(sxx), Some(sxy), Some(sxz), Some(syy), Some(syz), Some(szz)) = (
        scratch.sxx.as_slice(),
        scratch.sxy.as_slice(),
        scratch.sxz.as_slice(),
        scratch.syy.as_slice(),
        scratch.syz.as_slice(),
        scratch.szz.as_slice(),
    ) else {
        return false;
    };
    let (Some(div_x), Some(div_y), Some(div_z)) = (
        scratch.div_x.as_slice_mut(),
        scratch.div_y.as_slice_mut(),
        scratch.div_z.as_slice_mut(),
    ) else {
        return false;
    };
    let fields = [
        FlatField(sxx),
        FlatField(sxy),
        FlatField(sxz),
        FlatField(syy),
        FlatField(syz),
        FlatField(szz),
    ];
    let [_, ny, nz] = shape;
    for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
        div_x,
        div_y,
        div_z,
        STRESS_CHUNK,
        |chunk_idx, div_x, div_y, div_z| {
            let start = chunk_idx * STRESS_CHUNK;
            let mut position = GridPosition::from_flat(start, ny, nz);
            for offset in 0..div_x.len() {
                let [x, y, z] = divergence_components(
                    fields,
                    StencilPoint {
                        index: start + offset,
                        position: position.coordinates(),
                        shape,
                        spacing,
                    },
                );
                div_x[offset] = x;
                div_y[offset] = y;
                div_z[offset] = z;
                position.advance();
            }
        },
    );
    true
}

fn divergence_strided_layout(
    scratch: &mut ElasticStepScratch,
    shape: [usize; 3],
    spacing: [f64; 3],
) {
    let fields = [
        StridedField(scratch.sxx.view()),
        StridedField(scratch.sxy.view()),
        StridedField(scratch.sxz.view()),
        StridedField(scratch.syy.view()),
        StridedField(scratch.syz.view()),
        StridedField(scratch.szz.view()),
    ];
    let mut div_x = scratch.div_x.view_mut();
    let mut div_y = scratch.div_y.view_mut();
    let mut div_z = scratch.div_z.view_mut();
    let mut index = 0;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let point = StencilPoint {
                    index,
                    position: [i, j, k],
                    shape,
                    spacing,
                };
                let [x, y, z] = divergence_components(fields, point);
                div_x[[i, j, k]] = x;
                div_y[[i, j, k]] = y;
                div_z[[i, j, k]] = z;
                index += 1;
            }
        }
    }
}

/// Fill `scratch.{sxx,…,syz,div_x,div_y,div_z}` with the elastic stress
/// tensor divergence ∇·σ, reusing the caller's pre-allocated workspace.
///
/// ## Theorem (operator isolation)
///
/// `stress_divergence_into` is split into two independent chunk passes:
/// - Pass 1 writes `{sxx,syy,szz,sxy,sxz,syz}` from displacement views.
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
/// - `scratch`: pre-allocated workspace whose stress and divergence fields
///   match the grid shape; all fields are overwritten before use (no reads of
///   stale data).
///
/// # Panics
///
/// Panics if a caller-supplied shape or an internal solver state violates
/// the precondition required by this operation.
pub fn stress_divergence_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
) {
    validate_stress_divergence_shapes(grid, lambda, mu, field, scratch);

    let shape = field.ux.shape();
    let spacing = [grid.dx, grid.dy, grid.dz];
    if !try_stress_standard_layout(lambda, mu, field, scratch, shape, spacing) {
        stress_strided_layout(lambda, mu, field, scratch, shape, spacing);
    }
    if !try_divergence_standard_layout(scratch, shape, spacing) {
        divergence_strided_layout(scratch, shape, spacing);
    }
}

/// Fill the in-plane stress divergence for a plane-strain field.
///
/// This specialization requires a singleton z axis with `u_z = 0`. Under
/// those invariants every z derivative and the `xz`, `yz`, and `zz`
/// contributions to the divergence vanish exactly. The kernel therefore
/// computes only `{sxx, syy, sxy}` and `{div_x, div_y}`. The point-force
/// driver's fresh scratch storage keeps `div_z = 0`. Selection happens once at
/// the propagation boundary through a zero-sized stress mode; no dimensionality
/// branch enters the voxel loops.
///
/// # Panics
///
/// Panics in debug builds if the field is not a singleton-z plane-strain
/// field. The point-force driver establishes these invariants before choosing
/// this kernel.
pub(crate) fn stress_divergence_plane_strain_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
) {
    let [nx, ny, nz] = field.ux.shape();
    debug_assert_eq!(nz, 1);
    let dx = grid.dx;
    let dy = grid.dy;
    let ux = field.ux.view();
    let uy = field.uy.view();

    {
        let sxx_slice = scratch
            .sxx
            .as_slice_mut()
            .expect("invariant: sxx uses standard layout");
        let syy_slice = scratch
            .syy
            .as_slice_mut()
            .expect("invariant: syy uses standard layout");
        let sxy_slice = scratch
            .sxy
            .as_slice_mut()
            .expect("invariant: sxy uses standard layout");
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            sxx_slice,
            syy_slice,
            sxy_slice,
            STRESS_CHUNK,
            |chunk_idx, sxx_chunk, syy_chunk, sxy_chunk| {
                let start = chunk_idx * STRESS_CHUNK;
                for offset in 0..sxx_chunk.len() {
                    let idx = start + offset;
                    let i = idx / ny;
                    let j = idx % ny;
                    let exx = fd1_x(ux, i, j, 0, nx, dx);
                    let eyy = fd1_y(uy, i, j, 0, ny, dy);
                    let la = lambda[[i, j, 0]];
                    let mv = mu[[i, j, 0]];
                    let la2mu = 2.0f64.mul_add(mv, la);
                    sxx_chunk[offset] = la2mu.mul_add(exx, la * eyy);
                    syy_chunk[offset] = la2mu.mul_add(eyy, la * exx);
                    sxy_chunk[offset] =
                        mv * (fd1_y(ux, i, j, 0, ny, dy) + fd1_x(uy, i, j, 0, nx, dx));
                }
            },
        );
    }

    let sxx = scratch.sxx.view();
    let syy = scratch.syy.view();
    let sxy = scratch.sxy.view();
    let div_x = scratch
        .div_x
        .as_slice_mut()
        .expect("invariant: div_x uses standard layout");
    let div_y = scratch
        .div_y
        .as_slice_mut()
        .expect("invariant: div_y uses standard layout");
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        div_x,
        div_y,
        STRESS_CHUNK,
        |chunk_idx, div_x_chunk, div_y_chunk| {
            let start = chunk_idx * STRESS_CHUNK;
            for offset in 0..div_x_chunk.len() {
                let idx = start + offset;
                let i = idx / ny;
                let j = idx % ny;
                div_x_chunk[offset] = fd1_x(sxx, i, j, 0, nx, dx) + fd1_y(sxy, i, j, 0, ny, dy);
                div_y_chunk[offset] = fd1_x(sxy, i, j, 0, nx, dx) + fd1_y(syy, i, j, 0, ny, dy);
            }
        },
    );
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
