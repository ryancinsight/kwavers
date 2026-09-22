//! Elastic stress tensor and its divergence, from whole-field derivative
//! sweeps.
//!
//! Every derivative is one sweep of leto's fourth-order central operator
//! (ADR 128) over a whole field; the stresses and the divergence are then
//! assembled pointwise through [`kwavers_core::traversal`]. A point's value
//! depends only on its own inputs, so each pass is race-free under the
//! parallel traversals.
//!
//! Two scratch fields hold one derivative each between its sweep and the
//! assembly that reads it. The normal strains need a third; they borrow the
//! shear fields, which the shear pass overwrites right after.
//!
//! **Reference**: LeVeque (2002), "Finite Volume Methods for Hyperbolic
//! Problems", §2.13 (stress-velocity formulation for elastic waves).

use super::super::scratch::ElasticStepScratch;
use super::super::types::ElasticWaveField;
use core::num::NonZeroUsize;
use core::ops::Range;
use kwavers_core::arena::last_level_cache_bytes;
use kwavers_core::traversal::{zip_mut, zip_mut_pair};
use kwavers_grid::Grid;

use leto::{Array3, ArrayView3, ArrayViewMut3};
use leto_ops::{Axis, FiniteDifference3D, PlaneWindow, PlaneWindowMut};

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
        ("scratch.derivative", scratch.derivative.shape()),
        ("scratch.other_derivative", scratch.other_derivative.shape()),
    ] {
        assert!(
            actual == expected,
            "invariant: {name} shape {actual:?} must match grid shape {expected:?}"
        );
    }
}

/// Fourth-order central first derivatives on the grid's spacing.
fn derivatives(grid: &Grid) -> FiniteDifference3D<f64> {
    FiniteDifference3D::central_fourth_order(grid.dx, grid.dy, grid.dz)
        .expect("invariant: a grid has positive spacing")
}

/// `out = ∂field/∂axis`.
fn sweep(op: &FiniteDifference3D<f64>, axis: Axis, field: &Array3<f64>, out: &mut Array3<f64>) {
    let mut out = out.view_mut();
    match axis {
        Axis::X => op.apply_x_into(field.view(), &mut out),
        Axis::Y => op.apply_y_into(field.view(), &mut out),
        Axis::Z => op.apply_z_into(field.view(), &mut out),
    }
    .expect("invariant: validated elastic fields share the grid shape");
}

/// `shear = μ (∂first/∂first_axis + ∂second/∂second_axis)`.
///
/// One fused pass: leto reads both displacement components and μ once per
/// output lane, where sweeping each axis into a scratch buffer and scaling
/// the sum moves twice the traffic on a path measured memory-bound. The
/// values are bit-identical to the composed form.
fn shear(
    op: &FiniteDifference3D<f64>,
    planes: Range<usize>,
    (first_axis, first): (Axis, &Array3<f64>),
    (second_axis, second): (Axis, &Array3<f64>),
    mu: &Array3<f64>,
    mut shear: ArrayViewMut3<'_, f64>,
    origin: usize,
) {
    op.map_axis_derivatives_in_windows(
        mu.shape()[0],
        planes,
        [
            (first_axis, PlaneWindow::whole(first.view())),
            (second_axis, PlaneWindow::whole(second.view())),
        ],
        [PlaneWindow::whole(mu.view())],
        [PlaneWindowMut::new(&mut shear, origin)],
        |[a, b], [m]| [m * (a + b)],
    )
    .expect("invariant: validated elastic fields share the grid shape");
}

/// `out = ∂first/∂x + ∂second/∂y + ∂third/∂z`, summed left to right.
///
/// One fused pass: leto reads each stress field once per output lane rather
/// than writing three per-axis buffers this then adds back, which is the
/// difference between 8 MB and 20 MB of traffic per component at 64 cubed on
/// a path measured memory-bound. The values are unchanged to the bit.
fn divergence(
    op: &FiniteDifference3D<f64>,
    [first, second, third]: [&Array3<f64>; 3],
    out: &mut Array3<f64>,
) {
    op.divergence_into(
        [first.view(), second.view(), third.view()],
        &mut out.view_mut(),
    )
    .expect("invariant: validated elastic fields share the grid shape");
}

/// How the divergence becomes an acceleration: a uniform medium carries one
/// reciprocal, a heterogeneous one divides by its density field.
///
/// The two are not interchangeable at the bit level -- `d * (1/rho)` and
/// `d / rho` round differently -- so each route keeps the arithmetic its
/// caller had.
pub(crate) enum DensityScale<'a> {
    UniformReciprocal(f64),
    Field(ArrayView3<'a, f64>),
}

/// The six stress components from the displacement field and the Lamé
/// parameters, on the grid planes `planes`, written into destinations that
/// hold the grid planes from `origin` on.
///
/// One pass over the nine displacement gradients writes all six. As four
/// passes -- the diagonal, which reads all three normal strains, then one
/// per shear -- the displacement components were read nine times and `μ`
/// four, and each pass was its own parallel region. Each stress is the same
/// arithmetic on the same derivatives either way, so the values are
/// unchanged to the bit.
fn stress_components(
    op: &FiniteDifference3D<f64>,
    planes: Range<usize>,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    stresses: [ArrayViewMut3<'_, f64>; 6],
    origin: usize,
) {
    let [ux, uy, uz] = [&field.ux, &field.uy, &field.uz].map(|u| PlaneWindow::whole(u.view()));
    let mut stresses = stresses;
    op.map_axis_derivatives_in_windows(
        lambda.shape()[0],
        planes,
        [
            (Axis::X, ux),
            (Axis::Y, ux),
            (Axis::Z, ux),
            (Axis::X, uy),
            (Axis::Y, uy),
            (Axis::Z, uy),
            (Axis::X, uz),
            (Axis::Y, uz),
            (Axis::Z, uz),
        ],
        [
            PlaneWindow::whole(lambda.view()),
            PlaneWindow::whole(mu.view()),
        ],
        stresses
            .each_mut()
            .map(|stress| PlaneWindowMut::new(stress, origin)),
        |[xx, xy, xz, yx, yy, yz, zx, zy, zz], [la, mv]| {
            let la2mu = 2.0f64.mul_add(mv, la);
            [
                la2mu.mul_add(xx, la * (yy + zz)),
                la2mu.mul_add(yy, la * (xx + zz)),
                la2mu.mul_add(zz, la * (xx + yy)),
                mv * (xy + yx),
                mv * (xz + zx),
                mv * (yz + zy),
            ]
        },
    )
    .expect("invariant: validated elastic fields share the grid shape");
}

/// Compute the elastic stress tensor divergence ∇·σ into pre-allocated
/// scratch buffers (zero allocation).
///
/// Writes all six stress fields and the three divergence fields of `scratch`.
/// Since the shears and the divergences each take one fused pass, the two
/// derivative workspaces this used to sweep through are untouched here; the
/// plane-strain kernel still uses them.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's.
pub fn stress_divergence_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
) {
    validate_stress_divergence_shapes(grid, lambda, mu, field, scratch);
    let op = derivatives(grid);
    let ElasticStepScratch {
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        div_x,
        div_y,
        div_z,
        ..
    } = scratch;

    stress_components(
        &op,
        0..grid.nx,
        lambda,
        mu,
        field,
        [
            sxx.view_mut(),
            syy.view_mut(),
            szz.view_mut(),
            sxy.view_mut(),
            sxz.view_mut(),
            syz.view_mut(),
        ],
        0,
    );

    for (stresses, out) in [
        ([&*sxx, &*sxy, &*sxz], div_x),
        ([&*sxy, &*syy, &*syz], div_y),
        ([&*sxz, &*syz, &*szz], div_z),
    ] {
        divergence(&op, stresses, out);
    }
}

/// The accelerations of `field`: the stress divergence scaled by the density,
/// written into `scratch`'s `ax`, `ay` and `az`.
///
/// The scale rides the divergence pass. Computing the divergence into its own
/// fields and scaling them afterwards costs a second pass over three grids --
/// three reads and three writes, 24 MB at 64 cubed -- for arithmetic that is
/// one operation per lane. Values are unchanged: the divergence is summed in
/// x, y, z order and scaled exactly as the separate pass scaled it, and the
/// intermediate is a value the separate pass stored and reloaded without
/// rounding.
///
/// `scratch`'s divergence fields are untouched here; the body-force route
/// still writes and reads them, since `(divergence + force) / rho` is not the
/// same rounding as scaling the divergence alone.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's.
pub(crate) fn stress_acceleration_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scale: &DensityScale<'_>,
    scratch: &mut ElasticStepScratch,
) {
    let slab = slab_planes(grid, scale);
    stress_acceleration_in_slabs(grid, lambda, mu, field, scale, scratch, slab);
}

/// Fields a slab keeps in flight per plane with a uniform density: the six
/// stresses of the window, the three displacement components, the Lamé pair
/// and the three accelerations. A density field adds one.
const LIVE_FIELDS: usize = 14;

/// How many x-planes each slab of the acceleration evaluation covers.
///
/// Whole-grid while the evaluation's live fields fit the last-level cache:
/// then every intermediate is still resident when it is read back, and
/// slabs only add passes. Past it, the largest slab whose window and the
/// planes around it fit that cache, and no more than the worker count, since
/// each pass hands one plane to a task.
///
/// Measured through `swe_acceleration_slab_sweep` (release, fastest of 20
/// paired repeats, 36 MB last-level cache, 24 workers): at 64 cubed every
/// slab height is slower than whole-grid, at 80 cubed the best is level with
/// it, at 96 cubed 24-plane slabs run 1857-1904 us against 2742-3113 (1.5x),
/// and at 128 cubed 16-plane slabs run 5294-5634 us against 9712-10677
/// (1.8x). The rule gives 24 planes at 96 cubed and 16 at 128, the measured
/// best of 8, 12, 16, 24 and 32 at each; which of its two limits binds is
/// what moves the optimum between them.
///
/// A platform reporting no cache size evaluates whole-grid, the route that
/// wins whenever the fields fit.
fn slab_planes(grid: &Grid, scale: &DensityScale<'_>) -> NonZeroUsize {
    let whole = NonZeroUsize::new(grid.nx).unwrap_or(NonZeroUsize::MIN);
    let Some(cache) = last_level_cache_bytes() else {
        return whole;
    };
    let live = LIVE_FIELDS + usize::from(matches!(scale, DensityScale::Field(_)));
    let per_plane = live * grid.ny * grid.nz * size_of::<f64>();
    if per_plane.saturating_mul(grid.nx) <= cache {
        return whole;
    }
    let fitting = (cache / per_plane.max(1)).saturating_sub(2 * STENCIL_REACH);
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    NonZeroUsize::new(fitting.min(workers))
        .unwrap_or(NonZeroUsize::MIN)
        .min(whole)
}

/// Planes a fourth-order derivative reaches on either side of its own.
///
/// A slab's accelerations at planes `start..end` read stresses on
/// `start - STENCIL_REACH..end + STENCIL_REACH`, so those stress planes must
/// be held before the slab's divergence runs.
const STENCIL_REACH: usize = 2;

/// [`stress_acceleration_into`], evaluated `slab` x-planes at a time through
/// a stress window.
///
/// The window is the leading planes of `scratch`'s six stress fields: each
/// slab holds there the stress planes its divergence reads, starting at grid
/// plane `start - STENCIL_REACH`. The planes the previous slab already
/// computed and this one still needs slide to the front, and only the rest
/// are computed, so no stress plane is computed twice. The same few
/// megabytes are rewritten for every slab, so they stay in cache: the stress
/// never makes the round trip through DRAM that a grid-sized intermediate
/// does, where every line written is first read back from planes last
/// touched a whole pass earlier.
///
/// A `slab` of `nx` or more is one slab, and the window is then the whole of
/// each stress field: the whole-grid evaluation. Either way every plane of
/// every acceleration is the value the whole-grid evaluation writes, since
/// leto's windowed passes take the grid's stencil at each grid plane.
///
/// After a slabbed evaluation the stress fields hold the last slab's window,
/// not the grid's stress.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's.
pub(crate) fn stress_acceleration_in_slabs(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scale: &DensityScale<'_>,
    scratch: &mut ElasticStepScratch,
    slab: NonZeroUsize,
) {
    validate_stress_divergence_shapes(grid, lambda, mu, field, scratch);
    let op = derivatives(grid);
    let ElasticStepScratch {
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        ax,
        ay,
        az,
        ..
    } = scratch;
    let mut stresses = [sxx, syy, szz, sxy, sxz, syz];
    let [nx, ny, nz] = [grid.nx, grid.ny, grid.nz];
    let plane = ny * nz;
    let mut held = 0..0;
    for start in (0..nx).step_by(slab.get()) {
        let end = start.saturating_add(slab.get()).min(nx);
        let needed = start.saturating_sub(STENCIL_REACH)..end.saturating_add(STENCIL_REACH).min(nx);
        let kept = needed.start.max(held.start)..held.end;
        if kept.start < kept.end && kept.start > held.start {
            let from = (kept.start - held.start) * plane..(kept.end - held.start) * plane;
            for stress in &mut stresses {
                stress
                    .as_slice_mut()
                    .expect("invariant: scratch fields are C-contiguous")
                    .copy_within(from.clone(), 0);
            }
        }
        let fresh = kept.end.max(needed.start)..needed.end;
        let window = [(0, needed.len(), 1), (0, ny, 1), (0, nz, 1)];
        stress_components(
            &op,
            fresh,
            lambda,
            mu,
            field,
            stresses.each_mut().map(|stress| {
                stress
                    .slice_mut(&window)
                    .expect("invariant: the window lies within the stress field")
            }),
            needed.start,
        );
        accelerations(
            &op,
            start..end,
            stresses.each_ref().map(|stress| {
                stress
                    .slice(&window)
                    .expect("invariant: the window lies within the stress field")
            }),
            needed.start,
            scale,
            [&mut *ax, &mut *ay, &mut *az],
        );
        held = needed;
    }
}

/// The accelerations on the grid planes `planes`: the divergence of the
/// stress tensor, whose fields hold the grid planes from `origin` on, scaled
/// by the density.
///
/// One pass for all three. Taken separately they read nine stress lanes over
/// six distinct fields -- `sxy`, `sxz` and `syz` each feed two of the three
/// -- so each repeated field crossed the bus twice. Here every field is read
/// once per output lane: at 96 cubed that is 70 MB against 105, and one pass
/// rather than three.
fn accelerations(
    op: &FiniteDifference3D<f64>,
    planes: Range<usize>,
    [sxx, syy, szz, sxy, sxz, syz]: [ArrayView3<'_, f64>; 6],
    origin: usize,
    scale: &DensityScale<'_>,
    [ax, ay, az]: [&mut Array3<f64>; 3],
) {
    let grid_planes = ax.shape()[0];
    let (mut x_out, mut y_out, mut z_out) = (ax.view_mut(), ay.view_mut(), az.view_mut());
    let held = |stress| PlaneWindow::new(stress, origin);
    let terms = [
        (Axis::X, held(sxx)),
        (Axis::Y, held(sxy)),
        (Axis::Z, held(sxz)),
        (Axis::X, held(sxy)),
        (Axis::Y, held(syy)),
        (Axis::Z, held(syz)),
        (Axis::X, held(sxz)),
        (Axis::Y, held(syz)),
        (Axis::Z, held(szz)),
    ];
    let destinations = [
        PlaneWindowMut::whole(&mut x_out),
        PlaneWindowMut::whole(&mut y_out),
        PlaneWindowMut::whole(&mut z_out),
    ];
    match scale {
        DensityScale::UniformReciprocal(reciprocal) => op.map_axis_derivatives_in_windows(
            grid_planes,
            planes,
            terms,
            [],
            destinations,
            |[xx, xy, xz, yx, yy, yz, zx, zy, zz], []| {
                [
                    ((xx + xy) + xz) * reciprocal,
                    ((yx + yy) + yz) * reciprocal,
                    ((zx + zy) + zz) * reciprocal,
                ]
            },
        ),
        DensityScale::Field(density) => op.map_axis_derivatives_in_windows(
            grid_planes,
            planes,
            terms,
            [PlaneWindow::whole(*density)],
            destinations,
            |[xx, xy, xz, yx, yy, yz, zx, zy, zz], [rho]| {
                [
                    ((xx + xy) + xz) / rho,
                    ((yx + yy) + yz) / rho,
                    ((zx + zy) + zz) / rho,
                ]
            },
        ),
    }
    .expect("invariant: validated elastic fields share the grid shape");
}

/// Fill the in-plane stress divergence for a plane-strain field.
///
/// This specialization requires a singleton z axis with `u_z = 0`. Under
/// those invariants every z derivative and the `xz`, `yz`, and `zz`
/// contributions to the divergence vanish exactly. The kernel therefore
/// computes only `{sxx, syy, sxy}` and `{div_x, div_y}`. The point-force
/// driver's fresh scratch storage keeps `div_z = 0`. Selection happens once at
/// the propagation boundary through a zero-sized stress mode.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's, and in debug
/// builds if the field is not a singleton-z plane-strain field. The
/// point-force driver establishes these invariants before choosing this
/// kernel.
pub(crate) fn stress_divergence_plane_strain_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    scratch: &mut ElasticStepScratch,
) {
    debug_assert_eq!(field.ux.shape()[2], 1);
    validate_stress_divergence_shapes(grid, lambda, mu, field, scratch);
    let op = derivatives(grid);
    let ElasticStepScratch {
        sxx,
        syy,
        sxy,
        div_x,
        div_y,
        derivative,
        other_derivative,
        ..
    } = scratch;

    sweep(&op, Axis::X, &field.ux, derivative);
    sweep(&op, Axis::Y, &field.uy, other_derivative);
    plane_diagonal_stresses(sxx, syy, derivative, other_derivative, lambda, mu);
    shear(
        &op,
        0..grid.nx,
        (Axis::Y, &field.ux),
        (Axis::X, &field.uy),
        mu,
        sxy.view_mut(),
        0,
    );

    for ([first, second], out) in [([&*sxx, &*sxy], div_x), ([&*sxy, &*syy], div_y)] {
        sweep(&op, Axis::X, first, out);
        sweep(&op, Axis::Y, second, derivative);
        zip_mut(out.view_mut(), derivative.view(), |value, &b| {
            *value += b;
        });
    }
}

/// The in-plane diagonal stresses from their two normal strains.
fn plane_diagonal_stresses(
    sxx: &mut Array3<f64>,
    syy: &mut Array3<f64>,
    exx: &Array3<f64>,
    eyy: &Array3<f64>,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
) {
    zip_mut_pair(
        sxx.view_mut(),
        syy.view_mut(),
        (exx.view(), eyy.view(), lambda.view(), mu.view()),
        |xx, yy, (&exx, &eyy, &la, &mv)| {
            let la2mu = 2.0f64.mul_add(mv, la);
            *xx = la2mu.mul_add(exx, la * eyy);
            *yy = la2mu.mul_add(eyy, la * exx);
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
