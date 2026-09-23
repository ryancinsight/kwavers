//! Elastic stress tensor and its divergence, from fused derivative passes.
//!
//! Every derivative is leto's fourth-order central operator (ADR 128). The
//! six stresses come from one fused pass over the nine displacement
//! gradients and the Lamé parameters, and each divergence component from one
//! pass summing three stress derivatives. A lane's outputs depend only on
//! its own inputs and their stencil neighbourhoods, so each pass is
//! race-free under leto's parallel traversal.
//!
//! The plane-strain kernel keeps its sweeps: two scratch fields hold one
//! derivative each between its sweep and the pointwise assembly that reads
//! it.
//!
//! **Reference**: LeVeque (2002), "Finite Volume Methods for Hyperbolic
//! Problems", §2.13 (stress-velocity formulation for elastic waves).

use super::super::scratch::ElasticStepScratch;
use super::super::types::ElasticWaveField;
use core::ops::Range;
use kwavers_core::traversal::{zip_mut, zip_mut_pair};
use kwavers_grid::Grid;

use leto::{Array3, ArrayView3, ArrayViewMut3};
use leto_ops::{Axis, FiniteDifference3D, PlaneWindow, PlaneWindowMut};

pub(super) fn validate_stress_divergence_shapes(
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
pub(super) fn derivatives(grid: &Grid) -> FiniteDifference3D<f64> {
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
    (first_axis, first): (Axis, &Array3<f64>),
    (second_axis, second): (Axis, &Array3<f64>),
    mu: &Array3<f64>,
    shear: &mut Array3<f64>,
) {
    op.map_axis_derivatives(
        [(first_axis, first.view()), (second_axis, second.view())],
        [mu.view()],
        &mut shear.view_mut(),
        |[a, b], [m], _| m * (a + b),
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

/// A velocity-Verlet half-step kick: the stress divergence, scaled by the
/// density into an acceleration, advances each velocity by `half_dt` of it.
pub(crate) struct VelocityKick<'a> {
    pub(crate) scale: DensityScale<'a>,
    pub(crate) half_dt: f64,
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
pub(super) fn stress_components(
    op: &FiniteDifference3D<f64>,
    planes: Range<usize>,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    displacement: [&Array3<f64>; 3],
    stresses: [ArrayViewMut3<'_, f64>; 6],
    origin: usize,
) {
    let [ux, uy, uz] = displacement.map(|u| PlaneWindow::whole(u.view()));
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
        |[xx, xy, xz, yx, yy, yz, zx, zy, zz], [la, mv], _| {
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
        [&field.ux, &field.uy, &field.uz],
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
    shear(&op, (Axis::Y, &field.ux), (Axis::X, &field.uy), mu, sxy);

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
