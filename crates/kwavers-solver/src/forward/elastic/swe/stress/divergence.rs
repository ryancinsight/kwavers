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
use kwavers_core::traversal::{zip_mut, zip_mut_pair};
use kwavers_grid::Grid;
use leto::{Array3, ArrayView3};
use leto_ops::{Axis, FiniteDifference3D};

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
    (first_axis, first): (Axis, &Array3<f64>),
    (second_axis, second): (Axis, &Array3<f64>),
    mu: &Array3<f64>,
    shear: &mut Array3<f64>,
) {
    op.map_axis_derivatives(
        [(first_axis, first.view()), (second_axis, second.view())],
        [mu.view()],
        &mut shear.view_mut(),
        |[a, b], [m]| m * (a + b),
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
/// parameters.
fn stress_components(
    op: &FiniteDifference3D<f64>,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
    [sxx, syy, szz, sxy, sxz, syz]: [&mut Array3<f64>; 6],
) {
    // All three diagonal stresses read the same three normal strains, so one
    // fused pass sweeps the strains once and writes the three: 16 MB of
    // traffic at 64 cubed where sweeping them into the shear fields and
    // combining afterwards moved 28 MB. The shear fields no longer hold
    // strains on the way, so the shears below are their only writer.
    let (mut xx, mut yy, mut zz) = (sxx.view_mut(), syy.view_mut(), szz.view_mut());
    op.map_axis_derivatives_many(
        [
            (Axis::X, field.ux.view()),
            (Axis::Y, field.uy.view()),
            (Axis::Z, field.uz.view()),
        ],
        [lambda.view(), mu.view()],
        [&mut xx, &mut yy, &mut zz],
        |[exx, eyy, ezz], [la, mv]| {
            let la2mu = 2.0f64.mul_add(mv, la);
            [
                la2mu.mul_add(exx, la * (eyy + ezz)),
                la2mu.mul_add(eyy, la * (exx + ezz)),
                la2mu.mul_add(ezz, la * (exx + eyy)),
            ]
        },
    )
    .expect("invariant: validated elastic fields share the grid shape");

    for (first, second, out) in [
        ((Axis::Y, &field.ux), (Axis::X, &field.uy), &mut *sxy),
        ((Axis::Z, &field.ux), (Axis::X, &field.uz), &mut *sxz),
        ((Axis::Z, &field.uy), (Axis::Y, &field.uz), &mut *syz),
    ] {
        shear(op, first, second, mu, out);
    }
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

    stress_components(&op, lambda, mu, field, [sxx, syy, szz, sxy, sxz, syz]);

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
    stress_into(grid, lambda, mu, field, scratch);
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
    // One pass for all three accelerations. Taken separately they read nine
    // stress lanes over six distinct fields -- `sxy`, `sxz` and `syz` each
    // feed two of the three -- so each repeated field crossed the bus twice.
    // Here every field is read once per output lane: at 96 cubed that is
    // 70 MB against 105, and one pass rather than three.
    let (mut x_out, mut y_out, mut z_out) = (ax.view_mut(), ay.view_mut(), az.view_mut());
    let terms = [
        (Axis::X, sxx.view()),
        (Axis::Y, sxy.view()),
        (Axis::Z, sxz.view()),
        (Axis::X, sxy.view()),
        (Axis::Y, syy.view()),
        (Axis::Z, syz.view()),
        (Axis::X, sxz.view()),
        (Axis::Y, syz.view()),
        (Axis::Z, szz.view()),
    ];
    let destinations = [&mut x_out, &mut y_out, &mut z_out];
    match scale {
        DensityScale::UniformReciprocal(reciprocal) => op.map_axis_derivatives_many(
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
        DensityScale::Field(density) => op.map_axis_derivatives_many(
            terms,
            [*density],
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

/// The six stress components of `field`, written into `scratch`.
///
/// [`stress_divergence_into`] is this followed by the three divergences. A
/// caller that scales the divergence -- the acceleration divides it by the
/// density -- takes them apart, so the scale rides the divergence pass
/// instead of costing a second one over three fields.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's.
pub(crate) fn stress_into(
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
        ..
    } = scratch;
    stress_components(&op, lambda, mu, field, [sxx, syy, szz, sxy, sxz, syz]);
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
