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
use kwavers_core::traversal::{zip_mut, zip_mut_pair, zip_mut_triple};
use kwavers_grid::Grid;
use leto::Array3;
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
fn shear(
    op: &FiniteDifference3D<f64>,
    (first_axis, first): (Axis, &Array3<f64>),
    (second_axis, second): (Axis, &Array3<f64>),
    mu: &Array3<f64>,
    shear: &mut Array3<f64>,
    [derivative, other_derivative]: [&mut Array3<f64>; 2],
) {
    sweep(op, first_axis, first, derivative);
    sweep(op, second_axis, second, other_derivative);
    zip_mut(
        shear.view_mut(),
        (derivative.view(), other_derivative.view(), mu.view()),
        |value, (&a, &b, &mv)| *value = mv * (a + b),
    );
}

/// `out = ∂first/∂x + ∂second/∂y + ∂third/∂z`, summed left to right.
fn divergence(
    op: &FiniteDifference3D<f64>,
    [first, second, third]: [&Array3<f64>; 3],
    out: &mut Array3<f64>,
    [derivative, other_derivative]: [&mut Array3<f64>; 2],
) {
    sweep(op, Axis::X, first, out);
    sweep(op, Axis::Y, second, derivative);
    sweep(op, Axis::Z, third, other_derivative);
    zip_mut(
        out.view_mut(),
        (derivative.view(), other_derivative.view()),
        |value, (&b, &c)| *value = (*value + b) + c,
    );
}

/// Compute the elastic stress tensor divergence ∇·σ into pre-allocated
/// scratch buffers (zero allocation).
///
/// Writes all six stress fields and the three divergence fields of `scratch`;
/// its two derivative fields are overwritten as workspace. Stale contents are
/// never read.
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
        derivative,
        other_derivative,
        ..
    } = scratch;

    // Normal strains, held in the shear fields until the diagonal stresses
    // have read them.
    sweep(&op, Axis::X, &field.ux, sxy);
    sweep(&op, Axis::Y, &field.uy, sxz);
    sweep(&op, Axis::Z, &field.uz, syz);
    zip_mut_triple(
        sxx.view_mut(),
        syy.view_mut(),
        szz.view_mut(),
        (sxy.view(), sxz.view(), syz.view(), lambda.view(), mu.view()),
        |xx, yy, zz, (&exx, &eyy, &ezz, &la, &mv)| {
            let la2mu = 2.0f64.mul_add(mv, la);
            *xx = la2mu.mul_add(exx, la * (eyy + ezz));
            *yy = la2mu.mul_add(eyy, la * (exx + ezz));
            *zz = la2mu.mul_add(ezz, la * (exx + eyy));
        },
    );

    for (first, second, out) in [
        ((Axis::Y, &field.ux), (Axis::X, &field.uy), &mut *sxy),
        ((Axis::Z, &field.ux), (Axis::X, &field.uz), &mut *sxz),
        ((Axis::Z, &field.uy), (Axis::Y, &field.uz), &mut *syz),
    ] {
        shear(
            &op,
            first,
            second,
            mu,
            out,
            [&mut *derivative, &mut *other_derivative],
        );
    }

    for (stresses, out) in [
        ([&*sxx, &*sxy, &*sxz], div_x),
        ([&*sxy, &*syy, &*syz], div_y),
        ([&*sxz, &*syz, &*szz], div_z),
    ] {
        divergence(
            &op,
            stresses,
            out,
            [&mut *derivative, &mut *other_derivative],
        );
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
    shear(
        &op,
        (Axis::Y, &field.ux),
        (Axis::X, &field.uy),
        mu,
        sxy,
        [&mut *derivative, &mut *other_derivative],
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
