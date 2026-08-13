//! Adjointness and accuracy of the high-order staggered pair.

use super::*;

fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    let mut field = Array3::<f64>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let t = i as f64 * 0.7 + j as f64 * 1.3 + k as f64 * 2.1 + salt;
                field[[i, j, k]] = t.sin() * 1.7 + t.cos() * 0.4 + 0.35;
            }
        }
    }
    field
}

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// **`⟨Gp, u⟩ = −⟨p, Du⟩`** at every supported order and axis.
///
/// The identity the leapfrog's energy conservation rests on. The divergence is
/// *defined* as `−Gᵀ`, so this holds identically at every order, grid size and
/// boundary — the bound is round-off, not a discretization tolerance.
#[test]
fn gradient_and_divergence_are_negative_adjoints() {
    let shape = [9usize, 7, 6];
    for order in [2usize, 4, 6, 8] {
        let op = StaggeredLeapfrogOperator::new(order, 1.5e-4, 3.0e-4, 7.0e-4)
            .expect("valid order and spacings");
        assert_eq!(op.order(), order);
        assert_eq!(op.halo_width(), order / 2);

        let pressure = seeded(shape, 0.0);
        let velocity = seeded(shape, 4.2);

        for axis in AXES {
            let mut gradient = Array3::<f64>::zeros(shape);
            let mut divergence = Array3::<f64>::zeros(shape);
            op.gradient_into(axis, pressure.view(), &mut gradient);
            op.divergence_into(axis, velocity.view(), &mut divergence);

            let left: f64 = gradient
                .iter()
                .zip(velocity.iter())
                .map(|(g, u)| g * u)
                .sum();
            let right: f64 = pressure
                .iter()
                .zip(divergence.iter())
                .map(|(p, d)| p * d)
                .sum();
            let scale = left.abs().max(right.abs()).max(1.0);
            assert!(
                (left + right).abs() < 1e-10 * scale,
                "order {order}, axis {axis:?}: ⟨Gp,u⟩ = {left:.9e} but ⟨p,Du⟩ = {right:.9e}"
            );
        }
    }
}

/// The identity must not be an artefact of a symmetric test field. A field with
/// a vanishing low-face value hid a defective closure once before
/// (KW-SOL-081), so the operands here are checked to be non-degenerate at both
/// faces along every axis.
#[test]
fn adjointness_test_fields_are_non_degenerate() {
    let shape = [9usize, 7, 6];
    let pressure = seeded(shape, 0.0);
    let velocity = seeded(shape, 4.2);
    for field in [&pressure, &velocity] {
        for axis in 0..3 {
            let last = shape[axis] - 1;
            let mut low = 0.0_f64;
            let mut high = 0.0_f64;
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        let index = [i, j, k];
                        if index[axis] == 0 {
                            low = low.max(field[index].abs());
                        }
                        if index[axis] == last {
                            high = high.max(field[index].abs());
                        }
                    }
                }
            }
            assert!(low > 1e-3 && high > 1e-3, "axis {axis} face values vanish");
        }
    }
}

/// `N = 1` reproduces the plain half-grid difference exactly, so the general
/// form is a strict generalization rather than a replacement.
#[test]
fn second_order_reduces_to_the_plain_half_grid_difference() {
    let shape = [8usize, 1, 1];
    let dx = 2.5e-4;
    let op = StaggeredLeapfrogOperator::new(2, dx, dx, dx).expect("valid");
    let field = seeded(shape, 1.1);

    let mut gradient = Array3::<f64>::zeros(shape);
    op.gradient_into(Axis::X, field.view(), &mut gradient);
    for i in 0..shape[0] - 1 {
        let expected = (field[[i + 1, 0, 0]] - field[[i, 0, 0]]) / dx;
        assert!((gradient[[i, 0, 0]] - expected).abs() < 1e-12 * expected.abs().max(1.0));
    }
    // The far face is the wall, and reflection makes it rigid: `p[nx] = p[nx−1]`
    // cancels the only tap pair. It read `−p[nx−1]/Δx` while the closure was
    // zero-extension, which was a pressure-release wall (KW-SOL-085).
    let last = shape[0] - 1;
    assert!(gradient[[last, 0, 0]].abs() < 1e-12);

    let mut divergence = Array3::<f64>::zeros(shape);
    op.divergence_into(Axis::X, field.view(), &mut divergence);
    // The near wall carries no flux either, so the first cell sees only `u[0]`.
    assert!((divergence[[0, 0, 0]] - field[[0, 0, 0]] / dx).abs() < 1e-12);
    for j in 1..shape[0] - 1 {
        let expected = (field[[j, 0, 0]] - field[[j - 1, 0, 0]]) / dx;
        assert!((divergence[[j, 0, 0]] - expected).abs() < 1e-12 * expected.abs().max(1.0));
    }
    // The last cell is bounded by the rigid far wall. The transpose has no
    // sensitivity to whatever is stored at that face — the gradient can never
    // produce a non-zero there — so the flux out of the last cell is zero
    // regardless of the value fed in, and only `u[nx−2]` flows in.
    {
        let last = shape[0] - 1;
        let expected = -field[[last - 1, 0, 0]] / dx;
        assert!((divergence[[last, 0, 0]] - expected).abs() < 1e-12 * expected.abs().max(1.0));
    }
}

/// Each order delivers **its claimed order of accuracy**, measured by grid
/// refinement well inside the grid so the wall closure never enters.
///
/// The observed slope is the falsifiable claim; an absolute error level is not,
/// since it depends on the resolution chosen. (An earlier draft asserted the
/// eighth-order stencil would reach `1e-9` at twelve points per wavelength. It
/// reaches `6.4e-7`, which is exactly `(kΔx)^8 = 0.5236^8` times an `O(1e-4)`
/// coefficient — the guess was wrong, not the operator.)
#[test]
fn each_order_converges_at_its_claimed_rate() {
    /// Relative error of the gradient at a probe cell, for a sinusoid resolved
    /// at `points_per_wavelength`.
    fn error_at(order: usize, points_per_wavelength: f64) -> f64 {
        // Long enough that the probe sits many stencil widths from either face.
        const N: usize = 600;
        let probe = N / 2;
        let dx = 1.0e-4;
        let k = std::f64::consts::TAU / (points_per_wavelength * dx);
        let op = StaggeredLeapfrogOperator::new(order, dx, dx, dx).expect("valid");
        let field = Array3::from_shape_fn((N, 1, 1), |[i, _, _]| (k * i as f64 * dx).sin());
        let mut gradient = Array3::<f64>::zeros([N, 1, 1]);
        op.gradient_into(Axis::X, field.view(), &mut gradient);
        // The gradient at index i approximates the derivative at the half point.
        let exact = k * (k * (probe as f64 + 0.5) * dx).cos();
        (gradient[[probe, 0, 0]] - exact).abs() / exact.abs()
    }

    for order in [2usize, 4, 6, 8] {
        let coarse = error_at(order, 12.0);
        let fine = error_at(order, 24.0);
        let observed = (coarse / fine).log2();
        assert!(
            (observed - order as f64).abs() < 0.3,
            "order {order}: observed convergence {observed:.3}, errors {coarse:.3e} -> {fine:.3e}"
        );
    }

    // And higher order is genuinely more accurate at a fixed resolution.
    let mut previous = f64::INFINITY;
    for order in [2usize, 4, 6, 8] {
        let error = error_at(order, 12.0);
        assert!(
            error < previous,
            "order {order} error {error:.3e} is not better than {previous:.3e}"
        );
        previous = error;
    }
}

/// A constant field has zero gradient in the interior at every order — the
/// consistency condition. At the faces the closure sees the step to zero, which
/// is the wall, not an error.
#[test]
fn constant_field_has_zero_interior_gradient() {
    let shape = [16usize, 1, 1];
    for order in [2usize, 4, 6, 8] {
        let op = StaggeredLeapfrogOperator::new(order, 1e-4, 1e-4, 1e-4).expect("valid");
        let field = Array3::<f64>::from_elem(shape, 2.5);
        let mut gradient = Array3::<f64>::zeros(shape);
        op.gradient_into(Axis::X, field.view(), &mut gradient);
        let halo = op.halo_width();
        for i in halo..shape[0] - halo {
            assert!(
                gradient[[i, 0, 0]].abs() < 1e-9,
                "order {order} cell {i}: constant field gave {}",
                gradient[[i, 0, 0]]
            );
        }
    }
}

#[test]
fn rejects_invalid_orders_and_spacings() {
    assert!(StaggeredLeapfrogOperator::new(0, 1e-4, 1e-4, 1e-4).is_err());
    assert!(StaggeredLeapfrogOperator::new(3, 1e-4, 1e-4, 1e-4).is_err());
    assert!(StaggeredLeapfrogOperator::new(2, 0.0, 1e-4, 1e-4).is_err());
    assert!(StaggeredLeapfrogOperator::new(2, 1e-4, 1e-4, -1.0).is_err());
    // Past the coefficient derivation's verified range.
    assert!(StaggeredLeapfrogOperator::new(64, 1e-4, 1e-4, 1e-4).is_err());
}

/// The Courant limit follows the derivation in the module docs, and reproduces
/// the familiar `1/√3` at second order in 3-D.
///
/// The higher-order values are *less* restrictive than the collocated table
/// (`1/√15 = 0.258` at fourth order) by roughly a factor of two — the staggered
/// stencil's symbol grows more slowly with order. Reusing the collocated number
/// here would halve the step for nothing.
#[test]
fn cfl_limit_matches_its_derivation() {
    // Σ|cₙ| for orders 2, 4, 6, 8.
    let sums = [
        1.0_f64,
        9.0 / 8.0 + 1.0 / 24.0,
        75.0 / 64.0 + 25.0 / 384.0 + 3.0 / 640.0,
        1225.0 / 1024.0 + 245.0 / 3072.0 + 49.0 / 5120.0 + 5.0 / 7168.0,
    ];
    for (index, order) in [2usize, 4, 6, 8].into_iter().enumerate() {
        let op = StaggeredLeapfrogOperator::new(order, 1e-4, 1e-4, 1e-4).expect("valid");
        for dimensions in 1..=3 {
            let expected = 1.0 / ((dimensions as f64).sqrt() * sums[index]);
            let got = op.cfl_limit(dimensions);
            assert!(
                (got - expected).abs() < 1e-12 * expected,
                "order {order}, {dimensions}-D: {got} vs {expected}"
            );
        }
    }

    // Second order in 3-D is the familiar 1/sqrt(3).
    let second = StaggeredLeapfrogOperator::new(2, 1e-4, 1e-4, 1e-4).expect("valid");
    assert!((second.cfl_limit(3) - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);

    // The limit relaxes monotonically with order, and stays well above the
    // collocated table's 1/sqrt(15) at fourth order.
    let fourth = StaggeredLeapfrogOperator::new(4, 1e-4, 1e-4, 1e-4).expect("valid");
    assert!(fourth.cfl_limit(3) > 1.0 / 15.0_f64.sqrt());
    let mut previous = f64::INFINITY;
    for order in [2usize, 4, 6, 8] {
        let limit = StaggeredLeapfrogOperator::new(order, 1e-4, 1e-4, 1e-4)
            .expect("valid")
            .cfl_limit(3);
        assert!(
            limit < previous,
            "order {order} limit {limit} did not tighten"
        );
        previous = limit;
    }
}

/// A uniform field has *exactly* zero gradient, on every axis, at every order,
/// down to a singleton extent.
///
/// This is the property a rigid wall has and a pressure-release wall does not,
/// and it is what keeps a thin `N × 4 × 4` slab behaving as a 1-D line instead
/// of a soft-walled waveguide. Zero-extension failed it badly — order 4 on a
/// 4-cell axis gave `[-417, 0, 417, -10833]` for a field of ones, and a purely
/// axial packet launched into such a grid put more energy into transverse
/// velocity than axial within 150 steps (KW-SOL-085).
#[test]
fn a_uniform_field_has_no_gradient() {
    for order in [2usize, 4, 6, 8] {
        let op = StaggeredLeapfrogOperator::new(order, 1e-4, 1e-4, 1e-4).unwrap();
        for extent in [1usize, 2, 4, 8, 17] {
            for (axis, shape) in [
                (Axis::X, [extent, 3, 3]),
                (Axis::Y, [3, extent, 3]),
                (Axis::Z, [3, 3, extent]),
            ] {
                let field = Array3::<f64>::from_elem(shape, 2.5);
                let mut gradient = Array3::<f64>::zeros(shape);
                op.gradient_into(axis, field.view(), &mut gradient);
                let worst = gradient.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                assert!(
                    worst < 1e-9,
                    "order {order}, {axis:?}, extent {extent}: uniform field \
                     produced a gradient of {worst:.4e}"
                );
            }
        }
    }
}

/// `⟨Gp, u⟩ = −⟨p, Du⟩` to rounding, including at the walls.
///
/// The divergence is *defined* as `−Gᵀ`, so this is a check that the scatter
/// implements the transpose it claims to, not a check that a boundary closure
/// happened to work out. The fields are random and dense at the boundary
/// precisely so a wall-only error cannot hide.
#[test]
fn the_divergence_is_the_negative_adjoint_of_the_gradient() {
    for order in [2usize, 4, 6, 8] {
        let op = StaggeredLeapfrogOperator::new(order, 3e-4, 7e-4, 1.1e-3).unwrap();
        for (axis, shape) in [
            (Axis::X, [11usize, 3, 3]),
            (Axis::Y, [3, 9, 4]),
            (Axis::Z, [4, 3, 13]),
        ] {
            // Deterministic, non-degenerate, and non-zero at every boundary.
            let fill = |seed: f64| {
                Array3::from_shape_fn(shape, |[i, j, k]| {
                    let t = seed + i as f64 * 1.7 + j as f64 * 2.9 + k as f64 * 0.53;
                    t.sin() + 0.5 * (2.0 * t).cos() + 0.25
                })
            };
            let p = fill(0.3);
            let u = fill(1.9);

            let mut gradient = Array3::<f64>::zeros(shape);
            op.gradient_into(axis, p.view(), &mut gradient);
            let mut divergence = Array3::<f64>::zeros(shape);
            op.divergence_into(axis, u.view(), &mut divergence);

            let left: f64 = gradient.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            let right: f64 = p.iter().zip(divergence.iter()).map(|(a, b)| a * b).sum();
            let magnitude = left.abs().max(right.abs());
            assert!(
                magnitude > 1e-3,
                "degenerate test: both inner products vanish"
            );
            assert!(
                (left + right).abs() <= 1e-9 * magnitude,
                "order {order}, {axis:?}: <Gp,u> = {left:.6e} against -<p,Du> = {:.6e}",
                -right
            );
        }
    }
}

/// The far velocity face vanishes on its own, which is the rigid wall.
///
/// `StaggeredGridOperator` forced this face to zero as a separate step; under
/// reflection it is a consequence of the stencil, so there is nothing to forget.
#[test]
fn the_far_face_is_rigid_without_being_forced() {
    for order in [2usize, 4, 6, 8] {
        let op = StaggeredLeapfrogOperator::new(order, 1e-4, 1e-4, 1e-4).unwrap();
        let shape = [12usize, 1, 1];
        let field = Array3::from_shape_fn(shape, |[i, _, _]| (i as f64 * 0.9).sin() + 1.3);
        let mut gradient = Array3::<f64>::zeros(shape);
        op.gradient_into(Axis::X, field.view(), &mut gradient);
        assert!(
            gradient[[shape[0] - 1, 0, 0]].abs() < 1e-9,
            "order {order}: far face carried {:.4e}",
            gradient[[shape[0] - 1, 0, 0]]
        );
    }
}
