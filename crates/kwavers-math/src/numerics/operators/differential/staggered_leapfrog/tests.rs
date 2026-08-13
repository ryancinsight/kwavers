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
/// The identity the leapfrog's energy conservation rests on. It is exact for
/// every order and grid size under zero-extension, not asymptotic, so the bound
/// is round-off rather than a discretization tolerance.
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
    // The far face sees zero outside.
    let last = shape[0] - 1;
    assert!((gradient[[last, 0, 0]] + field[[last, 0, 0]] / dx).abs() < 1e-12);

    let mut divergence = Array3::<f64>::zeros(shape);
    op.divergence_into(Axis::X, field.view(), &mut divergence);
    assert!((divergence[[0, 0, 0]] - field[[0, 0, 0]] / dx).abs() < 1e-12);
    for j in 1..shape[0] {
        let expected = (field[[j, 0, 0]] - field[[j - 1, 0, 0]]) / dx;
        assert!((divergence[[j, 0, 0]] - expected).abs() < 1e-12 * expected.abs().max(1.0));
    }
}

/// Each order delivers **its claimed order of accuracy**, measured by grid
/// refinement well inside the grid so the zero-extension closure never enters.
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
