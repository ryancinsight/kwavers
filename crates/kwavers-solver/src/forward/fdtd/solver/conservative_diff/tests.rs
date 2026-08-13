//! Skew-symmetry of the conservative collocated operator, asserted directly.

use super::*;

fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    let mut field = Array3::<f64>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let t = i as f64 * 0.7 + j as f64 * 1.3 + k as f64 * 2.1 + salt;
                field[[i, j, k]] = t.sin() * 1.7 + t.cos() * 0.4 + 0.3;
            }
        }
    }
    field
}

/// **`⟨Ga, b⟩ = −⟨a, Gb⟩`** for every supported order and axis.
///
/// Skew-symmetry is the property that makes the collocated leapfrog
/// energy-conserving: with `D = G`, `D = −Gᵀ` reduces to `Gᵀ = −G`. The
/// one-sided closure the general central difference uses fails this with a
/// residual of order unity, so the bound below is not a near miss.
#[test]
fn operator_is_skew_symmetric_on_every_axis_and_order() {
    let shape = [7usize, 6, 5];
    for order in [2usize, 4, 6] {
        let op = ConservativeCentralDifference::new(order, 1.5e-4, 3.0e-4, 7.0e-4)
            .expect("valid order and spacings");
        let a = seeded(shape, 0.0);
        let b = seeded(shape, 3.3);

        for axis in 0..3 {
            let mut ga = Array3::<f64>::zeros(shape);
            let mut gb = Array3::<f64>::zeros(shape);
            match axis {
                0 => {
                    op.apply_x_into(a.view(), &mut ga);
                    op.apply_x_into(b.view(), &mut gb);
                }
                1 => {
                    op.apply_y_into(a.view(), &mut ga);
                    op.apply_y_into(b.view(), &mut gb);
                }
                _ => {
                    op.apply_z_into(a.view(), &mut ga);
                    op.apply_z_into(b.view(), &mut gb);
                }
            }

            let left: f64 = ga.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let right: f64 = a.iter().zip(gb.iter()).map(|(x, y)| x * y).sum();
            let scale = left.abs().max(right.abs()).max(1.0);
            assert!(
                (left + right).abs() < 1e-9 * scale,
                "order {order}, axis {axis}: skew-symmetry violated, \
                 {left:.9e} vs {right:.9e}"
            );
        }
    }
}

/// The interior reproduces the standard central difference, so the closure is
/// the only thing that differs from the general operator.
#[test]
fn interior_matches_the_standard_central_stencil() {
    let shape = [9usize, 1, 1];
    let dx = 2.5e-4;
    let op = ConservativeCentralDifference::new(2, dx, dx, dx).expect("valid");
    let field = seeded(shape, 1.7);

    let mut out = Array3::<f64>::zeros(shape);
    op.apply_x_into(field.view(), &mut out);

    for i in 1..shape[0] - 1 {
        let expected = (field[[i + 1, 0, 0]] - field[[i - 1, 0, 0]]) / (2.0 * dx);
        assert!(
            (out[[i, 0, 0]] - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "interior cell {i}: {} vs {expected}",
            out[[i, 0, 0]]
        );
    }

    // The low face uses the zero-extension closure, not a one-sided formula.
    let expected_low = field[[1, 0, 0]] / (2.0 * dx);
    assert!((out[[0, 0, 0]] - expected_low).abs() < 1e-12 * expected_low.abs().max(1.0));
}

/// A constant field differentiates to zero in the interior. At the faces the
/// zero-extension closure sees a step to zero and reports it — that is the
/// physical content of a rigid wall, not an error.
#[test]
fn constant_field_is_flat_in_the_interior() {
    let shape = [7usize, 1, 1];
    let dx = 1.0e-4;
    let op = ConservativeCentralDifference::new(2, dx, dx, dx).expect("valid");
    let field = Array3::<f64>::from_elem(shape, 3.0);
    let mut out = Array3::<f64>::zeros(shape);
    op.apply_x_into(field.view(), &mut out);

    for i in 1..shape[0] - 1 {
        assert!(
            out[[i, 0, 0]].abs() < 1e-12,
            "interior cell {i} is not flat"
        );
    }
    assert!(out[[0, 0, 0]] > 0.0, "low face must see the wall");
    assert!(
        out[[shape[0] - 1, 0, 0]] < 0.0,
        "high face must see the wall"
    );
}

#[test]
fn rejects_odd_orders_and_bad_spacings() {
    assert!(ConservativeCentralDifference::new(3, 1e-4, 1e-4, 1e-4).is_err());
    assert!(ConservativeCentralDifference::new(0, 1e-4, 1e-4, 1e-4).is_err());
    assert!(ConservativeCentralDifference::new(2, 0.0, 1e-4, 1e-4).is_err());
    assert!(ConservativeCentralDifference::new(2, 1e-4, -1.0, 1e-4).is_err());
}
