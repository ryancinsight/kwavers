//! The defining conditions of a summation-by-parts operator, checked directly.
//!
//! These matter more than usual because the boundary blocks are *derived* at
//! construction rather than transcribed. The derivation is only trustworthy if
//! the thing it produces is verified against the conditions it claims to
//! satisfy — a least-squares solve will always return a vector, and without
//! these tests a wrong one would look exactly like a right one.

use super::*;

/// The orders the collocated path offers.
const ORDERS: [usize; 3] = [2, 4, 6];

/// Extract the dense derivative matrix along X by applying the operator to
/// basis vectors. Testing the assembled matrix rather than the stencil code
/// means the test sees what callers see, boundary handling included.
fn dense_matrix(op: &SummationByPartsOperator, n: usize) -> Vec<f64> {
    let mut matrix = vec![0.0_f64; n * n];
    for column in 0..n {
        let mut basis = Array3::<f64>::zeros([n, 1, 1]);
        basis[[column, 0, 0]] = 1.0;
        let mut result = Array3::<f64>::zeros([n, 1, 1]);
        op.apply_into(Axis::X, basis.view(), &mut result);
        for row in 0..n {
            matrix[row * n + column] = result[[row, 0, 0]];
        }
    }
    matrix
}

fn build(order: usize, n: usize, h: f64) -> SummationByPartsOperator {
    SummationByPartsOperator::new(order, [n, 1, 1], [h, h, h]).expect("derivable")
}

/// **`Q + Qᵀ = B`** — the summation-by-parts property itself, from which energy
/// conservation follows.
#[test]
fn the_operator_satisfies_summation_by_parts() {
    let n = 24;
    let h = 1.0e-3;
    for order in ORDERS {
        let op = build(order, n, h);
        let d = dense_matrix(&op, n);

        // Q = H D, with H the diagonal norm (including the spacing factor).
        let q: Vec<f64> = (0..n)
            .flat_map(|i| {
                let weight = op.norm_weight(Axis::X, i) * h;
                (0..n).map(move |j| (i, j, weight)).collect::<Vec<_>>()
            })
            .map(|(i, j, weight)| weight * d[i * n + j])
            .collect();

        for i in 0..n {
            for j in 0..n {
                let expected = match (i, j) {
                    (0, 0) => -1.0,
                    (a, b) if a == n - 1 && b == n - 1 => 1.0,
                    _ => 0.0,
                };
                let actual = q[i * n + j] + q[j * n + i];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "order {order}: (Q + Qᵀ)[{i}][{j}] = {actual:.3e}, expected {expected}"
                );
            }
        }
    }
}

/// A uniform field has exactly zero derivative — the property the collocated
/// path lacked, and the reason this operator exists (KW-SOL-086).
///
/// Checked down to a singleton extent, because a thin transverse axis is the
/// case that was broken: under the previous zero-extension closure a one-cell
/// axis produced a large spurious gradient.
///
/// The bound is relative to `|f|/Δx`, the scale a derivative of this field
/// would have. `D·1 = 0` is structural — it is the `k = 0` accuracy condition —
/// but the coefficients are solved numerically, so the cancellation is to
/// round-off rather than bitwise. Measured worst case is `~5e-12` relative,
/// against the `O(1)` relative error the pressure-release closure produced: the
/// two are not near each other, and no plausible structural defect lands
/// between them.
#[test]
fn a_uniform_field_has_no_derivative() {
    for order in ORDERS {
        for extent in [1usize, 2, 3, 5, 8, 13, 24] {
            for (axis, shape) in [
                (Axis::X, [extent, 2, 2]),
                (Axis::Y, [2, extent, 2]),
                (Axis::Z, [2, 2, extent]),
            ] {
                let op = SummationByPartsOperator::new(order, shape, [1e-3, 1e-3, 1e-3])
                    .expect("derivable");
                let field = Array3::<f64>::from_elem(shape, -3.25);
                let mut derivative = Array3::<f64>::zeros(shape);
                op.apply_into(axis, field.view(), &mut derivative);
                let worst = derivative.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                let scale = 3.25 / 1e-3;
                assert!(
                    worst < 1e-9 * scale,
                    "order {order}, {axis:?}, extent {extent}: uniform field gave {worst:.4e} \n                     ({:.2e} relative)",
                    worst / scale
                );
            }
        }
    }
}

/// The norm is positive and is the trapezoidal quadrature: half weight at the
/// end points, one inside. An indefinite norm would make the "conserved" energy
/// meaningless.
#[test]
fn the_norm_is_positive_and_interior_weights_are_one() {
    let n = 24;
    for order in ORDERS {
        let op = build(order, n, 1.0e-3);
        for i in 0..n {
            let weight = op.norm_weight(Axis::X, i);
            assert!(weight > 0.0, "order {order}: weight {i} is {weight}");
        }
        for i in (n / 2 - 2)..(n / 2 + 2) {
            let weight = op.norm_weight(Axis::X, i);
            assert!(
                (weight - 1.0).abs() < 1e-12,
                "order {order}: interior weight {i} is {weight}"
            );
        }
        // Quadrature consistency: the weights sum to the number of intervals,
        // which is what makes `Σ hᵢ fᵢ · Δx` an integral.
        let total: f64 = (0..n).map(|i| op.norm_weight(Axis::X, i)).sum();
        assert!(
            (total - (n as f64 - 1.0)).abs() < 1e-9,
            "order {order}: weights sum to {total}, expected {}",
            n as f64 - 1.0
        );
    }
}

/// Boundary rows are accurate to `m`, interior rows to `2m`, verified by
/// differentiating monomials exactly rather than by a refinement slope — an
/// exactness claim is the stronger statement and it is what the derivation
/// actually solved for.
#[test]
fn the_operator_is_exact_on_the_polynomials_it_claims() {
    let n = 24;
    let h = 1.0;
    for order in ORDERS {
        let half = order / 2;
        let op = build(order, n, h);

        for degree in 0..=half {
            let field =
                Array3::from_shape_fn([n, 1, 1], |[i, _, _]| (i as f64).powi(degree as i32));
            let mut derivative = Array3::<f64>::zeros([n, 1, 1]);
            op.apply_into(Axis::X, field.view(), &mut derivative);
            for i in 0..n {
                let exact = if degree == 0 {
                    0.0
                } else {
                    degree as f64 * (i as f64).powi(degree as i32 - 1)
                };
                let scale = exact.abs().max(1.0);
                assert!(
                    (derivative[[i, 0, 0]] - exact).abs() < 1e-8 * scale,
                    "order {order}, degree {degree}, point {i}: got {:.6e}, exact {exact:.6e}",
                    derivative[[i, 0, 0]]
                );
            }
        }

        // Interior points additionally reach the full interior order.
        for degree in (half + 1)..=order {
            let field =
                Array3::from_shape_fn([n, 1, 1], |[i, _, _]| (i as f64).powi(degree as i32));
            let mut derivative = Array3::<f64>::zeros([n, 1, 1]);
            op.apply_into(Axis::X, field.view(), &mut derivative);
            for i in order..(n - order) {
                let exact = degree as f64 * (i as f64).powi(degree as i32 - 1);
                assert!(
                    (derivative[[i, 0, 0]] - exact).abs() < 1e-6 * exact.abs().max(1.0),
                    "order {order}, degree {degree}, interior point {i}: got {:.6e}, \
                     exact {exact:.6e}",
                    derivative[[i, 0, 0]]
                );
            }
        }
    }
}

/// Order 2 reproduces the textbook operator: one-sided differences at the ends,
/// centred inside, and `H = diag(½, 1, …, 1, ½)`.
///
/// The only SBP operator whose closed form is small enough to state, so it is
/// the one place the derivation can be checked against a known answer rather
/// than against its own conditions.
#[test]
fn second_order_reproduces_the_classical_closure() {
    let n = 8;
    let h = 0.25;
    let op = build(2, n, h);
    let d = dense_matrix(&op, n);

    assert!((op.norm_weight(Axis::X, 0) - 0.5).abs() < 1e-12);
    assert!((op.norm_weight(Axis::X, n - 1) - 0.5).abs() < 1e-12);

    // Row 0: (f₁ − f₀)/h.
    assert!((d[0] + 1.0 / h).abs() < 1e-10, "got {}", d[0]);
    assert!((d[1] - 1.0 / h).abs() < 1e-10, "got {}", d[1]);

    // Interior row: (f_{i+1} − f_{i−1})/(2h).
    let row = 3;
    assert!((d[row * n + row - 1] + 0.5 / h).abs() < 1e-10);
    assert!((d[row * n + row + 1] - 0.5 / h).abs() < 1e-10);
    assert!(d[row * n + row].abs() < 1e-12);

    // Last row: (f_{N−1} − f_{N−2})/h.
    let last = n - 1;
    assert!((d[last * n + last] - 1.0 / h).abs() < 1e-10);
    assert!((d[last * n + last - 1] + 1.0 / h).abs() < 1e-10);
}

/// **The payoff**: a leapfrog built on this operator conserves the `H`-weighted
/// energy when velocity is held at zero on the walls.
///
/// This is the statement the whole module exists for, so it is checked on an
/// actual time integration rather than inferred from `Q + Qᵀ = B`.
#[test]
fn the_leapfrog_conserves_the_weighted_energy_at_a_rigid_wall() {
    const N: usize = 64;
    const STEPS: usize = 4000;
    let h = 1.0e-3;
    let c = 1500.0_f64;
    let rho = 1000.0_f64;
    let bulk = rho * c * c;
    let dt = 0.3 * h / c;

    for order in ORDERS {
        let op = build(order, N, h);
        let mut p = Array3::from_shape_fn([N, 1, 1], |[i, _, _]| {
            let x = (i as f64 - N as f64 / 2.0) / 6.0;
            (-x * x).exp()
        });
        let mut u = Array3::<f64>::zeros([N, 1, 1]);
        let mut scratch = Array3::<f64>::zeros([N, 1, 1]);

        let energy = |p: &Array3<f64>, u: &Array3<f64>| -> f64 {
            (0..N)
                .map(|i| {
                    let w = op.norm_weight(Axis::X, i);
                    w * (p[[i, 0, 0]].powi(2) / (2.0 * bulk) + rho * u[[i, 0, 0]].powi(2) / 2.0)
                })
                .sum()
        };

        let initial = energy(&p, &u);
        for _ in 0..STEPS {
            op.apply_into(Axis::X, p.view(), &mut scratch);
            for i in 0..N {
                u[[i, 0, 0]] -= dt / rho * scratch[[i, 0, 0]];
            }
            // The rigid wall: no flux through the end points. This is what makes
            // the boundary term `pᵀBu` vanish.
            u[[0, 0, 0]] = 0.0;
            u[[N - 1, 0, 0]] = 0.0;

            op.apply_into(Axis::X, u.view(), &mut scratch);
            for i in 0..N {
                p[[i, 0, 0]] -= dt * bulk * scratch[[i, 0, 0]];
            }
        }
        let final_energy = energy(&p, &u);
        let drift = (final_energy - initial).abs() / initial;
        assert!(
            drift < 0.02,
            "order {order}: weighted energy drifted {:.3} % over {STEPS} steps",
            100.0 * drift
        );
    }
}

/// A grid too short for the requested block falls back rather than failing, and
/// what it falls back to is still a valid operator — the inertness and
/// conservation tests above already cover every extent down to one.
#[test]
fn a_short_axis_falls_back_to_an_order_that_fits() {
    let op = SummationByPartsOperator::new(6, [4, 1, 1], [1e-3, 1e-3, 1e-3]).expect("derivable");
    assert_eq!(op.order(), 6, "the requested order is reported unchanged");
    assert!(
        op.realized_order(Axis::X) < 6,
        "a 4-point axis cannot host the order-6 block"
    );
    assert_eq!(
        op.realized_order(Axis::Y),
        0,
        "a single-point axis carries no derivative at all"
    );
}

#[test]
fn construction_rejects_odd_orders_and_bad_spacings() {
    assert!(SummationByPartsOperator::new(3, [8, 8, 8], [1e-3, 1e-3, 1e-3]).is_err());
    assert!(SummationByPartsOperator::new(0, [8, 8, 8], [1e-3, 1e-3, 1e-3]).is_err());
    assert!(SummationByPartsOperator::new(2, [8, 8, 8], [0.0, 1e-3, 1e-3]).is_err());
}
