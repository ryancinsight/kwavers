//! Verification of the derived staggered stencil coefficients.
//!
//! Two independent oracles: the published closed-form coefficients for orders
//! 2–8, and a measured order of accuracy on an analytic function. The published
//! values check the derivation's algebra; the convergence test checks that the
//! stencil actually delivers the order the algebra claims, which a correct-
//! looking but mis-scaled coefficient set would fail.

use super::*;

/// Published staggered first-derivative coefficients (Fornberg 1988, Table 1;
/// the 4th-order pair is Levander 1988).
fn published(half_order: usize) -> Vec<f64> {
    match half_order {
        1 => vec![1.0],
        2 => vec![9.0 / 8.0, -1.0 / 24.0],
        3 => vec![75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
        4 => vec![
            1225.0 / 1024.0,
            -245.0 / 3072.0,
            49.0 / 5120.0,
            -5.0 / 7168.0,
        ],
        _ => unreachable!("no published table entry for half-order {half_order}"),
    }
}

#[test]
fn matches_published_coefficients_through_eighth_order() {
    for half_order in 1..=4 {
        let derived = staggered_first_derivative_coefficients(half_order).expect("derivable");
        let expected = published(half_order);
        assert_eq!(derived.len(), expected.len());
        for (n, (&got, &want)) in derived.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-13 * want.abs().max(1.0),
                "half-order {half_order}, c{}: {got:.17e} vs published {want:.17e}",
                n + 1
            );
        }
    }
}

/// The Taylor conditions the derivation imposes must hold in the solution:
/// `Σ cₙ aₙ = ½` and `Σ cₙ aₙ^{2m+1} = 0` for `m = 1…N−1`.
///
/// The residual is judged **relative to the magnitude of the terms being
/// summed**, not against a fixed absolute floor. The high moments cancel terms
/// as large as `a^{15} ≈ 4·10^12` against each other, so an absolute tolerance
/// would be measuring `f64` round-off in the cancellation rather than any
/// property of the derivation. `1e-11` of the summed magnitude is five orders
/// above machine epsilon and still tight enough to catch a wrong coefficient.
#[test]
fn satisfies_its_own_taylor_conditions() {
    for half_order in 1..=MAX_HALF_ORDER {
        let c = staggered_first_derivative_coefficients(half_order).expect("derivable");
        for m in 0..half_order {
            let power = (2 * m + 1) as i32;
            let terms = c
                .iter()
                .enumerate()
                .map(|(j, &cj)| cj * (j as f64 + 0.5).powi(power));
            let sum: f64 = terms.clone().sum();
            let scale: f64 = terms.map(f64::abs).sum::<f64>().max(0.5);
            let want = if m == 0 { 0.5 } else { 0.0 };
            assert!(
                (sum - want).abs() < 1e-11 * scale,
                "half-order {half_order}, moment {m}: residual {:.3e} against                  term scale {scale:.3e}",
                sum - want
            );
        }
    }
}

/// Apply the stencil to `sin` on a uniform grid and measure the observed order
/// of accuracy by grid refinement. Each half-order must deliver its claimed
/// `2N` — the check that the coefficients are not merely self-consistent but
/// correctly scaled.
#[test]
fn delivers_its_claimed_order_of_accuracy() {
    // Beyond 6th order the error reaches f64 round-off on this function before
    // the asymptotic regime is left, so the measured slope stops being
    // meaningful; orders 2-6 are where refinement can still resolve it.
    for half_order in 1..=3 {
        let c = staggered_first_derivative_coefficients(half_order).expect("derivable");
        let expected_order = 2.0 * half_order as f64;

        let error_at = |h: f64| -> f64 {
            // Derivative of sin at the half point x0 + h/2 is cos(x0 + h/2).
            let x0 = 0.3_f64;
            let mid = x0 + 0.5 * h;
            let approx: f64 = c
                .iter()
                .enumerate()
                .map(|(j, &cj)| {
                    let n = j as f64 + 1.0;
                    cj * ((x0 + n * h).sin() - (x0 - (n - 1.0) * h).sin())
                })
                .sum::<f64>()
                / h;
            (approx - mid.cos()).abs()
        };

        let coarse = 0.1_f64;
        let fine = coarse / 2.0;
        let observed = (error_at(coarse) / error_at(fine)).log2();
        assert!(
            (observed - expected_order).abs() < 0.2,
            "half-order {half_order}: observed order {observed:.3}, expected {expected_order}"
        );
    }
}

/// Higher order is genuinely more accurate at a fixed spacing.
#[test]
fn accuracy_improves_with_order() {
    let h = 0.2_f64;
    let x0 = 0.3_f64;
    let exact = (x0 + 0.5 * h).cos();
    let mut previous = f64::INFINITY;
    for half_order in 1..=4 {
        let c = staggered_first_derivative_coefficients(half_order).expect("derivable");
        let approx: f64 = c
            .iter()
            .enumerate()
            .map(|(j, &cj)| {
                let n = j as f64 + 1.0;
                cj * ((x0 + n * h).sin() - (x0 - (n - 1.0) * h).sin())
            })
            .sum::<f64>()
            / h;
        let error = (approx - exact).abs();
        assert!(
            error < previous,
            "half-order {half_order} error {error:.3e} is not better than {previous:.3e}"
        );
        previous = error;
    }
}

/// A constant field has zero derivative, and a linear field has exactly its
/// slope, for every order — the two exactness conditions every consistent
/// first-derivative stencil must satisfy identically.
#[test]
fn is_exact_for_constant_and_linear_fields() {
    for half_order in 1..=MAX_HALF_ORDER {
        let c = staggered_first_derivative_coefficients(half_order).expect("derivable");
        let h = 0.37_f64;

        let constant: f64 = c.iter().map(|&cj| cj * (5.0 - 5.0)).sum::<f64>() / h;
        assert_eq!(constant, 0.0, "half-order {half_order} on a constant field");

        // f(x) = 3x + 1 at taps around the half point; the answer must be 3.
        let slope: f64 = c
            .iter()
            .enumerate()
            .map(|(j, &cj)| {
                let n = j as f64 + 1.0;
                let hi = 3.0 * (n * h) + 1.0;
                let lo = 3.0 * (-(n - 1.0) * h) + 1.0;
                cj * (hi - lo)
            })
            .sum::<f64>()
            / h;
        assert!(
            (slope - 3.0).abs() < 1e-10,
            "half-order {half_order} on a linear field: {slope}"
        );
    }
}

/// Coefficients alternate in sign and decay in magnitude — the structural
/// signature of a valid central stencil.
#[test]
fn coefficients_alternate_and_decay() {
    for half_order in 2..=MAX_HALF_ORDER {
        let c = staggered_first_derivative_coefficients(half_order).expect("derivable");
        assert!(
            c[0] > 1.0,
            "half-order {half_order}: leading tap must exceed 1"
        );
        for (j, pair) in c.windows(2).enumerate() {
            assert!(
                pair[0] * pair[1] < 0.0,
                "half-order {half_order}: taps {j} and {} share a sign",
                j + 1
            );
            assert!(
                pair[1].abs() < pair[0].abs(),
                "half-order {half_order}: tap {} does not decay",
                j + 1
            );
        }
    }
}

#[test]
fn rejects_unsupported_orders() {
    assert!(staggered_first_derivative_coefficients(0).is_err());
    assert!(staggered_first_derivative_coefficients(MAX_HALF_ORDER + 1).is_err());
    assert!(staggered_first_derivative_coefficients(MAX_HALF_ORDER).is_ok());
}
