//! Legendre polynomials Pₙ(x) and their derivatives — workspace SSOT.
//!
//! Consolidates the copies formerly duplicated across the solver's
//! discontinuous-Galerkin basis/quadrature/matrix modules. Uses the standard
//! three-term recurrence `(n+1)Pₙ₊₁ = (2n+1)x·Pₙ − n·Pₙ₋₁`; the derivative uses
//! the identity `(1−x²)Pₙ′ = n(Pₙ₋₁ − x·Pₙ)` with the exact closed form at the
//! `x = ±1` endpoints to avoid the `1/(1−x²)` singularity (needed for
//! Gauss–Lobatto nodes).
//!
//! Reference: Abramowitz & Stegun (1964) §8; DLMF 14.

/// Legendre polynomial Pₙ(x) via the three-term recurrence.
#[must_use]
pub fn legendre_poly(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut l_prev = 1.0;
    let mut l_curr = x;
    for i in 1..n {
        let l_next =
            ((2 * i + 1) as f64 * x).mul_add(l_curr, -(i as f64 * l_prev)) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

/// Exact endpoint derivative `Pₙ′(±1) = ±n(n+1)/2` (parity sign at x = −1).
#[must_use]
pub fn endpoint_derivative(n: usize, x: f64) -> f64 {
    let magnitude = (n * (n + 1)) as f64 / 2.0;
    if x == 1.0 || !n.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

/// `(Pₙ(x), Pₙ′(x))`. Returns the exact closed-form derivative at `x = ±1`;
/// interior values use the recurrence and the standard derivative identity.
#[must_use]
pub fn legendre_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    let mut l_prev = 1.0;
    let mut l_curr = x;
    for i in 1..n {
        let l_next =
            ((2 * i + 1) as f64 * x).mul_add(l_curr, -(i as f64 * l_prev)) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }
    if x == 1.0 {
        return (1.0, endpoint_derivative(n, 1.0));
    }
    if x == -1.0 {
        return (
            if n.is_multiple_of(2) { 1.0 } else { -1.0 },
            endpoint_derivative(n, -1.0),
        );
    }
    let deriv = (n as f64) * x.mul_add(-l_curr, l_prev) / x.mul_add(-x, 1.0);
    (l_curr, deriv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_orders() {
        // P0=1, P1=x, P2=(3x²−1)/2, P3=(5x³−3x)/2.
        for &x in &[-0.9, -0.3, 0.0, 0.5, 0.8] {
            assert!((legendre_poly(0, x) - 1.0).abs() < 1e-14);
            assert!((legendre_poly(1, x) - x).abs() < 1e-14);
            assert!((legendre_poly(2, x) - (1.5 * x * x - 0.5)).abs() < 1e-13);
            assert!((legendre_poly(3, x) - (2.5 * x * x * x - 1.5 * x)).abs() < 1e-13);
        }
    }

    #[test]
    fn endpoints() {
        for n in 0..8 {
            assert!((legendre_poly(n, 1.0) - 1.0).abs() < 1e-13);
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((legendre_poly(n, -1.0) - expected).abs() < 1e-13);
            // Derivative is finite (no 1/(1−x²) blow-up) and matches n(n+1)/2.
            let (_, d1) = legendre_poly_and_deriv(n, 1.0);
            assert!(d1.is_finite());
            assert!((d1 - (n * (n + 1)) as f64 / 2.0).abs() < 1e-13);
        }
    }

    #[test]
    fn derivative_matches_finite_difference() {
        let h = 1e-6;
        for n in 2..6 {
            for &x in &[-0.6, 0.1, 0.7] {
                let (_, d) = legendre_poly_and_deriv(n, x);
                let fd = (legendre_poly(n, x + h) - legendre_poly(n, x - h)) / (2.0 * h);
                assert!((d - fd).abs() < 1e-5, "n={n} x={x} d={d} fd={fd}");
            }
        }
    }
}
