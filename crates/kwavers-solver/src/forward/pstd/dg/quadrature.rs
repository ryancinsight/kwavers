//! Quadrature rules for DG methods
//!
//! ## Theorem: Trapezoidal-rule exactness for periodic functions (Davis & Rabinowitz 1984, §2.9)
//!
//! For a function `f` that is periodic and has continuous derivatives of all orders on `[-1, 1)`,
//! the N-point trapezoidal rule with equispaced nodes `x_j = -1 + 2j/N`, `j = 0,...,N-1`, and
//! weights `w_j = 2/N` integrates the real trigonometric polynomials of degree ≤ ⌊N/2⌋ exactly:
//! ```text
//!   (2/N) Σ_{j=0}^{N-1} f(x_j) = ∫₋₁¹ f(x) dx    for deg(f) ≤ ⌊N/2⌋.
//! ```
//! This makes equispaced nodes the natural quadrature for the real Fourier DG basis, analogous
//! to GLL nodes for Legendre DG.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{ConfigError, KwaversError};
use ndarray::Array1;

/// Compute Gauss-Lobatto-Legendre (GLL) quadrature nodes and weights
///
/// Returns (nodes, weights) for N points on [-1, 1]
/// # Errors
/// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
///
pub fn gauss_lobatto_quadrature(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 2 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "quadrature_points".to_owned(),
            value: n.to_string(),
            constraint: ">= 2".to_owned(),
        }));
    }

    let mut nodes = Array1::zeros(n);
    let mut weights = Array1::zeros(n);

    // Endpoints
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;
    weights[0] = 2.0 / ((n * (n - 1)) as f64);
    weights[n - 1] = weights[0];

    // Compute interior nodes using Newton-Raphson on P'_{n-1}(x)
    let p = n - 1;

    for i in 1..=(n - 1) / 2 {
        // Initial guess (Chebyshev nodes)
        let mut x = -((TWO_PI * i as f64) / 2.0f64.mul_add(p as f64, 1.0)).cos();

        // Newton iterations
        for _ in 0..100 {
            let (ln, dln) = legendre_poly_and_deriv(p, x);
            // P''_n
            let ddln = (2.0 * x).mul_add(dln, -((p * (p + 1)) as f64 * ln)) / (1.0 - x * x);

            let delta = dln / ddln;
            x -= delta;

            if delta.abs() < 1e-14 {
                break;
            }
        }

        nodes[i] = x;
        nodes[n - 1 - i] = -x;

        // Weight: w_i = 2 / (N(N-1) [P_{N-1}(x_i)]^2)
        let ln = legendre_poly(p, x);
        let w = 2.0 / ((p * (p + 1)) as f64 * ln * ln);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    // Middle point for odd N
    if !n.is_multiple_of(2) {
        nodes[n / 2] = 0.0;
        let ln = legendre_poly(p, 0.0);
        weights[n / 2] = 2.0 / ((p * (p + 1)) as f64 * ln * ln);
    }

    Ok((nodes, weights))
}

/// Compute equispaced periodic quadrature nodes and weights for the Fourier DG basis.
///
/// Returns `(nodes, weights)` where nodes are `x_j = -1 + 2j/N` for `j = 0,...,N-1`
/// (N equispaced points on `[-1, 1)`) and all weights equal `2/N`.
///
/// By the trapezoidal exactness theorem these nodes and weights integrate real trigonometric
/// polynomials of degree ≤ `⌊N/2⌋` exactly. They do NOT include the right endpoint `x=1`,
/// satisfying the Fourier basis periodicity constraint (Hesthaven & Warburton 2008, §5.3).
///
/// # Errors
/// Returns [`KwaversError::Config`] when `n < 2`.
pub fn fourier_periodic_nodes(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 2 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "fourier_nodes".to_owned(),
            value: n.to_string(),
            constraint: ">= 2".to_owned(),
        }));
    }
    let weight = 2.0 / n as f64;
    let nodes = Array1::from_iter((0..n).map(|j| -1.0 + 2.0 * j as f64 / n as f64));
    let weights = Array1::from_elem(n, weight);
    Ok((nodes, weights))
}

fn legendre_poly(n: usize, x: f64) -> f64 {
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

fn legendre_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    let ln = legendre_poly(n, x);
    let l_prev = legendre_poly(n - 1, x);

    let dln = (n as f64) * x.mul_add(-ln, l_prev) / x.mul_add(-x, 1.0);
    (ln, dln)
}
