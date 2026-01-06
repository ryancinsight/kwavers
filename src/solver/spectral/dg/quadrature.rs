//! Quadrature rules for DG methods

use crate::error::{ConfigError, KwaversError};
use crate::KwaversResult;
use ndarray::Array1;

/// Compute Gauss-Lobatto-Legendre (GLL) quadrature nodes and weights
///
/// Returns (nodes, weights) for N points on [-1, 1]
pub fn gauss_lobatto_quadrature(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 2 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "quadrature_points".to_string(),
            value: n.to_string(),
            constraint: ">= 2".to_string(),
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
        let mut x = -((2.0 * std::f64::consts::PI * i as f64) / (2.0 * p as f64 + 1.0)).cos();

        // Newton iterations
        for _ in 0..100 {
            let (ln, dln) = legendre_poly_and_deriv(p, x);
            // P''_n
            let ddln = (2.0 * x * dln - (p * (p + 1)) as f64 * ln) / (1.0 - x * x);

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
        let l_next = ((2 * i + 1) as f64 * x * l_curr - i as f64 * l_prev) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

fn legendre_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    let ln = legendre_poly(n, x);
    let l_prev = legendre_poly(n - 1, x);

    let dln = (n as f64) * (l_prev - x * ln) / (1.0 - x * x);
    (ln, dln)
}
