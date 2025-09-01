//! Quadrature rules for DG methods
//!
//! This module provides Gaussian quadrature rules for numerical integration
//! in discontinuous Galerkin methods.

use crate::{KwaversError, KwaversResult};
use ndarray::Array1;

/// Maximum iterations for Newton-Raphson in quadrature computation
const MAX_NEWTON_ITERATIONS: usize = 50;

/// Tolerance for Newton-Raphson convergence
const NEWTON_TOLERANCE: f64 = 1e-14;

/// Generate Gauss-Lobatto quadrature nodes and weights
///
/// Returns nodes on [-1, 1] and corresponding weights for integration
pub fn gauss_lobatto_quadrature(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 2 {
        return Err(KwaversError::InvalidInput(
            "Gauss-Lobatto quadrature requires at least 2 nodes".to_string(),
        ));
    }

    let mut nodes = Array1::zeros(n);
    let mut weights = Array1::zeros(n);

    // Endpoints
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;

    if n == 2 {
        weights[0] = 1.0;
        weights[1] = 1.0;
        return Ok((nodes, weights));
    }

    // Interior nodes are roots of P'_{n-1}(x)
    // Use Chebyshev nodes as initial guess
    for i in 1..n - 1 {
        let theta = std::f64::consts::PI * (n - 1 - i) as f64 / (n - 1) as f64;
        nodes[i] = theta.cos();
    }

    // Special case for n=3: interior node is exactly 0
    if n == 3 {
        nodes[1] = 0.0;
    } else {
        // Newton-Raphson iteration for interior nodes
        // Find roots of P'_{n-1}(x) where P is Legendre polynomial
        for i in 1..n - 1 {
            let mut x = nodes[i];
            for _ in 0..MAX_NEWTON_ITERATIONS {
                // We need the derivative and second derivative of P_{n-1}
                let (p_nm1, dp_nm1) = legendre_poly_deriv(n - 1, x);

                // For Newton iteration on P'_{n-1} = 0, we need P' and P''
                // P''_{n-1} can be computed from recurrence
                let ddp = ((n - 1) as f64 * (x * dp_nm1 - p_nm1)) / (x * x - 1.0);

                let dx = -dp_nm1 / ddp;
                x += dx;
                if dx.abs() < NEWTON_TOLERANCE {
                    break;
                }
            }
            nodes[i] = x;
        }
    }

    // Compute weights: w_i = 2 / (n(n-1) [P_{n-1}(x_i)]^2)
    let n_f = n as f64;
    for (i, &xi) in nodes.iter().enumerate() {
        let (p, _) = legendre_poly_deriv(n - 1, xi);
        weights[i] = 2.0 / (n_f * (n_f - 1.0) * p * p);
    }

    Ok((nodes, weights))
}

/// Compute Legendre polynomial and derivative (internal helper)
fn legendre_poly_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    // Recurrence relation
    let mut p_nm2 = 1.0;
    let mut p_nm1 = x;
    let mut dp_nm2 = 0.0;
    let mut dp_nm1 = 1.0;

    for k in 2..=n {
        let k_f = k as f64;
        let p_n = ((2.0 * k_f - 1.0) * x * p_nm1 - (k_f - 1.0) * p_nm2) / k_f;
        let dp_n = dp_nm2 + (2.0 * k_f - 1.0) * p_nm1;

        p_nm2 = p_nm1;
        p_nm1 = p_n;
        dp_nm2 = dp_nm1;
        dp_nm1 = dp_n;
    }

    (p_nm1, dp_nm1)
}

/// Generate Gauss-Legendre quadrature nodes and weights
pub fn gauss_legendre_quadrature(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 1 {
        return Err(KwaversError::InvalidInput(
            "Gauss-Legendre quadrature requires at least 1 node".to_string(),
        ));
    }

    let mut nodes = Array1::zeros(n);
    let mut weights = Array1::zeros(n);

    // Use symmetry
    let m = n.div_ceil(2);

    for i in 0..m {
        // Initial guess using Chebyshev nodes
        let z = ((2 * i + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos();

        // Newton-Raphson to find roots of P_n(x)
        let mut x = z;
        for _ in 0..MAX_NEWTON_ITERATIONS {
            let (p, dp) = legendre_poly_deriv(n, x);
            let dx = -p / dp;
            x += dx;
            if dx.abs() < NEWTON_TOLERANCE {
                break;
            }
        }

        nodes[i] = -x;
        nodes[n - 1 - i] = x;

        // Weight formula: w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
        let (_, dp) = legendre_poly_deriv(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    Ok((nodes, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gauss_lobatto_2_points() {
        let (nodes, weights) = gauss_lobatto_quadrature(2).unwrap();
        assert_eq!(nodes[0], -1.0);
        assert_eq!(nodes[1], 1.0);
        assert_eq!(weights[0], 1.0);
        assert_eq!(weights[1], 1.0);
    }

    #[test]
    fn test_gauss_lobatto_3_points() {
        let (nodes, weights) = gauss_lobatto_quadrature(3).unwrap();
        assert_eq!(nodes[0], -1.0);
        assert_relative_eq!(nodes[1], 0.0, epsilon = 1e-10);
        assert_eq!(nodes[2], 1.0);

        // Weights for 3-point Gauss-Lobatto
        assert_relative_eq!(weights[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(weights[1], 4.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(weights[2], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrature_exactness() {
        // Test that quadrature is exact for polynomials up to degree 2n-3
        let (nodes, weights) = gauss_lobatto_quadrature(4).unwrap();

        // Integrate x^2 from -1 to 1 (exact value = 2/3)
        let mut integral = 0.0;
        for (i, &xi) in nodes.iter().enumerate() {
            integral += weights[i] * xi * xi;
        }
        assert_relative_eq!(integral, 2.0 / 3.0, epsilon = 1e-10);
    }
}
