//! Spectral Element Basis Functions
//!
//! Implements Lagrange polynomial basis functions on Gauss-Lobatto-Legendre (GLL)
//! quadrature points for spectral element methods.
//!
//! ## Mathematical Foundation
//!
//! The basis functions are Lagrange polynomials defined on GLL points:
//!
//! ```text
//! ℓ_i(ξ) = ∏_{j=0,j≠i}^N (ξ - ξ_j) / (ξ_i - ξ_j)
//! ```
//!
//! where ξ_i are the Gauss-Lobatto-Legendre points.
//!
//! ## GLL Points
//!
//! Gauss-Lobatto-Legendre points include the endpoints (-1, 1) and are optimal
//! for spectral element methods due to their interpolation properties.

use crate::core::error::{KwaversResult, KwaversError};
use ndarray::Array1;

/// Spectral element basis functions using Lagrange polynomials on GLL points
#[derive(Debug, Clone)]
pub struct SemBasis {
    /// Polynomial degree (N)
    pub degree: usize,
    /// Gauss-Lobatto-Legendre points in [-1, 1]
    pub gll_points: Array1<f64>,
    pub gll_weights: Array1<f64>,
    /// Lagrange basis function values at GLL points
    /// Shape: (n_points, n_points) where n_points = degree + 1
    pub lagrange_values: ndarray::Array2<f64>,
    /// Derivatives of Lagrange functions at GLL points
    /// Shape: (n_points, n_points)
    pub lagrange_derivatives: ndarray::Array2<f64>,
}

impl SemBasis {
    /// Create spectral element basis for given polynomial degree
    #[must_use]
    pub fn new(degree: usize) -> Self {
        let n_points = degree + 1;
        let gll_points = Self::compute_gll_points(n_points);
        let gll_weights = Self::compute_gll_weights(n_points, &gll_points);
        let lagrange_values = Self::compute_lagrange_matrix(&gll_points);
        let lagrange_derivatives = Self::compute_lagrange_derivatives(&gll_points);

        Self {
            degree,
            gll_points,
            gll_weights,
            lagrange_values,
            lagrange_derivatives,
        }
    }

    /// Evaluate Lagrange polynomial ℓ_i at point ξ
    #[must_use]
    pub fn lagrange(&self, i: usize, xi: f64) -> f64 {
        let mut result = 1.0;

        for j in 0..self.gll_points.len() {
            if j != i {
                result *= (xi - self.gll_points[j]) / (self.gll_points[i] - self.gll_points[j]);
            }
        }

        result
    }

    /// Evaluate derivative of Lagrange polynomial dℓ_i/dξ at point ξ
    #[must_use]
    pub fn lagrange_derivative(&self, i: usize, xi: f64) -> f64 {
        let mut numerator = 0.0;
        let mut denominator = 1.0;

        for j in 0..self.gll_points.len() {
            if j != i {
                let mut term = 1.0;
                for k in 0..self.gll_points.len() {
                    if k != i && k != j {
                        term *= xi - self.gll_points[k];
                    }
                }
                numerator += term;
                denominator *= self.gll_points[i] - self.gll_points[j];
            }
        }

        numerator / denominator
    }

    /// Get the number of GLL points
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.degree + 1
    }

    /// Compute Gauss-Lobatto-Legendre points using Newton iteration
    ///
    /// GLL points are the roots of: (1-ξ²)P'_N(ξ) = 0
    /// where P_N is the Legendre polynomial of degree N
    fn compute_gll_points(n_points: usize) -> Array1<f64> {
        let mut points = Array1::zeros(n_points);

        // Endpoints are always -1 and 1
        points[0] = -1.0;
        if n_points > 1 {
            points[n_points - 1] = 1.0;
        }

        // Interior points computed via Newton iteration
        if n_points > 2 {
            let n_interior = n_points - 2;
            for i in 0..n_interior {
                // Initial guess using Chebyshev points
                let xi = -((std::f64::consts::PI * (i + 1) as f64) /
                          (n_interior + 1) as f64).cos();

                // Newton iteration to find GLL point
                let mut x = xi;
                for _ in 0..10 {
                    let (p, dp) = Self::legendre_and_derivative(n_points - 1, x);
                    let numerator = (1.0 - x * x) * dp;
                    let denominator = -2.0 * x * dp + (n_points - 1) as f64 *
                                    (n_points) as f64 * p;

                    if denominator.abs() > 1e-15 {
                        x += numerator / denominator;
                    }
                }

                points[i + 1] = x;
            }
        }

        points
    }

    fn compute_gll_weights(n_points: usize, points: &Array1<f64>) -> Array1<f64> {
        let n = n_points.saturating_sub(1);
        if n == 0 {
            return Array1::from_vec(vec![2.0]);
        }

        let mut weights = Array1::zeros(n_points);
        let n_f = n as f64;
        let denom_base = n_f * (n_f + 1.0);

        for i in 0..n_points {
            let (p_n, _) = Self::legendre_and_derivative(n, points[i]);
            weights[i] = 2.0 / (denom_base * p_n * p_n);
        }

        weights
    }

    /// Compute Legendre polynomial and its derivative
    fn legendre_and_derivative(n: usize, x: f64) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        }

        let mut p0 = 1.0;
        let mut p1 = x;
        let mut dp0 = 0.0;
        let mut dp1 = 1.0;

        for k in 1..n {
            let p2 = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
            let dp2 = ((2 * k + 1) as f64 * p1 + (2 * k + 1) as f64 * x * dp1 -
                      k as f64 * dp0) / (k + 1) as f64;

            p0 = p1;
            p1 = p2;
            dp0 = dp1;
            dp1 = dp2;
        }

        (p1, dp1)
    }

    /// Compute Lagrange interpolation matrix
    ///
    /// Returns matrix L where L[i,j] = ℓ_i(ξ_j)
    fn compute_lagrange_matrix(points: &Array1<f64>) -> ndarray::Array2<f64> {
        let n = points.len();
        let mut matrix = ndarray::Array2::<f64>::eye(n);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[[i, j]] = 1.0;
                    for k in 0..n {
                        if k != i && k != j {
                            matrix[[i, j]] *= (points[j] - points[k]) /
                                             (points[i] - points[k]);
                        }
                    }
                }
            }
        }

        matrix
    }

    /// Compute derivatives of Lagrange functions at GLL points
    ///
    /// Returns matrix D where D[i,j] = dℓ_i/dξ|_(ξ_j)
    fn compute_lagrange_derivatives(points: &Array1<f64>) -> ndarray::Array2<f64> {
        let n = points.len();
        let mut derivatives = ndarray::Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let mut prod = 1.0;
                    for k in 0..n {
                        if k != i && k != j {
                            prod *= (points[j] - points[k]) / (points[i] - points[k]);
                        }
                    }

                    let mut sum = 0.0;
                    for k in 0..n {
                        if k != i && k != j {
                            let mut term = 1.0;
                            for m in 0..n {
                                if m != i && m != k && m != j {
                                    term *= (points[j] - points[m]) /
                                           (points[i] - points[m]);
                                }
                            }
                            sum += term;
                        }
                    }

                    derivatives[[i, j]] = prod * sum;
                } else {
                    // Diagonal terms: dℓ_i/dξ|_(ξ_i) = sum_{j≠i} 1/(ξ_i - ξ_j)
                    let mut sum = 0.0;
                    for k in 0..n {
                        if k != i {
                            sum += 1.0 / (points[i] - points[k]);
                        }
                    }
                    derivatives[[i, i]] = sum;
                }
            }
        }

        derivatives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gll_points_endpoints() {
        let basis = SemBasis::new(4);
        assert_eq!(basis.gll_points[0], -1.0);
        assert_eq!(basis.gll_points[4], 1.0);
    }

    #[test]
    fn test_lagrange_property() {
        let basis = SemBasis::new(3);

        // Test Kronecker delta property: ℓ_i(ξ_j) = δ_ij
        for i in 0..basis.n_points() {
            for j in 0..basis.n_points() {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = basis.lagrange(i, basis.gll_points[j]);
                assert_relative_eq!(actual, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_lagrange_derivative() {
        let basis = SemBasis::new(2);

        // For degree 2, we expect specific derivative values
        // At ξ = -1: dℓ₀/dξ = -3/2, dℓ₁/dξ = 2, dℓ₂/dξ = -1/2
        assert_relative_eq!(basis.lagrange_derivative(0, -1.0), -1.5, epsilon = 1e-12);
        assert_relative_eq!(basis.lagrange_derivative(1, -1.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(basis.lagrange_derivative(2, -1.0), -0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_partition_of_unity() {
        let basis = SemBasis::new(3);

        // Test that ∑ℓ_i(ξ) = 1 for all ξ in [-1, 1]
        let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];

        for &xi in &test_points {
            let sum: f64 = (0..basis.n_points()).map(|i| basis.lagrange(i, xi)).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_gll_weights_sum_to_two() {
        let basis = SemBasis::new(6);
        assert_relative_eq!(basis.gll_weights.sum(), 2.0, epsilon = 1e-12);
    }
}
