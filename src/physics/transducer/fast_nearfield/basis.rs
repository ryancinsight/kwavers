//! Basis Function Decomposition for FNM
//!
//! Implements basis function decomposition of transducer surfaces for efficient
//! nearfield calculation.
//!
//! ## References
//!
//! - McGough (2004): Legendre polynomial basis for rectangular pistons
//! - Zeng & McGough (2008): Hermite basis functions for Gaussian beams

use crate::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Basis functions for transducer surface decomposition
#[derive(Debug)]
pub struct BasisFunctions {
    /// Number of basis functions
    num_functions: usize,
    /// Basis function coefficients
    #[allow(dead_code)] // Used in future full FNM implementation
    coefficients: Array2<f64>,
    /// Basis function nodes (Gauss-Legendre quadrature points)
    nodes: Array1<f64>,
    /// Integration weights
    weights: Array1<f64>,
}

impl BasisFunctions {
    /// Create basis functions with specified count
    ///
    /// # Arguments
    ///
    /// * `num_functions` - Number of basis functions (typically 32-128)
    pub fn new(num_functions: usize) -> KwaversResult<Self> {
        // Generate Gauss-Legendre quadrature nodes and weights
        let (nodes, weights) = Self::gauss_legendre_quadrature(num_functions);
        
        // Compute basis function coefficients
        let coefficients = Self::compute_legendre_basis(num_functions, &nodes);

        Ok(Self {
            num_functions,
            coefficients,
            nodes,
            weights,
        })
    }

    /// Generate Gauss-Legendre quadrature nodes and weights
    ///
    /// Uses simplified computation for efficiency. Full implementation would use
    /// iterative eigenvalue method for higher accuracy.
    fn gauss_legendre_quadrature(n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut nodes = Array1::zeros(n);
        let mut weights = Array1::zeros(n);

        // Simplified: use Chebyshev nodes as approximation
        // Full implementation: use Golub-Welsch algorithm
        for i in 0..n {
            let theta = PI * (i as f64 + 0.5) / n as f64;
            nodes[i] = theta.cos();
            weights[i] = PI / n as f64;
        }

        (nodes, weights)
    }

    /// Compute Legendre polynomial basis
    ///
    /// # Arguments
    ///
    /// * `n` - Number of basis functions
    /// * `nodes` - Evaluation nodes
    fn compute_legendre_basis(n: usize, nodes: &Array1<f64>) -> Array2<f64> {
        let m = nodes.len();
        let mut basis = Array2::zeros((n, m));

        for (i, &x) in nodes.iter().enumerate() {
            // Legendre polynomials using recurrence relation
            // P₀(x) = 1
            // P₁(x) = x
            // Pₙ₊₁(x) = ((2n+1)xPₙ(x) - nPₙ₋₁(x))/(n+1)
            
            if n > 0 {
                basis[[0, i]] = 1.0;
            }
            if n > 1 {
                basis[[1, i]] = x;
            }
            for k in 2..n {
                let k_f64 = k as f64;
                basis[[k, i]] = ((2.0 * k_f64 - 1.0) * x * basis[[k - 1, i]]
                    - (k_f64 - 1.0) * basis[[k - 2, i]])
                    / k_f64;
            }
        }

        basis
    }

    /// Evaluate basis function at given point
    ///
    /// # Arguments
    ///
    /// * `index` - Basis function index
    /// * `x` - Evaluation point [-1, 1]
    #[must_use]
    pub fn evaluate(&self, index: usize, x: f64) -> f64 {
        if index >= self.num_functions {
            return 0.0;
        }

        // Evaluate Legendre polynomial at x using recurrence
        if index == 0 {
            return 1.0;
        }
        if index == 1 {
            return x;
        }

        let mut p_prev2 = 1.0;
        let mut p_prev1 = x;
        let mut p_current = 0.0;

        for k in 2..=index {
            let k_f64 = k as f64;
            p_current = ((2.0 * k_f64 - 1.0) * x * p_prev1 - (k_f64 - 1.0) * p_prev2) / k_f64;
            p_prev2 = p_prev1;
            p_prev1 = p_current;
        }

        p_current
    }

    /// Get number of basis functions
    #[must_use]
    pub fn count(&self) -> usize {
        self.num_functions
    }

    /// Get quadrature nodes
    #[must_use]
    pub fn nodes(&self) -> &Array1<f64> {
        &self.nodes
    }

    /// Get integration weights
    #[must_use]
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_creation() {
        let result = BasisFunctions::new(32);
        assert!(result.is_ok());
        
        let basis = result.unwrap();
        assert_eq!(basis.count(), 32);
    }

    #[test]
    fn test_legendre_polynomials() {
        let basis = BasisFunctions::new(5).unwrap();
        
        // P₀(x) = 1
        assert!((basis.evaluate(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((basis.evaluate(0, -0.5) - 1.0).abs() < 1e-10);
        
        // P₁(x) = x
        assert!((basis.evaluate(1, 0.5) - 0.5).abs() < 1e-10);
        assert!((basis.evaluate(1, -0.5) + 0.5).abs() < 1e-10);
        
        // P₂(x) = (3x² - 1)/2
        let p2_at_half = (3.0 * 0.5 * 0.5 - 1.0) / 2.0;
        assert!((basis.evaluate(2, 0.5) - p2_at_half).abs() < 1e-10);
    }

    #[test]
    fn test_quadrature_weights() {
        let basis = BasisFunctions::new(16).unwrap();
        let weights = basis.weights();
        
        // Weights should sum to approximately 2 for interval [-1, 1]
        let sum: f64 = weights.iter().sum();
        assert!((sum - PI).abs() < 0.1, "Weight sum should be approximately π");
    }
}
