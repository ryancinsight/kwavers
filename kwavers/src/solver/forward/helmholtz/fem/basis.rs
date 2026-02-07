//! FEM Basis Functions for Helmholtz Elements
//!
//! Provides polynomial basis functions and their derivatives for tetrahedral elements.
//! Supports linear (P1) and quadratic (P2) Lagrange basis functions.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::mesh::Tetrahedron;
use ndarray::Array2;

/// Gauss quadrature points and weights for tetrahedral elements
#[derive(Debug, Clone)]
pub struct GaussPoint {
    pub local_coords: [f64; 3], // ξ, η, ζ barycentric coordinates
    pub weight: f64,
}

/// Lagrange basis functions for tetrahedral elements
#[derive(Debug, Clone)]
pub struct BasisFunction {
    degree: usize,
    num_functions: usize,
}

impl BasisFunction {
    /// Create basis functions of given polynomial degree
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if degree is not 1 or 2.
    pub fn new(degree: usize) -> KwaversResult<Self> {
        let num_functions = match degree {
            1 => 4,  // Linear tetrahedron: 4 nodes
            2 => 10, // Quadratic tetrahedron: 10 nodes
            _ => {
                return Err(KwaversError::InvalidInput(format!(
                    "Unsupported polynomial degree: {degree}. Use 1 (linear) or 2 (quadratic)"
                )))
            }
        };

        Ok(Self {
            degree,
            num_functions,
        })
    }

    /// Get number of basis functions
    #[must_use]
    pub fn num_functions(&self) -> usize {
        self.num_functions
    }

    /// Evaluate all basis functions at local coordinates ξ ∈ [0,1]³
    pub fn evaluate(&self, xi: [f64; 3]) -> KwaversResult<Vec<f64>> {
        match self.degree {
            1 => self.evaluate_linear(xi),
            2 => self.evaluate_quadratic(xi),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unsupported polynomial degree: {}",
                self.degree
            ))),
        }
    }

    /// Evaluate derivatives of all basis functions
    pub fn evaluate_derivatives(&self, xi: [f64; 3]) -> KwaversResult<Array2<f64>> {
        match self.degree {
            1 => self.evaluate_linear_derivatives(xi),
            2 => self.evaluate_quadratic_derivatives(xi),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unsupported polynomial degree: {}",
                self.degree
            ))),
        }
    }

    /// Linear (P1) tetrahedral basis functions
    /// φ₁ = 1 - ξ - η - ζ
    /// φ₂ = ξ
    /// φ₃ = η
    /// φ₄ = ζ
    fn evaluate_linear(&self, xi: [f64; 3]) -> KwaversResult<Vec<f64>> {
        let [xi_xi, eta, zeta] = xi;

        // Validate coordinates are in valid range
        if xi_xi < 0.0 || eta < 0.0 || zeta < 0.0 || xi_xi + eta + zeta > 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Invalid barycentric coordinates: ({}, {}, {})",
                xi_xi, eta, zeta
            )));
        }

        Ok(vec![
            1.0 - xi_xi - eta - zeta, // φ₁: vertex 0
            xi_xi,                    // φ₂: vertex 1
            eta,                      // φ₃: vertex 2
            zeta,                     // φ₄: vertex 3
        ])
    }

    /// Derivatives of linear basis functions
    /// dφ/dξ = [-1, 1, 0, 0]
    /// dφ/dη = [-1, 0, 1, 0]
    /// dφ/dζ = [-1, 0, 0, 1]
    fn evaluate_linear_derivatives(&self, _xi: [f64; 3]) -> KwaversResult<Array2<f64>> {
        // Derivatives are constant for linear elements
        let mut derivatives = Array2::<f64>::zeros((4, 3));

        // dφ/dξ
        derivatives[[0, 0]] = -1.0; // φ₁
        derivatives[[1, 0]] = 1.0; // φ₂
        derivatives[[2, 0]] = 0.0; // φ₃
        derivatives[[3, 0]] = 0.0; // φ₄

        // dφ/dη
        derivatives[[0, 1]] = -1.0; // φ₁
        derivatives[[1, 1]] = 0.0; // φ₂
        derivatives[[2, 1]] = 1.0; // φ₃
        derivatives[[3, 1]] = 0.0; // φ₄

        // dφ/dζ
        derivatives[[0, 2]] = -1.0; // φ₁
        derivatives[[1, 2]] = 0.0; // φ₂
        derivatives[[2, 2]] = 0.0; // φ₃
        derivatives[[3, 2]] = 1.0; // φ₄

        Ok(derivatives)
    }

    /// Quadratic (P2) tetrahedral basis functions
    /// 10-node tetrahedron with edge and face nodes
    fn evaluate_quadratic(&self, xi: [f64; 3]) -> KwaversResult<Vec<f64>> {
        let [xi_xi, eta, zeta] = xi;

        // Validate coordinates
        if xi_xi < 0.0 || eta < 0.0 || zeta < 0.0 || xi_xi + eta + zeta > 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Invalid barycentric coordinates: ({}, {}, {})",
                xi_xi, eta, zeta
            )));
        }

        let lambda1 = 1.0 - xi_xi - eta - zeta;
        let lambda2 = xi_xi;
        let lambda3 = eta;
        let lambda4 = zeta;

        // Vertex functions (same as linear)
        let phi1 = lambda1 * (2.0 * lambda1 - 1.0);
        let phi2 = lambda2 * (2.0 * lambda2 - 1.0);
        let phi3 = lambda3 * (2.0 * lambda3 - 1.0);
        let phi4 = lambda4 * (2.0 * lambda4 - 1.0);

        // Edge functions
        let phi5 = 4.0 * lambda1 * lambda2; // Edge 1-2
        let phi6 = 4.0 * lambda2 * lambda3; // Edge 2-3
        let phi7 = 4.0 * lambda3 * lambda1; // Edge 3-1
        let phi8 = 4.0 * lambda1 * lambda4; // Edge 1-4
        let phi9 = 4.0 * lambda2 * lambda4; // Edge 2-4
        let phi10 = 4.0 * lambda3 * lambda4; // Edge 3-4

        Ok(vec![
            phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10,
        ])
    }

    /// Derivatives of quadratic basis functions
    fn evaluate_quadratic_derivatives(&self, xi: [f64; 3]) -> KwaversResult<Array2<f64>> {
        let [xi_xi, eta, zeta] = xi;
        let lambda1 = 1.0 - xi_xi - eta - zeta;
        let lambda2 = xi_xi;
        let lambda3 = eta;
        let lambda4 = zeta;

        let mut derivatives = Array2::<f64>::zeros((10, 3));

        // Derivatives with respect to ξ (lambda2 direction)
        derivatives[[0, 0]] = 4.0 * lambda1 - 1.0; // φ₁
        derivatives[[1, 0]] = 4.0 * lambda2 - 1.0; // φ₂
        derivatives[[2, 0]] = 0.0; // φ₃
        derivatives[[3, 0]] = 0.0; // φ₄
        derivatives[[4, 0]] = 4.0 * (lambda1 - lambda2); // φ₅
        derivatives[[5, 0]] = 4.0 * lambda3; // φ₆
        derivatives[[6, 0]] = 4.0 * (lambda3 - lambda1); // φ₇
        derivatives[[7, 0]] = 4.0 * (lambda4 - lambda1); // φ₈
        derivatives[[8, 0]] = 4.0 * lambda4; // φ₉
        derivatives[[9, 0]] = 0.0; // φ₁₀

        // Derivatives with respect to η (lambda3 direction)
        derivatives[[0, 1]] = 4.0 * lambda1 - 1.0; // φ₁
        derivatives[[1, 1]] = 0.0; // φ₂
        derivatives[[2, 1]] = 4.0 * lambda3 - 1.0; // φ₃
        derivatives[[3, 1]] = 0.0; // φ₄
        derivatives[[4, 1]] = -4.0 * lambda2; // φ₅
        derivatives[[5, 1]] = 4.0 * (lambda2 - lambda3); // φ₆
        derivatives[[6, 1]] = 4.0 * lambda1; // φ₇
        derivatives[[7, 1]] = -4.0 * lambda1; // φ₈
        derivatives[[8, 1]] = 0.0; // φ₉
        derivatives[[9, 1]] = 4.0 * lambda4; // φ₁₀

        // Derivatives with respect to ζ (lambda4 direction)
        derivatives[[0, 2]] = 4.0 * lambda1 - 1.0; // φ₁
        derivatives[[1, 2]] = 0.0; // φ₂
        derivatives[[2, 2]] = 0.0; // φ₃
        derivatives[[3, 2]] = 4.0 * lambda4 - 1.0; // φ₄
        derivatives[[4, 2]] = 0.0; // φ₅
        derivatives[[5, 2]] = 0.0; // φ₆
        derivatives[[6, 2]] = -4.0 * lambda3; // φ₇
        derivatives[[7, 2]] = 4.0 * lambda1; // φ₈
        derivatives[[8, 2]] = 4.0 * lambda2; // φ₉
        derivatives[[9, 2]] = 4.0 * (lambda3 - lambda4); // φ₁₀

        Ok(derivatives)
    }
}

/// Gauss quadrature for tetrahedral integration
#[derive(Debug)]
pub struct GaussQuadrature;

impl GaussQuadrature {
    /// Create new assembly helper
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Get Gauss quadrature points for tetrahedral integration
    #[must_use]
    pub fn get_gauss_points(&self, polynomial_degree: usize) -> Vec<GaussPoint> {
        match polynomial_degree {
            1 => self.gauss_points_linear(),
            2 => self.gauss_points_quadratic(),
            _ => self.gauss_points_linear(), // Default to linear
        }
    }

    /// 1-point Gauss quadrature for linear elements
    fn gauss_points_linear(&self) -> Vec<GaussPoint> {
        vec![GaussPoint {
            local_coords: [0.25, 0.25, 0.25], // Centroid
            weight: 1.0 / 6.0,                // Volume of reference tetrahedron is 1/6
        }]
    }

    /// 4-point Gauss quadrature for quadratic elements
    fn gauss_points_quadratic(&self) -> Vec<GaussPoint> {
        let a = (5.0 - 3.0_f64.sqrt()) / 20.0;
        let b = (5.0 + 3.0_f64.sqrt()) / 20.0;
        let w_a = (1.0 / 6.0) * (5.0 / 12.0);

        vec![
            // Point 1
            GaussPoint {
                local_coords: [a, a, a],
                weight: w_a,
            },
            // Point 2
            GaussPoint {
                local_coords: [b, a, a],
                weight: w_a,
            },
            // Point 3
            GaussPoint {
                local_coords: [a, b, a],
                weight: w_a,
            },
            // Point 4
            GaussPoint {
                local_coords: [a, a, b],
                weight: w_a,
            },
        ]
    }

    /// Add element contribution to global matrices
    pub fn add_element_contribution(
        &self,
        global_stiffness: &mut crate::math::linear_algebra::sparse::CompressedSparseRowMatrix<
            num_complex::Complex64,
        >,
        global_mass: &mut crate::math::linear_algebra::sparse::CompressedSparseRowMatrix<
            num_complex::Complex64,
        >,
        global_rhs: &mut ndarray::Array1<num_complex::Complex64>,
        elem_stiffness: &ndarray::Array2<num_complex::Complex64>,
        elem_mass: &ndarray::Array2<num_complex::Complex64>,
        elem_rhs: &ndarray::Array1<num_complex::Complex64>,
        element: &Tetrahedron,
        _basis: &BasisFunction,
    ) -> KwaversResult<()> {
        // Add element contributions to global matrices
        for i in 0..elem_stiffness.nrows() {
            let global_i = element.nodes[i];

            // Right-hand side
            global_rhs[global_i] += elem_rhs[i];

            for j in 0..elem_stiffness.ncols() {
                let global_j = element.nodes[j];

                // Stiffness matrix
                global_stiffness.add_value(global_i, global_j, elem_stiffness[[i, j]]);

                // Mass matrix
                global_mass.add_value(global_i, global_j, elem_mass[[i, j]]);
            }
        }

        Ok(())
    }
}

impl Default for GaussQuadrature {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BasisFunction {
    fn default() -> Self {
        Self::new(1).expect("Linear elements (degree=1) always valid")
    }
}
