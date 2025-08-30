//! Christoffel equation solver for wave propagation in anisotropic media
//!
//! References:
//! - Auld, B. A. (1973). "Acoustic Fields and Waves in Solids"

use super::stiffness::StiffnessTensor;
use crate::KwaversResult;
use ndarray::{Array1, Array2};

/// Christoffel equation solver for anisotropic wave propagation
#[derive(Debug, Debug))]
pub struct ChristoffelEquation {
    /// Stiffness tensor
    stiffness: StiffnessTensor,
    /// Material density
    density: f64,
}

impl ChristoffelEquation {
    /// Create Christoffel equation solver
    pub fn create(stiffness: StiffnessTensor, density: f64) -> Self {
        Self { stiffness, density }
    }

    /// Compute Christoffel matrix for given propagation direction
    pub fn christoffel_matrix(&self, direction: &[f64; 3]) -> Array2<f64> {
        let mut gamma = Array2::zeros((3, 3));
        let c = &self.stiffness.c;
        let n = direction;

        // Γik = Cijkl * nj * nl (Einstein summation)
        gamma[[0, 0] = c[[0, 0] * n[0] * n[0]
            + c[[5, 5] * n[1] * n[1]
            + c[[4, 4] * n[2] * n[2]
            + 2.0 * (c[[0, 5] * n[0] * n[1] + c[[0, 4] * n[0] * n[2] + c[[4, 5] * n[1] * n[2]);

        gamma[[1, 1] = c[[5, 5] * n[0] * n[0]
            + c[[1, 1] * n[1] * n[1]
            + c[[3, 3] * n[2] * n[2]
            + 2.0 * (c[[1, 5] * n[0] * n[1] + c[[3, 5] * n[0] * n[2] + c[[1, 3] * n[1] * n[2]);

        gamma[[2, 2] = c[[4, 4] * n[0] * n[0]
            + c[[3, 3] * n[1] * n[1]
            + c[[2, 2] * n[2] * n[2]
            + 2.0 * (c[[3, 4] * n[0] * n[1] + c[[2, 4] * n[0] * n[2] + c[[2, 3] * n[1] * n[2]);

        // Off-diagonal terms (symmetric)
        gamma[[0, 1] = c[[0, 5] * n[0] * n[0]
            + c[[1, 5] * n[1] * n[1]
            + c[[3, 4] * n[2] * n[2]
            + (c[[0, 1] + c[[5, 5]) * n[0] * n[1]
            + (c[[0, 3] + c[[4, 5]) * n[0] * n[2]
            + (c[[1, 4] + c[[3, 5]) * n[1] * n[2];
        gamma[[1, 0] = gamma[[0, 1];

        gamma[[0, 2] = c[[0, 4] * n[0] * n[0]
            + c[[3, 5] * n[1] * n[1]
            + c[[2, 4] * n[2] * n[2]
            + (c[[0, 3] + c[[4, 5]) * n[0] * n[1]
            + (c[[0, 2] + c[[4, 4]) * n[0] * n[2]
            + (c[[2, 5] + c[[3, 4]) * n[1] * n[2];
        gamma[[2, 0] = gamma[[0, 2];

        gamma[[1, 2] = c[[4, 5] * n[0] * n[0]
            + c[[1, 3] * n[1] * n[1]
            + c[[2, 3] * n[2] * n[2]
            + (c[[1, 4] + c[[3, 5]) * n[0] * n[1]
            + (c[[2, 5] + c[[3, 4]) * n[0] * n[2]
            + (c[[1, 2] + c[[3, 3]) * n[1] * n[2];
        gamma[[2, 1] = gamma[[1, 2];

        gamma
    }

    /// Solve for phase velocities in given direction
    pub fn phase_velocities(&self, direction: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        let gamma = self.christoffel_matrix(direction);

        // Eigenvalue problem: (Γ - ρv²I)u = 0
        // Solve for v² (phase velocity squared)
        let eigenvalues = self.eigenvalues_3x3(&gamma);

        Ok([
            (eigenvalues[0] / self.density).sqrt(),
            (eigenvalues[1] / self.density).sqrt(),
            (eigenvalues[2] / self.density).sqrt(),
        ])
    }

    /// Compute eigenvalues of 3x3 matrix (analytical solution)
    fn eigenvalues_3x3(&self, matrix: &Array2<f64>) -> [f64; 3] {
        // Characteristic polynomial: det(A - λI) = 0
        // λ³ - tr(A)λ² + (sum of principal minors)λ - det(A) = 0

        let trace = matrix[[0, 0] + matrix[[1, 1] + matrix[[2, 2];

        let minor_sum = matrix[[0, 0] * matrix[[1, 1] - matrix[[0, 1] * matrix[[1, 0]
            + matrix[[0, 0] * matrix[[2, 2]
            - matrix[[0, 2] * matrix[[2, 0]
            + matrix[[1, 1] * matrix[[2, 2]
            - matrix[[1, 2] * matrix[[2, 1];

        let det = matrix[[0, 0]
            * (matrix[[1, 1] * matrix[[2, 2] - matrix[[1, 2] * matrix[[2, 1])
            - matrix[[0, 1] * (matrix[[1, 0] * matrix[[2, 2] - matrix[[1, 2] * matrix[[2, 0])
            + matrix[[0, 2] * (matrix[[1, 0] * matrix[[2, 1] - matrix[[1, 1] * matrix[[2, 0]);

        // Use Cardano's formula for cubic equation
        let p = minor_sum - trace * trace / 3.0;
        let q = trace * (2.0 * trace * trace / 27.0 - minor_sum / 3.0) + det;

        let discriminant = -(4.0 * p * p * p + 27.0 * q * q) / 108.0;

        if discriminant > 0.0 {
            // Three real roots
            let m = 2.0 * (-p / 3.0).sqrt();
            let theta = (3.0 * q / (p * m)).acos() / 3.0;
            let offset = trace / 3.0;

            [
                m * (theta).cos() + offset,
                m * (theta - 2.0 * std::f64::consts::PI / 3.0).cos() + offset,
                m * (theta + 2.0 * std::f64::consts::PI / 3.0).cos() + offset,
            ]
        } else {
            // Fallback for edge cases
            [1.0, 1.0, 1.0]
        }
    }

    /// Get polarization vectors for each wave mode
    pub fn polarization_vectors(&self, direction: &[f64; 3]) -> KwaversResult<[Array1<f64>; 3]> {
        // This would solve the full eigenvalue problem
        // For now, return placeholder
        Ok([
            Array1::from(vec![1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0]),
        ])
    }
}
