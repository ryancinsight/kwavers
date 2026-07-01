//! Elastic stiffness tensor operations
//!
//! Implements stiffness matrices in Voigt notation for various symmetries

use super::types::AnisotropyType;
use kwavers_core::constants::{LAME_TO_STIFFNESS_FACTOR, SYMMETRY_TOLERANCE};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::Array2;

/// Full elastic stiffness tensor (6x6 in Voigt notation)
#[derive(Debug, Clone)]
pub struct AnisotropicStiffnessTensor {
    /// Stiffness matrix in Voigt notation
    pub c: Array2<f64>,
    /// Anisotropy type
    pub anisotropy_type: AnisotropyType,
}

impl AnisotropicStiffnessTensor {
    /// Create isotropic stiffness tensor from Lamé parameters
    #[must_use]
    pub fn isotropic(lambda: f64, mu: f64) -> Self {
        let mut c = Array2::zeros((6, 6));

        // Diagonal terms
        c[[0, 0]] = LAME_TO_STIFFNESS_FACTOR.mul_add(mu, lambda); // C11
        c[[1, 1]] = LAME_TO_STIFFNESS_FACTOR.mul_add(mu, lambda); // C22
        c[[2, 2]] = LAME_TO_STIFFNESS_FACTOR.mul_add(mu, lambda); // C33
        c[[3, 3]] = mu; // C44
        c[[4, 4]] = mu; // C55
        c[[5, 5]] = mu; // C66

        // Off-diagonal terms
        c[[0, 1]] = lambda;
        c[[1, 0]] = lambda;
        c[[0, 2]] = lambda;
        c[[2, 0]] = lambda;
        c[[1, 2]] = lambda;
        c[[2, 1]] = lambda;

        Self {
            c,
            anisotropy_type: AnisotropyType::Isotropic,
        }
    }

    /// Create transversely isotropic tensor (fiber direction along z)
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn transversely_isotropic(
        c11: f64,
        c12: f64,
        c13: f64,
        c33: f64,
        c44: f64,
    ) -> KwaversResult<Self> {
        // Validate parameters
        if c11 <= 0.0 || c33 <= 0.0 || c44 <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "stiffness_components".to_owned(),
                value: format!("c11={c11}, c33={c33}, c44={c44}"),
                constraint: "Diagonal components must be positive".to_owned(),
            }));
        }

        // C66 = (C11 - C12) / 2 for transverse isotropy
        let c66 = (c11 - c12) / 2.0;

        let mut c = Array2::zeros((6, 6));

        // Fill symmetric matrix
        c[[0, 0]] = c11;
        c[[1, 1]] = c11;
        c[[2, 2]] = c33;
        c[[3, 3]] = c44;
        c[[4, 4]] = c44;
        c[[5, 5]] = c66;

        c[[0, 1]] = c12;
        c[[1, 0]] = c12;
        c[[0, 2]] = c13;
        c[[2, 0]] = c13;
        c[[1, 2]] = c13;
        c[[2, 1]] = c13;

        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::TransverselyIsotropic,
        })
    }

    /// Create orthotropic stiffness tensor
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn orthotropic(components: [[f64; 6]; 6]) -> KwaversResult<Self> {
        let mut c = Array2::zeros((6, 6));

        // Copy components ensuring symmetry
        for i in 0..6 {
            for j in 0..6 {
                c[[i, j]] = components[i][j];
            }
        }

        // Verify symmetry
        for i in 0..6 {
            for j in i + 1..6 {
                if (c[[i, j]] - c[[j, i]]).abs() > SYMMETRY_TOLERANCE {
                    return Err(KwaversError::Validation(ValidationError::FieldValidation {
                        field: "stiffness_matrix".to_owned(),
                        value: format!(
                            "C[{},{}]={}, C[{},{}]={}",
                            i,
                            j,
                            c[[i, j]],
                            j,
                            i,
                            c[[j, i]]
                        ),
                        constraint: "Matrix must be symmetric".to_owned(),
                    }));
                }
            }
        }

        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::Orthotropic,
        })
    }

    /// Phase velocities `[qP, qS1, qS2]` \[m·s⁻¹] along unit `direction` at
    /// `density`, computed from this stiffness tensor via the **Christoffel
    /// equation** (`Γ = Cₙ`, eigenvalues `ρv²`). Sorted descending
    /// (quasi-longitudinal first).
    /// # Errors
    /// - Returns [`Err`] if `density ≤ 0`.
    pub fn phase_velocities(&self, direction: &[f64; 3], density: f64) -> KwaversResult<[f64; 3]> {
        super::christoffel::ChristoffelEquation::create(self.clone(), density)
            .phase_velocities(direction)
    }

    /// Group (energy) velocity vector for each mode along `direction` at
    /// `density`, via the Christoffel equation. Energy walks off the phase
    /// direction in anisotropic media.
    /// # Errors
    /// - Returns [`Err`] for non-positive density or zero direction.
    pub fn group_velocities(
        &self,
        direction: &[f64; 3],
        density: f64,
    ) -> KwaversResult<[[f64; 3]; 3]> {
        super::christoffel::ChristoffelEquation::create(self.clone(), density)
            .group_velocities(direction)
    }

    /// Maximum quasi-longitudinal phase speed over all propagation directions
    /// \[m·s⁻¹] — the CFL reference speed for this anisotropic medium (isotropic
    /// → `√((λ+2μ)/ρ)` exactly). See [`ChristoffelEquation::max_phase_velocity`].
    ///
    /// [`ChristoffelEquation::max_phase_velocity`]: super::christoffel::ChristoffelEquation::max_phase_velocity
    /// # Errors
    /// - Returns [`Err`] if `density ≤ 0`.
    pub fn max_phase_velocity(&self, density: f64) -> KwaversResult<f64> {
        super::christoffel::ChristoffelEquation::create(self.clone(), density).max_phase_velocity()
    }

    /// Check if tensor is positive definite
    #[must_use]
    pub fn is_positive_definite(&self) -> bool {
        // Use Sylvester's criterion: all leading principal minors must be positive
        for k in 1..=6 {
            let submatrix = self.c.slice(ndarray::s![0..k, 0..k]);
            let det = Self::determinant_2d(&submatrix.to_owned());
            if det <= 0.0 {
                return false;
            }
        }
        true
    }

    /// Calculate determinant of 2D array using nalgebra LU decomposition
    ///
    /// Uses nalgebra's robust LU decomposition with partial pivoting for numerical stability.
    ///
    /// # References
    /// - Golub & Van Loan (2013): "Matrix Computations", Algorithm 3.4.1
    fn determinant_2d(matrix: &Array2<f64>) -> f64 {
        use nalgebra::DMatrix;

        let n = matrix.shape()[0];

        // Fast path for small matrices
        if n == 1 {
            return matrix[[0, 0]];
        } else if n == 2 {
            return matrix[[0, 0]].mul_add(matrix[[1, 1]], -(matrix[[0, 1]] * matrix[[1, 0]]));
        }

        // For larger matrices (3x3 and above), use nalgebra's LU decomposition
        // Convert ndarray to nalgebra DMatrix
        let mut na_matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                na_matrix[(i, j)] = matrix[[i, j]];
            }
        }

        // Compute LU decomposition with partial pivoting
        // det(A) = det(P) * det(L) * det(U) = (-1)^p * ∏ u_ii
        // where p is the number of permutations in P
        match na_matrix.clone().lu().determinant() {
            det if det.is_finite() => det,
            _ => 0.0, // Handle NaN/Inf cases
        }
    }

    /// Get compliance matrix (inverse of stiffness)
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn compliance_matrix(&self) -> KwaversResult<Array2<f64>> {
        use nalgebra::DMatrix;

        // Convert to nalgebra matrix
        let mut matrix = DMatrix::zeros(6, 6);
        for i in 0..6 {
            for j in 0..6 {
                matrix[(i, j)] = self.c[[i, j]];
            }
        }

        // Compute inverse
        match matrix.try_inverse() {
            Some(inv) => {
                // Convert back to ndarray
                let mut compliance = Array2::zeros((6, 6));
                for i in 0..6 {
                    for j in 0..6 {
                        compliance[[i, j]] = inv[(i, j)];
                    }
                }
                Ok(compliance)
            }
            None => Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "stiffness_matrix".to_owned(),
                value: "singular".to_owned(),
                constraint: "Stiffness matrix must be invertible".to_owned(),
            })),
        }
    }

    /// Apply rotation to stiffness tensor
    #[must_use]
    pub fn rotate(&self, rotation: &super::rotation::RotationMatrix) -> Self {
        let rotated = rotation.apply_to_stiffness(&self.c);
        Self {
            c: rotated,
            anisotropy_type: self.anisotropy_type,
        }
    }
}
