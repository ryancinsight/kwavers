//! Elastic stiffness tensor operations
//!
//! Implements stiffness matrices in Voigt notation for various symmetries

use super::types::AnisotropyType;
use crate::constants::elastic::{LAME_TO_STIFFNESS_FACTOR, SYMMETRY_TOLERANCE};
use crate::{KwaversError, KwaversResult, ValidationError};
use ndarray::Array2;

/// Full elastic stiffness tensor (6x6 in Voigt notation)
#[derive(Debug, Clone))]
pub struct StiffnessTensor {
    /// Stiffness matrix in Voigt notation
    pub c: Array2<f64>,
    /// Anisotropy type
    pub anisotropy_type: AnisotropyType,
}

impl StiffnessTensor {
    /// Create isotropic stiffness tensor from Lamé parameters
    pub fn isotropic(lambda: f64, mu: f64) -> Self {
        let mut c = Array2::zeros((6, 6));

        // Diagonal terms
        c[[0, 0] = lambda + LAME_TO_STIFFNESS_FACTOR * mu; // C11
        c[[1, 1] = lambda + LAME_TO_STIFFNESS_FACTOR * mu; // C22
        c[[2, 2] = lambda + LAME_TO_STIFFNESS_FACTOR * mu; // C33
        c[[3, 3] = mu; // C44
        c[[4, 4] = mu; // C55
        c[[5, 5] = mu; // C66

        // Off-diagonal terms
        c[[0, 1] = lambda;
        c[[1, 0] = lambda;
        c[[0, 2] = lambda;
        c[[2, 0] = lambda;
        c[[1, 2] = lambda;
        c[[2, 1] = lambda;

        Self {
            c,
            anisotropy_type: AnisotropyType::Isotropic,
        }
    }

    /// Create transversely isotropic tensor (fiber direction along z)
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
                field: "stiffness_components".to_string(),
                value: format!("c11={}, c33={}, c44={}", c11, c33, c44),
                constraint: "Diagonal components must be positive".to_string(),
            }));
        }

        // C66 = (C11 - C12) / 2 for transverse isotropy
        let c66 = (c11 - c12) / 2.0;

        let mut c = Array2::zeros((6, 6));

        // Fill symmetric matrix
        c[[0, 0] = c11;
        c[[1, 1] = c11;
        c[[2, 2] = c33;
        c[[3, 3] = c44;
        c[[4, 4] = c44;
        c[[5, 5] = c66;

        c[[0, 1] = c12;
        c[[1, 0] = c12;
        c[[0, 2] = c13;
        c[[2, 0] = c13;
        c[[1, 2] = c13;
        c[[2, 1] = c13;

        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::TransverselyIsotropic,
        })
    }

    /// Create orthotropic stiffness tensor
    pub fn orthotropic(components: [[f64; 6]; 6]) -> KwaversResult<Self> {
        let mut c = Array2::zeros((6, 6));

        // Copy components ensuring symmetry
        for i in 0..6 {
            for j in 0..6 {
                c[[i, j] = components[i][j];
            }
        }

        // Verify symmetry
        for i in 0..6 {
            for j in i + 1..6 {
                if (c[[i, j] - c[[j, i]).abs() > SYMMETRY_TOLERANCE {
                    return Err(KwaversError::Validation(ValidationError::FieldValidation {
                        field: "stiffness_matrix".to_string(),
                        value: format!(
                            "C[{},{}]={}, C[{},{}]={}",
                            i,
                            j,
                            c[[i, j],
                            j,
                            i,
                            c[[j, i]
                        ),
                        constraint: "Matrix must be symmetric".to_string(),
                    }));
                }
            }
        }

        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::Orthotropic,
        })
    }

    /// Check if tensor is positive definite
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

    /// Calculate determinant of 2D array (simplified for small matrices)
    fn determinant_2d(matrix: &Array2<f64>) -> f64 {
        let n = matrix.shape()[0];
        if n == 1 {
            matrix[[0, 0]
        } else if n == 2 {
            matrix[[0, 0] * matrix[[1, 1] - matrix[[0, 1] * matrix[[1, 0]
        } else {
            // LU decomposition for larger matrices
            // Simplified implementation - in production use nalgebra or similar
            let mut det = 1.0;
            let mut work = matrix.clone();

            for i in 0..n {
                let pivot = work[[i, i];
                if pivot.abs() < 1e-10 {
                    return 0.0;
                }
                det *= pivot;

                for j in i + 1..n {
                    let factor = work[[j, i] / pivot;
                    for k in i + 1..n {
                        work[[j, k] -= factor * work[[i, k];
                    }
                }
            }
            det
        }
    }

    /// Get compliance matrix (inverse of stiffness)
    pub fn compliance_matrix(&self) -> KwaversResult<Array2<f64>> {
        // In production, use proper matrix inversion
        // This is a placeholder
        Err(KwaversError::Validation(ValidationError::FieldValidation {
            field: "compliance".to_string(),
            value: "not_implemented".to_string(),
            constraint: "Matrix inversion not yet implemented".to_string(),
        }))
    }

    /// Apply rotation to stiffness tensor
    pub fn rotate(&self, rotation: &super::rotation::RotationMatrix) -> Self {
        let rotated = rotation.apply_to_stiffness(&self.c);
        Self {
            c: rotated,
            anisotropy_type: self.anisotropy_type,
        }
    }
}
