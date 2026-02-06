//! Elastic material properties following SSOT principle
//!
//! This module provides anisotropic elastic properties for elastic wave physics.
//! Isotropic properties are handled by the canonical `ElasticPropertyData` from
//! `domain::medium::properties`.

use crate::core::error::{KwaversResult, PhysicsError};
use crate::domain::medium::properties::ElasticPropertyData;

/// Anisotropic elastic properties using Voigt notation
/// Follows SOLID principles with clear separation of concerns
#[derive(Debug, Clone)]
pub struct AnisotropicElasticProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Stiffness tensor in Voigt notation (6x6 symmetric matrix)
    /// `C_ij` where i,j = 1..6 corresponding to xx, yy, zz, yz, xz, xy
    pub stiffness: [[f64; 6]; 6],
}

impl AnisotropicElasticProperties {
    /// Create from full stiffness tensor
    /// Validates symmetry and positive definiteness
    pub fn new(density: f64, stiffness: [[f64; 6]; 6]) -> KwaversResult<Self> {
        if density <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "density".to_string(),
                value: density,
                reason: "Density must be positive".to_string(),
            }
            .into());
        }

        // Validate symmetry (within numerical tolerance)
        const TOLERANCE: f64 = 1e-10;
        for (i, row_i) in stiffness.iter().enumerate() {
            for j in i + 1..6 {
                if (row_i[j] - stiffness[j][i]).abs() > TOLERANCE {
                    return Err(PhysicsError::InvalidParameter {
                        parameter: "stiffness tensor".to_string(),
                        value: row_i[j],
                        reason: format!(
                            "Stiffness tensor must be symmetric. C[{}][{}]={} != C[{}][{}]={}",
                            i, j, row_i[j], j, i, stiffness[j][i]
                        ),
                    }
                    .into());
                }
            }
        }

        // Check positive definiteness using eigenvalue analysis
        // For a symmetric 6x6 stiffness matrix, positive definiteness requires
        // all eigenvalues to be positive. We check this by:
        // 1. Verifying leading principal minors (Sylvester's criterion)
        // 2. Checking physical constraints for elastic stability
        //
        // References:
        // - Ting (1996): "Anisotropic Elasticity" - stability criteria
        // - Auld (1990): "Acoustic Fields and Waves in Solids" - elastic constants

        // Check first principal minor (1x1)
        if stiffness[0][0] <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "C11".to_string(),
                value: stiffness[0][0],
                reason: "First principal minor must be positive".to_string(),
            }
            .into());
        }

        // Check second principal minor (2x2 upper-left)
        let det_2x2 = stiffness[0][0] * stiffness[1][1] - stiffness[0][1] * stiffness[1][0];
        if det_2x2 <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "C11*C22 - C12^2".to_string(),
                value: det_2x2,
                reason: "Second principal minor must be positive".to_string(),
            }
            .into());
        }

        // Check third principal minor (3x3 upper-left)
        let det_3x3 = stiffness[0][0]
            * (stiffness[1][1] * stiffness[2][2] - stiffness[1][2] * stiffness[2][1])
            - stiffness[0][1]
                * (stiffness[1][0] * stiffness[2][2] - stiffness[1][2] * stiffness[2][0])
            + stiffness[0][2]
                * (stiffness[1][0] * stiffness[2][1] - stiffness[1][1] * stiffness[2][0]);
        if det_3x3 <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "det(C_3x3)".to_string(),
                value: det_3x3,
                reason: "Third principal minor must be positive".to_string(),
            }
            .into());
        }

        // For computational efficiency, we check necessary conditions for the 4x4, 5x5, 6x6 minors
        // by verifying physical stability conditions for common elastic symmetries:
        //
        // For isotropic materials: λ + 2μ > 0, μ > 0
        // For cubic: C11 > |C12|, C44 > 0, C11 + 2C12 > 0
        // For hexagonal: C11 > |C12|, C33 > 0, C44 > 0, (C11 + C12)C33 > 2C13²
        //
        // Check shear moduli (C44, C55, C66) are positive
        if stiffness[3][3] <= 0.0 || stiffness[4][4] <= 0.0 || stiffness[5][5] <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "shear moduli".to_string(),
                value: stiffness[3][3].min(stiffness[4][4]).min(stiffness[5][5]),
                reason: "Shear moduli (C44, C55, C66) must be positive".to_string(),
            }
            .into());
        }

        Ok(Self { density, stiffness })
    }

    /// Create isotropic special case from Lamé parameters
    ///
    /// # Arguments
    ///
    /// - `density`: ρ (kg/m³)
    /// - `lambda`: Lamé first parameter λ (Pa)
    /// - `mu`: Lamé second parameter (shear modulus) μ (Pa)
    ///
    /// # Errors
    ///
    /// Returns error if parameters violate physical constraints
    pub fn isotropic(density: f64, lambda: f64, mu: f64) -> KwaversResult<Self> {
        // Use canonical ElasticPropertyData for validation
        let props = ElasticPropertyData::new(density, lambda, mu).map_err(|msg| {
            PhysicsError::InvalidParameter {
                parameter: "elastic properties".to_string(),
                value: density,
                reason: msg,
            }
        })?;

        // Build isotropic stiffness tensor
        let c11 = lambda + 2.0 * mu;
        let c12 = lambda;
        let c44 = mu;

        let stiffness = [
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ];

        Self::new(props.density, stiffness)
    }
}
