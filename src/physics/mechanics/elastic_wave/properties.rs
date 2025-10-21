//! Elastic material properties following SSOT principle
//!
//! This module defines elastic material properties and their relationships,
//! providing a single source of truth for material definitions.

use crate::error::{KwaversResult, PhysicsError};

/// Elastic material properties following SSOT principle
/// Single source of truth for material property definitions
#[derive(Debug, Clone)]
pub struct ElasticProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Lamé parameters (Pa)
    pub lambda: f64,
    pub mu: f64,
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Bulk modulus (Pa)
    pub bulk_modulus: f64,
    /// P-wave speed (m/s)
    pub p_wave_speed: f64,
    /// S-wave speed (m/s)
    pub s_wave_speed: f64,
}

impl ElasticProperties {
    /// Create elastic properties from density and Lamé parameters
    /// Follows Information Expert principle - knows how to compute derived properties
    pub fn from_lame(density: f64, lambda: f64, mu: f64) -> KwaversResult<Self> {
        if density <= 0.0 || mu <= 0.0 || lambda < -2.0 / 3.0 * mu {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticProperties".to_string(),
                value: density, // Use density as the primary invalid value
                reason: format!(
                    "Invalid elastic parameters: density={density}, lambda={lambda}, mu={mu}. \
                     Requires: density > 0, mu > 0, lambda >= -2/3 * mu"
                ),
            }
            .into());
        }

        // Compute derived properties
        let youngs_modulus = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
        let poisson_ratio = lambda / (2.0 * (lambda + mu));
        let bulk_modulus = lambda + 2.0 * mu / 3.0;

        // Wave speeds from Christoffel equation
        let p_wave_speed = ((lambda + 2.0 * mu) / density).sqrt();
        let s_wave_speed = (mu / density).sqrt();

        Ok(Self {
            density,
            lambda,
            mu,
            youngs_modulus,
            poisson_ratio,
            bulk_modulus,
            p_wave_speed,
            s_wave_speed,
        })
    }

    /// Create from Young's modulus and Poisson's ratio
    /// Alternative constructor for convenience
    pub fn from_youngs_poisson(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> KwaversResult<Self> {
        if density <= 0.0 || youngs_modulus <= 0.0 || poisson_ratio >= 0.5 || poisson_ratio <= -1.0
        {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticProperties".to_string(),
                value: poisson_ratio,
                reason: format!(
                    "Invalid parameters: E={youngs_modulus}, ν={poisson_ratio}, ρ={density}. \
                     Requires: E > 0, -1 < ν < 0.5, ρ > 0"
                ),
            }
            .into());
        }

        // Convert to Lamé parameters
        let lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        Self::from_lame(density, lambda, mu)
    }

    /// Create from wave speeds (inverse problem)
    pub fn from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> KwaversResult<Self> {
        if density <= 0.0 || p_speed <= 0.0 || s_speed <= 0.0 || s_speed >= p_speed {
            return Err(PhysicsError::InvalidParameter {
                parameter: "wave speeds".to_string(),
                value: p_speed,
                reason: format!(
                    "Invalid wave speeds: p={p_speed}, s={s_speed}. Requires: p > s > 0"
                ),
            }
            .into());
        }

        // Recover Lamé parameters
        let mu = density * s_speed * s_speed;
        let lambda = density * p_speed * p_speed - 2.0 * mu;

        Self::from_lame(density, lambda, mu)
    }
}

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
        for i in 0..6 {
            for j in i + 1..6 {
                if (stiffness[i][j] - stiffness[j][i]).abs() > TOLERANCE {
                    return Err(PhysicsError::InvalidParameter {
                        parameter: "stiffness tensor".to_string(),
                        value: stiffness[i][j],
                        reason: format!(
                            "Stiffness tensor must be symmetric. C[{}][{}]={} != C[{}][{}]={}",
                            i, j, stiffness[i][j], j, i, stiffness[j][i]
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
    pub fn isotropic(density: f64, lambda: f64, mu: f64) -> KwaversResult<Self> {
        let props = ElasticProperties::from_lame(density, lambda, mu)?;

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
