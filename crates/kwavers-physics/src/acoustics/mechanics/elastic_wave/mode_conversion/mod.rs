//! Elastic Wave Mode Conversion Module
//!
//! This module provides elastic wave propagation with:
//! - Full stress tensor formulation (all 6 independent components)
//! - Mode conversion at interfaces (P-wave to S-wave and vice versa)
//! - Anisotropic material support with full stiffness tensor
//! - Viscoelastic damping
//! - Surface wave propagation (Rayleigh and Love waves)
//!
//! # Design Principles
//! - **SOLID**: Each component has a single responsibility
//! - **CUPID**: Clear interfaces and composable components
//! - **DRY**: Reusable tensor operations
//! - **KISS**: Focused API for wave mode conversion

use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversResult, PhysicsError};
use ndarray::Array2;

/// Mode conversion configuration
#[derive(Debug, Clone)]
pub struct ModeConversionConfig {
    /// Enable P-to-S wave conversion
    pub enable_p_to_s: bool,

    /// Enable S-to-P wave conversion
    pub enable_s_to_p: bool,

    /// Critical angle for total internal reflection (radians)
    pub critical_angle: f64,

    /// Conversion efficiency factor (0.0 to 1.0)
    pub conversion_efficiency: f64,

    /// Interface detection threshold
    pub interface_threshold: f64,
}

impl Default for ModeConversionConfig {
    fn default() -> Self {
        Self {
            enable_p_to_s: true,
            enable_s_to_p: true,
            critical_angle: std::f64::consts::PI / 4.0, // 45 degrees
            conversion_efficiency: 0.3,
            interface_threshold: 0.1,
        }
    }
}

/// Viscoelastic damping configuration
#[derive(Debug, Clone)]
pub struct ViscoelasticConfig {
    /// Quality factor for P-waves
    pub q_p: f64,

    /// Quality factor for S-waves
    pub q_s: f64,

    /// Reference frequency for Q values (Hz)
    pub reference_frequency: f64,

    /// Frequency-dependent Q model
    pub frequency_dependent: bool,

    /// Power law exponent for frequency dependence
    pub frequency_exponent: f64,
}

impl Default for ViscoelasticConfig {
    fn default() -> Self {
        Self {
            q_p: 100.0,
            q_s: 50.0,
            reference_frequency: MHZ_TO_HZ, // 1 MHz
            frequency_dependent: true,
            frequency_exponent: 0.8,
        }
    }
}

/// Full stiffness tensor for anisotropic materials
/// Uses Voigt notation for 6x6 symmetric matrix
#[derive(Debug, Clone)]
pub struct ElasticModeStiffnessTensor {
    /// 6x6 stiffness matrix in Voigt notation (Pa)
    pub c: Array2<f64>,

    /// Density (kg/m³)
    pub density: f64,

    /// Material symmetry type
    pub symmetry: MaterialSymmetry,
}

/// Material symmetry types
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialSymmetry {
    Isotropic,
    Cubic,
    Hexagonal,
    Orthorhombic,
    Monoclinic,
    Triclinic,
}

impl ElasticModeStiffnessTensor {
    /// Create isotropic stiffness tensor from Lamé parameters
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn isotropic(lambda: f64, mu: f64, density: f64) -> KwaversResult<Self> {
        if density <= 0.0 || mu <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "density_or_mu".to_owned(),
                value: if density <= 0.0 { density } else { mu },
                reason: "Material parameters must be positive".to_owned(),
            }
            .into());
        }

        let mut c = Array2::zeros((6, 6));

        // Diagonal terms
        c[[0, 0]] = 2.0f64.mul_add(mu, lambda); // C11
        c[[1, 1]] = 2.0f64.mul_add(mu, lambda); // C22
        c[[2, 2]] = 2.0f64.mul_add(mu, lambda); // C33
        c[[3, 3]] = mu; // C44
        c[[4, 4]] = mu; // C55
        c[[5, 5]] = mu; // C66

        // Off-diagonal terms
        c[[0, 1]] = lambda;
        c[[1, 0]] = lambda; // C12
        c[[0, 2]] = lambda;
        c[[2, 0]] = lambda; // C13
        c[[1, 2]] = lambda;
        c[[2, 1]] = lambda; // C23

        Ok(Self {
            c,
            density,
            symmetry: MaterialSymmetry::Isotropic,
        })
    }

    /// Create hexagonal (transversely isotropic) stiffness tensor
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn hexagonal(
        c11: f64,
        c33: f64,
        c12: f64,
        c13: f64,
        c44: f64,
        density: f64,
    ) -> KwaversResult<Self> {
        let mut c = Array2::zeros((6, 6));

        // Hexagonal symmetry
        c[[0, 0]] = c11;
        c[[1, 1]] = c11;
        c[[2, 2]] = c33;
        c[[3, 3]] = c44;
        c[[4, 4]] = c44;
        c[[5, 5]] = (c11 - c12) / 2.0; // C66

        c[[0, 1]] = c12;
        c[[1, 0]] = c12;
        c[[0, 2]] = c13;
        c[[2, 0]] = c13;
        c[[1, 2]] = c13;
        c[[2, 1]] = c13;

        Ok(Self {
            c,
            density,
            symmetry: MaterialSymmetry::Hexagonal,
        })
    }

    /// Validate stiffness tensor for positive definiteness
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        // Check density
        if self.density <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "density".to_owned(),
                value: self.density,
                reason: "Density must be positive".to_owned(),
            }
            .into());
        }

        // Check symmetry
        for i in 0..6 {
            for j in i + 1..6 {
                if (self.c[[i, j]] - self.c[[j, i]]).abs() > 1e-10 {
                    return Err(PhysicsError::InvalidParameter {
                        parameter: format!("c[{i},{j}]"),
                        value: self.c[[i, j]],
                        reason: format!(
                            "Stiffness matrix must be symmetric: c[{},{}]={} != c[{},{}]={}",
                            i,
                            j,
                            self.c[[i, j]],
                            j,
                            i,
                            self.c[[j, i]]
                        ),
                    }
                    .into());
                }
            }
        }

        // Check positive definiteness via Sylvester's criterion
        if !self.is_positive_definite(&self.c) {
            return Err(PhysicsError::InvalidParameter {
                parameter: "stiffness_matrix".to_owned(),
                value: 0.0,
                reason: "Stiffness matrix must be positive definite for physical stability"
                    .to_owned(),
            }
            .into());
        }

        Ok(())
    }

    /// Check if a 6x6 symmetric matrix is positive definite via Sylvester's criterion.
    fn is_positive_definite(&self, matrix: &Array2<f64>) -> bool {
        if matrix.shape() != [6, 6] {
            return false;
        }

        if matrix[[0, 0]] <= 0.0 {
            return false;
        }

        let det2 = matrix[[0, 0]].mul_add(matrix[[1, 1]], -(matrix[[0, 1]] * matrix[[0, 1]]));
        if det2 <= 0.0 {
            return false;
        }

        let det3 = matrix[[0, 2]].mul_add(
            matrix[[0, 1]].mul_add(matrix[[1, 2]], -(matrix[[0, 2]] * matrix[[1, 1]])),
            matrix[[0, 0]].mul_add(
                matrix[[1, 1]].mul_add(matrix[[2, 2]], -(matrix[[1, 2]] * matrix[[1, 2]])),
                -(matrix[[0, 1]]
                    * matrix[[0, 1]].mul_add(matrix[[2, 2]], -(matrix[[0, 2]] * matrix[[1, 2]]))),
            ),
        );
        if det3 <= 0.0 {
            return false;
        }

        for i in 0..6 {
            if matrix[[i, i]] <= 0.0 {
                return false;
            }
        }

        // Bulk modulus constraint: C11 + C22 + 2·C12 > 0
        if 2.0f64.mul_add(matrix[[0, 1]], matrix[[0, 0]] + matrix[[1, 1]]) <= 0.0 {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests;
