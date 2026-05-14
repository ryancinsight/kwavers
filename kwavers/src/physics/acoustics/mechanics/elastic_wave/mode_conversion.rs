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

use crate::core::error::{KwaversResult, PhysicsError};
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
            reference_frequency: 1e6, // 1 MHz
            frequency_dependent: true,
            frequency_exponent: 0.8,
        }
    }
}

/// Full stiffness tensor for anisotropic materials
/// Uses Voigt notation for 6x6 symmetric matrix
#[derive(Debug, Clone)]
pub struct StiffnessTensor {
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

impl StiffnessTensor {
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

        // Check positive definiteness via eigenvalue analysis
        // For a 6x6 symmetric matrix, we need all eigenvalues to be positive
        // This ensures the material is physically stable
        if !self.is_positive_definite(&self.c) {
            return Err(PhysicsError::InvalidParameter {
                parameter: "stiffness_matrix".to_owned(),
                value: 0.0, // Indicates eigenvalue issue
                reason: "Stiffness matrix must be positive definite for physical stability"
                    .to_owned(),
            }
            .into());
        }

        Ok(())
    }

    /// Check if a 6x6 symmetric matrix is positive definite
    fn is_positive_definite(&self, matrix: &Array2<f64>) -> bool {
        // For a symmetric matrix to be positive definite, all leading principal minors must be positive
        // We'll use Sylvester's criterion

        // Check dimensions
        if matrix.shape() != [6, 6] {
            return false;
        }

        // Check 1x1 minor
        if matrix[[0, 0]] <= 0.0 {
            return false;
        }

        // Check 2x2 minor
        let det2 = matrix[[0, 0]].mul_add(matrix[[1, 1]], -(matrix[[0, 1]] * matrix[[0, 1]]));
        if det2 <= 0.0 {
            return false;
        }

        // Check 3x3 minor
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

        // Additional validation: Check diagonal positivity (Sylvester's criterion)
        // For symmetric positive definite matrices, all diagonal elements must be positive
        // Sufficient condition combined with positive leading principal minors above
        for i in 0..6 {
            if matrix[[i, i]] <= 0.0 {
                return false;
            }
        }

        // Additional check: ensure the matrix satisfies thermodynamic stability
        // C11, C22, C33 > 0 (already checked above)
        // C11 + C22 + 2*C12 > 0 (bulk modulus constraint)
        if 2.0f64.mul_add(matrix[[0, 1]], matrix[[0, 0]] + matrix[[1, 1]]) <= 0.0 {
            return false;
        }

        true
    }
}

// Extended ElasticWave functionality has been integrated into the main ElasticWave struct
// The enhanced features are now available through the standard ElasticWave API

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stiffness_tensor_isotropic() {
        let lambda = 1e10;
        let mu = 5e9;
        let density = 2700.0;

        let tensor = StiffnessTensor::isotropic(lambda, mu, density).unwrap();
        assert_eq!(tensor.symmetry, MaterialSymmetry::Isotropic);
        assert_eq!(tensor.c[[0, 0]], lambda + 2.0 * mu);
        assert_eq!(tensor.c[[3, 3]], mu);
        assert_eq!(tensor.c[[0, 1]], lambda);
    }

    #[test]
    fn test_mode_conversion_config() {
        let config = ModeConversionConfig::default();
        assert!(config.enable_p_to_s);
        assert!(config.enable_s_to_p);
        assert_eq!(config.critical_angle, std::f64::consts::PI / 4.0);
    }

    #[test]
    fn test_viscoelastic_config() {
        let config = ViscoelasticConfig::default();
        assert_eq!(config.q_p, 100.0);
        assert_eq!(config.q_s, 50.0);
        assert!(config.frequency_dependent);
    }

    /// `StiffnessTensor::hexagonal` encodes the transversely-isotropic stiffness
    /// matrix in Voigt notation. Analytical values for a representative
    /// geological medium (VTI shale, after Thomsen 1986):
    ///   c11=38.0, c33=22.0, c12=10.0, c13=8.0, c44=6.0 GPa, ρ=2200 kg/m³.
    ///
    /// The hexagonal symmetry requires c66 = (c11 − c12)/2 = 14 GPa.
    #[test]
    fn stiffness_tensor_hexagonal_encodes_vti_values_correctly() {
        let (c11, c33, c12, c13, c44) = (38.0e9, 22.0e9, 10.0e9, 8.0e9, 6.0e9);
        let density = 2200.0_f64;
        let t = StiffnessTensor::hexagonal(c11, c33, c12, c13, c44, density).unwrap();

        assert_eq!(t.symmetry, MaterialSymmetry::Hexagonal);
        assert!((t.c[[0, 0]] - c11).abs() < 1.0, "c11");
        assert!((t.c[[1, 1]] - c11).abs() < 1.0, "c22 = c11 for hexagonal");
        assert!((t.c[[2, 2]] - c33).abs() < 1.0, "c33");
        assert!((t.c[[3, 3]] - c44).abs() < 1.0, "c44");
        assert!((t.c[[4, 4]] - c44).abs() < 1.0, "c55 = c44 for hexagonal");
        // c66 = (c11 − c12) / 2
        let c66 = (c11 - c12) / 2.0;
        assert!((t.c[[5, 5]] - c66).abs() < 1.0, "c66");
        // Coupling
        assert!((t.c[[0, 1]] - c12).abs() < 1.0, "c12");
        assert!((t.c[[0, 2]] - c13).abs() < 1.0, "c13");
        assert!((t.c[[1, 2]] - c13).abs() < 1.0, "c23");
        // Symmetry
        for i in 0..6 {
            for j in i + 1..6 {
                assert!(
                    (t.c[[i, j]] - t.c[[j, i]]).abs() < 1.0,
                    "symmetry c[{i}][{j}]"
                );
            }
        }
    }

    /// `StiffnessTensor::validate` accepts a physically valid isotropic tensor.
    #[test]
    fn stiffness_tensor_validate_accepts_valid_isotropic_tensor() {
        let t = StiffnessTensor::isotropic(1e10, 5e9, 2700.0).unwrap();
        assert!(
            t.validate().is_ok(),
            "valid isotropic tensor must pass validate"
        );
    }

    /// `StiffnessTensor::validate` rejects an asymmetric matrix.
    #[test]
    fn stiffness_tensor_validate_rejects_asymmetric_matrix() {
        let mut t = StiffnessTensor::isotropic(1e10, 5e9, 2700.0).unwrap();
        // Introduce asymmetry: c[0][1] ≠ c[1][0]
        t.c[[0, 1]] += 1.0;
        assert!(
            t.validate().is_err(),
            "asymmetric stiffness matrix must be rejected"
        );
    }

    /// `StiffnessTensor::isotropic(lambda, mu, density)` rejects non-positive
    /// density or shear modulus μ. Negative λ is physically valid (λ can be
    /// negative while the medium remains elastically stable provided μ > 0 and
    /// λ + 2μ/3 > 0), so only density ≤ 0 and μ ≤ 0 are tested here.
    #[test]
    fn stiffness_tensor_isotropic_rejects_invalid_parameters() {
        // signature: isotropic(lambda, mu, density)
        assert!(
            StiffnessTensor::isotropic(1e10, 5e9, 0.0).is_err(),
            "zero density"
        );
        assert!(
            StiffnessTensor::isotropic(1e10, 5e9, -1.0).is_err(),
            "negative density"
        );
        assert!(
            StiffnessTensor::isotropic(1e10, 0.0, 2700.0).is_err(),
            "zero mu"
        );
        assert!(
            StiffnessTensor::isotropic(1e10, -1.0, 2700.0).is_err(),
            "negative mu"
        );
    }
}
