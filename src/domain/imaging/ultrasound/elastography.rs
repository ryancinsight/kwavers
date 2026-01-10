//! Elastography domain definitions

use ndarray::Array3;

/// Inversion method for elasticity reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    /// Time-of-flight method (simple, fast)
    TimeOfFlight,
    /// Phase gradient method (more accurate)
    PhaseGradient,
    /// Direct inversion (most accurate, computationally expensive)
    DirectInversion,
    /// 3D volumetric time-of-flight (for 3D SWE)
    VolumetricTimeOfFlight,
    /// 3D phase gradient with directional analysis
    DirectionalPhaseGradient,
}

/// Nonlinear inversion method for advanced parameter estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearInversionMethod {
    /// Harmonic ratio method (B/A from A₂/A₁)
    HarmonicRatio,
    /// Iterative nonlinear least squares
    NonlinearLeastSquares,
    /// Bayesian inversion with uncertainty quantification
    BayesianInversion,
}

/// Elasticity map containing reconstructed tissue properties
#[derive(Debug, Clone)]
pub struct ElasticityMap {
    /// Young's modulus (Pa)
    pub youngs_modulus: Array3<f64>,
    /// Shear modulus (Pa) - related to Young's modulus
    pub shear_modulus: Array3<f64>,
    /// Shear wave speed (m/s)
    pub shear_wave_speed: Array3<f64>,
}

impl ElasticityMap {
    /// Create elasticity map from shear wave speed
    ///
    /// # Physics
    ///
    /// For incompressible isotropic tissue:
    /// - Shear modulus: μ = ρcs²
    /// - Young's modulus: E = 3μ = 3ρcs² (Poisson's ratio ≈ 0.5)
    pub fn from_shear_wave_speed(shear_wave_speed: Array3<f64>, density: f64) -> Self {
        let (nx, ny, nz) = shear_wave_speed.dim();
        let mut shear_modulus = Array3::zeros((nx, ny, nz));
        let mut youngs_modulus = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let cs = shear_wave_speed[[i, j, k]];
                    shear_modulus[[i, j, k]] = density * cs * cs;
                    youngs_modulus[[i, j, k]] = 3.0 * density * cs * cs;
                }
            }
        }

        Self {
            youngs_modulus,
            shear_modulus,
            shear_wave_speed,
        }
    }

    /// Get elasticity statistics (min, max, mean)
    pub fn statistics(&self) -> (f64, f64, f64) {
        let min = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.youngs_modulus.mean().unwrap_or(0.0);
        (min, max, mean)
    }
}

/// Nonlinear parameter map for advanced tissue characterization
#[derive(Debug, Clone)]
pub struct NonlinearParameterMap {
    /// Acoustic nonlinearity parameter B/A (dimensionless)
    pub nonlinearity_parameter: Array3<f64>,
    /// Higher-order elastic constants A, B, C, D (Pa)
    pub elastic_constants: Vec<Array3<f64>>,
    /// Uncertainty in nonlinearity parameter estimation
    pub nonlinearity_uncertainty: Array3<f64>,
    /// Signal quality metrics for nonlinear estimation
    pub estimation_quality: Array3<f64>,
}
