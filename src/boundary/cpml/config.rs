//! CPML configuration and validation

use crate::error::{ConfigError, KwaversResult};

/// Minimum cosine theta value to prevent division by zero in reflection estimation
const MIN_COS_THETA_FOR_REFLECTION: f64 = 0.1;

/// Configuration for Convolutional PML
#[derive(Debug, Clone)]
pub struct CPMLConfig {
    /// Number of PML cells in each direction
    pub thickness: usize,

    /// Polynomial order for profile grading (typically 3-4)
    pub polynomial_order: f64,

    /// Maximum conductivity scaling factor
    pub sigma_factor: f64,

    /// Maximum κ (coordinate stretching) value
    pub kappa_max: f64,

    /// Maximum α (frequency shifting) value
    pub alpha_max: f64,

    /// Target reflection coefficient (e.g., 1e-6)
    pub target_reflection: f64,

    /// Enable grazing angle absorption
    pub grazing_angle_absorption: bool,
}

impl Default for CPMLConfig {
    fn default() -> Self {
        Self {
            thickness: 20,
            polynomial_order: 3.0,
            sigma_factor: 0.75,
            kappa_max: 15.0,
            alpha_max: 0.24,
            target_reflection: 1e-6,
            grazing_angle_absorption: true,
        }
    }
}

impl CPMLConfig {
    /// Create CPML configuration with specified thickness
    #[must_use]
    pub fn with_thickness(thickness: usize) -> Self {
        Self {
            thickness,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "thickness".to_string(),
                value: self.thickness.to_string(),
                constraint: "CPML thickness must be positive".to_string(),
            }
            .into());
        }

        if self.polynomial_order < 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "polynomial_order".to_string(),
                value: self.polynomial_order.to_string(),
                constraint: "Polynomial order must be >= 1.0".to_string(),
            }
            .into());
        }

        if self.sigma_factor <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_factor".to_string(),
                value: self.sigma_factor.to_string(),
                constraint: "Sigma factor must be positive".to_string(),
            }
            .into());
        }

        if self.kappa_max < 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "kappa_max".to_string(),
                value: self.kappa_max.to_string(),
                constraint: "Kappa max must be >= 1.0".to_string(),
            }
            .into());
        }

        if self.alpha_max < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "alpha_max".to_string(),
                value: self.alpha_max.to_string(),
                constraint: "Alpha max must be non-negative".to_string(),
            }
            .into());
        }

        if self.target_reflection <= 0.0 || self.target_reflection >= 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "target_reflection".to_string(),
                value: self.target_reflection.to_string(),
                constraint: "Target reflection must be between 0 and 1".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Calculate theoretical reflection coefficient for a given angle
    #[must_use]
    pub fn theoretical_reflection(&self, cos_theta: f64) -> f64 {
        let cos_theta = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);
        let m = self.polynomial_order;
        let thickness = self.thickness as f64;

        // Theoretical reflection based on Collino & Tsogka (2001)
        self.target_reflection * ((m + 1.0) * thickness * cos_theta).exp()
    }
}
