//! CPML configuration and validation

use crate::core::error::{ConfigError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Minimum cosine theta value to prevent division by zero in reflection estimation
const MIN_COS_THETA_FOR_REFLECTION: f64 = 0.1;

/// Per-dimension PML configuration for k-Wave parity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerDimensionPML {
    /// PML thickness in x-direction [grid cells]
    pub x: usize,
    /// PML thickness in y-direction [grid cells]
    pub y: usize,
    /// PML thickness in z-direction [grid cells]
    pub z: usize,
}

impl PerDimensionPML {
    /// Create uniform PML thickness for all dimensions
    #[must_use]
    pub fn uniform(thickness: usize) -> Self {
        Self {
            x: thickness,
            y: thickness,
            z: thickness,
        }
    }

    /// Create PML with different thickness per dimension
    #[must_use]
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    /// Get thickness for specific dimension (0=x, 1=y, 2=z)
    #[must_use]
    pub fn get(&self, dim: usize) -> usize {
        match dim {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Invalid dimension {} (must be 0, 1, or 2)", dim),
        }
    }

    /// Check if all dimensions have the same thickness
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.x == self.y && self.y == self.z
    }

    /// Get maximum thickness across all dimensions
    #[must_use]
    pub fn max_thickness(&self) -> usize {
        self.x.max(self.y).max(self.z)
    }
}

impl Default for PerDimensionPML {
    fn default() -> Self {
        Self::uniform(20)
    }
}

/// Per-dimension absorption (sigma_factor / pml_alpha) for asymmetric PML tuning.
///
/// # Theorem (Per-Dimension Sigma)
/// K-Wave's `pml_alpha` can be specified as a scalar (uniform) or a 3-vector
/// [alpha_x, alpha_y, alpha_z] to independently control how aggressively each
/// axis is absorbed. Setting different alpha per axis is useful for:
///   - Non-cubic domains where CFL constraints differ per axis.
///   - Reducing PML memory overhead on thin dimensions.
///
/// Ref: Treeby & Cox (2010), J. Biomed. Opt. 15(2) 021314.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerDimensionAlpha {
    /// Sigma factor for x-direction PML (k-Wave `pml_alpha_x`)
    pub x: f64,
    /// Sigma factor for y-direction PML (k-Wave `pml_alpha_y`)
    pub y: f64,
    /// Sigma factor for z-direction PML (k-Wave `pml_alpha_z`)
    pub z: f64,
}

impl PerDimensionAlpha {
    /// Uniform alpha across all dimensions (k-Wave default: 2.0)
    #[must_use]
    pub fn uniform(alpha: f64) -> Self {
        Self {
            x: alpha,
            y: alpha,
            z: alpha,
        }
    }

    /// Per-dimension alpha
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Get alpha for a specific dimension (0=x, 1=y, 2=z)
    #[must_use]
    pub fn get(&self, dim: usize) -> f64 {
        match dim {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Invalid dimension {} (must be 0, 1, or 2)", dim),
        }
    }
}

impl Default for PerDimensionAlpha {
    fn default() -> Self {
        Self::uniform(2.0) // k-Wave default pml_alpha
    }
}

/// Configuration for Convolutional PML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPMLConfig {
    /// Number of PML cells in each direction (legacy, use per_dimension instead)
    pub thickness: usize,

    /// Per-dimension PML thickness for k-Wave parity
    pub per_dimension: PerDimensionPML,

    /// Polynomial order for profile grading (typically 3-4)
    pub polynomial_order: f64,

    /// Maximum conductivity scaling factor (uniform, overridden by per_dimension_alpha)
    pub sigma_factor: f64,

    /// Per-dimension sigma scaling factors (k-Wave `pml_alpha` vector)
    /// When set, overrides `sigma_factor` per axis.
    pub per_dimension_alpha: PerDimensionAlpha,

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
        let thickness = 20;
        Self {
            thickness,
            per_dimension: PerDimensionPML::uniform(thickness),
            polynomial_order: 3.0,
            sigma_factor: 2.0,
            per_dimension_alpha: PerDimensionAlpha::default(),
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
            per_dimension: PerDimensionPML::uniform(thickness),
            ..Default::default()
        }
    }

    /// Create CPML configuration with per-dimension thickness
    #[must_use]
    pub fn with_per_dimension_thickness(x: usize, y: usize, z: usize) -> Self {
        let max_thickness = x.max(y).max(z);
        Self {
            thickness: max_thickness,
            per_dimension: PerDimensionPML::new(x, y, z),
            ..Default::default()
        }
    }

    /// Set per-dimension PML thickness
    #[must_use]
    pub fn with_pml_size(self, x: usize, y: usize, z: usize) -> Self {
        let max_thickness = x.max(y).max(z);
        Self {
            thickness: max_thickness,
            per_dimension: PerDimensionPML::new(x, y, z),
            ..self
        }
    }

    /// Set uniform sigma absorption factor (equivalent to k-Wave `pml_alpha` scalar).
    #[must_use]
    pub fn with_alpha(self, alpha: f64) -> Self {
        Self {
            sigma_factor: alpha,
            per_dimension_alpha: PerDimensionAlpha::uniform(alpha),
            ..self
        }
    }

    /// Set per-dimension sigma absorption factors (equivalent to k-Wave `pml_alpha` vector).
    #[must_use]
    pub fn with_alpha_xyz(self, ax: f64, ay: f64, az: f64) -> Self {
        // sigma_factor becomes the maximum for backwards-compatible code paths
        let max_alpha = ax.max(ay).max(az);
        Self {
            sigma_factor: max_alpha,
            per_dimension_alpha: PerDimensionAlpha::new(ax, ay, az),
            ..self
        }
    }

    /// Get sigma factor for a specific dimension (respects per_dimension_alpha).
    #[must_use]
    pub fn sigma_factor_for_dimension(&self, dim: usize) -> f64 {
        self.per_dimension_alpha.get(dim)
    }

    /// Get PML thickness for a specific dimension (0=x, 1=y, 2=z)
    #[must_use]
    pub fn thickness_for_dimension(&self, dim: usize) -> usize {
        self.per_dimension.get(dim)
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

        if self.per_dimension.x == 0 || self.per_dimension.y == 0 || self.per_dimension.z == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "per_dimension".to_string(),
                value: format!(
                    "({}, {}, {})",
                    self.per_dimension.x, self.per_dimension.y, self.per_dimension.z
                ),
                constraint: "All per-dimension PML thicknesses must be positive".to_string(),
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
    /// Uses maximum thickness across all dimensions
    #[must_use]
    pub fn theoretical_reflection(&self, cos_theta: f64, dx: f64, sound_speed: f64) -> f64 {
        let cos_theta = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);
        let m = self.polynomial_order;
        let thickness = self.per_dimension.max_thickness() as f64;

        // Corrected formula based on Collino & Tsogka (2001) with proper absorption
        // σ_max = (σ_factor·(m+1)·c) / (150π·dx) - includes sound speed
        let sigma_max =
            self.sigma_factor * (m + 1.0) * sound_speed / (150.0 * std::f64::consts::PI * dx);
        self.target_reflection * (-(m + 1.0) * sigma_max * thickness * cos_theta).exp()
    }

    /// Calculate theoretical reflection for a specific dimension
    #[must_use]
    pub fn theoretical_reflection_for_dimension(
        &self,
        dim: usize,
        cos_theta: f64,
        dx: f64,
        sound_speed: f64,
    ) -> f64 {
        let cos_theta = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);
        let m = self.polynomial_order;
        let thickness = self.per_dimension.get(dim) as f64;

        let sigma_max =
            self.sigma_factor * (m + 1.0) * sound_speed / (150.0 * std::f64::consts::PI * dx);
        self.target_reflection * (-(m + 1.0) * sigma_max * thickness * cos_theta).exp()
    }
}
