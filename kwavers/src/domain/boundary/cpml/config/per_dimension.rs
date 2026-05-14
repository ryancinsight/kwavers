//! Per-dimension PML thickness and absorption factor types.

use crate::core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Per-dimension PML configuration for k-Wave parity.
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
    /// Uniform PML thickness for all dimensions.
    #[must_use]
    pub fn uniform(thickness: usize) -> Self {
        Self {
            x: thickness,
            y: thickness,
            z: thickness,
        }
    }

    /// Per-dimension PML thickness.
    #[must_use]
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    /// Thickness for a specific dimension (0=x, 1=y, 2=z).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if `dim > 2`.
    ///
    pub fn get(&self, dim: usize) -> KwaversResult<usize> {
        match dim {
            0 => Ok(self.x),
            1 => Ok(self.y),
            2 => Ok(self.z),
            _ => Err(KwaversError::InvalidInput(format!(
                "CPML dimension {} out of range [0, 2]",
                dim
            ))),
        }
    }

    /// True when all three thicknesses are equal.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.x == self.y && self.y == self.z
    }

    /// Maximum thickness across all dimensions.
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
    /// Uniform alpha across all dimensions (k-Wave default: 2.0).
    #[must_use]
    pub fn uniform(alpha: f64) -> Self {
        Self {
            x: alpha,
            y: alpha,
            z: alpha,
        }
    }

    /// Per-dimension alpha.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Alpha for a specific dimension (0=x, 1=y, 2=z).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if `dim > 2`.
    ///
    pub fn get(&self, dim: usize) -> KwaversResult<f64> {
        match dim {
            0 => Ok(self.x),
            1 => Ok(self.y),
            2 => Ok(self.z),
            _ => Err(KwaversError::InvalidInput(format!(
                "CPML dimension {} out of range [0, 2]",
                dim
            ))),
        }
    }
}

impl Default for PerDimensionAlpha {
    fn default() -> Self {
        Self::uniform(2.0) // k-Wave default pml_alpha
    }
}
