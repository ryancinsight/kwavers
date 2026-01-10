//! Tissue region configuration and mapping

use crate::domain::core::error::{ConfigError, KwaversResult};
use crate::domain::medium::absorption::TissueType;

/// Configuration for setting tissue in a specific region
#[derive(Debug, Clone)]
pub struct TissueRegion {
    pub tissue_type: TissueType,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl TissueRegion {
    /// Create a new tissue region configuration
    #[must_use]
    pub fn new(
        tissue_type: TissueType,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> Self {
        Self {
            tissue_type,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }

    /// Validate the region bounds
    pub fn validate(&self) -> KwaversResult<()> {
        if self.x_min >= self.x_max {
            return Err(ConfigError::InvalidValue {
                parameter: "x bounds".to_string(),
                value: format!("[{}, {}]", self.x_min, self.x_max),
                constraint: "x_min must be less than x_max".to_string(),
            }
            .into());
        }
        if self.y_min >= self.y_max {
            return Err(ConfigError::InvalidValue {
                parameter: "y bounds".to_string(),
                value: format!("[{}, {}]", self.y_min, self.y_max),
                constraint: "y_min must be less than y_max".to_string(),
            }
            .into());
        }
        if self.z_min >= self.z_max {
            return Err(ConfigError::InvalidValue {
                parameter: "z bounds".to_string(),
                value: format!("[{}, {}]", self.z_min, self.z_max),
                constraint: "z_min must be less than z_max".to_string(),
            }
            .into());
        }
        Ok(())
    }

    /// Check if a point is within this region
    #[must_use]
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.x_min
            && x <= self.x_max
            && y >= self.y_min
            && y <= self.y_max
            && z >= self.z_min
            && z <= self.z_max
    }
}

/// 3D tissue type map
pub type TissueMap = ndarray::Array3<TissueType>;
