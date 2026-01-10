//! Output configuration parameters

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Output configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParameters {
    /// Output directory
    pub directory: PathBuf,
    /// Save interval (timesteps)
    pub save_interval: usize,
    /// Output format
    pub format: OutputFormat,
    /// Fields to save
    pub fields: Vec<FieldType>,
    /// Enable compression
    pub compress: bool,
    /// Save snapshots
    pub snapshots: bool,
}

/// Output file formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutputFormat {
    /// HDF5 format
    HDF5,
    /// `NumPy` format
    NumPy,
    /// VTK format for visualization
    VTK,
    /// Binary format
    Binary,
    /// NIFTI medical imaging format
    #[cfg(feature = "nifti")]
    NIFTI,
}

/// Field types to output
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FieldType {
    /// Pressure field
    Pressure,
    /// Velocity field
    Velocity,
    /// Intensity field
    Intensity,
    /// Temperature field
    Temperature,
    /// Density field
    Density,
}

impl OutputParameters {
    /// Validate output parameters
    pub fn validate(&self) -> crate::domain::core::error::KwaversResult<()> {
        if self.save_interval == 0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "save_interval".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.fields.is_empty() {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "fields".to_string(),
                value: "empty".to_string(),
                constraint: "Must specify at least one field to output".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for OutputParameters {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("output"),
            save_interval: 100,
            format: OutputFormat::HDF5,
            fields: vec![FieldType::Pressure],
            compress: true,
            snapshots: false,
        }
    }
}
