//! Simulation parameters configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Core simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    /// Simulation duration in seconds
    pub duration: f64,
    /// Time step (auto-calculated if None)
    pub dt: Option<f64>,
    /// CFL number for stability
    pub cfl: f64,
    /// Reference frequency in Hz
    pub frequency: f64,
    /// Enable nonlinear effects
    pub nonlinear: bool,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Simulation type
    pub simulation_type: SimulationType,
}

/// Types of simulations supported
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SimulationType {
    /// Forward wave propagation
    Forward,
    /// Time reversal reconstruction
    TimeReversal,
    /// Photoacoustic imaging
    Photoacoustic,
    /// Full waveform inversion
    FullWaveformInversion,
    /// Therapy planning
    Therapy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParameters {
    pub directory: PathBuf,
    pub save_interval: usize,
    pub format: OutputFormat,
    pub fields: Vec<FieldType>,
    pub compress: bool,
    pub snapshots: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutputFormat {
    HDF5,
    NumPy,
    VTK,
    Binary,
    #[cfg(feature = "nifti")]
    NIFTI,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FieldType {
    Pressure,
    Velocity,
    Intensity,
    Temperature,
    Density,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceParameters {
    pub num_threads: Option<usize>,
    pub use_gpu: bool,
    pub gpu_device: usize,
    pub cache_size: usize,
    pub chunk_size: usize,
    pub use_simd: bool,
    pub memory_pool: usize,
}

impl SimulationParameters {
    /// Validate simulation parameters
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.duration <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "duration".to_string(),
                value: self.duration.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if let Some(dt) = self.dt {
            if dt <= 0.0 {
                return Err(crate::core::error::ConfigError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "cfl".to_string(),
                value: self.cfl.to_string(),
                constraint: "Must be in (0, 1]".to_string(),
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.temperature < 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "temperature".to_string(),
                value: self.temperature.to_string(),
                constraint: "Must be non-negative (Kelvin)".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl OutputParameters {
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.save_interval == 0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "save_interval".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.fields.is_empty() {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "fields".to_string(),
                value: "empty".to_string(),
                constraint: "Must specify at least one field to output".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl PerformanceParameters {
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(crate::core::error::ConfigError::InvalidValue {
                    parameter: "num_threads".to_string(),
                    value: "0".to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if self.cache_size == 0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "cache_size".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.chunk_size == 0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "chunk_size".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            duration: 1e-3,
            dt: None,
            cfl: 0.3,
            frequency: 1e6,
            nonlinear: false,
            temperature: 293.15, // 20°C
            simulation_type: SimulationType::Forward,
        }
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

impl Default for PerformanceParameters {
    fn default() -> Self {
        Self {
            num_threads: None,
            use_gpu: false,
            gpu_device: 0,
            cache_size: 256,
            chunk_size: 1024,
            use_simd: true,
            memory_pool: 1024,
        }
    }
}
