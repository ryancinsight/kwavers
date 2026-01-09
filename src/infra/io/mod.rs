/// Data I/O module for loading and saving simulation data
///
/// This module provides loaders and savers for medical imaging and simulation data formats
/// following SSOT and SOLID principles.
pub mod nifti;

pub use nifti::{NiftiHeader, NiftiInfo, NiftiReader};

// Re-export output functions
pub mod config;
pub use config::OutputParameters;
mod output;
pub use output::{generate_summary, save_light_data, save_pressure_data};
