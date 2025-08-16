/// Data I/O module for loading and saving simulation data
/// 
/// This module provides loaders and savers for medical imaging and simulation data formats
/// following SSOT and SOLID principles.

pub mod nifti;

pub use nifti::{NiftiReader, NiftiHeader, NiftiInfo, load_nifti, load_nifti_with_header};

// Re-export output functions for backward compatibility
mod output;
pub use output::{save_pressure_data, save_light_data, generate_summary};