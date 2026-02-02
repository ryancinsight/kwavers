/// Data I/O module for loading and saving simulation data
///
/// This module provides loaders and savers for medical imaging and simulation data
/// following SSOT and SOLID principles.
pub mod dicom;
pub mod nifti;

pub use dicom::{DicomObject, DicomReader, DicomSeries, DicomStudy, DicomValue};
pub use nifti::{NiftiHeader, NiftiInfo, NiftiReader};

// Re-export output functions
mod output;
pub use output::{generate_summary, save_light_data, save_pressure_data};
