/// Data I/O module for loading and saving simulation data
///
/// This module provides loaders and savers for medical imaging and simulation data
/// following SSOT and SOLID principles.
pub mod dicom_ritk;
pub mod nifti;

pub use dicom_ritk::{
    load_series as load_dicom_series_ritk, load_series_from_dir as load_dicom_dir_ritk,
    load_series_with_uid as load_dicom_uid_ritk, select_unique_series as select_dicom_series_ritk,
    DicomSeriesVolume,
};
pub use nifti::{NiftiHeader, NiftiInfo, NiftiReader};

// Re-export output functions
mod output;
pub use output::{generate_summary, save_data_csv, save_light_data, save_pressure_data};
