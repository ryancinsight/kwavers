/// Data I/O module for loading various file formats
/// 
/// This module provides loaders for medical imaging and simulation data formats
/// following SSOT and SOLID principles.

pub mod nifti;

pub use nifti::{NiftiReader, NiftiHeader, NiftiInfo, load_nifti, load_nifti_with_header};