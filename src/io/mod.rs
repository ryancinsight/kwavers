/// Data I/O module for loading various file formats
/// 
/// This module provides loaders for medical imaging and simulation data formats
/// following SSOT and SOLID principles.

pub mod nifti;

pub use nifti::{NiftiLoader, NiftiHeader, BrainTissueLabel};