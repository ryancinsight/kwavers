/// Data I/O module for loading and saving simulation data
///
/// This module provides loaders and savers for medical imaging and simulation data
/// following SSOT and SOLID principles.

// NOTE: the DICOM `ritk_io` adapter moved to
// `domain::imaging::medical::dicom_loader::dicom_ritk` (ADR 009): it uses domain
// types and is consumed only by the domain DICOM loader, so it belongs in the
// domain layer, not this infrastructure facade module. The former re-export
// aliases here were unused.

// Re-export output functions
mod output;
pub use output::{generate_summary, save_data_csv, save_light_data, save_pressure_data};
