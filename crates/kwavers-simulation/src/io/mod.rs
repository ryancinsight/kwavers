//! Result/data output for simulation runs.
//!
//! Serialises recorded simulation outputs (CSV, pressure-field export, run
//! summaries). Relocated from the former `kwavers::infrastructure::io` facade
//! module into the simulation crate, where run output belongs (ADR 011).
//! Medical-image I/O (DICOM/NIfTI/…) is owned by ritk and bridged in
//! `kwavers_imaging::medical`.

mod output;
pub use output::{generate_summary, save_data_csv, save_light_data, save_pressure_data};
