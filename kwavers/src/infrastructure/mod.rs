//! Infrastructure layer — result/data output.
//!
//! Holds [`io`], which serialises simulation outputs (CSV, pressure-field export,
//! run summaries). Medical-image I/O (DICOM/NIfTI/…) is owned by ritk and bridged
//! in `domain::imaging::medical`; the former disabled REST-API, cloud-deployment,
//! and device-abstraction subtrees were removed as unused/incomplete.

pub mod io;
