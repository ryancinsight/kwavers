//! B-mode display pipeline: time-gain compensation → envelope detection →
//! log compression → scan conversion.
//!
//! These are the back-end stages that turn beamformed RF lines into a grayscale
//! image. Beamforming itself (DAS, MV, plane-wave compounding, …) lives in
//! [`crate::signal_processing::beamforming`]; this module consumes its output.
//!
//! # Pipeline
//! 1. [`tgc::TgcConfig`] — restore depth-uniform brightness.
//! 2. [`detection::envelope`] — Hilbert envelope (carrier removal).
//! 3. [`detection::log_compress`] — fit the echo dynamic range to `[0, 1]`.
//! 4. [`scan_conversion::ScanConverter`] — polar beams → Cartesian image.

pub mod detection;
pub mod scan_conversion;
pub mod tgc;

#[cfg(test)]
mod tests;

pub use detection::{envelope, log_compress};
pub use scan_conversion::{CartesianGrid, ScanConverter, ScanGeometry};
pub use tgc::TgcConfig;
