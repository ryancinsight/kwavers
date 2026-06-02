//! Real-Time SIRT Reconstruction Pipeline for Clinical Imaging.
//!
//! SRP split:
//! - `config`   — `RealTimeSirtConfig` and builder methods
//! - `types`    — `ReconstructionFrame` and `FrameQuality`
//! - `pipeline` — `RealTimeSirtPipeline` and all iterative logic

mod config;
mod pipeline;
#[cfg(test)]
mod tests;
mod types;

pub use config::RealTimeSirtConfig;
pub use pipeline::RealTimeSirtPipeline;
pub use types::{FrameQuality, ReconstructionFrame};
// Re-export for consumers that import AcousticProjectionGeometry via this path.
pub use super::acoustic_projection::AcousticProjectionGeometry;
