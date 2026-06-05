//! Super-Resolution Reconstruction for ULM.
//!
//! Accumulates microbubble localizations into a high-resolution density image,
//! achieving resolution ~5 μm — ~20× finer than the diffraction-limited acquisition.

pub mod reconstructor;
#[cfg(test)]
mod tests;
pub mod types;

pub use reconstructor::SuperResReconstructor;
pub use types::{RenderMode, SuperResConfig};
