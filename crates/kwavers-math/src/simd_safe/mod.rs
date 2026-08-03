//! Portable SIMD operations backed by `hermes_simd`
//!
//! ISA selection and feature dispatch are handled by `hermes_simd`.
//! The `auto_detect` submodule exposes the `SimdAuto` / `SimdCapability`
//! types for callers that need runtime-capability queries.

pub mod auto_detect;
pub mod operations;

pub use auto_detect::{SimdAuto, SimdCapability};
pub use operations::SimdOps;
