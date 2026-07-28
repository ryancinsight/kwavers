//! Safe, portable SIMD field operations backed by `hermes-simd` runtime dispatch.
//!
//! All unsafe intrinsics are encapsulated inside `hermes_simd_intrinsics`.
//! This module exposes only safe, architecture-agnostic wrappers.
//!
//! # Module layout
//!
//! - [`auto_detect`] — CPU-capability detection helpers (`SimdAuto`, `SimdCapability`)
//! - [`operations`] — `SimdOps` — field-level convenience operations

pub mod auto_detect;
pub mod operations;

pub use auto_detect::{SimdAuto, SimdCapability};
pub use operations::SimdOps;
