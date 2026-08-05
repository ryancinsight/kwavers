//! Portable SIMD operations backed by `hermes_simd`
//!
//! ISA selection and feature dispatch are handled by `hermes_simd`.
//! `SimdOps` is the single entry point for all SIMD field operations;
//! callers should not need to query the ISA level directly.

pub mod operations;

pub use operations::SimdOps;
