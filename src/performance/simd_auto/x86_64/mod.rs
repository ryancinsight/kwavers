//! x86_64 architecture-specific SIMD implementations
//!
//! This module follows the Information Expert GRASP principle by owning
//! all x86_64-specific SIMD knowledge and implementations.

pub mod avx512;
pub mod avx2;
pub mod sse42;