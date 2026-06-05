//! Safe, portable SIMD operations with architecture-conditional compilation
//!
//! This module provides SIMD acceleration with:
//! - Architecture-specific optimizations (`x86_64`, aarch64)
//! - SWAR (SIMD Within A Register) fallback for unsupported architectures
//! - Runtime auto-detection and dispatching
//! - Zero unsafe blocks in public API
//! - Compile-time feature detection

pub mod auto_detect;
pub mod avx2;
pub mod neon;
pub mod operations;
pub mod swar;

pub use auto_detect::{SimdAuto, SimdCapability};
pub use operations::SimdOps;
