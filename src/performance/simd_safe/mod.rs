//! Safe, portable SIMD operations with architecture-conditional compilation
//!
//! This module provides SIMD acceleration with:
//! - Architecture-specific optimizations (`x86_64`, aarch64)
//! - SWAR (SIMD Within A Register) fallback for unsupported architectures
//! - Zero unsafe blocks in public API
//! - Compile-time feature detection

pub mod avx2;
pub mod neon;
pub mod operations;
pub mod swar;

pub use operations::SimdOps;