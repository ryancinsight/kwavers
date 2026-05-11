//! SIMD (Single Instruction, Multiple Data) Optimizations
//!
//! This module provides SIMD-accelerated implementations for performance-critical
//! mathematical operations in acoustic wave simulations.
//!
//! ## Supported Architectures
//!
//! - **x86_64**: AVX2, AVX-512 (when available)
//! - **ARM**: NEON
//! - **Portable SIMD**: Rust's std::simd (nightly feature)
//!
//! ## Performance Optimizations
//!
//! ### FDTD Updates
//! - Stencil operations for pressure and velocity updates
//! - Boundary condition applications
//! - Medium property interpolations
//!
//! ### FFT Operations
//! - Complex arithmetic in frequency domain
//! - Convolution operations
//! - Spectral filtering
//!
//! ### Linear Algebra
//! - Matrix-vector multiplications
//! - Vector field operations
//! - Interpolation kernels
//!
//! ## Safety and Portability
//!
//! - **Runtime Detection**: Automatic SIMD level detection
//! - **Fallback**: Scalar implementations when SIMD unavailable
//! - **Alignment**: Proper memory alignment for SIMD operations
//! - **Bounds Checking**: Safe SIMD operations with bounds validation
//!
//! ## Module layout
//!
//! - `config`: `SimdConfig`, `SimdLevel` enum, runtime CPU-feature detection.
//! - `fdtd_ops`: `FdtdSimdOps` — SIMD pressure/velocity update kernels.
//! - `fft_ops`: `FftSimdOps` — SIMD complex multiplication for spectral kernels.
//! - `interpolation_ops`: `InterpolationSimdOps` — SIMD trilinear interpolation.
//! - `metrics`: `SimdPerformance`, `SimdMetrics` — speedup estimation.

mod config;
mod fdtd_ops;
mod fft_ops;
mod interpolation_ops;
mod metrics;

#[cfg(test)]
mod tests;

pub use config::{SimdConfig, SimdLevel};
pub use fdtd_ops::FdtdSimdOps;
pub use fft_ops::FftSimdOps;
pub use interpolation_ops::InterpolationSimdOps;
pub use metrics::{SimdMetrics, SimdPerformance};
