//! SIMD-accelerated operations for kwavers field kernels.
//!
//! ## Module layout
//!
//! - `config`  — `SimdConfig`, `MathSimdLevel` enum, runtime CPU-feature detection.
//!   Used by `kwavers-solver`'s FDTD dispatch layer.
//! - `fdtd_ops` — `FdtdSimdOps` — hermes-backed pressure/velocity update kernels.
//! - `interpolation_ops` — `InterpolationSimdOps` — trilinear interpolation helpers.
//! - `metrics` — `SimdPerformance`, `SimdMetrics` — speedup estimation.
//!
//! **FFT / complex-multiply** operations are owned by `apollo` (the Atlas SSOT for
//! spectral math).  `kwavers-math/fft` re-exports the Apollo API; there is no
//! separate FFT SIMD type in this module.

mod config;
mod fdtd_ops;
mod interpolation_ops;
mod metrics;

#[cfg(test)]
mod tests;

pub use config::{MathSimdLevel, SimdConfig};
pub use fdtd_ops::FdtdSimdOps;
pub use interpolation_ops::InterpolationSimdOps;
pub use metrics::{SimdMetrics, SimdPerformance};
