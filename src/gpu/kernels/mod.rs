//! GPU Compute Kernels Module
//!
//! This module provides high-performance GPU kernels for acoustic wave propagation,
//! thermal diffusion, and FFT operations.

mod acoustic;
mod boundary;
mod config;
mod generator;
mod manager;
mod thermal;
mod transforms;
mod types;

// Re-export key types
pub use acoustic::AcousticKernel;
pub use boundary::BoundaryKernel;
pub use config::{KernelConfig, OptimizationLevel};
pub use generator::{CudaKernelGenerator, OpenCLKernelGenerator, WebGPUKernelGenerator};
pub use manager::KernelManager;
pub use thermal::ThermalKernel;
pub use transforms::{FFTKernel, TransformDirection};
pub use types::{CompiledKernel, KernelType};

#[cfg(test)]
mod tests;
