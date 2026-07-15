//! GPU-accelerated Thermal-Acoustic Coupling Solver.
//!
//! SRP split:
//! - `config`  — `GpuThermalAcousticConfig` + CFL validation
//! - `buffers` — `GpuThermalAcousticBuffers` + field I/O
//! - `shader`  — fused WGSL compute kernel source
//! - `solver`  — `GpuThermalAcousticSolver` + `new`, `step`, accessors

mod buffers;
mod config;
mod shader;
mod solver;
#[cfg(test)]
mod tests;

pub use buffers::{
    GpuThermalAcousticBuffers, ThermalAcousticBufferProvider, WgpuThermalAcousticBuffers,
};
pub use config::GpuThermalAcousticConfig;
pub use solver::{
    GpuThermalAcousticSolver, ThermalAcousticSolverProvider, WgpuThermalAcousticSolverProvider,
};
