//! GPU-Accelerated 3D Shear Wave Elastography
//!
//! High-performance GPU implementations for 3D SWE algorithms including
//! volumetric wave propagation, multi-directional inversion, and real-time processing.
//!
//! ## References
//!
//! - Komatitsch, D., et al. (2009). "High-order finite-element seismic wave propagation
//!   modeling with MPI on a large GPU cluster." *Journal of Computational Physics*
//! - Mickievicius, P. (2009). "3D finite difference computation on GPUs using CUDA."
//!   *Proceedings of 2nd Workshop on General Purpose Processing on Graphics Processing Units*

mod adaptive;
mod device;
mod memory;
mod metrics;
mod solver;
#[cfg(test)]
mod tests;
mod types;

pub use adaptive::AdaptiveResolution;
pub use device::GPUDevice;
pub use memory::{GPUMemoryPool, MemoryStats};
pub use metrics::{PerformanceMetrics, PerformanceStatistics};
pub use solver::GPUElasticWaveSolver3D;
pub use types::{AdaptiveSolution, AdaptiveSolutionStep, GPUInversionResult, GPUPropagationResult};
