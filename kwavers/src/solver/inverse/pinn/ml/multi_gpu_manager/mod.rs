//! Multi-GPU Manager for Distributed PINN Training
//!
//! Provides comprehensive multi-GPU support for Physics-Informed Neural Networks,
//! including device discovery, domain decomposition, load balancing, and
//! communication protocols.

mod manager;
#[cfg(test)]
mod tests;
mod types;

pub use manager::MultiGpuManager;
#[cfg(feature = "gpu")]
pub use types::GpuCapabilities;
pub use types::{
    CommunicationChannel, DataTransfer, FaultTolerance, GpuDeviceInfo, LoadBalancingAlgorithm,
    MultiGpuDecompositionStrategy, MultiGpuPerformanceMonitor, PerformanceSummary,
    PinnMultiGpuTransferStatus, WorkUnit,
};
