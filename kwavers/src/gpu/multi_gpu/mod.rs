//! Multi-GPU context management, device affinity, and cross-GPU communication.

pub mod context;
#[cfg(test)]
mod tests;
pub mod types;

pub use context::MultiGpuContext;
pub use types::{
    CommunicationChannel, GpuAffinity, MultiGpuPerformanceSummary, PendingTransfer, TransferStatus,
};
