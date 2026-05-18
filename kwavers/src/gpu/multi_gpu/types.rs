//! Data types for multi-GPU context management.

/// GPU affinity configuration.
#[derive(Debug, Clone)]
pub enum GpuAffinity {
    /// No specific affinity.
    None,
    /// NUMA-aware affinity.
    NumaAware {
        /// NUMA node assignments for each GPU.
        numa_nodes: Vec<usize>,
    },
    /// Custom affinity mapping.
    Custom {
        /// Affinity groups.
        groups: Vec<Vec<usize>>,
    },
}

/// Communication channel between GPUs.
#[derive(Debug, Clone)]
pub struct CommunicationChannel {
    /// Channel bandwidth (GB/s).
    pub bandwidth: f64,
    /// Channel latency (microseconds).
    pub latency: f64,
    /// Supports peer-to-peer access.
    pub supports_p2p: bool,
    /// Transfer queue.
    pub transfer_queue: Vec<PendingTransfer>,
}

/// Pending data transfer.
#[derive(Debug, Clone)]
pub struct PendingTransfer {
    /// Transfer size (bytes).
    pub size: usize,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Transfer status.
    pub status: GpuTransferStatus,
}

/// Transfer status.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuTransferStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Performance summary for multi-GPU setup.
#[derive(Debug, Clone)]
pub struct MultiGpuPerformanceSummary {
    /// Number of GPUs.
    pub num_gpus: usize,
    /// Total GPU memory (GB).
    pub total_memory_gb: f64,
    /// Total inter-GPU bandwidth (GB/s).
    pub total_bandwidth_gbps: f64,
    /// Number of peer-to-peer pairs.
    pub p2p_pairs: usize,
    /// Affinity configuration type.
    pub affinity_type: String,
}
