//! Data types for the multi-GPU PINN manager.

use std::collections::VecDeque;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PinnGpuCapabilities {
    pub max_buffer_size: u64,
    pub max_workgroup_size: [u32; 3],
    pub max_compute_invocations: u32,
    pub supports_f64: bool,
    pub supports_atomics: bool,
}

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct PinnMultiGpuDeviceInfo {
    /// Unique device identifier
    pub id: usize,
    /// Device name
    pub name: String,
    /// Backend type (Vulkan, DirectX, Metal)
    pub backend: String,
    /// GPU capabilities (available when GPU feature is enabled)
    #[cfg(feature = "gpu")]
    pub capabilities: PinnGpuCapabilities,
    /// Current memory usage (bytes)
    pub memory_used: usize,
    /// Current computational load (0.0 to 1.0)
    pub compute_load: f32,
    /// Device health status
    pub healthy: bool,
}

/// Domain decomposition strategy
#[derive(Debug, Clone)]
pub enum MultiGpuDecompositionStrategy {
    /// Spatial decomposition across GPUs
    Spatial {
        /// Number of spatial dimensions to split
        dimensions: usize,
        /// Overlap size for boundary conditions
        overlap: f64,
    },
    /// Temporal decomposition (pipeline parallelism)
    Temporal {
        /// Number of time steps per GPU
        steps_per_gpu: usize,
    },
    /// Hybrid spatial-temporal decomposition
    Hybrid {
        /// Spatial split configuration
        spatial_dims: usize,
        /// Temporal steps per GPU
        temporal_steps: usize,
        /// Spatial overlap
        overlap: f64,
    },
}

/// Load balancing algorithm
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    /// Static load balancing (equal distribution)
    Static,
    /// Dynamic load balancing with work stealing
    Dynamic {
        /// Threshold for load imbalance detection
        imbalance_threshold: f32,
        /// Migration interval (seconds)
        migration_interval: f64,
    },
    /// Predictive load balancing using historical data
    Predictive {
        /// Historical window size
        history_window: usize,
        /// Prediction horizon
        prediction_horizon: usize,
    },
}

/// Work unit for distributed computation
#[derive(Debug, Clone)]
pub struct WorkUnit {
    /// Unique work identifier
    pub id: usize,
    /// GPU device assignment
    pub device_id: usize,
    /// Computational complexity estimate
    pub complexity: f64,
    /// Memory requirements (bytes)
    pub memory_required: usize,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Dependencies on other work units
    pub dependencies: Vec<usize>,
    /// Data range for this work unit (start..end indices for frames)
    pub data_range: Option<std::ops::Range<usize>>,
    /// Channel range for this work unit (start..end indices for channels)
    pub channel_range: Option<std::ops::Range<usize>>,
    /// Sample range for this work unit (start..end indices for time samples)
    pub sample_range: Option<std::ops::Range<usize>>,
}

/// Communication channel between two GPUs
#[derive(Debug, Clone)]
pub struct PinnMultiGpuCommunicationChannel {
    /// Bandwidth estimate (GB/s)
    pub bandwidth: f64,
    /// Latency estimate (microseconds)
    pub latency: f64,
    /// Active transfers
    pub active_transfers: usize,
    /// Transfer queue
    pub transfer_queue: VecDeque<DataTransfer>,
}

/// Data transfer operation
#[derive(Debug, Clone)]
pub struct DataTransfer {
    /// Transfer identifier
    pub id: usize,
    /// Source GPU
    pub source_gpu: usize,
    /// Destination GPU
    pub dest_gpu: usize,
    /// Data size (bytes)
    pub size: usize,
    /// Priority
    pub priority: u32,
    /// Transfer status
    pub status: PinnMultiGpuTransferStatus,
}

/// Transfer status
#[derive(Debug, Clone, PartialEq)]
pub enum PinnMultiGpuTransferStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Performance monitoring data
#[derive(Debug, Clone)]
pub struct MultiGpuPerformanceMonitor {
    /// GPU utilization history
    pub gpu_utilization: Vec<Vec<f32>>,
    /// Memory usage history
    pub memory_usage: Vec<Vec<usize>>,
    /// Communication overhead
    pub communication_overhead: f64,
    /// Load imbalance metrics
    pub load_imbalance: f32,
    /// Scaling efficiency
    pub scaling_efficiency: f64,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultTolerance {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Checkpoint interval (seconds)
    pub checkpoint_interval: f64,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Graceful degradation enabled
    pub graceful_degradation: bool,
}

/// Performance summary for multi-GPU training
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Number of GPUs used
    pub num_gpus: usize,
    /// Load imbalance metric (0.0 = perfect balance)
    pub load_imbalance: f32,
    /// Scaling efficiency (0.0 to 1.0)
    pub scaling_efficiency: f64,
    /// Communication overhead as fraction of training time
    pub communication_overhead: f64,
    /// Average GPU utilization
    pub average_utilization: f32,
}
