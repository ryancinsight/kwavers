//! Distributed multi-GPU neural beamforming.
//!
//! This module implements distributed neural beamforming across multiple GPUs
//! with sophisticated workload decomposition, fault tolerance, and communication
//! optimization strategies.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │               Distributed Beamforming System               │
//! ├────────────────────────────────────────────────────────────┤
//! │                                                            │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
//! │  │  GPU 0   │    │  GPU 1   │    │  GPU 2   │            │
//! │  │          │◄──►│          │◄──►│          │            │
//! │  │  PINN    │    │  PINN    │    │  PINN    │            │
//! │  │  Model   │    │  Model   │    │  Model   │            │
//! │  └─────┬────┘    └─────┬────┘    └─────┬────┘            │
//! │        │               │               │                  │
//! │        └───────────────┴───────────────┘                  │
//! │                   Aggregator                               │
//! │                                                            │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Parallelization Strategies
//!
//! ### 1. Data Parallelism
//!
//! Distribute independent data samples across GPUs:
//! - Each GPU processes different frames/volumes
//! - No inter-GPU communication during forward pass
//! - Gradients aggregated during training
//!
//! **Best for**: Large batch sizes, independent frames
//!
//! ### 2. Model Parallelism
//!
//! Partition neural network layers across GPUs:
//! - GPU 0: Layers 0-N/3
//! - GPU 1: Layers N/3-2N/3
//! - GPU 2: Layers 2N/3-N
//!
//! **Best for**: Large models exceeding single GPU memory
//!
//! ### 3. Spatial Decomposition
//!
//! Partition imaging volume spatially:
//! ```text
//! ┌─────┬─────┐
//! │ GPU │ GPU │
//! │  0  │  1  │
//! ├─────┼─────┤
//! │ GPU │ GPU │
//! │  2  │  3  │
//! └─────┴─────┘
//! ```
//!
//! **Best for**: Large volumes, localized computations
//!
//! ### 4. Temporal Decomposition
//!
//! Distribute frames across time:
//! - GPU 0: Frames [0, T/4)
//! - GPU 1: Frames [T/4, T/2)
//! - GPU 2: Frames [T/2, 3T/4)
//! - GPU 3: Frames [3T/4, T)
//!
//! **Best for**: Video sequences, streaming data
//!
//! ### 5. Hybrid Decomposition
//!
//! Combine multiple strategies based on workload characteristics and
//! hardware topology.
//!
//! ## Communication Optimization
//!
//! ### Topology-Aware Scheduling
//!
//! - **NUMA-aware**: Prefer GPUs on same NUMA node
//! - **NVLink detection**: Use high-bandwidth links when available
//! - **PCIe topology**: Minimize hops through CPU
//!
//! ### Overlap Communication & Computation
//!
//! Pipeline data transfers with computation:
//! ```text
//! Time:  GPU 0         GPU 1
//! t0:    Compute       Recv
//! t1:    Send          Compute
//! t2:    Recv          Send
//! ```
//!
//! ## Fault Tolerance
//!
//! ### GPU Health Monitoring
//!
//! - Memory allocation failures
//! - Computation timeouts
//! - Temperature/power anomalies
//!
//! ### Recovery Strategies
//!
//! 1. **Task retry**: Re-submit failed tasks (up to max_retries)
//! 2. **GPU blacklist**: Mark unhealthy GPUs as unavailable
//! 3. **Workload rebalancing**: Redistribute load to healthy GPUs
//! 4. **Graceful degradation**: Continue with fewer GPUs
//!
//! ## Load Balancing
//!
//! ### Static Load Balancing
//!
//! Equal partitioning assuming homogeneous GPUs:
//! ```text
//! Work per GPU = Total_Work / Num_GPUs
//! ```
//!
//! ### Dynamic Load Balancing
//!
//! Adapt to heterogeneous performance:
//! ```text
//! Work_i = Total_Work × (Perf_i / ∑_j Perf_j)
//! ```
//!
//! Rebalance when:
//! ```text
//! max(Load_i) - min(Load_i) > threshold
//! ```
//!
//! ## Module Organization
//!
//! - [`core`]: Distributed processor and initialization
//! - (Future) `decomposition`: Workload decomposition strategies
//! - (Future) `model_parallel`: Model parallelism pipeline
//! - (Future) `data_parallel`: Data parallelism implementation
//! - (Future) `fault_tolerance`: Health monitoring and recovery
//!
//! ## Performance Considerations
//!
//! ### Scalability
//!
//! Speedup with N GPUs (ideal):
//! ```text
//! S(N) = T(1) / T(N) ≈ N
//! ```
//!
//! Actual speedup with overhead:
//! ```text
//! S(N) = T(1) / (T(1)/N + T_comm + T_sync)
//! ```
//!
//! ### Communication Overhead
//!
//! Transfer time for data D over bandwidth B:
//! ```text
//! T_transfer = D / B + Latency
//! ```
//!
//! Minimize by:
//! - Overlap with computation
//! - Batch transfers
//! - Compress data when beneficial
//!
//! ### Memory Efficiency
//!
//! Per-GPU memory usage:
//! ```text
//! M_gpu = M_model + M_activations + M_gradients + M_data
//! ```
//!
//! Optimization strategies:
//! - Gradient checkpointing
//! - Activation recomputation
//! - Mixed precision training
//!
//! ## Usage Example
//!
//! ```ignore
//! use kwavers::analysis::signal_processing::beamforming::neural::distributed::DistributedNeuralBeamformingProcessor;
//! use kwavers::math::ml::pinn::multi_gpu_manager::{DecompositionStrategy, LoadBalancingAlgorithm};
//!
//! // Create distributed processor
//! let processor = DistributedNeuralBeamformingProcessor::new(
//!     config,
//!     4, // 4 GPUs
//!     DecompositionStrategy::Spatial,
//!     LoadBalancingAlgorithm::Dynamic,
//! ).await?;
//!
//! // Process volume
//! let result = processor.process_volume_distributed(&rf_data).await?;
//!
//! println!("Used {} GPUs", result.num_gpus_used);
//! println!("Load balance efficiency: {:.2}", result.load_balance_efficiency);
//! ```
//!
//! ## References
//!
//! ### Distributed Deep Learning
//!
//! - Raina, R., Madhavan, A., & Ng, A. Y. (2009).
//!   "Large-scale deep unsupervised learning using graphics processors."
//!   ICML 2009.
//!
//! - Dean, J., et al. (2012).
//!   "Large scale distributed deep networks."
//!   NIPS 2012.
//!
//! - Sergeev, A., & Del Balso, M. (2018).
//!   "Horovod: fast and easy distributed deep learning in TensorFlow."
//!   arXiv:1802.05799
//!
//! ### GPU Communication
//!
//! - NVIDIA (2020).
//!   "NCCL: Optimized primitives for collective multi-GPU communication."
//!
//! - Jia, Z., et al. (2018).
//!   "Beyond data and model parallelism for deep neural networks."
//!   SysML 2018.
//!
//! ### Fault Tolerance
//!
//! - Chen, T., et al. (2016).
//!   "Training deep nets with sublinear memory cost."
//!   arXiv:1604.06174
//!
//! - Moritz, P., et al. (2018).
//!   "Ray: A distributed framework for emerging AI applications."
//!   OSDI 2018.

pub mod core;

// Re-export primary types
#[cfg(feature = "pinn")]
pub use core::{
    DistributedNeuralBeamformingProcessor, FaultToleranceState, ModelParallelConfig, PipelineStage,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_distributed_module_available() {
        // Verify module compiles
        #[cfg(feature = "pinn")]
        {
            use super::core::FaultToleranceState;
            let _ = FaultToleranceState::default();
        }
    }
}
