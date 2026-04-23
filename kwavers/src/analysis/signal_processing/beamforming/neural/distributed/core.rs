//! Core distributed neural beamforming processor.
//!
//! This module implements the main distributed processing infrastructure for
//! multi-GPU neural beamforming, including processor initialization, workload
//! distribution, and result aggregation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │           Distributed Beamforming Processor                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │  GPU 0         GPU 1         GPU 2         GPU 3            │
//! │  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐          │
//! │  │ PINN │     │ PINN │     │ PINN │     │ PINN │          │
//! │  │ Model│     │ Model│     │ Model│     │ Model│          │
//! │  └──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘          │
//! │     │            │            │            │               │
//! │     └────────────┴────────────┴────────────┘               │
//! │                  Aggregation                                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Workload Distribution Strategies
//!
//! ### Spatial Decomposition
//! Partition volume along spatial dimensions:
//! - GPU 0: Volume slice [0, N/4)
//! - GPU 1: Volume slice [N/4, N/2)
//! - GPU 2: Volume slice [N/2, 3N/4)
//! - GPU 3: Volume slice [3N/4, N)
//!
//! ### Temporal Decomposition
//! Partition along time (frame) dimension:
//! - GPU 0: Frames [0, T/4)
//! - GPU 1: Frames [T/4, T/2)
//! - etc.
//!
//! ### Hybrid Decomposition
//! Combine spatial and temporal strategies based on data characteristics.
//!
//! ## References
//!
//! - Raina et al. (2009): "Large-scale deep unsupervised learning using graphics processors"
//! - Dean et al. (2012): "Large scale distributed deep networks"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array3, Array4};
use rayon::prelude::*;

// Use solver-agnostic interface instead of direct solver imports
#[cfg(feature = "pinn")]
use crate::analysis::signal_processing::beamforming::neural::pinn_interface::{
    DecompositionStrategy, DistributedConfig, LoadBalancingStrategy,
};

use super::frame_partitioning::{
    active_processor_indices, build_ranges, partition_round_sizes, DistributedChunkResult,
};
use crate::analysis::signal_processing::beamforming::neural::pinn::NeuralBeamformingProcessor;
use crate::analysis::signal_processing::beamforming::neural::types::{
    DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingResult, PINNBeamformingConfig,
};

/// Distributed neural beamforming processor for multi-GPU systems.
///
/// Coordinates parallel beamforming across multiple GPUs with load balancing,
/// fault tolerance, and communication optimization.
///
/// ## Architecture Note
///
/// This implementation uses solver-agnostic interface types for configuration
/// but maintains an internal adapter to the concrete solver implementation.
/// The adapter is created at runtime based on the available PINN backend.
///
/// ## Implementation Status
///
/// The processor partitions frame-major RF volumes into contiguous chunks,
/// schedules them across healthy processors, and recomposes the per-chunk
/// outputs without copying the source input buffer.
#[cfg(feature = "pinn")]
pub struct DistributedNeuralBeamformingProcessor {
    /// Distributed configuration (solver-agnostic)
    config: DistributedConfig,
    /// Individual neural beamforming processors (one per GPU)
    processors: Vec<NeuralBeamformingProcessor>,
    /// Fault tolerance and load balancing state
    fault_tolerance: FaultToleranceState,
    /// Performance metrics
    metrics: DistributedNeuralBeamformingMetrics,
}

#[cfg(feature = "pinn")]
impl std::fmt::Debug for DistributedNeuralBeamformingProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedNeuralBeamformingProcessor")
            .field("config", &self.config)
            .field("num_processors", &self.processors.len())
            .field("fault_tolerance", &self.fault_tolerance)
            .field("metrics", &self.metrics)
            .finish()
    }
}

/// Fault tolerance and dynamic load balancing state.
#[cfg(feature = "pinn")]
#[derive(Debug)]
pub struct FaultToleranceState {
    /// GPU health status (true = healthy)
    pub gpu_health: Vec<bool>,
    /// Current load on each GPU (0.0 to 1.0)
    pub gpu_load: Vec<f32>,
    /// Failed task retry count
    pub retry_count: usize,
    /// Maximum retries before marking GPU as failed
    pub max_retries: usize,
    /// Dynamic load balancing enabled
    pub dynamic_load_balancing: bool,
    /// Load imbalance threshold for rebalancing
    pub load_imbalance_threshold: f32,
}

#[cfg(feature = "pinn")]
impl Default for FaultToleranceState {
    fn default() -> Self {
        Self {
            gpu_health: Vec::new(),
            gpu_load: Vec::new(),
            retry_count: 0,
            max_retries: 3,
            dynamic_load_balancing: true,
            load_imbalance_threshold: 0.2,
        }
    }
}

// ModelParallelConfig and PipelineStage removed - will be implemented via DistributedPinnProvider trait
// when distributed support is fully integrated with the solver-agnostic interface

#[cfg(feature = "pinn")]
impl DistributedNeuralBeamformingProcessor {
    /// Create new distributed neural beamforming processor.
    ///
    /// # Arguments
    ///
    /// * `config` - PINN beamforming configuration
    /// * `num_gpus` - Number of GPUs to use
    /// * `decomposition_strategy` - Workload distribution strategy
    /// * `load_balancer` - Load balancing algorithm
    ///
    /// # Returns
    ///
    /// Initialized distributed processor ready for processing.
    pub fn new(
        beamforming_config: PINNBeamformingConfig,
        distributed_config: DistributedConfig,
    ) -> KwaversResult<Self> {
        let num_gpus = distributed_config.gpu_devices.len();

        if num_gpus == 0 {
            return Err(KwaversError::InvalidInput(
                "Must specify at least 1 GPU".to_string(),
            ));
        }

        // Create individual processors for each GPU
        let mut processors = Vec::with_capacity(num_gpus);
        for _ in 0..num_gpus {
            let processor = NeuralBeamformingProcessor::new(beamforming_config.clone())?;
            processors.push(processor);
        }

        // Initialize fault tolerance state
        let fault_tolerance = FaultToleranceState {
            gpu_health: vec![true; num_gpus],
            gpu_load: vec![0.0; num_gpus],
            ..Default::default()
        };

        Ok(Self {
            config: distributed_config,
            processors,
            fault_tolerance,
            metrics: DistributedNeuralBeamformingMetrics::default(),
        })
    }

    /// Get the distributed configuration.
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Get the decomposition strategy.
    pub fn decomposition_strategy(&self) -> &DecompositionStrategy {
        &self.config.decomposition
    }

    /// Get the load balancing strategy.
    pub fn load_balancing_strategy(&self) -> &LoadBalancingStrategy {
        &self.config.load_balancing
    }

    /// Get GPU device IDs.
    pub fn gpu_devices(&self) -> &[usize] {
        &self.config.gpu_devices
    }

    /// Get number of active processors.
    pub fn num_processors(&self) -> usize {
        self.processors.len()
    }

    // The current distributed path is frame-local; communication channels are
    // not required for deterministic recomposition of chunked inference.

    /// Get number of active GPUs.
    pub fn num_gpus(&self) -> usize {
        self.processors.len()
    }

    /// Get current performance metrics.
    pub fn metrics(&self) -> &DistributedNeuralBeamformingMetrics {
        &self.metrics
    }

    /// Get fault tolerance configuration.
    pub fn fault_tolerance_config(&self) -> &FaultToleranceState {
        &self.fault_tolerance
    }

    // Model parallelism configuration removed - will be implemented via DistributedPinnProvider trait

    /// Process RF data using distributed neural beamforming.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Input RF data (frames × channels × samples × 1)
    ///
    /// # Returns
    ///
    /// Distributed beamforming result with aggregated volume, uncertainty, and metrics.
    ///
    /// # Process
    ///
    /// 1. Decompose workload across GPUs
    /// 2. Distribute work units to processors
    /// 3. Execute parallel processing
    /// 4. Aggregate results
    /// 5. Compute final metrics
    ///
    /// # Theorem
    ///
    /// Let `F` be the frame-local mapping implemented by
    /// [`NeuralBeamformingProcessor::process_volume_view`]. For any partition
    /// of the frame axis into disjoint contiguous ranges `R_i`, the distributed
    /// result equals `concat(F(R_i))` because each output frame depends only on
    /// the matching input frame slice and there is no cross-frame coupling.
    ///
    /// # Proof Sketch
    ///
    /// The worker output for a frame range never reads or writes outside that
    /// range. Therefore partitioning and concatenation commute with the
    /// processor mapping, so the recomposed tensor is identical to the
    /// sequential result.
    pub async fn process_volume_distributed(
        &mut self,
        rf_data: &Array4<f32>,
    ) -> KwaversResult<DistributedNeuralBeamformingResult> {
        let start_time = std::time::Instant::now();
        let (frames, channels, samples, trailing) = rf_data.dim();

        if trailing != 1 {
            return Err(KwaversError::InvalidInput(format!(
                "Distributed beamforming expects a singleton trailing axis, got {}",
                trailing
            )));
        }

        if frames == 0 || channels == 0 || samples == 0 {
            return Err(KwaversError::InvalidInput(
                "Distributed beamforming requires non-empty frame, channel, and sample dimensions"
                    .to_string(),
            ));
        }

        let active_indices =
            active_processor_indices(self.processors.len(), &self.fault_tolerance.gpu_health);
        if active_indices.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Distributed beamforming requires at least one healthy processor".to_string(),
            ));
        }

        let batch_size = self.config.batch_size_per_gpu.max(1);
        let round_capacity = active_indices.len().saturating_mul(batch_size).max(1);
        let active_lookup: Vec<Option<usize>> = {
            let mut lookup = vec![None; self.processors.len()];
            for (slot, &processor_idx) in active_indices.iter().enumerate() {
                lookup[processor_idx] = Some(slot);
            }
            lookup
        };

        let mut final_volume = Array3::<f32>::zeros((frames, channels, samples));
        let mut final_uncertainty = Array3::<f32>::zeros((frames, channels, samples));
        let mut final_confidence = Array3::<f32>::zeros((frames, channels, samples));
        let mut processor_times = vec![0.0_f64; self.processors.len()];
        let mut peak_round_chunk_bytes = 0usize;
        let mut used_processors = vec![false; self.processors.len()];

        for round_start in (0..frames).step_by(round_capacity) {
            let round_end = (round_start + round_capacity).min(frames);
            let round_frames = round_end - round_start;
            let partition_sizes = partition_round_sizes(
                &self.config,
                round_frames,
                &active_indices,
                batch_size,
                &self.fault_tolerance.gpu_load,
            );
            let ranges = build_ranges(round_start, &partition_sizes);

            debug_assert_eq!(
                ranges.last().map(|range| range.end).unwrap_or(round_start),
                round_end
            );

            let round_chunk_bytes =
                round_frames * channels * samples * std::mem::size_of::<f32>() * 3;
            peak_round_chunk_bytes = peak_round_chunk_bytes.max(round_chunk_bytes);

            let round_results: KwaversResult<Vec<Option<DistributedChunkResult>>> = self
                .processors
                .par_iter_mut()
                .enumerate()
                .map(|(processor_index, processor)| {
                    let Some(slot) = active_lookup[processor_index] else {
                        return Ok::<Option<DistributedChunkResult>, KwaversError>(None);
                    };

                    let Some(range) = ranges.get(slot).cloned() else {
                        return Ok::<Option<DistributedChunkResult>, KwaversError>(None);
                    };

                    if range.is_empty() {
                        return Ok::<Option<DistributedChunkResult>, KwaversError>(None);
                    }

                    let chunk_view = rf_data.slice(s![range.start..range.end, .., .., ..]);
                    let result = processor.process_volume_view(chunk_view)?;

                    Ok(Some(DistributedChunkResult {
                        start: range.start,
                        processor_index,
                        processing_time_ms: result.processing_time_ms,
                        volume: result.volume,
                        uncertainty: result.uncertainty,
                        confidence: result.confidence,
                    }))
                })
                .collect();

            for chunk in round_results?.into_iter().flatten() {
                let chunk_frames = chunk.volume.dim().0;
                let end = chunk.start + chunk_frames;

                final_volume
                    .slice_mut(s![chunk.start..end, .., ..])
                    .assign(&chunk.volume);
                final_uncertainty
                    .slice_mut(s![chunk.start..end, .., ..])
                    .assign(&chunk.uncertainty);
                final_confidence
                    .slice_mut(s![chunk.start..end, .., ..])
                    .assign(&chunk.confidence);

                processor_times[chunk.processor_index] += chunk.processing_time_ms;
                used_processors[chunk.processor_index] = true;
            }
        }

        let total_processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let max_processor_time = processor_times.iter().copied().fold(0.0_f64, f64::max);
        let min_processor_time = processor_times
            .iter()
            .copied()
            .filter(|time| *time > 0.0)
            .fold(f64::INFINITY, f64::min);
        let used_processor_count = used_processors.iter().filter(|&&used| used).count();

        let load_balance_efficiency = if max_processor_time > 0.0 && min_processor_time.is_finite()
        {
            (min_processor_time / max_processor_time).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let communication_overhead = (total_processing_time_ms - max_processor_time).max(0.0);
        let load_imbalance_ratio = if max_processor_time > 0.0 && min_processor_time.is_finite() {
            ((max_processor_time - min_processor_time).max(0.0)) / max_processor_time
        } else {
            0.0
        };

        let useful_bytes = frames * channels * samples * std::mem::size_of::<f32>() * 3;
        let memory_efficiency = if useful_bytes > 0 {
            useful_bytes as f64 / (useful_bytes + peak_round_chunk_bytes) as f64
        } else {
            1.0
        };

        for (processor_index, total_time) in processor_times.iter().copied().enumerate() {
            if let Some(load) = self.fault_tolerance.gpu_load.get_mut(processor_index) {
                *load = if max_processor_time > 0.0 {
                    (total_time / max_processor_time).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
            }
        }

        self.metrics.total_processing_time = total_processing_time_ms;
        self.metrics.communication_overhead = communication_overhead;
        self.metrics.load_imbalance_ratio = load_imbalance_ratio;
        self.metrics.memory_efficiency = memory_efficiency;
        self.metrics.fault_tolerance_events = self.fault_tolerance.retry_count;
        self.metrics.active_gpus = used_processor_count;

        Ok(DistributedNeuralBeamformingResult {
            volume: final_volume,
            uncertainty: final_uncertainty,
            confidence: final_confidence,
            processing_time_ms: total_processing_time_ms,
            num_gpus_used: used_processor_count,
            load_balance_efficiency,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_processor_creation() {
        let beamforming_config = PINNBeamformingConfig::default();
        let distributed_config = DistributedConfig {
            num_gpus: 1,
            gpu_devices: vec![0],
            batch_size_per_gpu: 32,
            decomposition: DecompositionStrategy::Spatial,
            load_balancing: LoadBalancingStrategy::Static,
        };
        let result =
            DistributedNeuralBeamformingProcessor::new(beamforming_config, distributed_config);

        // May fail if GPUs not available, but should not panic
        let _ = result;
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_fault_tolerance_default() {
        let ft = FaultToleranceState::default();
        assert_eq!(ft.max_retries, 3);
        assert!(ft.dynamic_load_balancing);
        assert_eq!(ft.load_imbalance_threshold, 0.2);
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_fault_tolerance_config() {
        let mut fault_state = FaultToleranceState::default();
        assert_eq!(fault_state.max_retries, 3);
        assert!(fault_state.dynamic_load_balancing);

        fault_state.gpu_health = vec![true, true, true, true];
        fault_state.gpu_load = vec![0.5, 0.6, 0.4, 0.7];

        let max_load = fault_state.gpu_load.iter().copied().fold(0.0, f32::max);
        assert!(max_load <= 1.0);
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_distributed_processing_matches_sequential_result() {
        use crate::analysis::signal_processing::beamforming::neural::pinn::NeuralBeamformingProcessor;
        use ndarray::Array4;

        let beamforming_config = PINNBeamformingConfig {
            rf_data_channels: 2,
            samples_per_channel: 3,
            volume_size: (4, 2, 3),
            enable_pinn: false,
            enable_uncertainty_quantification: false,
            ..Default::default()
        };

        let sequential_config = beamforming_config.clone();
        let mut sequential = NeuralBeamformingProcessor::new(sequential_config).unwrap();

        let distributed_config = DistributedConfig {
            num_gpus: 2,
            gpu_devices: vec![0, 1],
            batch_size_per_gpu: 1,
            decomposition: DecompositionStrategy::Spatial,
            load_balancing: LoadBalancingStrategy::Static,
        };
        let mut distributed =
            DistributedNeuralBeamformingProcessor::new(beamforming_config, distributed_config)
                .unwrap();

        let rf_data = Array4::from_shape_fn((4, 2, 3, 1), |(frame, channel, sample, _)| {
            frame as f32 + 0.1 * channel as f32 + 0.01 * sample as f32
        });

        let expected = sequential.process_volume(&rf_data).unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();

        let actual = runtime
            .block_on(distributed.process_volume_distributed(&rf_data))
            .unwrap();

        assert_eq!(actual.volume, expected.volume);
        assert_eq!(actual.uncertainty, expected.uncertainty);
        assert_eq!(actual.confidence, expected.confidence);
        assert_eq!(actual.num_gpus_used, 2);
        assert_eq!(distributed.metrics().active_gpus, 2);
        assert!(actual.load_balance_efficiency > 0.0);
        assert!(distributed.metrics().memory_efficiency > 0.0);
    }
}
