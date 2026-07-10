//! DistributedNeuralBeamformingProcessor and FaultToleranceState.

use kwavers_core::error::{KwaversError, KwaversResult};
use moirai_parallel::{map_collect_mut_with, Adaptive};
use leto::{
    Array3,
    Array4,
};

#[cfg(feature = "pinn")]
use kwavers_solver::interface::pinn_beamforming::{
    DistributedConfig, LoadBalancingStrategy, PinnBeamformingDecompositionStrategy,
};

use super::super::frame_partitioning::{
    active_processor_indices, build_ranges, partition_round_sizes, DistributedChunkResult,
};
use crate::signal_processing::beamforming::neural::pinn::NeuralBeamformingProcessor;
use crate::signal_processing::beamforming::neural::types::{
    DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingResult, PINNBeamformingConfig,
};

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

/// Distributed neural beamforming processor for multi-GPU systems.
#[cfg(feature = "pinn")]
pub struct DistributedNeuralBeamformingProcessor {
    config: DistributedConfig,
    processors: Vec<NeuralBeamformingProcessor>,
    fault_tolerance: FaultToleranceState,
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

#[cfg(feature = "pinn")]
impl DistributedNeuralBeamformingProcessor {
    /// Create new distributed neural beamforming processor.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

        let mut processors = Vec::with_capacity(num_gpus);
        for _ in 0..num_gpus {
            let processor = NeuralBeamformingProcessor::new(beamforming_config.clone())?;
            processors.push(processor);
        }

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
    pub fn decomposition_strategy(&self) -> &PinnBeamformingDecompositionStrategy {
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

    /// Get number of active GPUs.
    pub fn num_gpus(&self) -> usize {
        self.processors.len()
    }

    /// Get current performance metrics.
    pub fn metrics(&self) -> &DistributedNeuralBeamformingMetrics {
        &self.metrics
    }

    /// Get fault tolerance configuration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fault_tolerance_config(&self) -> &FaultToleranceState {
        &self.fault_tolerance
    }

    /// Process RF data using distributed neural beamforming.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn process_volume_distributed(
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

            let round_results: KwaversResult<Vec<Option<DistributedChunkResult>>> =
                map_collect_mut_with::<Adaptive, _, _, _>(
                    &mut self.processors,
                    |processor_index, processor| {
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
                    },
                )
                .into_iter()
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
