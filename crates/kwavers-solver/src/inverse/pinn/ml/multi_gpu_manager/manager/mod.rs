//! `MultiGpuManager` struct and core construction.

pub mod assign;
pub mod decompose;
pub mod monitor;

use super::types::{
    FaultTolerance, LoadBalancingAlgorithm, MultiGpuDecompositionStrategy,
    MultiGpuPerformanceMonitor, PinnMultiGpuCommunicationChannel, PinnMultiGpuDeviceInfo, WorkUnit,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::{HashMap, VecDeque};

/// Multi-GPU manager for PINN training.
#[derive(Debug)]
pub struct MultiGpuManager {
    /// Available GPU devices.
    pub(super) devices: Vec<PinnMultiGpuDeviceInfo>,
    /// Decomposition strategy.
    pub(super) decomposition: MultiGpuDecompositionStrategy,
    /// Load balancing algorithm.
    pub(super) load_balancer: LoadBalancingAlgorithm,
    /// Work queue for distribution.
    pub(super) work_queue: VecDeque<WorkUnit>,
    /// Active work assignments.
    pub(super) active_work: HashMap<usize, WorkUnit>,
    /// Communication channels between GPUs.
    pub(super) _communication_channels: HashMap<(usize, usize), PinnMultiGpuCommunicationChannel>,
    /// Performance monitoring.
    pub(super) performance_monitor: MultiGpuPerformanceMonitor,
    /// Fault tolerance state.
    pub(super) fault_tolerance: FaultTolerance,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager.
    /// # Errors
    /// - Returns [`crate::KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn new(
        decomposition: MultiGpuDecompositionStrategy,
        load_balancer: LoadBalancingAlgorithm,
    ) -> KwaversResult<Self> {
        let devices = Self::discover_gpu_devices()?;

        if (devices.len()) < 2 {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "Multiple GPU devices required for multi-GPU training".to_string(),
                },
            ));
        }

        let communication_channels = Self::initialize_communication_channels(&devices);
        let device_count = devices.len();

        Ok(Self {
            devices,
            decomposition,
            load_balancer,
            work_queue: VecDeque::new(),
            active_work: HashMap::new(),
            _communication_channels: communication_channels,
            performance_monitor: MultiGpuPerformanceMonitor {
                gpu_utilization: vec![vec![]; device_count],
                memory_usage: vec![vec![]; device_count],
                communication_overhead: 0.0,
                load_imbalance: 0.0,
                scaling_efficiency: 1.0,
            },
            fault_tolerance: FaultTolerance {
                auto_recovery: true,
                checkpoint_interval: 300.0,
                max_retries: 3,
                graceful_degradation: true,
            },
        })
    }

    fn discover_gpu_devices() -> KwaversResult<Vec<PinnMultiGpuDeviceInfo>> {
        Err(KwaversError::System(
            kwavers_core::error::SystemError::ResourceUnavailable {
                resource: "PINN multi-GPU discovery requires a Coeus training provider routed through Hephaestus WGPU/CUDA device traits".to_string(),
            },
        ))
    }

    pub(super) fn initialize_communication_channels(
        devices: &[PinnMultiGpuDeviceInfo],
    ) -> HashMap<(usize, usize), PinnMultiGpuCommunicationChannel> {
        let mut channels = HashMap::new();

        for i in 0..(devices.len()) {
            for j in (i + 1)..(devices.len()) {
                let channel = PinnMultiGpuCommunicationChannel {
                    bandwidth: 50.0,
                    latency: 5.0,
                    active_transfers: 0,
                    transfer_queue: VecDeque::new(),
                };
                channels.insert((i, j), channel.clone());
                channels.insert((j, i), channel);
            }
        }

        channels
    }

    /// Get device information.
    pub fn get_devices(&self) -> &[PinnMultiGpuDeviceInfo] {
        &self.devices
    }

    /// Get active work assignments.
    pub fn get_active_work(&self) -> &HashMap<usize, WorkUnit> {
        &self.active_work
    }

    /// Check if a GPU is healthy and available.
    pub fn is_gpu_healthy(&self, gpu_id: usize) -> bool {
        gpu_id < (self.devices.len()) && self.devices[gpu_id].healthy
    }
}
