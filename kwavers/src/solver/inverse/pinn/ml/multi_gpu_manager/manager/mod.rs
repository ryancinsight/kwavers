//! `MultiGpuManager` struct and core construction.

pub mod assign;
pub mod decompose;
pub mod monitor;

use super::types::{
    CommunicationChannel, FaultTolerance, GpuDeviceInfo, LoadBalancingAlgorithm,
    MultiGpuDecompositionStrategy, MultiGpuPerformanceMonitor, WorkUnit,
};
use crate::core::error::{KwaversError, KwaversResult};
use std::collections::{HashMap, VecDeque};

/// Multi-GPU manager for PINN training.
#[derive(Debug)]
pub struct MultiGpuManager {
    /// Available GPU devices.
    pub(super) devices: Vec<GpuDeviceInfo>,
    /// Decomposition strategy.
    pub(super) decomposition: MultiGpuDecompositionStrategy,
    /// Load balancing algorithm.
    pub(super) load_balancer: LoadBalancingAlgorithm,
    /// Work queue for distribution.
    pub(super) work_queue: VecDeque<WorkUnit>,
    /// Active work assignments.
    pub(super) active_work: HashMap<usize, WorkUnit>,
    /// Communication channels between GPUs.
    pub(super) _communication_channels: HashMap<(usize, usize), CommunicationChannel>,
    /// Performance monitoring.
    pub(super) performance_monitor: MultiGpuPerformanceMonitor,
    /// Fault tolerance state.
    pub(super) fault_tolerance: FaultTolerance,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn new(
        decomposition: MultiGpuDecompositionStrategy,
        load_balancer: LoadBalancingAlgorithm,
    ) -> KwaversResult<Self> {
        let devices = Self::discover_gpu_devices().await?;

        if devices.len() < 2 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
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

    #[cfg(feature = "gpu")]
    async fn discover_gpu_devices() -> KwaversResult<Vec<GpuDeviceInfo>> {
        use super::types::GpuCapabilities;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let mut devices = Vec::new();
        let mut device_id = 0;

        let adapters: Vec<_> = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .collect();
        for adapter in adapters {
            let info = adapter.get_info();
            let limits = adapter.limits();

            if matches!(info.backend, wgpu::Backend::BrowserWebGpu) {
                continue;
            }

            let capabilities = GpuCapabilities {
                max_buffer_size: limits.max_buffer_size,
                max_workgroup_size: [
                    limits.max_compute_workgroup_size_x,
                    limits.max_compute_workgroup_size_y,
                    limits.max_compute_workgroup_size_z,
                ],
                max_compute_invocations: limits.max_compute_invocations_per_workgroup,
                supports_f64: adapter.features().contains(wgpu::Features::SHADER_F64),
                supports_atomics: true,
            };

            devices.push(GpuDeviceInfo {
                id: device_id,
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                capabilities,
                memory_used: 0,
                compute_load: 0.0,
                healthy: true,
            });

            device_id += 1;
        }

        Ok(devices)
    }

    #[cfg(not(feature = "gpu"))]
    async fn discover_gpu_devices() -> KwaversResult<Vec<GpuDeviceInfo>> {
        let devices = vec![GpuDeviceInfo {
            id: 0,
            name: "CPU Backend".to_string(),
            backend: "CPU".to_string(),
            memory_used: 0,
            compute_load: 0.0,
            healthy: true,
        }];
        Ok(devices)
    }

    pub(super) fn initialize_communication_channels(
        devices: &[GpuDeviceInfo],
    ) -> HashMap<(usize, usize), CommunicationChannel> {
        let mut channels = HashMap::new();

        for i in 0..devices.len() {
            for j in (i + 1)..devices.len() {
                let channel = CommunicationChannel {
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
    pub fn get_devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Get active work assignments.
    pub fn get_active_work(&self) -> &HashMap<usize, WorkUnit> {
        &self.active_work
    }

    /// Check if a GPU is healthy and available.
    pub fn is_gpu_healthy(&self, gpu_id: usize) -> bool {
        gpu_id < self.devices.len() && self.devices[gpu_id].healthy
    }
}
