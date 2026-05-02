//! `MultiGpuManager` struct and impl.

use super::types::{
    CommunicationChannel, DecompositionStrategy, FaultTolerance, GpuDeviceInfo,
    LoadBalancingAlgorithm, PerformanceMonitor, PerformanceSummary, WorkUnit,
};
use crate::core::error::{KwaversError, KwaversResult};
use std::collections::{HashMap, VecDeque};

/// Multi-GPU manager for PINN training
#[derive(Debug)]
pub struct MultiGpuManager {
    /// Available GPU devices
    devices: Vec<GpuDeviceInfo>,
    /// Decomposition strategy
    decomposition: DecompositionStrategy,
    /// Load balancing algorithm
    load_balancer: LoadBalancingAlgorithm,
    /// Work queue for distribution
    work_queue: VecDeque<WorkUnit>,
    /// Active work assignments
    active_work: HashMap<usize, WorkUnit>,
    /// Communication channels between GPUs
    _communication_channels: HashMap<(usize, usize), CommunicationChannel>,
    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
    /// Fault tolerance state
    fault_tolerance: FaultTolerance,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager
    pub async fn new(
        decomposition: DecompositionStrategy,
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
            performance_monitor: PerformanceMonitor {
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

    fn initialize_communication_channels(
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

    /// Decompose a PINN training task across GPUs
    pub fn decompose_training_task(
        &self,
        total_collocation_points: usize,
        geometry_bounds: (f64, f64, f64, f64),
    ) -> KwaversResult<Vec<WorkUnit>> {
        match &self.decomposition {
            DecompositionStrategy::Spatial {
                dimensions,
                overlap,
            } => self.decompose_spatially(
                total_collocation_points,
                geometry_bounds,
                *dimensions,
                *overlap,
            ),
            DecompositionStrategy::Temporal { steps_per_gpu } => {
                self.decompose_temporally(total_collocation_points, *steps_per_gpu)
            }
            DecompositionStrategy::Hybrid {
                spatial_dims,
                temporal_steps,
                overlap,
            } => self.decompose_hybrid(
                total_collocation_points,
                geometry_bounds,
                *spatial_dims,
                *temporal_steps,
                *overlap,
            ),
        }
    }

    fn decompose_spatially(
        &self,
        total_points: usize,
        bounds: (f64, f64, f64, f64),
        _dimensions: usize,
        overlap: f64,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let n_gpus = self.devices.len();
        let points_per_gpu = total_points / n_gpus;
        let mut work_units = Vec::new();

        let (x_min, x_max, _y_min, _y_max) = bounds;

        for gpu_id in 0..n_gpus {
            let x_start = x_min + (x_max - x_min) * (gpu_id as f64) / (n_gpus as f64);
            let x_end = x_min + (x_max - x_min) * ((gpu_id + 1) as f64) / (n_gpus as f64);

            let _x_start_overlap = (x_start - overlap).max(x_min);
            let _x_end_overlap = (x_end + overlap).min(x_max);

            work_units.push(WorkUnit {
                id: gpu_id,
                device_id: gpu_id,
                complexity: points_per_gpu as f64,
                memory_required: points_per_gpu * std::mem::size_of::<f32>() * 4,
                priority: 1,
                dependencies: vec![],
                data_range: None,
                channel_range: None,
                sample_range: None,
            });
        }

        Ok(work_units)
    }

    fn decompose_temporally(
        &self,
        total_points: usize,
        _steps_per_gpu: usize,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let n_gpus = self.devices.len();
        let mut work_units = Vec::new();

        for gpu_id in 0..n_gpus {
            work_units.push(WorkUnit {
                id: gpu_id,
                device_id: gpu_id,
                complexity: (total_points / n_gpus) as f64,
                memory_required: (total_points / n_gpus) * std::mem::size_of::<f32>() * 2,
                priority: 1,
                dependencies: if gpu_id > 0 { vec![gpu_id - 1] } else { vec![] },
                data_range: None,
                channel_range: None,
                sample_range: None,
            });
        }

        Ok(work_units)
    }

    fn decompose_hybrid(
        &self,
        total_points: usize,
        bounds: (f64, f64, f64, f64),
        spatial_dims: usize,
        temporal_steps: usize,
        overlap: f64,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let spatial_work = self.decompose_spatially(total_points, bounds, spatial_dims, overlap)?;
        let temporal_work = self.decompose_temporally(total_points, temporal_steps)?;

        let mut work_units = Vec::new();
        let mut id_counter = 0;

        for spatial_unit in spatial_work {
            for temporal_unit in &temporal_work {
                work_units.push(WorkUnit {
                    id: id_counter,
                    device_id: spatial_unit.device_id,
                    complexity: spatial_unit.complexity * temporal_unit.complexity,
                    memory_required: spatial_unit.memory_required + temporal_unit.memory_required,
                    priority: 1,
                    dependencies: temporal_unit.dependencies.clone(),
                    data_range: spatial_unit.data_range.clone(),
                    channel_range: spatial_unit.channel_range.clone(),
                    sample_range: temporal_unit.sample_range.clone(),
                });
                id_counter += 1;
            }
        }

        Ok(work_units)
    }

    /// Assign work to GPUs using load balancing
    pub fn assign_work(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        match &self.load_balancer {
            LoadBalancingAlgorithm::Static => self.assign_work_static(work_units),
            LoadBalancingAlgorithm::Dynamic { .. } => self.assign_work_dynamic(work_units),
            LoadBalancingAlgorithm::Predictive { .. } => self.assign_work_predictive(work_units),
        }
    }

    fn assign_work_static(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        let n_gpus = self.devices.len();
        for (i, work_unit) in work_units.into_iter().enumerate() {
            let gpu_id = i % n_gpus;
            let mut assigned_work = work_unit;
            assigned_work.device_id = gpu_id;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    fn assign_work_dynamic(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        let mut sorted_work = work_units;
        sorted_work.sort_by(|a, b| {
            b.complexity
                .partial_cmp(&a.complexity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for work_unit in sorted_work {
            let best_gpu = self.find_least_loaded_gpu();
            let mut assigned_work = work_unit;
            assigned_work.device_id = best_gpu;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    fn assign_work_predictive(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        for work_unit in work_units {
            let predicted_gpu = self.predict_optimal_gpu(&work_unit);
            let mut assigned_work = work_unit;
            assigned_work.device_id = predicted_gpu;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    fn find_least_loaded_gpu(&self) -> usize {
        let mut min_load = f32::INFINITY;
        let mut best_gpu = 0;
        for (i, device) in self.devices.iter().enumerate() {
            if device.healthy && device.compute_load < min_load {
                min_load = device.compute_load;
                best_gpu = i;
            }
        }
        best_gpu
    }

    fn predict_optimal_gpu(&self, _work_unit: &WorkUnit) -> usize {
        self.find_least_loaded_gpu()
    }

    /// Update performance metrics
    pub fn update_performance_metrics(
        &mut self,
        gpu_id: usize,
        utilization: f32,
        memory_used: usize,
    ) {
        if gpu_id < self.devices.len() {
            self.devices[gpu_id].compute_load = utilization;
            self.devices[gpu_id].memory_used = memory_used;

            self.performance_monitor.gpu_utilization[gpu_id].push(utilization);
            self.performance_monitor.memory_usage[gpu_id].push(memory_used);

            const MAX_HISTORY: usize = 100;
            if self.performance_monitor.gpu_utilization[gpu_id].len() > MAX_HISTORY {
                self.performance_monitor.gpu_utilization[gpu_id].remove(0);
                self.performance_monitor.memory_usage[gpu_id].remove(0);
            }
        }
    }

    /// Calculate load imbalance metric
    pub fn calculate_load_imbalance(&self) -> f32 {
        if self.devices.is_empty() {
            return 0.0;
        }
        let loads: Vec<f32> = self.devices.iter().map(|d| d.compute_load).collect();
        let mean_load = loads.iter().sum::<f32>() / loads.len() as f32;
        if mean_load == 0.0 {
            return 0.0;
        }
        let variance = loads
            .iter()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f32>()
            / loads.len() as f32;
        variance.sqrt() / mean_load
    }

    /// Calculate scaling efficiency
    pub fn calculate_scaling_efficiency(&self) -> f64 {
        let n_gpus = self.devices.len() as f64;
        let avg_utilization = self
            .devices
            .iter()
            .map(|d| d.compute_load as f64)
            .sum::<f64>()
            / n_gpus;
        avg_utilization
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            num_gpus: self.devices.len(),
            load_imbalance: self.calculate_load_imbalance(),
            scaling_efficiency: self.calculate_scaling_efficiency(),
            communication_overhead: self.performance_monitor.communication_overhead,
            average_utilization: self.devices.iter().map(|d| d.compute_load).sum::<f32>()
                / self.devices.len() as f32,
        }
    }

    /// Handle GPU failure
    pub fn handle_gpu_failure(&mut self, failed_gpu_id: usize) -> KwaversResult<()> {
        if failed_gpu_id >= self.devices.len() {
            return Ok(());
        }
        self.devices[failed_gpu_id].healthy = false;

        if self.fault_tolerance.graceful_degradation {
            let failed_work: Vec<_> = self
                .active_work
                .values()
                .filter(|work| work.device_id == failed_gpu_id)
                .cloned()
                .collect();

            for work in failed_work {
                self.work_queue.push_back(work);
            }

            self.active_work
                .retain(|_, work| work.device_id != failed_gpu_id);

            self.reassign_work_after_failure()?;
        }

        Ok(())
    }

    fn reassign_work_after_failure(&mut self) -> KwaversResult<()> {
        let healthy_gpus: Vec<usize> = self
            .devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy_gpus.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "No healthy GPUs remaining".to_string(),
                },
            ));
        }

        while let Some(work) = self.work_queue.pop_front() {
            let best_gpu = healthy_gpus[self.work_queue.len() % healthy_gpus.len()];
            let mut reassigned_work = work;
            reassigned_work.device_id = best_gpu;
            self.active_work.insert(reassigned_work.id, reassigned_work);
        }

        Ok(())
    }

    /// Get device information
    pub fn get_devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Get active work assignments
    pub fn get_active_work(&self) -> &HashMap<usize, WorkUnit> {
        &self.active_work
    }

    /// Check if a GPU is healthy and available
    pub fn is_gpu_healthy(&self, gpu_id: usize) -> bool {
        gpu_id < self.devices.len() && self.devices[gpu_id].healthy
    }
}
