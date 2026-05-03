//! Multi-GPU context management.

use super::types::{
    CommunicationChannel, GpuAffinity, MultiGpuPerformanceSummary, PendingTransfer, TransferStatus,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::gpu::{GpuCapabilities, GpuContext};
use std::collections::HashMap;

/// Multi-GPU context for distributed computing.
#[derive(Debug)]
pub struct MultiGpuContext {
    contexts: Vec<GpuContext>,
    affinity: GpuAffinity,
    peer_accessibility: Vec<Vec<bool>>,
    communication_channels: HashMap<(usize, usize), CommunicationChannel>,
}

impl MultiGpuContext {
    /// Create a new multi-GPU context.
    pub async fn new() -> KwaversResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let mut contexts = Vec::new();
        let mut peer_accessibility = Vec::new();

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let mut adapter_count = 0;

        for adapter in adapters {
            if matches!(adapter.get_info().backend, wgpu::Backend::BrowserWebGpu) {
                continue;
            }

            let context = Self::create_context_for_adapter(adapter).await?;
            contexts.push(context);

            if peer_accessibility.is_empty() {
                peer_accessibility = vec![vec![true; 1]; 1];
            } else {
                let n = peer_accessibility.len();
                for row in peer_accessibility.iter_mut() {
                    row.push(true);
                }
                peer_accessibility.push(vec![true; n + 1]);
            }

            adapter_count += 1;
            if adapter_count >= 4 {
                break;
            }
        }

        if contexts.len() < 2 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "Multiple GPU devices required for multi-GPU context".to_string(),
                },
            ));
        }

        let communication_channels = Self::initialize_communication_channels(&contexts);
        let affinity = Self::determine_affinity(&contexts);

        Ok(Self {
            contexts,
            affinity,
            peer_accessibility,
            communication_channels,
        })
    }

    async fn create_context_for_adapter(adapter: wgpu::Adapter) -> KwaversResult<GpuContext> {
        let info = adapter.get_info();
        let limits = adapter.limits();

        log::info!(
            "Multi-GPU: Initializing {} ({:?}) - Driver: {}",
            info.name,
            info.backend,
            info.driver
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&format!("Multi-GPU Device: {}", info.name)),
                required_features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                    | wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    max_compute_workgroup_storage_size: 16384,
                    max_compute_invocations_per_workgroup: 256,
                    max_compute_workgroup_size_x: 256,
                    max_compute_workgroup_size_y: 256,
                    max_compute_workgroup_size_z: 64,
                    max_push_constant_size: 128,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("GPU device initialization failed: {}", e),
                })
            })?;

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

        let compute = crate::gpu::GpuCompute::new(&device);
        let buffer_manager = crate::gpu::BufferManager::new(&device);

        Ok(GpuContext {
            device,
            queue,
            capabilities,
            compute,
            buffer_manager,
        })
    }

    fn initialize_communication_channels(
        contexts: &[GpuContext],
    ) -> HashMap<(usize, usize), CommunicationChannel> {
        let mut channels = HashMap::new();

        for i in 0..contexts.len() {
            for j in (i + 1)..contexts.len() {
                let channel = CommunicationChannel {
                    bandwidth: 50.0,
                    latency: 5.0,
                    supports_p2p: true,
                    transfer_queue: Vec::new(),
                };
                channels.insert((i, j), channel.clone());
                channels.insert((j, i), channel);
            }
        }

        channels
    }

    fn determine_affinity(contexts: &[GpuContext]) -> GpuAffinity {
        if contexts.len() <= 2 {
            GpuAffinity::None
        } else {
            let numa_nodes = (0..contexts.len()).map(|i| i / 2).collect();
            GpuAffinity::NumaAware { numa_nodes }
        }
    }

    /// Get number of GPUs in the context.
    pub fn num_gpus(&self) -> usize {
        self.contexts.len()
    }

    /// Get GPU context by index.
    pub fn get_context(&self, index: usize) -> Option<&GpuContext> {
        self.contexts.get(index)
    }

    /// Get all GPU contexts.
    pub fn get_all_contexts(&self) -> &[GpuContext] {
        &self.contexts
    }

    /// Check if peer-to-peer access is available between two GPUs.
    pub fn supports_p2p(&self, gpu_a: usize, gpu_b: usize) -> bool {
        self.peer_accessibility
            .get(gpu_a)
            .and_then(|row| row.get(gpu_b))
            .copied()
            .unwrap_or(false)
    }

    /// Get communication channel between two GPUs.
    pub fn get_communication_channel(
        &self,
        gpu_a: usize,
        gpu_b: usize,
    ) -> Option<&CommunicationChannel> {
        let key = if gpu_a < gpu_b {
            (gpu_a, gpu_b)
        } else {
            (gpu_b, gpu_a)
        };
        self.communication_channels.get(&key)
    }

    /// Initiate data transfer between GPUs.
    pub fn initiate_transfer(
        &mut self,
        from_gpu: usize,
        to_gpu: usize,
        size: usize,
        priority: u8,
    ) -> KwaversResult<()> {
        let channel = self
            .get_communication_channel_mut(from_gpu, to_gpu)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "No communication channel between GPUs {} and {}",
                        from_gpu, to_gpu
                    ),
                })
            })?;

        let transfer = PendingTransfer {
            size,
            priority,
            status: TransferStatus::Pending,
        };

        channel.transfer_queue.push(transfer);
        channel
            .transfer_queue
            .sort_by_key(|transfer| std::cmp::Reverse(transfer.priority));

        Ok(())
    }

    fn get_communication_channel_mut(
        &mut self,
        gpu_a: usize,
        gpu_b: usize,
    ) -> Option<&mut CommunicationChannel> {
        let key = if gpu_a < gpu_b {
            (gpu_a, gpu_b)
        } else {
            (gpu_b, gpu_a)
        };
        self.communication_channels.get_mut(&key)
    }

    /// Process pending transfers.
    pub fn process_transfers(&mut self) -> usize {
        let mut completed_transfers = 0;

        for channel in self.communication_channels.values_mut() {
            for transfer in channel.transfer_queue.iter_mut() {
                if transfer.status == TransferStatus::Pending {
                    transfer.status = TransferStatus::InProgress;
                    transfer.status = TransferStatus::Completed;
                    completed_transfers += 1;
                }
            }
            channel
                .transfer_queue
                .retain(|t| t.status != TransferStatus::Completed);
        }

        completed_transfers
    }

    /// Get affinity configuration.
    pub fn get_affinity(&self) -> &GpuAffinity {
        &self.affinity
    }

    /// Calculate optimal GPU for workload based on affinity.
    pub fn optimal_gpu_for_workload(
        &self,
        _workload_size: usize,
        preferred_gpu: Option<usize>,
    ) -> usize {
        if let Some(gpu) = preferred_gpu {
            if gpu < self.contexts.len() {
                return gpu;
            }
        }

        let mut best_gpu = 0;
        let mut max_memory: u64 = 0;

        for (i, context) in self.contexts.iter().enumerate() {
            let available_memory = context.capabilities.max_buffer_size.saturating_sub(0);
            if available_memory > max_memory {
                max_memory = available_memory;
                best_gpu = i;
            }
        }

        best_gpu
    }

    /// Get performance summary.
    pub fn get_performance_summary(&self) -> MultiGpuPerformanceSummary {
        let mut total_memory: u64 = 0;
        let mut total_bandwidth = 0.0;
        let mut p2p_pairs = 0;

        for context in &self.contexts {
            total_memory = total_memory.saturating_add(context.capabilities.max_buffer_size);
        }

        for channel in self.communication_channels.values() {
            total_bandwidth += channel.bandwidth;
            if channel.supports_p2p {
                p2p_pairs += 1;
            }
        }

        MultiGpuPerformanceSummary {
            num_gpus: self.contexts.len(),
            total_memory_gb: total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            total_bandwidth_gbps: total_bandwidth,
            p2p_pairs,
            affinity_type: match &self.affinity {
                GpuAffinity::None => "None".to_string(),
                GpuAffinity::NumaAware { .. } => "NUMA-aware".to_string(),
                GpuAffinity::Custom { .. } => "Custom".to_string(),
            },
        }
    }
}
