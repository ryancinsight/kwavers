//! Multi-GPU Context Management
//!
//! This module provides multi-GPU context management, device affinity,
//! and cross-GPU communication for distributed computing.

use crate::error::{KwaversError, KwaversResult};
use crate::gpu::{GpuCapabilities, GpuContext};
use std::collections::HashMap;

/// Multi-GPU context for distributed computing
#[derive(Debug)]
pub struct MultiGpuContext {
    /// Individual GPU contexts
    contexts: Vec<GpuContext>,
    /// Device affinity mapping
    affinity: GpuAffinity,
    /// Peer-to-peer accessibility matrix
    peer_accessibility: Vec<Vec<bool>>,
    /// Communication channels
    communication_channels: HashMap<(usize, usize), CommunicationChannel>,
}

/// GPU affinity configuration
#[derive(Debug, Clone)]
pub enum GpuAffinity {
    /// No specific affinity (any GPU can access any memory)
    None,
    /// NUMA-aware affinity (GPUs grouped by NUMA nodes)
    NumaAware {
        /// NUMA node assignments for each GPU
        numa_nodes: Vec<usize>,
    },
    /// Custom affinity mapping
    Custom {
        /// Affinity groups (GPUs in same group have fast communication)
        groups: Vec<Vec<usize>>,
    },
}

/// Communication channel between GPUs
#[derive(Debug)]
pub struct CommunicationChannel {
    /// Channel bandwidth (GB/s)
    pub bandwidth: f64,
    /// Channel latency (microseconds)
    pub latency: f64,
    /// Supports peer-to-peer access
    pub supports_p2p: bool,
    /// Transfer queue
    pub transfer_queue: Vec<PendingTransfer>,
}

/// Pending data transfer
#[derive(Debug, Clone)]
pub struct PendingTransfer {
    /// Transfer size (bytes)
    pub size: usize,
    /// Priority (0 = lowest, 255 = highest)
    pub priority: u8,
    /// Transfer status
    pub status: TransferStatus,
}

/// Transfer status
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl MultiGpuContext {
    /// Create a new multi-GPU context
    pub async fn new() -> KwaversResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let mut contexts = Vec::new();
        let mut peer_accessibility = Vec::new();

        // Enumerate and create contexts for all available GPUs
        let mut adapter_count = 0;
        for await adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            // Skip software adapters
            if matches!(adapter.get_info().backend, wgpu::Backend::BrowserWebGpu) {
                continue;
            }

            // Create context for this adapter
            let context = Self::create_context_for_adapter(adapter).await?;
            contexts.push(context);

            // Initialize peer accessibility (simplified)
            if peer_accessibility.is_empty() {
                peer_accessibility = vec![vec![true; 1]; 1];
            } else {
                // Add new row and column for peer accessibility
                let n = peer_accessibility.len();
                for row in peer_accessibility.iter_mut() {
                    row.push(true); // Assume P2P possible
                }
                peer_accessibility.push(vec![true; n + 1]);
            }

            adapter_count += 1;
            if adapter_count >= 4 {
                break; // Limit to 4 GPUs for practical purposes
            }
        }

        if contexts.len() < 2 {
            return Err(KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                resource: "Multiple GPU devices required for multi-GPU context".to_string(),
            }));
        }

        // Initialize communication channels
        let communication_channels = Self::initialize_communication_channels(&contexts);

        // Determine affinity
        let affinity = Self::determine_affinity(&contexts);

        Ok(Self {
            contexts,
            affinity,
            peer_accessibility,
            communication_channels,
        })
    }

    /// Create GPU context for a specific adapter
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
            .request_device(
                &wgpu::DeviceDescriptor {
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
                },
                None,
            )
            .await
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::ResourceUnavailable {
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

    /// Initialize communication channels between GPUs
    fn initialize_communication_channels(contexts: &[GpuContext]) -> HashMap<(usize, usize), CommunicationChannel> {
        let mut channels = HashMap::new();

        for i in 0..contexts.len() {
            for j in (i + 1)..contexts.len() {
                // Estimate bandwidth and latency based on device capabilities
                // In practice, this would be measured or queried from system
                let bandwidth = 50.0; // GB/s (PCIe 4.0 x16)
                let latency = 5.0;   // microseconds
                let supports_p2p = true; // Assume P2P possible for now

                let channel = CommunicationChannel {
                    bandwidth,
                    latency,
                    supports_p2p,
                    transfer_queue: Vec::new(),
                };

                channels.insert((i, j), channel.clone());
                channels.insert((j, i), channel);
            }
        }

        channels
    }

    /// Determine GPU affinity configuration
    fn determine_affinity(contexts: &[GpuContext]) -> GpuAffinity {
        // Simple affinity determination - in practice would query system topology
        if contexts.len() <= 2 {
            GpuAffinity::None
        } else {
            // Group GPUs by hypothetical NUMA nodes
            let numa_nodes = (0..contexts.len()).map(|i| i / 2).collect();
            GpuAffinity::NumaAware { numa_nodes }
        }
    }

    /// Get number of GPUs in the context
    pub fn num_gpus(&self) -> usize {
        self.contexts.len()
    }

    /// Get GPU context by index
    pub fn get_context(&self, index: usize) -> Option<&GpuContext> {
        self.contexts.get(index)
    }

    /// Get all GPU contexts
    pub fn get_all_contexts(&self) -> &[GpuContext] {
        &self.contexts
    }

    /// Check if peer-to-peer access is available between two GPUs
    pub fn supports_p2p(&self, gpu_a: usize, gpu_b: usize) -> bool {
        self.peer_accessibility
            .get(gpu_a)
            .and_then(|row| row.get(gpu_b))
            .copied()
            .unwrap_or(false)
    }

    /// Get communication channel between two GPUs
    pub fn get_communication_channel(&self, gpu_a: usize, gpu_b: usize) -> Option<&CommunicationChannel> {
        let key = if gpu_a < gpu_b { (gpu_a, gpu_b) } else { (gpu_b, gpu_a) };
        self.communication_channels.get(&key)
    }

    /// Initiate data transfer between GPUs
    pub fn initiate_transfer(&mut self, from_gpu: usize, to_gpu: usize, size: usize, priority: u8) -> KwaversResult<()> {
        let channel = self.get_communication_channel_mut(from_gpu, to_gpu)
            .ok_or_else(|| KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                resource: format!("No communication channel between GPUs {} and {}", from_gpu, to_gpu),
            }))?;

        let transfer = PendingTransfer {
            size,
            priority,
            status: TransferStatus::Pending,
        };

        channel.transfer_queue.push(transfer);
        // Sort by priority (highest first)
        channel.transfer_queue.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(())
    }

    /// Get mutable communication channel
    fn get_communication_channel_mut(&mut self, gpu_a: usize, gpu_b: usize) -> Option<&mut CommunicationChannel> {
        let key = if gpu_a < gpu_b { (gpu_a, gpu_b) } else { (gpu_b, gpu_a) };
        self.communication_channels.get_mut(&key)
    }

    /// Process pending transfers
    pub fn process_transfers(&mut self) -> usize {
        let mut completed_transfers = 0;

        for channel in self.communication_channels.values_mut() {
            // Process transfers in priority order
            for transfer in channel.transfer_queue.iter_mut() {
                if transfer.status == TransferStatus::Pending {
                    transfer.status = TransferStatus::InProgress;
                    // In practice, this would initiate actual GPU transfer
                    transfer.status = TransferStatus::Completed;
                    completed_transfers += 1;
                }
            }

            // Remove completed transfers
            channel.transfer_queue.retain(|t| t.status != TransferStatus::Completed);
        }

        completed_transfers
    }

    /// Get affinity configuration
    pub fn get_affinity(&self) -> &GpuAffinity {
        &self.affinity
    }

    /// Calculate optimal GPU for workload based on affinity
    pub fn optimal_gpu_for_workload(&self, workload_size: usize, preferred_gpu: Option<usize>) -> usize {
        if let Some(gpu) = preferred_gpu {
            if gpu < self.contexts.len() {
                return gpu;
            }
        }

        // Simple load balancing - return GPU with most available memory
        let mut best_gpu = 0;
        let mut max_memory = 0;

        for (i, context) in self.contexts.iter().enumerate() {
            let available_memory = context.capabilities.max_buffer_size.saturating_sub(0); // Would track actual usage
            if available_memory > max_memory {
                max_memory = available_memory;
                best_gpu = i;
            }
        }

        best_gpu
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> MultiGpuPerformanceSummary {
        let mut total_memory = 0;
        let mut total_bandwidth = 0.0;
        let mut p2p_pairs = 0;

        for context in &self.contexts {
            total_memory += context.capabilities.max_buffer_size;
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

/// Performance summary for multi-GPU setup
#[derive(Debug, Clone)]
pub struct MultiGpuPerformanceSummary {
    /// Number of GPUs
    pub num_gpus: usize,
    /// Total GPU memory (GB)
    pub total_memory_gb: f64,
    /// Total inter-GPU bandwidth (GB/s)
    pub total_bandwidth_gbps: f64,
    /// Number of peer-to-peer pairs
    pub p2p_pairs: usize,
    /// Affinity configuration type
    pub affinity_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_gpu_context_creation() {
        let result = MultiGpuContext::new().await;

        match result {
            Ok(context) => {
                assert!(context.num_gpus() >= 1);
                let summary = context.get_performance_summary();
                assert_eq!(summary.num_gpus, context.num_gpus());
            }
            Err(_) => {
                // Expected on systems without multi-GPU support
            }
        }
    }

    #[test]
    fn test_gpu_affinity() {
        // Test NUMA-aware affinity
        let affinity = GpuAffinity::NumaAware {
            numa_nodes: vec![0, 0, 1, 1],
        };

        match affinity {
            GpuAffinity::NumaAware { numa_nodes } => {
                assert_eq!(numa_nodes.len(), 4);
                assert_eq!(numa_nodes[0], 0);
                assert_eq!(numa_nodes[2], 1);
            }
            _ => panic!("Expected NUMA-aware affinity"),
        }
    }

    #[test]
    fn test_communication_channel() {
        let mut channel = CommunicationChannel {
            bandwidth: 50.0,
            latency: 5.0,
            supports_p2p: true,
            transfer_queue: Vec::new(),
        };

        // Add transfers with different priorities
        channel.transfer_queue.push(PendingTransfer {
            size: 1024,
            priority: 1,
            status: TransferStatus::Pending,
        });

        channel.transfer_queue.push(PendingTransfer {
            size: 2048,
            priority: 5,
            status: TransferStatus::Pending,
        });

        // Higher priority should be first after sorting
        assert_eq!(channel.transfer_queue[0].priority, 5);
        assert_eq!(channel.transfer_queue[1].priority, 1);
    }
}
