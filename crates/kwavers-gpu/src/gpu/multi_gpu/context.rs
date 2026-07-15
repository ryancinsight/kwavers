//! Multi-GPU context management.

use super::types::{
    GpuAffinity, GpuCommunicationChannel, GpuTransferStatus, MultiGpuPerformanceSummary,
    PendingTransfer,
};
use crate::gpu::{CoreGpuContext, GpuDeviceProvider};
use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Multi-GPU context for distributed computing.
///
/// `P` is the concrete Hephaestus GPU provider. The default preserves the
/// current WGPU/WGSL implementation, while CUDA can use the same topology and
/// scheduling surface once selected at the type level.
#[derive(Debug)]
pub struct MultiGpuContext<P = WgpuDevice>
where
    P: GpuDeviceProvider,
{
    contexts: Vec<CoreGpuContext<P>>,
    affinity: GpuAffinity,
    peer_accessibility: Vec<Vec<bool>>,
    communication_channels: HashMap<(usize, usize), GpuCommunicationChannel>,
}

impl<P> MultiGpuContext<P>
where
    P: GpuDeviceProvider,
{
    #[cfg(test)]
    pub(super) fn with_test_channels(
        communication_channels: HashMap<(usize, usize), GpuCommunicationChannel>,
    ) -> Self {
        Self {
            contexts: Vec::new(),
            affinity: GpuAffinity::None,
            peer_accessibility: vec![vec![true; 2]; 2],
            communication_channels,
        }
    }

    /// Create a provider-generic multi-GPU context with explicit acquisition
    /// requirements.
    ///
    /// This constructor is the generic WGPU/CUDA seam. It binds only to the
    /// Hephaestus acquisition trait exposed through [`GpuDeviceProvider`], so a
    /// provider either returns real acquired devices or reports an acquisition
    /// error.
    ///
    /// # Errors
    ///
    /// Returns a system resource error when provider discovery fails or fewer
    /// than two matching devices are available.
    ///
    // The constructor remains async to preserve the existing public API even
    // though Hephaestus owns the blocking provider discovery path.
    #[allow(clippy::unused_async)]
    pub async fn new_with_requirements(
        label_prefix: &str,
        max_devices: usize,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> KwaversResult<Self> {
        let providers = P::try_acquire_devices(
            label_prefix,
            max_devices,
            device_preference,
            optional_features,
            required_limits,
        )
        .map_err(|e| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                resource: format!(
                    "{:?} GPU device initialization failed: {e}",
                    P::provider_kind()
                ),
            })
        })?;

        let contexts: Vec<_> = providers
            .into_iter()
            .map(CoreGpuContext::<P>::from_provider)
            .collect();

        if contexts.len() < 2 {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "Multiple {:?} GPU devices required for multi-GPU context",
                        P::provider_kind()
                    ),
                },
            ));
        }

        let peer_accessibility = vec![vec![true; contexts.len()]; contexts.len()];
        let communication_channels = Self::initialize_communication_channels(&contexts);
        let affinity = Self::determine_affinity(&contexts);

        Ok(Self {
            contexts,
            affinity,
            peer_accessibility,
            communication_channels,
        })
    }

    fn initialize_communication_channels(
        contexts: &[CoreGpuContext<P>],
    ) -> HashMap<(usize, usize), GpuCommunicationChannel> {
        let mut channels = HashMap::new();

        for i in 0..contexts.len() {
            for j in (i + 1)..contexts.len() {
                let channel = GpuCommunicationChannel {
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

    fn determine_affinity(contexts: &[CoreGpuContext<P>]) -> GpuAffinity {
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
    pub fn get_context(&self, index: usize) -> Option<&CoreGpuContext<P>> {
        self.contexts.get(index)
    }

    /// Get all GPU contexts.
    pub fn get_all_contexts(&self) -> &[CoreGpuContext<P>] {
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
    ) -> Option<&GpuCommunicationChannel> {
        let key = if gpu_a < gpu_b {
            (gpu_a, gpu_b)
        } else {
            (gpu_b, gpu_a)
        };
        self.communication_channels.get(&key)
    }

    /// Initiate data transfer between GPUs.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "No communication channel between GPUs {} and {}",
                        from_gpu, to_gpu
                    ),
                })
            })?;

        let transfer = PendingTransfer {
            size,
            priority,
            status: GpuTransferStatus::Pending,
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
    ) -> Option<&mut GpuCommunicationChannel> {
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
                if transfer.status == GpuTransferStatus::Pending {
                    transfer.status = GpuTransferStatus::InProgress;
                    transfer.status = GpuTransferStatus::Completed;
                    completed_transfers += 1;
                }
            }
            channel
                .transfer_queue
                .retain(|t| t.status != GpuTransferStatus::Completed);
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

impl MultiGpuContext<WgpuDevice> {
    /// Create a new WGPU multi-GPU context for the current WGSL kernels.
    ///
    /// # Errors
    ///
    /// Returns a system resource error when WGPU discovery fails or fewer than
    /// two matching devices are available.
    pub async fn new() -> KwaversResult<Self> {
        Self::new_with_requirements(
            "Multi-GPU Device",
            4,
            DevicePreference::HighPerformance,
            &[
                DeviceFeature::MappablePrimaryBuffers,
                DeviceFeature::ImmediateData,
            ],
            CoreGpuContext::required_limits(),
        )
        .await
    }
}
