//! Tests for multi-GPU context management.

use super::context::MultiGpuContext;
use super::types::{GpuAffinity, GpuCommunicationChannel};
use std::collections::HashMap;

fn assert_multi_gpu_context_accepts_provider<P>()
where
    P: crate::gpu::GpuDeviceProvider,
{
    let _ = core::mem::size_of::<MultiGpuContext<P>>();
}

#[test]
fn multi_gpu_context_is_provider_generic() {
    assert_multi_gpu_context_accepts_provider::<hephaestus_wgpu::WgpuDevice>();
}

#[cfg(feature = "cuda-provider")]
#[test]
fn multi_gpu_context_accepts_cuda_provider() {
    assert_multi_gpu_context_accepts_provider::<hephaestus_cuda::CudaDevice>();
}

#[test]
fn test_multi_gpu_context_creation() {
    let result = pollster::block_on(MultiGpuContext::new());

    match result {
        Ok(context) => {
            assert!(context.num_gpus() >= 1);
            let summary = context.get_performance_summary();
            assert_eq!(summary.num_gpus, context.num_gpus());
        }
        Err(_) => {
            // Expected on systems without multi-GPU support.
        }
    }
}

#[test]
fn test_gpu_affinity() {
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
    let mut channels = HashMap::new();
    channels.insert(
        (0, 1),
        GpuCommunicationChannel {
            bandwidth: 50.0,
            latency: 5.0,
            supports_p2p: true,
            transfer_queue: Vec::new(),
        },
    );

    let mut context = MultiGpuContext::<hephaestus_wgpu::WgpuDevice>::with_test_channels(channels);
    context
        .initiate_transfer(0, 1, 1024, 1)
        .expect("invariant: test channel exists for GPU pair 0 -> 1");
    context
        .initiate_transfer(0, 1, 2048, 5)
        .expect("invariant: test channel exists for GPU pair 0 -> 1");

    let channel = context
        .get_communication_channel(0, 1)
        .expect("invariant: test channel exists for GPU pair 0 -> 1");
    assert_eq!(channel.transfer_queue[0].priority, 5);
    assert_eq!(channel.transfer_queue[0].size, 2048);
    assert_eq!(channel.transfer_queue[1].priority, 1);
    assert_eq!(channel.transfer_queue[1].size, 1024);
}
