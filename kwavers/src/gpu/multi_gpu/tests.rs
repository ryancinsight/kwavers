//! Tests for multi-GPU context management.

use super::context::MultiGpuContext;
use super::types::{CommunicationChannel, GpuAffinity, PendingTransfer, TransferStatus};

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
    let mut channel = CommunicationChannel {
        bandwidth: 50.0,
        latency: 5.0,
        supports_p2p: true,
        transfer_queue: Vec::new(),
    };

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

    assert_eq!(channel.transfer_queue[0].priority, 5);
    assert_eq!(channel.transfer_queue[1].priority, 1);
}
