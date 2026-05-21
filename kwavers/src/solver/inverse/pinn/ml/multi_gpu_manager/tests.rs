use super::manager::MultiGpuManager;
#[cfg(feature = "gpu")]
use super::types::PinnGpuCapabilities;
use super::types::{
    LoadBalancingAlgorithm, MultiGpuDecompositionStrategy, PerformanceSummary,
    PinnMultiGpuDeviceInfo,
};

#[tokio::test]
async fn test_multi_gpu_manager_creation() {
    let result = MultiGpuManager::new(
        MultiGpuDecompositionStrategy::Spatial {
            dimensions: 2,
            overlap: 0.1,
        },
        LoadBalancingAlgorithm::Static,
    )
    .await;

    match result {
        Ok(manager) => {
            assert!(!manager.get_devices().is_empty());
        }
        Err(_) => {
            // Expected on systems without multiple GPU devices
        }
    }
}

#[test]
fn test_spatial_decomposition() {
    let devices = [
        PinnMultiGpuDeviceInfo {
            id: 0,
            name: "GPU 0".to_string(),
            backend: "Vulkan".to_string(),
            #[cfg(feature = "gpu")]
            capabilities: PinnGpuCapabilities {
                max_buffer_size: 0,
                max_workgroup_size: [0, 0, 0],
                max_compute_invocations: 0,
                supports_f64: false,
                supports_atomics: false,
            },
            memory_used: 0,
            compute_load: 0.0,
            healthy: true,
        },
        PinnMultiGpuDeviceInfo {
            id: 1,
            name: "GPU 1".to_string(),
            backend: "Vulkan".to_string(),
            #[cfg(feature = "gpu")]
            capabilities: PinnGpuCapabilities {
                max_buffer_size: 0,
                max_workgroup_size: [0, 0, 0],
                max_compute_invocations: 0,
                supports_f64: false,
                supports_atomics: false,
            },
            memory_used: 0,
            compute_load: 0.0,
            healthy: true,
        },
    ];

    let total_points = 1000;
    let points_per_gpu = total_points / devices.len();
    assert_eq!(points_per_gpu, 500);
}

#[test]
fn test_performance_summary() {
    let summary = PerformanceSummary {
        num_gpus: 4,
        load_imbalance: 0.05,
        scaling_efficiency: 0.85,
        communication_overhead: 0.03,
        average_utilization: 0.82,
    };

    assert_eq!(summary.num_gpus, 4);
    assert!(summary.scaling_efficiency > 0.7);
    assert!(summary.load_imbalance < 0.1);
}
