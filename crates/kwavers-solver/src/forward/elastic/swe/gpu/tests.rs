use super::adaptive::AdaptiveResolution;
use super::device::GPUDevice;
use super::memory::GPUMemoryPool;
use super::metrics::SweGpuStepMetrics;
use super::solver::GPUElasticWaveSolver3D;
use kwavers_domain::grid::Grid;
use ndarray::Array3;

#[test]
fn test_gpu_device_capabilities() {
    let device = GPUDevice {
        name: "Test GPU".to_string(),
        global_memory: 8 * 1024 * 1024 * 1024,
        shared_memory: 48 * 1024,
        max_threads_per_block: 1024,
        max_grid_dims: [2147483647, 65535, 65535],
        compute_capability: (7, 5),
        memory_bandwidth: 448.0,
    };

    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
    assert!(device.can_handle_volume(&grid));

    let block_size = device.optimal_block_size([100, 100, 100]);
    assert!(block_size.iter().all(|&x| x > 0));
    assert!(block_size.iter().product::<usize>() <= device.max_threads_per_block);
}

#[test]
fn test_memory_pool_allocation() {
    let mut pool = GPUMemoryPool::new(1024 * 1024 * 1024, 256);

    let block1 = pool.allocate(1024).unwrap();
    let _block2 = pool.allocate(2048).unwrap();

    let stats = pool.memory_stats();
    assert_eq!(stats.total_blocks, 2);
    assert!(stats.total_allocated >= 1024 + 2048);

    pool.free(block1);
    let stats_after = pool.memory_stats();
    assert_eq!(stats_after.total_blocks, 1);
    assert!(stats_after.total_allocated < stats.total_allocated);
}

#[test]
fn test_gpu_solver_initialization() {
    let device = GPUDevice {
        name: "Test GPU".to_string(),
        global_memory: 8 * 1024 * 1024 * 1024,
        shared_memory: 48 * 1024,
        max_threads_per_block: 1024,
        max_grid_dims: [2147483647, 65535, 65535],
        compute_capability: (7, 5),
        memory_bandwidth: 448.0,
    };

    let mut solver = GPUElasticWaveSolver3D::new(device).unwrap();
    solver.initialize_kernels().unwrap();
    assert!(!solver.kernel_cache.is_empty());
}

#[test]
fn test_adaptive_resolution() {
    let base_grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let adaptive = AdaptiveResolution::new(&base_grid, 3);

    assert_eq!(adaptive.resolution_levels.len(), 3);
    assert!(
        adaptive.resolution_levels[0].scale_factor <= adaptive.resolution_levels[1].scale_factor
    );

    let initial_disp = Array3::zeros((64, 64, 64));
    let solution = adaptive.adaptive_solve(&initial_disp, 0.85).unwrap();
    assert!(!solution.steps.is_empty());
    assert!(solution.final_quality > 0.0);
}

#[test]
fn test_performance_metrics() {
    let metrics = SweGpuStepMetrics {
        total_kernel_time: 1.0,
        total_execution_time: 1.5,
        kernels_executed: 10,
        ..SweGpuStepMetrics::default()
    };

    let stats = metrics.statistics();
    assert!(stats.average_kernel_time > 0.0);
    assert!(stats.kernel_efficiency > 0.0 && stats.kernel_efficiency <= 1.0);
    assert!(stats.total_throughput > 0.0);
}
