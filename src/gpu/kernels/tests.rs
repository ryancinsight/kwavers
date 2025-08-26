//! Tests for GPU kernels

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::gpu::GpuBackend;
    use crate::grid::Grid;

    #[test]
    fn test_kernel_manager_creation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Level2);
        assert_eq!(manager.backend, GpuBackend::Cuda);
        assert_eq!(manager.optimization_level, OptimizationLevel::Level2);
        assert!(manager.kernels.is_empty());
    }

    #[test]
    fn test_kernel_compilation() {
        let mut manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Level1);
        let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1);

        let result = manager.compile_kernels(&grid);
        assert!(result.is_ok());
        assert_eq!(manager.kernels.len(), 5); // Should compile 5 kernel types
    }

    #[test]
    fn test_acoustic_kernel_generation() {
        let config = KernelConfig::default();
        let kernel = acoustic::AcousticKernel::new(config);
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1);

        let cuda_source = kernel.generate_cuda(&grid);
        assert!(cuda_source.contains("acoustic_wave_kernel"));
        assert!(cuda_source.contains("divergence"));

        let opencl_source = kernel.generate_opencl(&grid);
        assert!(opencl_source.contains("__kernel"));
        assert!(opencl_source.contains("acoustic_wave_kernel"));

        let wgsl_source = kernel.generate_wgsl(&grid);
        assert!(wgsl_source.contains("@compute"));
        assert!(wgsl_source.contains("acoustic_wave"));
    }

    #[test]
    fn test_thermal_kernel_generation() {
        let config = KernelConfig::default();
        let kernel = thermal::ThermalKernel::new(config);
        let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1);

        let cuda_source = kernel.generate_cuda(&grid);
        assert!(cuda_source.contains("thermal_diffusion_kernel"));
        assert!(cuda_source.contains("laplacian"));
        assert!(cuda_source.contains("temperature"));

        let opencl_source = kernel.generate_opencl(&grid);
        assert!(opencl_source.contains("thermal_diffusion_kernel"));

        let wgsl_source = kernel.generate_wgsl(&grid);
        assert!(wgsl_source.contains("thermal_diffusion"));
    }

    #[test]
    fn test_fft_kernel_generation() {
        let config = KernelConfig::default();
        let kernel = transforms::FFTKernel::new(config, transforms::TransformDirection::Forward);
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1);

        let cuda_source = kernel.generate_cuda(&grid);
        assert!(cuda_source.contains("fft_kernel"));
        assert!(cuda_source.contains("Cooley-Tukey"));

        let opencl_source = kernel.generate_opencl(&grid);
        assert!(opencl_source.contains("fft_kernel"));

        let wgsl_source = kernel.generate_wgsl(&grid);
        assert!(wgsl_source.contains("Complex"));
    }

    #[test]
    fn test_boundary_kernel_generation() {
        let config = KernelConfig::default();
        let kernel = boundary::BoundaryKernel::new(config);
        let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1);

        let cuda_source = kernel.generate_cuda(&grid);
        assert!(cuda_source.contains("boundary_kernel"));
        assert!(cuda_source.contains("Dirichlet"));
        assert!(cuda_source.contains("Neumann"));

        let opencl_source = kernel.generate_opencl(&grid);
        assert!(opencl_source.contains("boundary_kernel"));

        let wgsl_source = kernel.generate_wgsl(&grid);
        assert!(wgsl_source.contains("BoundaryParams"));
    }

    #[test]
    fn test_optimization_levels() {
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1);

        for level in vec![
            OptimizationLevel::Level1,
            OptimizationLevel::Level2,
            OptimizationLevel::Level3,
        ] {
            let manager = KernelManager::new(GpuBackend::Cuda, level);
            assert_eq!(manager.optimization_level, level);

            let block_size = manager.calculate_block_size(&grid);
            match level {
                OptimizationLevel::Level1 => assert_eq!(block_size, (8, 8, 8)),
                OptimizationLevel::Level2 => assert_eq!(block_size, (16, 16, 4)),
                OptimizationLevel::Level3 => {
                    assert!(block_size.0 >= 4);
                    assert!(block_size.1 >= 4);
                    assert!(block_size.2 >= 1);
                }
            }
        }
    }

    #[test]
    fn test_kernel_type_enum() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(KernelType::AcousticWave, "acoustic");
        map.insert(KernelType::ThermalDiffusion, "thermal");

        assert_eq!(map.get(&KernelType::AcousticWave), Some(&"acoustic"));
        assert_eq!(map.get(&KernelType::ThermalDiffusion), Some(&"thermal"));
        assert_ne!(KernelType::AcousticWave, KernelType::ThermalDiffusion);
    }

    #[test]
    fn test_performance_metrics_update() {
        use crate::gpu::GpuPerformanceMetrics;

        let mut manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Level2);
        let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1);

        // Compile kernels first
        let _ = manager.compile_kernels(&grid);

        let metrics = GpuPerformanceMetrics::new(
            1_000_000, // 1M grid points
            10.0,      // 10ms kernel time
            5.0,       // 5ms transfer time
            500.0,     // 500 GB/s bandwidth
            0.1,       // 0.1 GB data
        );

        manager.update_performance_metrics(KernelType::AcousticWave, metrics.clone());

        let summary = manager.get_performance_summary();
        assert!(summary.contains_key(&KernelType::AcousticWave));

        if let Some(Some(retrieved)) = summary.get(&KernelType::AcousticWave) {
            assert_eq!(retrieved.kernel_execution_time_ms, 10.0);
            assert_eq!(retrieved.data_transfer_time_ms, 5.0);
        }
    }
}
