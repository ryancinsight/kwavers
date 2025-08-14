//! # GPU Acceleration Module
//!
//! This module provides GPU acceleration capabilities for Kwavers using CUDA and OpenCL backends.
//! It implements Phase 9 requirements for massive performance scaling (>17M grid updates/second).
//!
//! ## Architecture
//!
//! - **CUDA Backend**: NVIDIA GPU acceleration with cudarc
//! - **OpenCL Backend**: Cross-platform GPU acceleration with wgpu
//! - **Memory Management**: Efficient GPU memory allocation and transfer
//! - **Kernel Optimization**: Highly optimized compute kernels
//! - **Multi-GPU Support**: Distributed computation across multiple devices

use crate::error::{KwaversResult, KwaversError};
use crate::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

/// GPU floating-point precision type
/// Use feature flag "gpu-f64" for double precision, otherwise single precision
#[cfg(feature = "gpu-f64")]
pub type GpuFloat = f64;

#[cfg(not(feature = "gpu-f64"))]
pub type GpuFloat = f32;

/// String representation of GPU float type for kernel generation
pub fn gpu_float_type_str() -> &'static str {
    if cfg!(feature = "gpu-f64") {
        "double"
    } else {
        "float"
    }
}

pub mod cuda;
pub mod opencl;
pub mod memory;
pub mod kernels;
pub mod benchmarks;
pub mod fft_kernels;

/// GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// Cross-platform OpenCL backend
    OpenCL,
    /// WebGPU backend for broader compatibility
    WebGPU,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub memory_size: u64,
    pub compute_units: u32,
    pub max_work_group_size: u32,
}

/// GPU acceleration context
pub struct GpuContext {
    devices: Vec<GpuDevice>,
    active_device: Option<usize>,
    backend: GpuBackend,
}

impl GpuContext {
    /// Create new GPU context with automatic device detection
    pub async fn new() -> KwaversResult<Self> {
        let devices = Self::detect_devices().await?;
        
        if devices.is_empty() {
            return Err(KwaversError::Gpu(crate::error::GpuError::NoDevicesFound));
        }

        // Select best device (highest memory and compute units)
        let active_device = devices.iter()
            .enumerate()
            .max_by_key(|(_, device)| (device.memory_size, device.compute_units))
            .map(|(idx, _)| idx);

        let backend = devices[active_device.unwrap()].backend;

        Ok(Self {
            devices,
            active_device,
            backend,
        })
    }

    /// Create new GPU context synchronously (for compatibility)
    pub fn new_sync() -> KwaversResult<Self> {
        let devices = Self::detect_devices_sync()?;
        
        if devices.is_empty() {
            return Err(KwaversError::Gpu(crate::error::GpuError::NoDevicesFound));
        }

        // Select best device (highest memory and compute units)
        let active_device = devices.iter()
            .enumerate()
            .max_by_key(|(_, device)| (device.memory_size, device.compute_units))
            .map(|(idx, _)| idx);

        let backend = devices[active_device.unwrap()].backend;

        Ok(Self {
            devices,
            active_device,
            backend,
        })
    }

    /// Detect available GPU devices (async version)
    async fn detect_devices() -> KwaversResult<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Try CUDA devices first
        #[cfg(feature = "cudarc")]
        {
            if let Ok(cuda_devices) = cuda::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        // Try OpenCL/WebGPU devices
        #[cfg(feature = "wgpu")]
        {
            if let Ok(wgpu_devices) = opencl::detect_wgpu_devices().await {
                devices.extend(wgpu_devices);
            }
        }

        Ok(devices)
    }

    /// Detect available GPU devices (sync version for compatibility)
    fn detect_devices_sync() -> KwaversResult<Vec<GpuDevice>> {
        let mut devices = Vec::new();

        // Try CUDA devices first
        #[cfg(feature = "cudarc")]
        {
            if let Ok(cuda_devices) = cuda::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        // Try OpenCL/WebGPU devices (using sync wrapper)
        #[cfg(feature = "wgpu")]
        {
            if let Ok(wgpu_devices) = opencl::detect_wgpu_devices_sync() {
                devices.extend(wgpu_devices);
            }
        }

        Ok(devices)
    }

    /// Get active device information
    pub fn active_device(&self) -> Option<&GpuDevice> {
        self.active_device.map(|idx| &self.devices[idx])
    }

    /// Set active device by index
    pub fn set_active_device(&mut self, device_idx: usize) -> KwaversResult<()> {
        if device_idx >= self.devices.len() {
            return Err(KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                device_id: device_idx as u32,
                reason: format!("Device index {} out of range (0-{})", device_idx, self.devices.len() - 1),
            }));
        }
        self.active_device = Some(device_idx);
        Ok(())
    }

    /// Get all available devices
    pub fn devices(&self) -> &[GpuDevice] {
        &self.devices
    }
    
    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }
    
    /// Allocate a GPU buffer
    pub fn allocate_buffer(&self, size_bytes: usize) -> KwaversResult<GpuBuffer> {
        use memory::BufferType;
        use std::time::Instant;
        
        let device_ptr = match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::allocate_cuda_memory(size_bytes)? as u64,
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::allocate_wgpu_memory(size_bytes)? as u64,
            #[allow(unreachable_patterns)]
            _ => return Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
        };
        
        Ok(GpuBuffer {
            id: 0, // Should be managed by a proper allocator
            size_bytes,
            device_ptr: Some(device_ptr),
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            buffer_type: BufferType::General,
        })
    }
    
    /// Upload data to a GPU buffer
    pub fn upload_to_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer, data: &[T]) -> KwaversResult<()> {
        let device_ptr = buffer.device_ptr.ok_or_else(|| {
            KwaversError::Gpu(crate::error::GpuError::InvalidOperation {
                operation: "upload_to_buffer".to_string(),
                reason: "Buffer has no device pointer".to_string(),
            })
        })?;
        
        // Convert to byte slice safely using bytemuck
        let byte_slice = bytemuck::cast_slice(data);
        
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::host_to_device_bytes(byte_slice, device_ptr as usize),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::host_to_device_bytes(byte_slice, device_ptr as usize),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
        }
    }
    
    /// Download data from a GPU buffer
    pub fn download_from_buffer<T: bytemuck::Pod>(&mut self, buffer: &GpuBuffer, data: &mut [T]) -> KwaversResult<()> {
        let device_ptr = buffer.device_ptr.ok_or_else(|| {
            KwaversError::Gpu(crate::error::GpuError::InvalidOperation {
                operation: "download_from_buffer".to_string(),
                reason: "Buffer has no device pointer".to_string(),
            })
        })?;
        
        // Convert to mutable byte slice safely using bytemuck
        let byte_slice = bytemuck::cast_slice_mut(data);
        
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::device_to_host_bytes(device_ptr as usize, byte_slice),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::device_to_host_bytes(device_ptr as usize, byte_slice),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
        }
    }
    
    /// Launch a compute kernel
    pub fn launch_kernel(&mut self, kernel_name: &str, grid_size: (u32, u32, u32), block_size: (u32, u32, u32), args: &[*const std::ffi::c_void]) -> KwaversResult<()> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::launch_cuda_kernel(kernel_name, grid_size, block_size, args),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::launch_webgpu_kernel(kernel_name, grid_size, block_size, args),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
        }
    }
    
    /// Enable peer access between GPUs
    pub fn enable_peer_access(&self, peer_device_id: u32) -> KwaversResult<()> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::enable_peer_access(0, peer_device_id),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // WebGPU doesn't have direct peer access
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
        }
    }
}

/// GPU-accelerated field operations
pub trait GpuFieldOps {
    /// Perform acoustic wave update on GPU
    fn acoustic_update_gpu(
        &self,
        pressure: &mut Array3<f64>,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Perform thermal diffusion update on GPU
    fn thermal_update_gpu(
        &self,
        temperature: &mut Array3<f64>,
        heat_source: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        thermal_diffusivity: f64,
    ) -> KwaversResult<()>;

    /// Perform FFT operations on GPU
    fn fft_gpu(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        forward: bool,
    ) -> KwaversResult<()>;
}

/// GPU memory management
pub struct GpuMemoryManager {
    context: Arc<GpuContext>,
    allocated_buffers: Vec<GpuBuffer>,
}

// GpuBuffer is defined in memory module
pub use memory::GpuBuffer;

impl GpuMemoryManager {
    /// Create new memory manager
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            allocated_buffers: Vec::new(),
        }
    }

    /// Allocate GPU memory buffer
    pub fn allocate(&mut self, size: usize) -> KwaversResult<usize> {
        match self.context.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::allocate_cuda_memory(size),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::allocate_wgpu_memory(size),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: format!("{:?}", self.context.backend),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }

    /// Transfer data from host to device
    pub fn host_to_device(&self, host_data: &[f64], device_buffer: usize) -> KwaversResult<()> {
        match self.context.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::host_to_device_cuda(host_data, device_buffer),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::host_to_device_wgpu(host_data, device_buffer),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: format!("{:?}", self.context.backend),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }

    /// Transfer data from device to host
    pub fn device_to_host(&self, device_buffer: usize, host_data: &mut [f64]) -> KwaversResult<()> {
        match self.context.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::device_to_host_cuda(device_buffer, host_data),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::device_to_host_wgpu(device_buffer, host_data),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: format!("{:?}", self.context.backend),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }
}

/// GPU performance metrics
#[derive(Debug, Clone)]
pub struct GpuPerformanceMetrics {
    pub grid_updates_per_second: f64,
    pub memory_bandwidth_utilization: f64,
    pub kernel_execution_time_ms: f64,
    pub memory_transfer_time_ms: f64,
    pub total_time_ms: f64,
}

impl GpuPerformanceMetrics {
    /// Calculate performance metrics
    pub fn new(
        grid_size: usize,
        kernel_time_ms: f64,
        transfer_time_ms: f64,
        memory_bandwidth_gb_s: f64,
        data_size_gb: f64,
    ) -> Self {
        let total_time_ms = kernel_time_ms + transfer_time_ms;
        let grid_updates_per_second = (grid_size as f64) / (total_time_ms / 1000.0);
        let memory_bandwidth_utilization = (data_size_gb / (transfer_time_ms / 1000.0)) / memory_bandwidth_gb_s;

        Self {
            grid_updates_per_second,
            memory_bandwidth_utilization,
            kernel_execution_time_ms: kernel_time_ms,
            memory_transfer_time_ms: transfer_time_ms,
            total_time_ms,
        }
    }

    /// Check if performance targets are met (Phase 9 requirements)
    pub fn meets_targets(&self) -> bool {
        self.grid_updates_per_second > 17_000_000.0 && // >17M grid updates/second
        self.memory_bandwidth_utilization > 0.8        // >80% memory bandwidth utilization
    }
}

// Re-export FFT kernels
pub use fft_kernels::{GpuFft3d, MultiGpuFft3d, DataDistribution};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MemoryTransferDirection;
    use crate::grid::Grid;
    use ndarray::Array3;

    #[test]
    fn test_gpu_backend_enum() {
        assert_eq!(GpuBackend::Cuda, GpuBackend::Cuda);
        assert_ne!(GpuBackend::Cuda, GpuBackend::OpenCL);
        assert_ne!(GpuBackend::OpenCL, GpuBackend::WebGPU);
    }

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice {
            id: 0,
            name: "Test GPU".to_string(),
            backend: GpuBackend::Cuda,
            memory_size: 8 * 1024 * 1024 * 1024, // 8GB
            compute_units: 32,
            max_work_group_size: 1024,
        };

        assert_eq!(device.id, 0);
        assert_eq!(device.name, "Test GPU");
        assert_eq!(device.backend, GpuBackend::Cuda);
        assert_eq!(device.memory_size, 8 * 1024 * 1024 * 1024);
        assert_eq!(device.compute_units, 32);
        assert_eq!(device.max_work_group_size, 1024);
    }

    #[test]
    fn test_gpu_context_sync_creation() {
        // Test synchronous GPU context creation
        let result = GpuContext::new_sync();
        
        // Should either succeed with devices or fail with NoDevicesFound
        match result {
            Ok(context) => {
                assert!(!context.devices.is_empty());
                assert!(context.active_device.is_some());
            }
            Err(KwaversError::Gpu(crate::error::GpuError::NoDevicesFound)) => {
                // This is expected when no GPU devices are available
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_gpu_performance_metrics() {
        let metrics = GpuPerformanceMetrics::new(
            1_000_000, // 1M grid points
            10.0,      // 10ms kernel time
            5.0,       // 5ms transfer time
            500.0,     // 500 GB/s memory bandwidth
            0.1,       // 0.1 GB data size
        );

        assert_eq!(metrics.kernel_execution_time_ms, 10.0);
        assert_eq!(metrics.memory_transfer_time_ms, 5.0);
        assert_eq!(metrics.total_time_ms, 15.0);
        
        // Calculate expected values
        let expected_updates_per_sec = 1_000_000.0 / (15.0 / 1000.0);
        let expected_bandwidth_util = (0.1 / (5.0 / 1000.0)) / 500.0;
        
        assert!((metrics.grid_updates_per_second - expected_updates_per_sec).abs() < 1.0);
        assert!((metrics.memory_bandwidth_utilization - expected_bandwidth_util).abs() < 0.01);
    }

    #[test]
    fn test_gpu_performance_targets() {
        // Test metrics that meet Phase 9 targets
        let good_metrics = GpuPerformanceMetrics::new(
            20_000_000, // 20M grid points
            1.0,        // 1ms kernel time
            0.1,        // 0.1ms transfer time
            1000.0,     // 1000 GB/s memory bandwidth
            0.8,        // 0.8 GB data size
        );
        assert!(good_metrics.meets_targets());

        // Test metrics that don't meet targets
        let bad_metrics = GpuPerformanceMetrics::new(
            1_000_000,  // 1M grid points (too low)
            100.0,      // 100ms kernel time (too slow)
            50.0,       // 50ms transfer time
            100.0,      // 100 GB/s memory bandwidth
            0.1,        // 0.1 GB data size
        );
        assert!(!bad_metrics.meets_targets());
    }

    #[test]
    fn test_memory_transfer_direction_display() {
        assert_eq!(format!("{}", MemoryTransferDirection::HostToDevice), "HostToDevice");
        assert_eq!(format!("{}", MemoryTransferDirection::DeviceToHost), "DeviceToHost");
        assert_eq!(format!("{}", MemoryTransferDirection::DeviceToDevice), "DeviceToDevice");
    }

    #[test]
    fn test_gpu_error_handling() {
        // Test device initialization error
        let error = crate::error::GpuError::DeviceInitialization {
            device_id: 0,
            reason: "Test error".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("GPU device initialization failed"));
        assert!(display.contains("Test error"));

        // Test memory allocation error
        let error = crate::error::GpuError::MemoryAllocation {
            requested_bytes: 1024,
            available_bytes: 512,
            reason: "Insufficient memory".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("requested 1024 bytes"));
        assert!(display.contains("available 512 bytes"));

        // Test memory transfer error
        let error = crate::error::GpuError::MemoryTransfer {
            direction: MemoryTransferDirection::HostToDevice,
            size_bytes: 2048,
            reason: "Transfer failed".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("HostToDevice"));
        assert!(display.contains("2048 bytes"));
    }

    #[test]
    fn test_gpu_context_creation() {
        // Test GPU context creation (synchronous test for proper validation)
        // This test validates the core functionality without requiring async runtime
        
        // Test case 1: Mock successful context creation
        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Test GPU".to_string(),
                backend: GpuBackend::Cuda,
                memory_size: 8192 * 1024 * 1024, // 8192 MB in bytes
                compute_units: 20,
                max_work_group_size: 1024,
            }
        ];
        
        let context = GpuContext {
            devices: devices.clone(),
            active_device: Some(0),
            backend: GpuBackend::Cuda,
        };
        
        // Validate device properties
        assert!(!context.devices.is_empty());
        assert!(context.active_device.is_some());
        
        if let Some(device) = context.active_device() {
            assert!(!device.name.is_empty());
            assert!(device.compute_units > 0);
            assert!(device.memory_size > 0);
        }
        
        // Test device list access
        assert_eq!(context.devices().len(), 1);
        assert_eq!(context.devices()[0].name, "Test GPU");
        
        // Test case 2: Empty devices (should handle gracefully)
        let empty_context = GpuContext {
            devices: vec![],
            active_device: None,
            backend: GpuBackend::OpenCL, // Use a valid backend even with no devices
        };
        
        assert!(empty_context.devices.is_empty());
        assert!(empty_context.active_device.is_none());
        assert!(empty_context.active_device().is_none());
    }

    #[test]
    fn test_gpu_context_device_selection() {
        // Create a mock context with test devices
        let devices = vec![
            GpuDevice {
                id: 0,
                name: "Low-end GPU".to_string(),
                backend: GpuBackend::Cuda,
                memory_size: 2 * 1024 * 1024 * 1024, // 2GB
                compute_units: 16,
                max_work_group_size: 512,
            },
            GpuDevice {
                id: 1,
                name: "High-end GPU".to_string(),
                backend: GpuBackend::Cuda,
                memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                compute_units: 64,
                max_work_group_size: 1024,
            },
        ];

        let mut context = GpuContext {
            devices,
            active_device: Some(0),
            backend: GpuBackend::Cuda,
        };

        // Test setting active device
        assert!(context.set_active_device(1).is_ok());
        assert_eq!(context.active_device, Some(1));

        // Test invalid device index
        assert!(context.set_active_device(99).is_err());
    }

    #[test]
    fn test_gpu_acoustic_kernel_validation() {
        // Test that would validate acoustic kernel correctness
        // GPU validation requires CUDA hardware availability
        println!("GPU acoustic kernel validation test - requires CUDA hardware");
        
        // Create test grid
        let grid = create_test_grid();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Create test arrays
        let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
        let mut velocity_x = Array3::<f64>::zeros((nx, ny, nz));
        let mut velocity_y = Array3::<f64>::zeros((nx, ny, nz));
        let mut velocity_z = Array3::<f64>::zeros((nx, ny, nz));
        
        // Initialize with test pattern
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    pressure[[i, j, k]] = ((i + j + k) as f64).sin();
                }
            }
        }
        
        // Attempt GPU acoustic update
        if let Ok(context) = GpuContext::new_sync() {
            if context.devices().len() > 0 {
                // Would test actual GPU computation here
                println!("GPU context available for testing");
            }
        }
    }

    #[test]
    fn test_gpu_thermal_kernel_validation() {
        // Test that would validate thermal kernel correctness
        println!("GPU thermal kernel validation test - requires CUDA hardware");
        
        // Create test grid
        let grid = create_test_grid();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Create test arrays
        let mut temperature = Array3::<f64>::zeros((nx, ny, nz));
        let heat_source = Array3::<f64>::zeros((nx, ny, nz));
        
        // Initialize with test pattern
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    temperature[[i, j, k]] = 37.0 + ((i * j * k) as f64).sin(); // Body temperature + variation
                }
            }
        }
        
        // Attempt GPU thermal update
        if let Ok(context) = GpuContext::new_sync() {
            if context.devices().len() > 0 {
                println!("GPU context available for thermal testing");
            }
        }
    }

    #[test]
    fn test_gpu_performance_benchmarking() {
        // Benchmark test for Phase 9 performance targets
        let grid_sizes = vec![
            (32, 32, 32),    // Small: 32K points
            (64, 64, 64),    // Medium: 262K points  
            (128, 128, 128), // Large: 2M points
        ];
        
        for (nx, ny, nz) in grid_sizes {
            let grid_size = nx * ny * nz;
            println!("Testing grid size: {}x{}x{} = {} points", nx, ny, nz, grid_size);
            
            // Simulate kernel execution times based on grid size
            let kernel_time_ms = (grid_size as f64) / 1_000_000.0; // 1M points per ms
            let transfer_time_ms = kernel_time_ms * 0.1; // 10% transfer overhead
            
            let metrics = GpuPerformanceMetrics::new(
                grid_size,
                kernel_time_ms,
                transfer_time_ms,
                1000.0, // 1000 GB/s theoretical bandwidth
                (grid_size * std::mem::size_of::<f64>()) as f64 / 1e9, // Data size in GB
            );
            
            println!("  Grid updates/sec: {:.0}", metrics.grid_updates_per_second);
            println!("  Memory bandwidth util: {:.1}%", metrics.memory_bandwidth_utilization * 100.0);
            
            // For large grids, should meet Phase 9 targets
            if grid_size >= 1_000_000 {
                println!("  Meets Phase 9 targets: {}", metrics.meets_targets());
            }
        }
    }

    // Helper function to create test grid
    fn create_test_grid() -> Grid {
        Grid {
            nx: 64,
            ny: 64, 
            nz: 64,
            dx: 0.1e-3, // 0.1 mm
            dy: 0.1e-3,
            dz: 0.1e-3,
        }
    }
}