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

pub mod cuda;
pub mod opencl;
pub mod memory;
pub mod kernels;
pub mod benchmarks;

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
    pub fn new() -> KwaversResult<Self> {
        let devices = Self::detect_devices()?;
        
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

    /// Detect available GPU devices
    fn detect_devices() -> KwaversResult<Vec<GpuDevice>> {
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
            if let Ok(wgpu_devices) = opencl::detect_wgpu_devices() {
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

/// GPU memory buffer
pub struct GpuBuffer {
    pub size: usize,
    pub device_ptr: *mut u8,
    pub host_ptr: Option<*mut u8>,
}

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
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::GpuError("No GPU backend available".to_string())),
        }
    }

    /// Transfer data from host to device
    pub fn host_to_device(&self, host_data: &[f64], device_buffer: usize) -> KwaversResult<()> {
        match self.context.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::host_to_device_cuda(host_data, device_buffer),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::host_to_device_wgpu(host_data, device_buffer),
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::GpuError("No GPU backend available".to_string())),
        }
    }

    /// Transfer data from device to host
    pub fn device_to_host(&self, device_buffer: usize, host_data: &mut [f64]) -> KwaversResult<()> {
        match self.context.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => cuda::device_to_host_cuda(device_buffer, host_data),
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => opencl::device_to_host_wgpu(device_buffer, host_data),
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::GpuError("No GPU backend available".to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        // Test should work even without GPU hardware
        match GpuContext::new() {
            Ok(context) => {
                assert!(!context.devices().is_empty());
                assert!(context.active_device().is_some());
            }
            Err(_) => {
                // No GPU devices available - this is acceptable in CI/test environments
                println!("No GPU devices found - skipping GPU tests");
            }
        }
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = GpuPerformanceMetrics::new(
            1000_000, // 1M grid points
            10.0,     // 10ms kernel time
            5.0,      // 5ms transfer time
            500.0,    // 500 GB/s memory bandwidth
            1.0,      // 1 GB data size
        );

        assert!(metrics.grid_updates_per_second > 0.0);
        assert!(metrics.memory_bandwidth_utilization > 0.0);
        assert_eq!(metrics.total_time_ms, 15.0);
    }

    #[test]
    fn test_performance_targets() {
        let good_metrics = GpuPerformanceMetrics::new(
            20_000_000, // 20M grid points
            1.0,        // 1ms kernel time
            0.1,        // 0.1ms transfer time
            1000.0,     // 1000 GB/s memory bandwidth
            1.0,        // 1 GB data size
        );
        assert!(good_metrics.meets_targets());

        let bad_metrics = GpuPerformanceMetrics::new(
            1_000_000,  // 1M grid points (too low)
            100.0,      // 100ms kernel time (too slow)
            50.0,       // 50ms transfer time (too slow)
            100.0,      // 100 GB/s memory bandwidth
            1.0,        // 1 GB data size
        );
        assert!(!bad_metrics.meets_targets());
    }
}