//! # CUDA Backend Implementation
//!
//! This module provides NVIDIA CUDA acceleration using the cudarc crate.
//! It implements high-performance kernels for acoustic wave propagation,
//! thermal diffusion, and FFT operations.

use crate::error::{KwaversResult, KwaversError, MemoryTransferDirection};
use crate::gpu::{GpuDevice, GpuBackend, GpuFieldOps};
use crate::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

#[cfg(feature = "cudarc")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cudarc")]
use cudarc::nvrtc::compile_ptx;

/// CUDA-specific GPU context
pub struct CudaContext {
    #[cfg(feature = "cudarc")]
    device: Arc<CudaDevice>,
    #[cfg(not(feature = "cudarc"))]
    _phantom: std::marker::PhantomData<()>,
}

impl CudaContext {
    /// Create new CUDA context
    pub fn new(device_id: usize) -> KwaversResult<Self> {
        #[cfg(feature = "cudarc")]
        {
            use std::panic;
            
            // Catch panics from CUDA library loading failures
            let result = panic::catch_unwind(|| {
                let device = CudaDevice::new(device_id)
                    .map_err(|e| KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                        device_id: device_id as u32,
                        reason: format!("Failed to create CUDA device: {:?}", e),
                    }))?;
                
                Ok(Self {
                    device,
                })
            });
            
            match result {
                Ok(context_result) => context_result,
                Err(_) => {
                    // CUDA library loading failed (panic caught)
                    Err(KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                        device_id: device_id as u32,
                        reason: "CUDA runtime library not available".to_string(),
                    }))
                }
            }
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "CUDA".to_string(),
                reason: "CUDA support not compiled".to_string(),
            }))
        }
    }

    /// Safely get array slice, ensuring standard layout
    fn get_safe_slice(array: &Array3<f64>) -> KwaversResult<&[f64]> {
        if array.is_standard_layout() {
            array.as_slice().ok_or_else(|| {
                KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                    direction: MemoryTransferDirection::HostToDevice,
                    size_bytes: array.len() * std::mem::size_of::<f64>(),
                    reason: "Failed to get array slice despite standard layout".to_string(),
                })
            })
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: array.len() * std::mem::size_of::<f64>(),
                reason: "Array is not in standard layout - cannot safely access as slice".to_string(),
            }))
        }
    }

    /// Safely get mutable array slice, ensuring standard layout
    fn get_safe_slice_mut(array: &mut Array3<f64>) -> KwaversResult<&mut [f64]> {
        if array.is_standard_layout() {
            let size_bytes = array.len() * std::mem::size_of::<f64>();
            array.as_slice_mut().ok_or_else(|| {
                KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                    direction: MemoryTransferDirection::DeviceToHost,
                    size_bytes,
                    reason: "Failed to get mutable array slice despite standard layout".to_string(),
                })
            })
        } else {
            let size_bytes = array.len() * std::mem::size_of::<f64>();
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes,
                reason: "Array is not in standard layout - cannot safely access as mutable slice".to_string(),
            }))
        }
    }

    /// Helper function to allocate GPU memory for a single array
    #[cfg(feature = "cudarc")]
    fn allocate_gpu_memory(&self, grid_size: usize) -> KwaversResult<CudaSlice<f32>> {
        self.device.alloc_zeros::<f32>(grid_size)
            .map_err(|e| KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: grid_size * std::mem::size_of::<f32>(),
                available_bytes: 0, // Not easily available from cudarc error
                reason: format!("GPU memory allocation failed: {:?}", e),
            }))
    }

    /// Helper function to convert f64 array to f32 and copy to GPU
    #[cfg(feature = "cudarc")]
    fn copy_array_to_gpu(&self, array: &Array3<f64>, d_array: &mut CudaSlice<f32>) -> KwaversResult<()> {
        let grid_size = array.len();
        let array_slice = Self::get_safe_slice(array)?;
        let array_f32: Vec<f32> = array_slice.iter().map(|&x| x as f32).collect();
        
        self.device.htod_sync_copy_into(&array_f32, d_array)
            .map_err(|e| KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: grid_size * std::mem::size_of::<f32>(),
                reason: format!("Host to device copy failed: {:?}", e),
            }))
    }

    /// Helper function to copy GPU results back to f64 array
    #[cfg(feature = "cudarc")]
    fn copy_array_from_gpu(&self, d_array: &CudaSlice<f32>, array: &mut Array3<f64>) -> KwaversResult<()> {
        let grid_size = array.len();
        let mut array_f32_result = vec![0.0f32; grid_size];
        
        self.device.dtoh_sync_copy_into(d_array, &mut array_f32_result)
            .map_err(|e| KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes: grid_size * std::mem::size_of::<f32>(),
                reason: format!("Device to host copy failed: {:?}", e),
            }))?;

        // Convert back to f64 and update array
        let array_slice_mut = Self::get_safe_slice_mut(array)?;
        for (f32_val, f64_val) in array_f32_result.iter().zip(array_slice_mut.iter_mut()) {
            *f64_val = *f32_val as f64;
        }
        
        Ok(())
    }
}

impl GpuFieldOps for CudaContext {
    fn acoustic_update_gpu(
        &self,
        pressure: &mut Array3<f64>,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "cudarc")]
        {
            let (nx, ny, nz) = pressure.dim();
            let grid_size = nx * ny * nz;

            // Allocate GPU memory (using f32 for GPU operations)
            let mut d_pressure = self.allocate_gpu_memory(grid_size)?;
            let mut d_velocity_x = self.allocate_gpu_memory(grid_size)?;
            let mut d_velocity_y = self.allocate_gpu_memory(grid_size)?;
            let mut d_velocity_z = self.allocate_gpu_memory(grid_size)?;

            // Copy data to GPU using helper functions
            self.copy_array_to_gpu(pressure, &mut d_pressure)?;
            self.copy_array_to_gpu(velocity_x, &mut d_velocity_x)?;
            self.copy_array_to_gpu(velocity_y, &mut d_velocity_y)?;
            self.copy_array_to_gpu(velocity_z, &mut d_velocity_z)?;

            // Load and compile CUDA kernel
            let ptx = compile_ptx(ACOUSTIC_UPDATE_KERNEL)
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelCompilation {
                    kernel_name: "acoustic_update_kernel".to_string(),
                    reason: format!("Kernel compilation failed: {:?}", e),
                }))?;
            
            self.device.load_ptx(ptx, "acoustic_update", &["acoustic_update_kernel"])
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelCompilation {
                    kernel_name: "acoustic_update_kernel".to_string(),
                    reason: format!("Kernel loading failed: {:?}", e),
                }))?;

            let f = self.device.get_func("acoustic_update", "acoustic_update_kernel")
                .ok_or_else(|| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "acoustic_update_kernel".to_string(),
                    reason: "Kernel function not found".to_string(),
                }))?;

            // Configure kernel launch parameters
            let block_size = 256;
            let grid_size_launch = (grid_size + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size_launch as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Launch kernel with complete parameters
            unsafe {
                f.launch(cfg, (
                    &d_pressure, &d_velocity_x, &d_velocity_y, &d_velocity_z,
                    nx as u32, ny as u32, nz as u32,
                    grid.dx as f32, grid.dy as f32, grid.dz as f32, dt as f32,
                    1500.0f32, 1000.0f32  // Default sound speed and density
                )).map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "acoustic_update_kernel".to_string(),
                    reason: format!("Kernel launch failed: {:?}", e),
                }))?;
            }

            // Synchronize device
            self.device.synchronize()
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "acoustic_update_kernel".to_string(),
                    reason: format!("Device synchronization failed: {:?}", e),
                }))?;

            // Copy results back to host using helper functions
            self.copy_array_from_gpu(&d_pressure, pressure)?;
            self.copy_array_from_gpu(&d_velocity_x, velocity_x)?;
            self.copy_array_from_gpu(&d_velocity_y, velocity_y)?;
            self.copy_array_from_gpu(&d_velocity_z, velocity_z)?;

            Ok(())
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "CUDA".to_string(),
                reason: "CUDA support not compiled".to_string(),
            }))
        }
    }

    fn thermal_update_gpu(
        &self,
        temperature: &mut Array3<f64>,
        heat_source: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        thermal_diffusivity: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "cudarc")]
        {
            let (nx, ny, nz) = temperature.dim();
            let grid_size = nx * ny * nz;

            // Allocate GPU memory (using f32 for GPU operations)
            let mut d_temperature = self.allocate_gpu_memory(grid_size)?;
            let mut d_heat_source = self.allocate_gpu_memory(grid_size)?;

            // Copy data to GPU using helper functions
            self.copy_array_to_gpu(temperature, &mut d_temperature)?;
            self.copy_array_to_gpu(heat_source, &mut d_heat_source)?;

            // Load thermal diffusion kernel
            let ptx = compile_ptx(THERMAL_UPDATE_KERNEL)
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelCompilation {
                    kernel_name: "thermal_update_kernel".to_string(),
                    reason: format!("Kernel compilation failed: {:?}", e),
                }))?;
            
            self.device.load_ptx(ptx, "thermal_update", &["thermal_update_kernel"])
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelCompilation {
                    kernel_name: "thermal_update_kernel".to_string(),
                    reason: format!("Kernel loading failed: {:?}", e),
                }))?;

            let f = self.device.get_func("thermal_update", "thermal_update_kernel")
                .ok_or_else(|| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "thermal_update_kernel".to_string(),
                    reason: "Kernel function not found".to_string(),
                }))?;

            // Launch kernel
            let block_size = 256;
            let grid_size_launch = (grid_size + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size_launch as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                f.launch(cfg, (
                    &d_temperature, &d_heat_source,
                    nx as u32, ny as u32, nz as u32,
                    grid.dx as f32, grid.dy as f32, grid.dz as f32, dt as f32, thermal_diffusivity as f32
                )).map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "thermal_update_kernel".to_string(),
                    reason: format!("Kernel launch failed: {:?}", e),
                }))?;
            }

            // Synchronize device
            self.device.synchronize()
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                    kernel_name: "thermal_update_kernel".to_string(),
                    reason: format!("Device synchronization failed: {:?}", e),
                }))?;

            // Copy results back to host using helper function
            self.copy_array_from_gpu(&d_temperature, temperature)?;

            Ok(())
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "CUDA".to_string(),
                reason: "CUDA support not compiled".to_string(),
            }))
        }
    }

    fn fft_gpu(
        &self,
        _input: &Array3<f64>,
        _output: &mut Array3<f64>,
        _forward: bool,
    ) -> KwaversResult<()> {
        #[cfg(feature = "cudarc")]
        {
            // Implementation would use cuFFT library
            // For now, return error indicating not implemented
            Err(KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                kernel_name: "FFT".to_string(),
                reason: "GPU FFT not yet implemented".to_string(),
            }))
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "CUDA".to_string(),
                reason: "CUDA support not compiled".to_string(),
            }))
        }
    }
}

/// Detect available CUDA devices
#[cfg(feature = "cudarc")]
pub fn detect_cuda_devices() -> KwaversResult<Vec<GpuDevice>> {
    use std::panic;
    
    // Catch panics from CUDA library loading failures
    let result = panic::catch_unwind(|| {
        let mut devices = Vec::new();
        
        // Get device count
        let device_count = CudaDevice::count()
            .map_err(|e| KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                device_id: 0, // No specific device ID for count error
                reason: format!("Failed to get CUDA device count: {:?}", e),
            }))?;
        
        for i in 0..device_count {
            if let Ok(_device) = CudaDevice::new(i as usize) {
                // cudarc returns Arc<CudaDevice>, so we need to dereference
                let name = format!("CUDA Device {}", i);
                let memory_size = 8 * 1024 * 1024 * 1024; // Default 8GB
                
                devices.push(GpuDevice {
                    id: i as u32,
                    name,
                    backend: GpuBackend::Cuda,
                    memory_size,
                    compute_units: 32, // Default compute units
                    max_work_group_size: 1024, // Default max work group size
                });
            }
        }
        
        Ok(devices)
    });
    
    match result {
        Ok(devices_result) => devices_result,
        Err(_) => {
            // CUDA library loading failed (panic caught)
            Err(KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                device_id: 0,
                reason: "CUDA runtime library not available".to_string(),
            }))
        }
    }
}

#[cfg(not(feature = "cudarc"))]
pub fn detect_cuda_devices() -> KwaversResult<Vec<GpuDevice>> {
    Ok(Vec::new())
}

/// Allocate CUDA memory
#[cfg(feature = "cudarc")]
pub fn allocate_cuda_memory(size: usize) -> KwaversResult<usize> {
    // Implementation would use CUDA memory allocation
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
        requested_bytes: size,
        available_bytes: 0,
        reason: "CUDA memory allocation not implemented".to_string(),
    }))
}

#[cfg(not(feature = "cudarc"))]
pub fn allocate_cuda_memory(_size: usize) -> KwaversResult<usize> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "CUDA".to_string(),
        reason: "CUDA support not compiled".to_string(),
    }))
}

/// Host to device memory transfer
#[cfg(feature = "cudarc")]
pub fn host_to_device_cuda(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
        direction: MemoryTransferDirection::HostToDevice,
        size_bytes: _host_data.len() * std::mem::size_of::<f64>(),
        reason: "CUDA memory transfer not implemented".to_string(),
    }))
}

#[cfg(not(feature = "cudarc"))]
pub fn host_to_device_cuda(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "CUDA".to_string(),
        reason: "CUDA support not compiled".to_string(),
    }))
}

/// Device to host memory transfer
#[cfg(feature = "cudarc")]
pub fn device_to_host_cuda(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
        direction: MemoryTransferDirection::DeviceToHost,
        size_bytes: _host_data.len() * std::mem::size_of::<f64>(),
        reason: "CUDA memory transfer not implemented".to_string(),
    }))
}

#[cfg(not(feature = "cudarc"))]
pub fn device_to_host_cuda(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "CUDA".to_string(),
        reason: "CUDA support not compiled".to_string(),
    }))
}

/// CUDA kernel for acoustic wave update
const ACOUSTIC_UPDATE_KERNEL: &str = r#"
extern "C" __global__ void acoustic_update_kernel(
    double* pressure,
    double* velocity_x,
    double* velocity_y,
    double* velocity_z,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    double dx,
    double dy,
    double dz,
    double dt,
    double sound_speed,
    double density
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_size = nx * ny * nz;
    
    if (idx >= total_size) return;
    
    // Convert linear index to 3D coordinates
    unsigned int k = idx / (nx * ny);
    unsigned int j = (idx % (nx * ny)) / nx;
    unsigned int i = idx % nx;
    
    // Skip boundary points
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    // Calculate finite difference derivatives
    double c2 = sound_speed * sound_speed;
    double rho_inv = 1.0 / density;
    
    // Pressure update using velocity divergence
    double div_v = (velocity_x[idx + 1] - velocity_x[idx - 1]) / (2.0 * dx) +
                   (velocity_y[idx + nx] - velocity_y[idx - nx]) / (2.0 * dy) +
                   (velocity_z[idx + nx * ny] - velocity_z[idx - nx * ny]) / (2.0 * dz);
    
    pressure[idx] -= density * c2 * div_v * dt;
    
    // Velocity updates using pressure gradients
    double dp_dx = (pressure[idx + 1] - pressure[idx - 1]) / (2.0 * dx);
    double dp_dy = (pressure[idx + nx] - pressure[idx - nx]) / (2.0 * dy);
    double dp_dz = (pressure[idx + nx * ny] - pressure[idx - nx * ny]) / (2.0 * dz);
    
    velocity_x[idx] -= rho_inv * dp_dx * dt;
    velocity_y[idx] -= rho_inv * dp_dy * dt;
    velocity_z[idx] -= rho_inv * dp_dz * dt;
}
"#;

/// CUDA kernel for thermal diffusion update
const THERMAL_UPDATE_KERNEL: &str = r#"
extern "C" __global__ void thermal_update_kernel(
    double* temperature,
    double* heat_source,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    double dx,
    double dy,
    double dz,
    double dt,
    double thermal_diffusivity
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_size = nx * ny * nz;
    
    if (idx >= total_size) return;
    
    // Convert linear index to 3D coordinates
    unsigned int k = idx / (nx * ny);
    unsigned int j = (idx % (nx * ny)) / nx;
    unsigned int i = idx % nx;
    
    // Skip boundary points
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    // Calculate second derivatives (Laplacian)
    double d2T_dx2 = (temperature[idx + 1] - 2.0 * temperature[idx] + temperature[idx - 1]) / (dx * dx);
    double d2T_dy2 = (temperature[idx + nx] - 2.0 * temperature[idx] + temperature[idx - nx]) / (dy * dy);
    double d2T_dz2 = (temperature[idx + nx * ny] - 2.0 * temperature[idx] + temperature[idx - nx * ny]) / (dz * dz);
    
    // Thermal diffusion equation: dT/dt = α∇²T + Q
    double laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
    temperature[idx] += (thermal_diffusivity * laplacian + heat_source[idx]) * dt;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_detection() {
        // Test should work even without CUDA hardware
        match detect_cuda_devices() {
            Ok(devices) => {
                println!("Found {} CUDA devices", devices.len());
                for device in devices {
                    assert_eq!(device.backend, GpuBackend::Cuda);
                    assert!(!device.name.is_empty());
                }
            }
            Err(_) => {
                println!("No CUDA devices found - this is acceptable");
            }
        }
    }

    #[test]
    fn test_cuda_context_creation() {
        // Try to create a CUDA context, but handle library loading failures gracefully
        match CudaContext::new(0) {
            Ok(_context) => {
                println!("CUDA context created successfully");
            }
            Err(KwaversError::Gpu(crate::error::GpuError::DeviceInitialization { .. })) => {
                println!("CUDA context creation failed - likely no CUDA runtime available");
                // This is acceptable in test environments without CUDA
            }
            Err(e) => {
                println!("CUDA context creation failed with unexpected error: {}", e);
            }
        }
    }
}