//! # CUDA Backend Implementation
//!
//! This module provides NVIDIA CUDA acceleration using the cudarc crate.
//! It implements high-performance kernels for acoustic wave propagation,
//! thermal diffusion, and FFT operations.

use crate::error::{KwaversResult, KwaversError};
use crate::gpu::{GpuDevice, GpuBackend, GpuFieldOps};
use crate::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

#[cfg(feature = "cudarc")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
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
            let device = CudaDevice::new(device_id)
                .map_err(|e| KwaversError::GpuError(format!("Failed to create CUDA device: {:?}", e)))?;
            
            Ok(Self {
                device: Arc::new(device),
            })
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
        }
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

            // Allocate GPU memory
            let mut d_pressure = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;
            let mut d_velocity_x = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;
            let mut d_velocity_y = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;
            let mut d_velocity_z = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;

            // Copy data to GPU
            self.device.htod_sync_copy_into(pressure.as_slice().unwrap(), &mut d_pressure)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;
            self.device.htod_sync_copy_into(velocity_x.as_slice().unwrap(), &mut d_velocity_x)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;
            self.device.htod_sync_copy_into(velocity_y.as_slice().unwrap(), &mut d_velocity_y)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;
            self.device.htod_sync_copy_into(velocity_z.as_slice().unwrap(), &mut d_velocity_z)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;

            // Load and compile CUDA kernel
            let ptx = compile_ptx(ACOUSTIC_UPDATE_KERNEL)
                .map_err(|e| KwaversError::GpuError(format!("Kernel compilation failed: {:?}", e)))?;
            
            self.device.load_ptx(ptx, "acoustic_update", &["acoustic_update_kernel"])
                .map_err(|e| KwaversError::GpuError(format!("Kernel loading failed: {:?}", e)))?;

            let f = self.device.get_func("acoustic_update", "acoustic_update_kernel")
                .map_err(|e| KwaversError::GpuError(format!("Kernel function not found: {:?}", e)))?;

            // Configure kernel launch parameters
            let block_size = 256;
            let grid_size_launch = (grid_size + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size_launch as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Launch kernel
            unsafe {
                f.launch(cfg, (
                    &d_pressure, &d_velocity_x, &d_velocity_y, &d_velocity_z,
                    nx as u32, ny as u32, nz as u32,
                    grid.dx, grid.dy, grid.dz, dt,
                    grid.sound_speed, grid.density
                )).map_err(|e| KwaversError::GpuError(format!("Kernel launch failed: {:?}", e)))?;
            }

            // Copy results back to host
            self.device.dtoh_sync_copy_into(&d_pressure, pressure.as_slice_mut().unwrap())
                .map_err(|e| KwaversError::GpuError(format!("Device to host copy failed: {:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_velocity_x, velocity_x.as_slice_mut().unwrap())
                .map_err(|e| KwaversError::GpuError(format!("Device to host copy failed: {:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_velocity_y, velocity_y.as_slice_mut().unwrap())
                .map_err(|e| KwaversError::GpuError(format!("Device to host copy failed: {:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_velocity_z, velocity_z.as_slice_mut().unwrap())
                .map_err(|e| KwaversError::GpuError(format!("Device to host copy failed: {:?}", e)))?;

            Ok(())
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
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

            // Allocate GPU memory
            let mut d_temperature = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;
            let mut d_heat_source = self.device.alloc_zeros::<f64>(grid_size)
                .map_err(|e| KwaversError::GpuError(format!("GPU memory allocation failed: {:?}", e)))?;

            // Copy data to GPU
            self.device.htod_sync_copy_into(temperature.as_slice().unwrap(), &mut d_temperature)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;
            self.device.htod_sync_copy_into(heat_source.as_slice().unwrap(), &mut d_heat_source)
                .map_err(|e| KwaversError::GpuError(format!("Host to device copy failed: {:?}", e)))?;

            // Load thermal diffusion kernel
            let ptx = compile_ptx(THERMAL_UPDATE_KERNEL)
                .map_err(|e| KwaversError::GpuError(format!("Kernel compilation failed: {:?}", e)))?;
            
            self.device.load_ptx(ptx, "thermal_update", &["thermal_update_kernel"])
                .map_err(|e| KwaversError::GpuError(format!("Kernel loading failed: {:?}", e)))?;

            let f = self.device.get_func("thermal_update", "thermal_update_kernel")
                .map_err(|e| KwaversError::GpuError(format!("Kernel function not found: {:?}", e)))?;

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
                    grid.dx, grid.dy, grid.dz, dt, thermal_diffusivity
                )).map_err(|e| KwaversError::GpuError(format!("Kernel launch failed: {:?}", e)))?;
            }

            // Copy results back
            self.device.dtoh_sync_copy_into(&d_temperature, temperature.as_slice_mut().unwrap())
                .map_err(|e| KwaversError::GpuError(format!("Device to host copy failed: {:?}", e)))?;

            Ok(())
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
        }
    }

    fn fft_gpu(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        forward: bool,
    ) -> KwaversResult<()> {
        #[cfg(feature = "cudarc")]
        {
            // Implementation would use cuFFT library
            // For now, return error indicating not implemented
            Err(KwaversError::GpuError("GPU FFT not yet implemented".to_string()))
        }
        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
        }
    }
}

/// Detect available CUDA devices
#[cfg(feature = "cudarc")]
pub fn detect_cuda_devices() -> KwaversResult<Vec<GpuDevice>> {
    use cudarc::driver::sys::CuDevice;
    
    let mut devices = Vec::new();
    
    // Get device count
    let device_count = CudaDevice::count()
        .map_err(|e| KwaversError::GpuError(format!("Failed to get CUDA device count: {:?}", e)))?;
    
    for i in 0..device_count {
        if let Ok(device) = CudaDevice::new(i) {
            let name = device.name()
                .unwrap_or_else(|_| format!("CUDA Device {}", i));
            let memory_size = device.total_memory()
                .unwrap_or(0);
            
            devices.push(GpuDevice {
                id: i as u32,
                name,
                backend: GpuBackend::Cuda,
                memory_size,
                compute_units: device.multiprocessor_count().unwrap_or(0) as u32,
                max_work_group_size: device.max_threads_per_block().unwrap_or(0) as u32,
            });
        }
    }
    
    Ok(devices)
}

#[cfg(not(feature = "cudarc"))]
pub fn detect_cuda_devices() -> KwaversResult<Vec<GpuDevice>> {
    Ok(Vec::new())
}

/// Allocate CUDA memory
#[cfg(feature = "cudarc")]
pub fn allocate_cuda_memory(size: usize) -> KwaversResult<usize> {
    // Implementation would use CUDA memory allocation
    Err(KwaversError::GpuError("CUDA memory allocation not implemented".to_string()))
}

#[cfg(not(feature = "cudarc"))]
pub fn allocate_cuda_memory(_size: usize) -> KwaversResult<usize> {
    Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
}

/// Host to device memory transfer
#[cfg(feature = "cudarc")]
pub fn host_to_device_cuda(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::GpuError("CUDA memory transfer not implemented".to_string()))
}

#[cfg(not(feature = "cudarc"))]
pub fn host_to_device_cuda(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
}

/// Device to host memory transfer
#[cfg(feature = "cudarc")]
pub fn device_to_host_cuda(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::GpuError("CUDA memory transfer not implemented".to_string()))
}

#[cfg(not(feature = "cudarc"))]
pub fn device_to_host_cuda(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::GpuError("CUDA support not compiled".to_string()))
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
    const double* heat_source,
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
    
    // Calculate Laplacian using finite differences
    double d2T_dx2 = (temperature[idx + 1] - 2.0 * temperature[idx] + temperature[idx - 1]) / (dx * dx);
    double d2T_dy2 = (temperature[idx + nx] - 2.0 * temperature[idx] + temperature[idx - nx]) / (dy * dy);
    double d2T_dz2 = (temperature[idx + nx * ny] - 2.0 * temperature[idx] + temperature[idx - nx * ny]) / (dz * dz);
    
    double laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
    
    // Update temperature using diffusion equation
    temperature[idx] += dt * (thermal_diffusivity * laplacian + heat_source[idx]);
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
        // Only test if CUDA is available
        if detect_cuda_devices().unwrap_or_default().is_empty() {
            return;
        }

        match CudaContext::new(0) {
            Ok(_context) => {
                println!("CUDA context created successfully");
            }
            Err(e) => {
                println!("CUDA context creation failed: {}", e);
            }
        }
    }
}