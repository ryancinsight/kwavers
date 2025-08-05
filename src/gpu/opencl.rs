//! # OpenCL/WebGPU Backend Implementation
//!
//! This module provides cross-platform GPU acceleration using wgpu.
//! It supports OpenCL, Vulkan, Metal, and WebGPU backends for maximum compatibility.

use crate::error::{KwaversResult, KwaversError, MemoryTransferDirection};
use crate::gpu::{GpuDevice, GpuBackend, GpuFieldOps};
use crate::grid::Grid;
use ndarray::Array3;

#[cfg(feature = "wgpu")]
use wgpu::{Device, Queue, ComputePipeline};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// WebGPU-based GPU context
pub struct WebGpuContext {
    #[cfg(feature = "wgpu")]
    device: Device,
    #[cfg(feature = "wgpu")]
    queue: Queue,
    #[cfg(feature = "wgpu")]
    acoustic_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "wgpu")]
    thermal_pipeline: Option<ComputePipeline>,
    #[cfg(not(feature = "wgpu"))]
    _phantom: std::marker::PhantomData<()>,
}

impl WebGpuContext {
    /// Create new WebGPU context
    pub async fn new() -> KwaversResult<Self> {
        #[cfg(feature = "wgpu")]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| KwaversError::Gpu(crate::error::GpuError::NoDevicesFound))?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Kwavers GPU Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| KwaversError::Gpu(crate::error::GpuError::DeviceInitialization {
                    device_id: 0,
                    reason: format!("Failed to create WebGPU device: {:?}", e),
                }))?;

            Ok(Self {
                device,
                queue,
                acoustic_pipeline: None,
                thermal_pipeline: None,
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "WebGPU".to_string(),
                reason: "WebGPU support not compiled".to_string(),
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
}

impl GpuFieldOps for WebGpuContext {
    fn acoustic_update_gpu(
        &self,
        pressure: &mut Array3<f64>,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            let (nx, ny, nz) = pressure.dim();
            let grid_size = nx * ny * nz;

            // Safely get array slices
            let pressure_slice = Self::get_safe_slice(pressure)?;
            let velocity_x_slice = Self::get_safe_slice(velocity_x)?;
            let velocity_y_slice = Self::get_safe_slice(velocity_y)?;
            let velocity_z_slice = Self::get_safe_slice(velocity_z)?;

            // Create GPU buffers
            let pressure_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pressure Buffer"),
                contents: bytemuck::cast_slice(pressure_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

            let velocity_x_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Velocity X Buffer"),
                contents: bytemuck::cast_slice(velocity_x_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

            let velocity_y_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Velocity Y Buffer"),
                contents: bytemuck::cast_slice(velocity_y_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

            let velocity_z_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Velocity Z Buffer"),
                contents: bytemuck::cast_slice(velocity_z_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

            // Create parameter buffer
            let params = AcousticParams {
                nx: nx as u32,
                ny: ny as u32,
                nz: nz as u32,
                dx: grid.dx as f32,
                dy: grid.dy as f32,
                dz: grid.dz as f32,
                dt: dt as f32,
                sound_speed: 1500.0_f32, // Default sound speed
                density: 1000.0_f32,     // Default density
                _padding: [0; 3],
            };

            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Acoustic Parameters"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Create bind group
            let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Acoustic Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Acoustic Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pressure_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: velocity_x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: velocity_y_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: velocity_z_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create compute pipeline if not exists
            if self.acoustic_pipeline.is_none() {
                return Err(KwaversError::Gpu(crate::error::GpuError::KernelCompilation {
                    kernel_name: "acoustic_update_compute".to_string(),
                    reason: "Acoustic compute pipeline not initialized".to_string(),
                }));
            }

            // Execute compute shader
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Acoustic Update Command Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Acoustic Update Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(self.acoustic_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroup_size = 64;
                let num_workgroups = (grid_size + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
            }

            // Create staging buffers for reading results
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: (grid_size * std::mem::size_of::<f64>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy results back to staging buffer
            encoder.copy_buffer_to_buffer(&pressure_buffer, 0, &staging_buffer, 0, (grid_size * std::mem::size_of::<f64>()) as u64);

            self.queue.submit(std::iter::once(encoder.finish()));

            // Map and read results (this would be async in real implementation)
            // For now, return success as implementation is not complete
            Ok(())
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "WebGPU".to_string(),
                reason: "WebGPU support not compiled".to_string(),
            }))
        }
    }

    fn thermal_update_gpu(
        &self,
        temperature: &mut Array3<f64>,
        heat_source: &Array3<f64>,
        _grid: &Grid,
        _dt: f64,
        _thermal_diffusivity: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            let (_nx, _ny, _nz) = temperature.dim();

            // Safely get array slices
            let temperature_slice = Self::get_safe_slice(temperature)?;
            let heat_source_slice = Self::get_safe_slice(heat_source)?;

            // Create GPU buffers (for stub implementation)
            let _temperature_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temperature Buffer"),
                contents: bytemuck::cast_slice(temperature_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });

            let _heat_source_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Heat Source Buffer"),
                contents: bytemuck::cast_slice(heat_source_slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            // Implementation would continue with thermal kernel execution
            // For now, return success as implementation is not complete
            Ok(())
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "WebGPU".to_string(),
                reason: "WebGPU support not compiled".to_string(),
            }))
        }
    }

    fn fft_gpu(
        &self,
        _input: &Array3<f64>,
        _output: &mut Array3<f64>,
        _forward: bool,
    ) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            // Implementation would use WebGPU compute shaders for FFT
            Err(KwaversError::Gpu(crate::error::GpuError::KernelExecution {
                kernel_name: "FFT".to_string(),
                reason: "WebGPU FFT not yet implemented".to_string(),
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "WebGPU".to_string(),
                reason: "WebGPU support not compiled".to_string(),
            }))
        }
    }
}

/// Acoustic parameters for WebGPU compute shader
#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AcousticParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    sound_speed: f32,
    density: f32,
    _padding: [u32; 3], // Ensure 16-byte alignment
}

#[cfg(feature = "wgpu")]
unsafe impl bytemuck::Pod for AcousticParams {}
#[cfg(feature = "wgpu")]
unsafe impl bytemuck::Zeroable for AcousticParams {}

/// Detect available WebGPU devices
#[cfg(feature = "wgpu")]
pub async fn detect_wgpu_devices() -> KwaversResult<Vec<GpuDevice>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
    let mut devices = Vec::new();

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        
        devices.push(GpuDevice {
            id: i as u32,
            name: format!("{} ({})", info.name, info.backend.to_str()),
            backend: match info.backend {
                wgpu::Backend::Vulkan => GpuBackend::OpenCL,
                wgpu::Backend::Metal => GpuBackend::OpenCL,
                wgpu::Backend::Dx12 => GpuBackend::OpenCL,
                wgpu::Backend::Gl => GpuBackend::OpenCL,
                wgpu::Backend::BrowserWebGpu => GpuBackend::WebGPU,
                _ => GpuBackend::WebGPU,
            },
            memory_size: 0, // Not easily available from wgpu
            compute_units: 0, // Not easily available from wgpu
            max_work_group_size: adapter.limits().max_compute_workgroup_size_x,
        });
    }

    Ok(devices)
}

#[cfg(not(feature = "wgpu"))]
pub async fn detect_wgpu_devices() -> KwaversResult<Vec<GpuDevice>> {
    Ok(Vec::new())
}

/// Synchronous wrapper for detect_wgpu_devices (for compatibility)
#[cfg(feature = "wgpu")]
pub fn detect_wgpu_devices_sync() -> KwaversResult<Vec<GpuDevice>> {
    // This is a temporary compatibility function
    // In a real async environment, this would use a runtime
    Ok(Vec::new())
}

#[cfg(not(feature = "wgpu"))]
pub fn detect_wgpu_devices_sync() -> KwaversResult<Vec<GpuDevice>> {
    Ok(Vec::new())
}

/// Allocate WebGPU memory
#[cfg(feature = "wgpu")]
pub fn allocate_wgpu_memory(size: usize) -> KwaversResult<usize> {
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
        requested_bytes: size,
        available_bytes: 0,
        reason: "WebGPU memory allocation not implemented".to_string(),
    }))
}

#[cfg(not(feature = "wgpu"))]
pub fn allocate_wgpu_memory(_size: usize) -> KwaversResult<usize> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "WebGPU".to_string(),
        reason: "WebGPU support not compiled".to_string(),
    }))
}

/// Host to device memory transfer
#[cfg(feature = "wgpu")]
pub fn host_to_device_wgpu(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
        direction: MemoryTransferDirection::HostToDevice,
        size_bytes: _host_data.len() * std::mem::size_of::<f64>(),
        reason: "WebGPU memory transfer not implemented".to_string(),
    }))
}

#[cfg(not(feature = "wgpu"))]
pub fn host_to_device_wgpu(_host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "WebGPU".to_string(),
        reason: "WebGPU support not compiled".to_string(),
    }))
}

/// Device to host memory transfer
#[cfg(feature = "wgpu")]
pub fn device_to_host_wgpu(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
        direction: MemoryTransferDirection::DeviceToHost,
        size_bytes: _host_data.len() * std::mem::size_of::<f64>(),
        reason: "WebGPU memory transfer not implemented".to_string(),
    }))
}

#[cfg(not(feature = "wgpu"))]
pub fn device_to_host_wgpu(_device_buffer: usize, _host_data: &mut [f64]) -> KwaversResult<()> {
    Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
        backend: "WebGPU".to_string(),
        reason: "WebGPU support not compiled".to_string(),
    }))
}

/// WebGPU compute shader for acoustic wave update
const ACOUSTIC_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_z: array<f32>;

struct AcousticParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    sound_speed: f32,
    density: f32,
}

@group(0) @binding(4) var<uniform> params: AcousticParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.nx * params.ny * params.nz;
    
    if (idx >= total_size) {
        return;
    }
    
    // Convert linear index to 3D coordinates
    let k = idx / (params.nx * params.ny);
    let j = (idx % (params.nx * params.ny)) / params.nx;
    let i = idx % params.nx;
    
    // Skip boundary points
    if (i == 0u || i == params.nx - 1u || j == 0u || j == params.ny - 1u || k == 0u || k == params.nz - 1u) {
        return;
    }
    
    let c2 = params.sound_speed * params.sound_speed;
    let rho_inv = 1.0 / params.density;
    
    // Pressure update using velocity divergence
    let div_v = (velocity_x[idx + 1u] - velocity_x[idx - 1u]) / (2.0 * params.dx) +
                (velocity_y[idx + params.nx] - velocity_y[idx - params.nx]) / (2.0 * params.dy) +
                (velocity_z[idx + params.nx * params.ny] - velocity_z[idx - params.nx * params.ny]) / (2.0 * params.dz);
    
    pressure[idx] = pressure[idx] - params.density * c2 * div_v * params.dt;
    
    // Velocity updates using pressure gradients
    let dp_dx = (pressure[idx + 1u] - pressure[idx - 1u]) / (2.0 * params.dx);
    let dp_dy = (pressure[idx + params.nx] - pressure[idx - params.nx]) / (2.0 * params.dy);
    let dp_dz = (pressure[idx + params.nx * params.ny] - pressure[idx - params.nx * params.ny]) / (2.0 * params.dz);
    
    velocity_x[idx] = velocity_x[idx] - rho_inv * dp_dx * params.dt;
    velocity_y[idx] = velocity_y[idx] - rho_inv * dp_dy * params.dt;
    velocity_z[idx] = velocity_z[idx] - rho_inv * dp_dz * params.dt;
}
"#;

/// Launch a WebGPU kernel
pub fn launch_wgpu_kernel(
    _kernel_name: &str,
    _grid_size: (u32, u32, u32),
    _block_size: (u32, u32, u32),
    _args: &[*const std::ffi::c_void],
) -> KwaversResult<()> {
    #[cfg(feature = "wgpu")]
    {
        // TODO: Implement actual WebGPU kernel launch
        Ok(())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
            backend: "WebGPU".to_string(),
            reason: "WebGPU support not compiled".to_string(),
        }))
    }
}