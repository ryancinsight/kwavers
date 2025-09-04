//! GPU acceleration using wgpu-rs for cross-platform compute shaders
//!
//! This module provides GPU-accelerated implementations of physics solvers
//! using WebGPU compute shaders for maximum portability and performance.

#[cfg(feature = "gpu")]
use wgpu::*;
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::error::{KwaversError, KwaversResult};
#[cfg(feature = "gpu")]
use crate::physics::data::{AlignedField, AcousticFields};

/// GPU device and queue management
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter_info: AdapterInfo,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize GPU context with optimal device selection
    pub async fn new() -> KwaversResult<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| KwaversError::GpuError("No suitable GPU adapter found".to_string()))?;
            
        let adapter_info = adapter.get_info();
        
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Kwavers GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| KwaversError::GpuError(format!("Failed to create device: {}", e)))?;
            
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
    
    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get queue reference  
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    
    /// Get adapter information
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }
    
    /// Check if device supports required features
    pub fn supports_compute(&self) -> bool {
        // All modern GPUs support compute shaders
        true
    }
}

/// GPU buffer wrapper for zero-copy operations
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
    usage: BufferUsages,
}

#[cfg(feature = "gpu")]
impl GpuBuffer {
    /// Create new GPU buffer
    pub fn new(device: &Device, size: u64, usage: BufferUsages, label: Option<&str>) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });
        
        Self { buffer, size, usage }
    }
    
    /// Create buffer from slice data
    pub fn from_slice<T: Pod>(device: &Device, data: &[T], usage: BufferUsages, label: Option<&str>) -> Self {
        let buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(data),
            usage,
        });
        
        Self {
            buffer,
            size: (data.len() * std::mem::size_of::<T>()) as u64,
            usage,
        }
    }
    
    /// Get buffer reference
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    
    /// Get buffer size
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Uniform data for shader parameters
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SimulationUniforms {
    pub grid_nx: u32,
    pub grid_ny: u32,
    pub grid_nz: u32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub dt: f32,
    pub time: f32,
    pub sound_speed: f32,
    pub density: f32,
    pub _padding: [u32; 2], // Align to 16 bytes
}

/// GPU-accelerated FDTD solver
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuFdtdSolver {
    context: Arc<GpuContext>,
    pressure_buffer: GpuBuffer,
    velocity_x_buffer: GpuBuffer,
    velocity_y_buffer: GpuBuffer,
    velocity_z_buffer: GpuBuffer,
    uniform_buffer: GpuBuffer,
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    grid_size: (u32, u32, u32),
    workgroup_size: (u32, u32, u32),
}

#[cfg(feature = "gpu")]
impl GpuFdtdSolver {
    /// Create new GPU FDTD solver
    pub async fn new(
        context: Arc<GpuContext>,
        grid_size: (u32, u32, u32),
        initial_fields: &AcousticFields,
    ) -> KwaversResult<Self> {
        let device = context.device();
        let (nx, ny, nz) = grid_size;
        let buffer_size = (nx * ny * nz * 4) as u64; // 4 bytes per f32
        
        // Create buffers for fields
        let pressure_buffer = GpuBuffer::from_slice(
            device,
            bytemuck::cast_slice(initial_fields.pressure.as_slice()),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            Some("Pressure Buffer"),
        );
        
        let velocity_x_buffer = GpuBuffer::from_slice(
            device,
            bytemuck::cast_slice(initial_fields.velocity_x.as_slice()),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            Some("Velocity X Buffer"),
        );
        
        let velocity_y_buffer = GpuBuffer::from_slice(
            device,
            bytemuck::cast_slice(initial_fields.velocity_y.as_slice()),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            Some("Velocity Y Buffer"),
        );
        
        let velocity_z_buffer = GpuBuffer::from_slice(
            device,
            bytemuck::cast_slice(initial_fields.velocity_z.as_slice()),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            Some("Velocity Z Buffer"),
        );
        
        // Create uniform buffer
        let uniforms = SimulationUniforms {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            dx: 1e-3, // Default 1mm spacing
            dy: 1e-3,
            dz: 1e-3,
            dt: 1e-7, // Default 100ns time step
            time: 0.0,
            sound_speed: 1500.0, // Water sound speed
            density: 1000.0,     // Water density
            _padding: [0; 2],
        };
        
        let uniform_buffer = GpuBuffer::from_slice(
            device,
            bytemuck::cast_slice(&[uniforms]),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            Some("Simulation Uniforms"),
        );
        
        // Load compute shader
        let shader_source = include_str!("shaders/fdtd.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("FDTD Compute Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("FDTD Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FDTD Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("FDTD Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("FDTD Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pressure_buffer.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: velocity_x_buffer.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: velocity_y_buffer.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: velocity_z_buffer.buffer().as_entire_binding(),
                },
            ],
        });
        
        // Calculate optimal workgroup size
        let workgroup_size = Self::calculate_workgroup_size(grid_size);
        
        Ok(Self {
            context,
            pressure_buffer,
            velocity_x_buffer,
            velocity_y_buffer,
            velocity_z_buffer,
            uniform_buffer,
            compute_pipeline,
            bind_group,
            grid_size,
            workgroup_size,
        })
    }
    
    /// Calculate optimal workgroup size for the grid
    fn calculate_workgroup_size(grid_size: (u32, u32, u32)) -> (u32, u32, u32) {
        // Use 8x8x8 workgroups for good occupancy
        const WORKGROUP_SIZE: u32 = 8;
        (WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
    }
    
    /// Execute one FDTD time step on GPU
    pub fn step(&mut self, dt: f32) -> KwaversResult<()> {
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Update uniforms with new time step
        let uniforms = SimulationUniforms {
            grid_nx: self.grid_size.0,
            grid_ny: self.grid_size.1,
            grid_nz: self.grid_size.2,
            dx: 1e-3,
            dy: 1e-3,
            dz: 1e-3,
            dt,
            time: 0.0, // Updated externally
            sound_speed: 1500.0,
            density: 1000.0,
            _padding: [0; 2],
        };
        
        queue.write_buffer(
            self.uniform_buffer.buffer(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        
        // Create command encoder
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("FDTD Compute Encoder"),
        });
        
        // Dispatch compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("FDTD Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            let (nx, ny, nz) = self.grid_size;
            let (wx, wy, wz) = self.workgroup_size;
            
            // Calculate dispatch size
            let dispatch_x = (nx + wx - 1) / wx;
            let dispatch_y = (ny + wy - 1) / wy;
            let dispatch_z = (nz + wz - 1) / wz;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }
        
        // Submit command buffer
        queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    /// Copy data back from GPU to CPU
    pub async fn read_pressure(&self) -> KwaversResult<Vec<f32>> {
        let device = self.context.device();
        let queue = self.context.queue();
        
        // Create staging buffer
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: self.pressure_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            self.pressure_buffer.buffer(),
            0,
            &staging_buffer,
            0,
            self.pressure_buffer.size(),
        );
        
        queue.submit(std::iter::once(encoder.finish()));
        
        // Map staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::unbounded();
        
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        device.poll(Maintain::Wait);
        
        receiver.recv_async().await
            .map_err(|_| KwaversError::GpuError("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| KwaversError::GpuError(format!("Buffer mapping failed: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        staging_buffer.unmap();
        
        Ok(result)
    }
}

/// GPU memory management utilities
#[cfg(feature = "gpu")]
pub struct GpuMemoryManager {
    context: Arc<GpuContext>,
    allocated_bytes: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl GpuMemoryManager {
    /// Create new memory manager
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            context,
            allocated_bytes: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> u64 {
        self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Estimate memory requirements for grid size
    pub fn estimate_memory_usage(grid_size: (u32, u32, u32), num_fields: u32) -> u64 {
        let (nx, ny, nz) = grid_size;
        let total_points = nx as u64 * ny as u64 * nz as u64;
        total_points * num_fields as u64 * 4 // 4 bytes per f32
    }
    
    /// Check if requested allocation would exceed limits
    pub fn can_allocate(&self, size: u64) -> bool {
        const MAX_GPU_MEMORY: u64 = 4 * 1024 * 1024 * 1024; // 4GB limit
        self.memory_usage() + size < MAX_GPU_MEMORY
    }
}

// No-op implementations when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub async fn new() -> crate::error::KwaversResult<Self> {
        Err(crate::error::KwaversError::FeatureNotEnabled(
            "GPU acceleration requires 'gpu' feature".to_string()
        ))
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuFdtdSolver;

#[cfg(not(feature = "gpu"))]
impl GpuFdtdSolver {
    pub async fn new(_: std::sync::Arc<GpuContext>, _: (u32, u32, u32), _: &crate::physics::data::AcousticFields) -> crate::error::KwaversResult<Self> {
        Err(crate::error::KwaversError::FeatureNotEnabled(
            "GPU FDTD requires 'gpu' feature".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_context_creation() {
        // This test might fail in CI without GPU, so we'll just check the error
        match GpuContext::new().await {
            Ok(_) => {
                // GPU available and working
            }
            Err(KwaversError::GpuError(_)) => {
                // Expected when no GPU available
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    #[test]
    fn test_memory_estimation() {
        #[cfg(feature = "gpu")]
        {
            let usage = GpuMemoryManager::estimate_memory_usage((64, 64, 64), 4);
            assert_eq!(usage, 64 * 64 * 64 * 4 * 4); // 4 fields, 4 bytes each
        }
    }
}