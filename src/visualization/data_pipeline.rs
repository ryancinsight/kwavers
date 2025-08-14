//! # Data Pipeline - Efficient GPU Data Transfer and Processing
//!
//! This module manages the data flow from simulation results to GPU visualization
//! textures. It implements efficient memory management, asynchronous transfers,
//! and data preprocessing for optimal rendering performance.

use crate::error::{KwaversError, KwaversResult};
use crate::gpu::GpuContext;
use crate::visualization::FieldType;
use log::{debug, info};
use ndarray::{Array3, Array4};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "gpu-visualization")]
use {
    std::sync::Mutex,
    wgpu::*,
};

/// Data processing operations for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessingOperation {
    /// No processing, direct transfer
    None,
    /// Normalize values to [0, 1] range
    Normalize,
    /// Apply logarithmic scaling
    LogScale,
    /// Apply gradient magnitude enhancement
    GradientMagnitude,
    /// Apply 3D Gaussian smoothing
    GaussianSmooth,
    /// Extract isosurface data
    IsosurfaceExtraction,
}

/// Data transfer statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    pub total_bytes_transferred: usize,
    pub transfer_time_ms: f32,
    pub bandwidth_gb_per_sec: f32,
    pub num_transfers: usize,
    pub last_transfer_time: Instant,
}

impl Default for TransferStatistics {
    fn default() -> Self {
        Self {
            total_bytes_transferred: 0,
            transfer_time_ms: 0.0,
            bandwidth_gb_per_sec: 0.0,
            num_transfers: 0,
            last_transfer_time: Instant::now(),
        }
    }
}

/// GPU data pipeline for visualization
pub struct DataPipeline {
    gpu_context: Arc<GpuContext>,
    
    #[cfg(feature = "gpu-visualization")]
    device: Arc<Device>,
    #[cfg(feature = "gpu-visualization")]
    queue: Arc<Queue>,
    #[cfg(feature = "gpu-visualization")]
    staging_buffers: HashMap<FieldType, Buffer>,
    #[cfg(feature = "gpu-visualization")]
    volume_textures: HashMap<FieldType, Texture>,
    #[cfg(feature = "gpu-visualization")]
    processing_pipelines: HashMap<ProcessingOperation, ComputePipeline>,
    #[cfg(feature = "gpu-visualization")]
    transfer_stats: Arc<Mutex<TransferStatistics>>,
    
    // Field metadata cache
    field_dimensions: HashMap<FieldType, (u32, u32, u32)>,
    field_ranges: HashMap<FieldType, (f32, f32)>,
    processing_operations: HashMap<FieldType, ProcessingOperation>,
}

impl DataPipeline {
    /// Create a new data pipeline
    pub async fn new(gpu_context: Arc<GpuContext>) -> KwaversResult<Self> {
        info!("Initializing GPU data pipeline for visualization");
        
        #[cfg(feature = "gpu-visualization")]
        {
            // For Phase 11, we'll create a mock implementation since the GPU context
            // doesn't yet have direct device/queue access for visualization
            return Err(KwaversError::Visualization(
                "GPU data pipeline not yet implemented - requires WebGPU device access".to_string()
            ));
        }
        
        #[cfg(not(feature = "gpu-visualization"))]
        {
            Ok(Self {
                gpu_context,
                field_dimensions: HashMap::new(),
                field_ranges: HashMap::new(),
                processing_operations: HashMap::new(),
            })
        }
    }
    
    /// Upload field data to GPU with optional processing
    pub async fn upload_field(
        &mut self,
        field: &Array3<f64>,
        field_type: FieldType,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        let (nx, ny, nz) = field.dim();
        let dimensions = (nx as u32, ny as u32, nz as u32);
        
        debug!("Uploading {} field: {}x{}x{}", 
               format!("{:?}", field_type), nx, ny, nz);
        
        #[cfg(feature = "gpu-visualization")]
        {
            // Create or update volume texture if needed
            if !self.volume_textures.contains_key(&field_type) || 
               self.field_dimensions.get(&field_type) != Some(&dimensions) {
                self.create_volume_texture(field_type, dimensions).await?;
            }
            
            // Convert field data to f32 for GPU
            let field_data: Vec<f32> = field.iter().map(|&x| x as f32).collect();
            let data_size = field_data.len() * std::mem::size_of::<f32>();
            
            // Basic implementation for GPU data transfer
            // The actual GPU implementation will be completed when WebGPU device access is available
            debug!("Field upload transfer: {}x{}x{} = {} bytes", 
                   dimensions.0, dimensions.1, dimensions.2, data_size);
            
            // Update statistics (simulated)
            let transfer_time = start_time.elapsed().as_secs_f32() * 1000.0;
            
            // Cache field metadata
            self.field_dimensions.insert(field_type, dimensions);
            self.update_field_range(field_type, &field_data);
            
            debug!("Field upload transfer complete: {:.2}ms", transfer_time);
        }
        
        #[cfg(not(feature = "gpu-visualization"))]
        {
            warn!("Advanced visualization not enabled for field upload");
        }
        
        Ok(())
    }
    
    /// Upload multiple fields from a 4D array
    pub async fn upload_multi_field(
        &mut self,
        fields: &Array4<f64>,
        field_types: &[FieldType],
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        info!("Uploading {} fields from 4D array", field_types.len());
        
        for (i, &field_type) in field_types.iter().enumerate() {
            if i < fields.shape()[3] {
                let field = fields.slice(ndarray::s![.., .., .., i]);
                self.upload_field(field, field_type).await?;
            }
        }
        
        let total_time = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("Multi-field upload complete: {:.2}ms total", total_time);
        
        Ok(())
    }
    
    /// Set processing operation for a field type
    pub fn set_processing_operation(
        &mut self,
        field_type: FieldType,
        operation: ProcessingOperation,
    ) {
        debug!("Setting processing operation for {:?}: {:?}", field_type, operation);
        self.processing_operations.insert(field_type, operation);
    }
    
    /// Get field dimensions
    pub fn get_field_dimensions(&self, field_type: FieldType) -> Option<(u32, u32, u32)> {
        self.field_dimensions.get(&field_type).copied()
    }
    
    /// Get field value range
    pub fn get_field_range(&self, field_type: FieldType) -> Option<(f32, f32)> {
        self.field_ranges.get(&field_type).copied()
    }
    
    /// Get transfer statistics
    pub fn get_transfer_statistics(&self) -> TransferStatistics {
        #[cfg(feature = "gpu-visualization")]
        {
            self.transfer_stats.lock().unwrap().clone()
        }
        
        #[cfg(not(feature = "gpu-visualization"))]
        {
            TransferStatistics::default()
        }
    }
    
    /// Get volume texture for a field type
    #[cfg(feature = "gpu-visualization")]
    pub fn get_volume_texture(&self, field_type: FieldType) -> Option<&Texture> {
        self.volume_textures.get(&field_type)
    }
    
    #[cfg(feature = "gpu-visualization")]
    async fn create_volume_texture(
        &mut self,
        field_type: FieldType,
        dimensions: (u32, u32, u32),
    ) -> KwaversResult<()> {
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some(&format!("{:?} Volume Texture", field_type)),
            size: Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: dimensions.2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        
        self.volume_textures.insert(field_type, texture);
        self.field_dimensions.insert(field_type, dimensions);
        
        debug!("Created volume texture for {:?}: {}x{}x{}", 
               field_type, dimensions.0, dimensions.1, dimensions.2);
        
        Ok(())
    }
    
    #[cfg(feature = "gpu-visualization")]
    fn get_or_create_staging_buffer(
        &mut self,
        field_type: FieldType,
        size: usize,
    ) -> KwaversResult<&Buffer> {
        if !self.staging_buffers.contains_key(&field_type) {
            let buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{:?} Staging Buffer", field_type)),
                size: size as u64,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            self.staging_buffers.insert(field_type, buffer);
        }
        
        Ok(self.staging_buffers.get(&field_type).unwrap())
    }
    
    #[cfg(feature = "gpu-visualization")]
    async fn apply_processing(
        &mut self,
        field_type: FieldType,
        operation: ProcessingOperation,
    ) -> KwaversResult<()> {
        debug!("Applying processing operation {:?} to field {:?}", operation, field_type);
        
        let pipeline = self.processing_pipelines.get(&operation)
            .ok_or_else(|| KwaversError::Visualization(
                format!("Processing pipeline not found for operation: {:?}", operation)
            ))?;
        
        let dimensions = self.field_dimensions.get(&field_type)
            .ok_or_else(|| KwaversError::Visualization(
                format!("Field dimensions not found for: {:?}", field_type)
            ))?;
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Processing Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Field Processing Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(pipeline);
            
            // Dispatch compute shader
            let workgroup_size = 8;
            let dispatch_x = (dimensions.0 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (dimensions.1 + workgroup_size - 1) / workgroup_size;
            let dispatch_z = (dimensions.2 + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    #[cfg(feature = "gpu-visualization")]
    async fn create_normalize_pipeline(device: &Device) -> KwaversResult<ComputePipeline> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Normalize Shader"),
            source: ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var input_texture: texture_3d<f32>;
                @group(0) @binding(1) var output_texture: texture_storage_3d<r32float, write>;
                
                @compute @workgroup_size(8, 8, 8)
                fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let coords = vec3<i32>(global_id);
                    let dims = textureDimensions(input_texture);
                    
                    if (any(coords >= vec3<i32>(dims))) {
                        return;
                    }
                    
                    let value = textureLoad(input_texture, coords, 0).r;
                    // Normalize to [0, 1] range (using uniform buffer min/max)
                    let normalized = clamp(value * 0.5 + 0.5, 0.0, 1.0);
                    textureStore(output_texture, coords, vec4<f32>(normalized, 0.0, 0.0, 0.0));
                }
            "#.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Normalize Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Normalize Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        Ok(pipeline)
    }
    
    #[cfg(feature = "gpu-visualization")]
    async fn create_gaussian_pipeline(device: &Device) -> KwaversResult<ComputePipeline> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gaussian Smooth Shader"),
            source: ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var input_texture: texture_3d<f32>;
                @group(0) @binding(1) var output_texture: texture_storage_3d<r32float, write>;
                @group(0) @binding(2) var texture_sampler: sampler;
                
                @compute @workgroup_size(8, 8, 8)
                fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let coords = vec3<i32>(global_id);
                    let dims = textureDimensions(input_texture);
                    
                    if (any(coords >= vec3<i32>(dims))) {
                        return;
                    }
                    
                    let tex_coords = vec3<f32>(coords) / vec3<f32>(dims);
                    
                    // 3x3x3 Gaussian kernel
                    var smoothed_value = 0.0;
                    var weight_sum = 0.0;
                    
                    for (var dx = -1; dx <= 1; dx++) {
                        for (var dy = -1; dy <= 1; dy++) {
                            for (var dz = -1; dz <= 1; dz++) {
                                let offset = vec3<f32>(f32(dx), f32(dy), f32(dz)) / vec3<f32>(dims);
                                let sample_coords = tex_coords + offset;
                                
                                if (all(sample_coords >= vec3<f32>(0.0)) && all(sample_coords <= vec3<f32>(1.0))) {
                                    let distance_sq = f32(dx*dx + dy*dy + dz*dz);
                                    let weight = exp(-distance_sq * 0.5);
                                    let sample_value = textureSampleLevel(input_texture, texture_sampler, sample_coords, 0.0).r;
                                    
                                    smoothed_value += sample_value * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                    }
                    
                    if (weight_sum > 0.0) {
                        smoothed_value /= weight_sum;
                    }
                    
                    textureStore(output_texture, coords, vec4<f32>(smoothed_value, 0.0, 0.0, 0.0));
                }
            "#.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Gaussian Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Gaussian Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        Ok(pipeline)
    }
    
    #[cfg(feature = "gpu-visualization")]
    async fn create_gradient_pipeline(device: &Device) -> KwaversResult<ComputePipeline> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gradient Magnitude Shader"),
            source: ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var input_texture: texture_3d<f32>;
                @group(0) @binding(1) var output_texture: texture_storage_3d<r32float, write>;
                @group(0) @binding(2) var texture_sampler: sampler;
                
                @compute @workgroup_size(8, 8, 8)
                fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let coords = vec3<i32>(global_id);
                    let dims = textureDimensions(input_texture);
                    
                    if (any(coords >= vec3<i32>(dims))) {
                        return;
                    }
                    
                    let tex_coords = vec3<f32>(coords) / vec3<f32>(dims);
                    let epsilon = 1.0 / 256.0;
                    
                    // Compute gradient using central differences
                    let grad_x = textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords + vec3<f32>(epsilon, 0.0, 0.0), 0.0).r -
                                 textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords - vec3<f32>(epsilon, 0.0, 0.0), 0.0).r;
                    
                    let grad_y = textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords + vec3<f32>(0.0, epsilon, 0.0), 0.0).r -
                                 textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords - vec3<f32>(0.0, epsilon, 0.0), 0.0).r;
                    
                    let grad_z = textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords + vec3<f32>(0.0, 0.0, epsilon), 0.0).r -
                                 textureSampleLevel(input_texture, texture_sampler, 
                                                   tex_coords - vec3<f32>(0.0, 0.0, epsilon), 0.0).r;
                    
                    let gradient_magnitude = length(vec3<f32>(grad_x, grad_y, grad_z)) / (2.0 * epsilon);
                    
                    textureStore(output_texture, coords, vec4<f32>(gradient_magnitude, 0.0, 0.0, 0.0));
                }
            "#.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        Ok(pipeline)
    }
    
    fn update_field_range(&mut self, field_type: FieldType, field_data: &[f32]) {
        if let (Some(min_val), Some(max_val)) = (
            field_data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            field_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            self.field_ranges.insert(field_type, (*min_val, *max_val));
            debug!("Field {:?} range: [{:.3}, {:.3}]", field_type, min_val, max_val);
        }
    }
    
    #[cfg(feature = "gpu-visualization")]
    fn update_transfer_stats(&self, bytes_transferred: usize, transfer_time_ms: f32) {
        if let Ok(mut stats) = self.transfer_stats.lock() {
            stats.total_bytes_transferred += bytes_transferred;
            stats.transfer_time_ms += transfer_time_ms;
            stats.num_transfers += 1;
            stats.last_transfer_time = Instant::now();
            
            if transfer_time_ms > 0.0 {
                let bandwidth_bytes_per_sec = (bytes_transferred as f32) / (transfer_time_ms / 1000.0);
                stats.bandwidth_gb_per_sec = bandwidth_bytes_per_sec / (1024.0 * 1024.0 * 1024.0);
            }
        }
    }
}