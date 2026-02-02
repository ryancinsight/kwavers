# Sprint 214 Session 3: GPU Beamforming Pipeline & Benchmark Remediation

**Date**: 2026-02-02  
**Sprint**: 214 Session 3  
**Status**: 🔄 IN PROGRESS  
**Priority**: P0 (Critical Infrastructure)  
**Estimated Effort**: 12-17 hours  

---

## Executive Summary

Sprint 214 Session 3 focuses on two critical P0 infrastructure items:

1. **GPU-Accelerated Beamforming Pipeline** (10-14 hours)
   - WGPU compute shaders for delay-and-sum beamforming
   - CPU/GPU equivalence validation
   - Performance benchmarking and optimization
   - Integration with existing beamforming infrastructure

2. **Benchmark Stub Remediation** (2-3 hours)
   - Remove/disable 18+ stub benchmarks in `performance_benchmark.rs`
   - Implement production-grade benchmarks using real solver code
   - Align with "no placeholders" principle

**Research Integration**: This session directly implements patterns from jwave (GPU acceleration), k-Wave (high-performance computing), and dbua (real-time beamforming).

---

## Session 3 Part 1: GPU Beamforming Pipeline (10-14 hours)

### Objective

Implement GPU-accelerated delay-and-sum (DAS) beamforming using WGPU compute shaders, achieving:
- Mathematical equivalence to CPU implementation
- 10-100× speedup for large arrays (128+ channels)
- Real-time imaging capability (30+ FPS for clinical applications)
- Clean Architecture compliance with SSOT enforcement

### Background

**Current State**:
- ✅ CPU DAS implementation exists: `analysis::signal_processing::beamforming::time_domain`
- ✅ GPU infrastructure exists: `gpu::mod.rs` with WGPU context, compute pipelines
- ✅ Mathematical specification documented (Van Trees 2002)
- ❌ No GPU beamforming implementation yet

**Research Patterns** (jwave, k-Wave, dbua):
- **jwave**: JAX-based GPU acceleration with automatic differentiation
- **k-Wave**: MATLAB GPU acceleration via CUDA (10-50× speedup)
- **dbua**: Real-time neural beamforming with GPU inference

**Mathematical Foundation**:

Delay-and-Sum (DAS) beamforming for focal point **r**:

```
y(r, t) = Σᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
```

where:
- `N` = number of sensors
- `wᵢ` = apodization weight for sensor i
- `xᵢ(t)` = received RF signal at sensor i
- `τᵢ(r)` = time-of-flight delay from focal point r to sensor i
- `y(r, t)` = beamformed output

Time-of-flight calculation:
```
τᵢ(r) = ||rᵢ - r|| / c
```

where:
- `rᵢ` = position of sensor i
- `r` = focal point position
- `c` = sound speed (typically 1540 m/s for soft tissue)

### Architecture Design

#### Layer Responsibilities

```
Clinical Layer (7)
    ↓
Analysis Layer (6) - Beamforming Algorithms
    ├── analysis::signal_processing::beamforming::time_domain (CPU)
    └── analysis::signal_processing::beamforming::gpu (NEW - GPU)
    ↓
Infrastructure Layer (1) - GPU Compute
    └── gpu::compute (WGPU pipelines, shaders)
    ↓
Core Layer (0) - Error Handling
```

**SSOT Enforcement**:
- CPU implementation: `analysis::signal_processing::beamforming::time_domain::delay_and_sum`
- GPU implementation: `analysis::signal_processing::beamforming::gpu::delay_and_sum_gpu`
- Both implementations must produce mathematically equivalent results (within floating-point tolerance)
- Shared configuration: `BeamformingConfig` (single source of truth for parameters)

#### Module Structure

```
src/analysis/signal_processing/beamforming/
├── time_domain/          # CPU implementation (existing)
│   ├── mod.rs
│   └── das.rs
└── gpu/                  # GPU implementation (NEW)
    ├── mod.rs            # Module exports and configuration
    ├── das_gpu.rs        # GPU DAS orchestration (Rust)
    ├── shaders/          # WGSL compute shaders
    │   ├── das.wgsl      # Delay-and-sum kernel
    │   └── utils.wgsl    # Shared utilities (interpolation, etc.)
    └── tests.rs          # CPU/GPU equivalence tests
```

### Implementation Plan

#### Task 1: Audit Existing GPU Infrastructure (30 minutes)

**Objective**: Understand existing GPU compute patterns and reusable components

**Files to Review**:
- `src/gpu/mod.rs` - GpuContext, capabilities
- `src/gpu/compute.rs` - GpuCompute, pipeline creation
- `src/gpu/buffer.rs` - Buffer management
- `src/gpu/fdtd.rs` - Example GPU solver (FDTD)
- `src/gpu/shaders/` - Existing WGSL shaders

**Key Questions**:
1. How are compute pipelines created and managed?
2. What buffer types and layouts are used?
3. How is CPU ↔ GPU data transfer handled?
4. What shader utilities exist (interpolation, atomics)?

**Deliverable**: Architecture notes for GPU beamforming design

---

#### Task 2: Design GPU Beamforming Architecture (1 hour)

**Objective**: Define clean architecture for GPU DAS with SSOT compliance

**Design Decisions**:

1. **Parallelization Strategy**:
   - **Option A**: Parallelize over focal points (grid) - Each workgroup processes one focal point
   - **Option B**: Parallelize over sensors - Each workgroup processes one sensor
   - **Option C**: Hybrid - 2D workgroups (focal points × sensors)
   - **Decision**: Option A (focal point parallelism) for better cache locality

2. **Data Layout**:
   - **Input**: RF data `Array3<f32>` shape `(n_sensors, n_frames, n_samples)`
   - **Output**: Beamformed image `Array3<f32>` shape `(n_x, n_y, n_z)`
   - **GPU Layout**: Flatten to 1D buffers (WGPU requirement)

3. **Delay Calculation**:
   - **Option A**: Pre-compute delays on CPU, upload to GPU
   - **Option B**: Compute delays on GPU in shader
   - **Decision**: Option B (GPU computation) - reduces CPU/GPU transfer

4. **Interpolation**:
   - **Option A**: Nearest-neighbor (integer sample shifts)
   - **Option B**: Linear interpolation (sub-sample accuracy)
   - **Option C**: Cubic interpolation (higher accuracy, slower)
   - **Decision**: Start with Option A (nearest-neighbor), add Option B in future

**Deliverables**:
- Architecture document with design rationale
- Interface specification for `GpuDasBeamformer`
- WGSL shader function signatures

---

#### Task 3: Create WGSL Compute Shader (2-3 hours)

**Objective**: Implement delay-and-sum kernel in WGSL

**File**: `src/analysis/signal_processing/beamforming/gpu/shaders/das.wgsl`

**Shader Structure**:

```wgsl
// Delay-and-Sum Beamforming Compute Shader
// 
// Parallelization: Each workgroup processes one focal point
// Input: RF data (n_sensors × n_samples), sensor positions
// Output: Beamformed pixel intensity

@group(0) @binding(0) var<storage, read> rf_data: array<f32>;           // Input RF data
@group(0) @binding(1) var<storage, read> sensor_positions: array<vec3f>; // Sensor positions [m]
@group(0) @binding(2) var<storage, read> apodization: array<f32>;       // Apodization weights
@group(0) @binding(3) var<storage, write> output: array<f32>;           // Beamformed output

struct PushConstants {
    n_sensors: u32,
    n_samples: u32,
    sampling_rate: f32,
    sound_speed: f32,
    focal_x: f32,
    focal_y: f32,
    focal_z: f32,
}

var<push_constant> params: PushConstants;

@compute @workgroup_size(256, 1, 1)
fn das_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sensor_idx = global_id.x;
    if (sensor_idx >= params.n_sensors) {
        return;
    }
    
    // 1. Compute time-of-flight delay
    let focal_point = vec3f(params.focal_x, params.focal_y, params.focal_z);
    let sensor_pos = sensor_positions[sensor_idx];
    let distance = length(focal_point - sensor_pos);
    let delay_seconds = distance / params.sound_speed;
    let delay_samples = delay_seconds * params.sampling_rate;
    
    // 2. Apply delay and sum (nearest-neighbor interpolation)
    let sample_idx = u32(delay_samples);
    if (sample_idx < params.n_samples) {
        let rf_idx = sensor_idx * params.n_samples + sample_idx;
        let weighted_value = rf_data[rf_idx] * apodization[sensor_idx];
        
        // 3. Atomic accumulation (thread-safe summation)
        atomicAdd(&output[0], weighted_value);
    }
}
```

**Key Implementation Details**:

1. **Memory Layout**:
   - RF data: Flattened 2D array `[sensor_0_sample_0, ..., sensor_0_sample_N, sensor_1_sample_0, ...]`
   - Sensor positions: Array of `vec3<f32>` (x, y, z coordinates in meters)
   - Apodization: Array of `f32` weights (typically normalized to sum = 1)

2. **Atomic Operations**:
   - Use `atomicAdd` for thread-safe accumulation across sensors
   - Alternative: Use workgroup shared memory and reduce

3. **Numerical Precision**:
   - Use `f32` for compatibility (most GPUs don't support `f64` in compute shaders)
   - Validate precision loss vs CPU `f64` implementation

4. **Bounds Checking**:
   - Guard against out-of-bounds access (sensor_idx, sample_idx)
   - Handle edge cases (delays outside valid sample range)

**Testing Strategy**:
- Unit tests for delay calculation (compare CPU vs GPU)
- Integration tests for full beamforming pipeline
- Property-based tests (monotonicity, energy conservation)

**Deliverables**:
- `das.wgsl` compute shader (150-200 lines)
- `utils.wgsl` shared utilities (interpolation functions)
- Shader validation tests

---

#### Task 4: Implement GPU Beamforming Orchestration (3-4 hours)

**Objective**: Rust-side GPU pipeline orchestration and buffer management

**File**: `src/analysis/signal_processing/beamforming/gpu/das_gpu.rs`

**Structure**:

```rust
//! GPU-accelerated delay-and-sum beamforming
//!
//! # Architecture
//!
//! This module implements GPU-accelerated DAS beamforming using WGPU compute shaders.
//! It maintains mathematical equivalence with the CPU implementation while achieving
//! 10-100× speedup for large arrays.
//!
//! # Mathematical Specification
//!
//! See `time_domain::das` for complete mathematical specification.
//! GPU implementation uses identical algorithm with f32 precision.
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::gpu::GpuDasBeamformer;
//! use kwavers::gpu::GpuContext;
//!
//! let gpu_context = GpuContext::new().await?;
//! let beamformer = GpuDasBeamformer::new(&gpu_context)?;
//!
//! let beamformed = beamformer.process(
//!     &rf_data,           // Array3<f32>: (n_sensors, n_frames, n_samples)
//!     &sensor_positions,  // Vec<[f64; 3]>: sensor coordinates
//!     &focal_points,      // Vec<[f64; 3]>: imaging grid points
//!     sampling_rate,      // f64: Hz
//!     sound_speed,        // f64: m/s
//!     &apodization,       // Vec<f64>: weights (optional)
//! )?;
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use crate::gpu::{GpuContext, GpuBuffer};
use ndarray::{Array3, Array1};
use wgpu::{self, util::DeviceExt};

/// GPU delay-and-sum beamformer
pub struct GpuDasBeamformer {
    /// GPU context
    gpu_context: Arc<GpuContext>,
    
    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,
    
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
    
    /// Pipeline layout (with push constants)
    pipeline_layout: wgpu::PipelineLayout,
}

impl GpuDasBeamformer {
    /// Create new GPU DAS beamformer
    pub fn new(gpu_context: &Arc<GpuContext>) -> KwaversResult<Self> {
        // 1. Load shader module
        let shader_source = include_str!("shaders/das.wgsl");
        let shader_module = gpu_context.device().create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("DAS Beamforming Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            }
        );
        
        // 2. Create bind group layout
        let bind_group_layout = gpu_context.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DAS Bind Group Layout"),
                entries: &[
                    // Binding 0: RF data (read-only storage buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: Sensor positions
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: Apodization weights
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: Output buffer
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
                ],
            }
        );
        
        // 3. Create pipeline layout with push constants
        let pipeline_layout = gpu_context.device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("DAS Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..64, // 7 × f32 + 1 × u32 = 32 bytes (aligned to 64)
                }],
            }
        );
        
        // 4. Create compute pipeline
        let pipeline = gpu_context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("DAS Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "das_kernel",
            }
        );
        
        Ok(Self {
            gpu_context: Arc::clone(gpu_context),
            pipeline,
            bind_group_layout,
            pipeline_layout,
        })
    }
    
    /// Process RF data with GPU beamforming
    pub fn process(
        &self,
        rf_data: &Array3<f32>,
        sensor_positions: &[[f64; 3]],
        focal_points: &[[f64; 3]],
        sampling_rate: f64,
        sound_speed: f64,
        apodization: Option<&[f64]>,
    ) -> KwaversResult<Array3<f32>> {
        let (n_sensors, n_frames, n_samples) = rf_data.dim();
        let n_focal_points = focal_points.len();
        
        // Validate inputs
        if sensor_positions.len() != n_sensors {
            return Err(KwaversError::invalid_argument(
                "sensor_positions length must match n_sensors"
            ));
        }
        
        // Default apodization (uniform weights)
        let apod_weights: Vec<f32> = if let Some(weights) = apodization {
            weights.iter().map(|&w| w as f32).collect()
        } else {
            vec![1.0 / n_sensors as f32; n_sensors]
        };
        
        // 1. Create GPU buffers
        let rf_buffer = self.create_buffer_from_array(rf_data)?;
        let sensor_buffer = self.create_sensor_position_buffer(sensor_positions)?;
        let apod_buffer = self.create_buffer_from_slice(&apod_weights)?;
        let output_buffer = self.create_output_buffer(n_focal_points)?;
        
        // 2. Create bind group
        let bind_group = self.gpu_context.device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("DAS Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: rf_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sensor_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: apod_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            }
        );
        
        // 3. Create command encoder
        let mut encoder = self.gpu_context.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("DAS Command Encoder"),
            }
        );
        
        // 4. Dispatch compute pass for each focal point
        for focal_point in focal_points {
            let mut compute_pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("DAS Compute Pass"),
                }
            );
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Set push constants
            let push_constants = [
                n_sensors as u32,
                n_samples as u32,
                sampling_rate as f32,
                sound_speed as f32,
                focal_point[0] as f32,
                focal_point[1] as f32,
                focal_point[2] as f32,
            ];
            compute_pass.set_push_constants(
                0,
                bytemuck::cast_slice(&push_constants),
            );
            
            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_count = (n_sensors + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }
        
        drop(compute_pass);
        
        // 5. Submit commands and wait
        self.gpu_context.queue().submit(std::iter::once(encoder.finish()));
        self.gpu_context.device().poll(wgpu::Maintain::Wait);
        
        // 6. Read back results
        let output_data = self.read_buffer_to_vec(&output_buffer, n_focal_points)?;
        
        // 7. Reshape to output array
        // TODO: Proper 3D reshaping based on focal_points grid structure
        let output = Array3::from_shape_vec((n_focal_points, 1, 1), output_data)?;
        
        Ok(output)
    }
    
    // Helper methods for buffer creation
    fn create_buffer_from_array(&self, data: &Array3<f32>) -> KwaversResult<wgpu::Buffer> {
        let flat_data: Vec<f32> = data.iter().copied().collect();
        Ok(self.gpu_context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("RF Data Buffer"),
                contents: bytemuck::cast_slice(&flat_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        ))
    }
    
    fn create_sensor_position_buffer(&self, positions: &[[f64; 3]]) -> KwaversResult<wgpu::Buffer> {
        let flat_positions: Vec<f32> = positions.iter()
            .flat_map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect();
        Ok(self.gpu_context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Sensor Position Buffer"),
                contents: bytemuck::cast_slice(&flat_positions),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        ))
    }
    
    fn create_buffer_from_slice(&self, data: &[f32]) -> KwaversResult<wgpu::Buffer> {
        Ok(self.gpu_context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Apodization Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        ))
    }
    
    fn create_output_buffer(&self, size: usize) -> KwaversResult<wgpu::Buffer> {
        Ok(self.gpu_context.device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: (size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE 
                     | wgpu::BufferUsages::COPY_SRC 
                     | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ))
    }
    
    fn read_buffer_to_vec(&self, buffer: &wgpu::Buffer, size: usize) -> KwaversResult<Vec<f32>> {
        // Create staging buffer for readback
        let staging_buffer = self.gpu_context.device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: (size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        
        // Copy GPU buffer to staging buffer
        let mut encoder = self.gpu_context.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            }
        );
        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );
        self.gpu_context.queue().submit(std::iter::once(encoder.finish()));
        
        // Map and read staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.gpu_context.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("Buffer mapping failed: {:?}", e),
            })
        })?;
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
}
```

**Key Design Decisions**:

1. **Buffer Management**: Use WGPU's `create_buffer_init` for convenience, add custom buffer pool in future optimization
2. **Push Constants**: Use for per-focal-point parameters (avoids buffer updates)
3. **Synchronization**: Use `poll(Maintain::Wait)` for simplicity (async in future)
4. **Error Handling**: All operations return `KwaversResult` with detailed error contexts

**Deliverables**:
- `das_gpu.rs` (400-500 lines)
- `mod.rs` module exports
- Integration with existing beamforming infrastructure

---

#### Task 5: CPU/GPU Equivalence Tests (2 hours)

**Objective**: Validate mathematical equivalence between CPU and GPU implementations

**File**: `src/analysis/signal_processing/beamforming/gpu/tests.rs`

**Test Strategy**:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_cpu_gpu_equivalence_single_focal_point() {
        // 1. Setup test case
        let n_sensors = 32;
        let n_samples = 1000;
        let sampling_rate = 10e6; // 10 MHz
        let sound_speed = 1540.0; // m/s
        
        // Linear array: 32 elements, 0.3mm pitch
        let sensor_positions: Vec<[f64; 3]> = (0..n_sensors)
            .map(|i| [i as f64 * 0.0003, 0.0, 0.0])
            .collect();
        
        // Focal point at 20mm depth
        let focal_point = [0.0048, 0.0, 0.02]; // Center of array, 20mm depth
        
        // Generate synthetic RF data (plane wave + noise)
        let rf_data = generate_synthetic_rf_data(n_sensors, n_samples, sampling_rate);
        
        // 2. Run CPU beamforming
        let cpu_result = delay_and_sum(
            &rf_data,
            sampling_rate,
            &compute_delays(&sensor_positions, &focal_point, sound_speed),
            &vec![1.0 / n_sensors as f64; n_sensors],
            DelayReference::SensorIndex(0),
        ).unwrap();
        
        // 3. Run GPU beamforming
        let gpu_context = GpuContext::new().await.unwrap();
        let gpu_beamformer = GpuDasBeamformer::new(&Arc::new(gpu_context)).unwrap();
        let gpu_result = gpu_beamformer.process(
            &rf_data.mapv(|x| x as f32),
            &sensor_positions,
            &[focal_point],
            sampling_rate,
            sound_speed,
            None,
        ).unwrap();
        
        // 4. Compare results (allow 1e-5 relative error for f32 precision)
        let cpu_value = cpu_result[[0, 0, 0]];
        let gpu_value = gpu_result[[0, 0, 0]] as f64;
        
        assert_relative_eq!(cpu_value, gpu_value, epsilon = 1e-5);
    }
    
    #[tokio::test]
    async fn test_cpu_gpu_equivalence_multiple_focal_points() {
        // Test 10×10 imaging grid (100 focal points)
        // Verify all pixels match between CPU and GPU
    }
    
    #[tokio::test]
    async fn test_gpu_beamforming_with_apodization() {
        // Test with Hamming window apodization
        // Verify sidelobe suppression
    }
    
    #[tokio::test]
    async fn test_gpu_beamforming_edge_cases() {
        // Test with focal point outside sensor array bounds
        // Test with zero RF data
        // Test with single sensor
    }
    
    fn generate_synthetic_rf_data(
        n_sensors: usize,
        n_samples: usize,
        sampling_rate: f64,
    ) -> Array3<f64> {
        // Generate plane wave arriving from 45° angle
        // Add Gaussian noise (SNR = 20 dB)
        // Return Array3 with shape (n_sensors, 1, n_samples)
        todo!("Implement synthetic data generation")
    }
    
    fn compute_delays(
        sensor_positions: &[[f64; 3]],
        focal_point: &[f64; 3],
        sound_speed: f64,
    ) -> Vec<f64> {
        sensor_positions.iter()
            .map(|&[x, y, z]| {
                let dx = focal_point[0] - x;
                let dy = focal_point[1] - y;
                let dz = focal_point[2] - z;
                (dx * dx + dy * dy + dz * dz).sqrt() / sound_speed
            })
            .collect()
    }
}
```

**Test Categories**:

1. **Correctness Tests**:
   - CPU/GPU equivalence (single focal point)
   - CPU/GPU equivalence (multiple focal points)
   - Apodization correctness
   - Edge cases (boundary conditions)

2. **Property-Based Tests**:
   - Energy conservation (output energy ≤ input energy)
   - Monotonicity (closer sensors → earlier samples)
   - Linearity (beamform(αx) = α·beamform(x))

3. **Performance Tests**:
   - Throughput (focal points/second)
   - Latency (end-to-end processing time)
   - Memory usage (GPU buffer allocation)

**Deliverables**:
- `tests.rs` (300-400 lines)
- 10+ comprehensive tests
- 100% pass rate

---

#### Task 6: Performance Validation & Benchmarks (2-3 hours)

**Objective**: Measure GPU speedup and validate real-time imaging capability

**File**: `benches/gpu_beamforming_benchmark.rs`

**Benchmark Structure**:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::analysis::signal_processing::beamforming::{
    time_domain::delay_and_sum,
    gpu::GpuDasBeamformer,
};
use kwavers::gpu::GpuContext;

fn bench_cpu_vs_gpu_beamforming(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu_beamforming");
    
    // Test configurations: (n_sensors, n_focal_points)
    let configs = vec![
        (32, 100),      // Small: 32 channels, 10×10 grid
        (64, 400),      // Medium: 64 channels, 20×20 grid
        (128, 1600),    // Large: 128 channels, 40×40 grid
        (256, 6400),    // XL: 256 channels, 80×80 grid
    ];
    
    for (n_sensors, n_focal_points) in configs {
        let n_samples = 2000;
        let sampling_rate = 10e6;
        let sound_speed = 1540.0;
        
        // Generate test data
        let sensor_positions = generate_linear_array(n_sensors, 0.0003);
        let focal_points = generate_imaging_grid(n_focal_points);
        let rf_data = generate_synthetic_rf(n_sensors, n_samples);
        
        // Benchmark CPU
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}ch_{}px", n_sensors, n_focal_points)),
            &(n_sensors, n_focal_points),
            |b, _| {
                b.iter(|| {
                    for focal_point in &focal_points {
                        let delays = compute_delays(&sensor_positions, focal_point, sound_speed);
                        delay_and_sum(
                            black_box(&rf_data),
                            sampling_rate,
                            &delays,
                            &vec![1.0 / n_sensors as f64; n_sensors],
                            DelayReference::SensorIndex(0),
                        ).unwrap();
                    }
                });
            },
        );
        
        // Benchmark GPU
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let gpu_context = runtime.block_on(GpuContext::new()).unwrap();
        let gpu_beamformer = GpuDasBeamformer::new(&Arc::new(gpu_context)).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("gpu", format!("{}ch_{}px", n_sensors, n_focal_points)),
            &(n_sensors, n_focal_points),
            |b, _| {
                b.iter(|| {
                    gpu_beamformer.process(
                        black_box(&rf_data.mapv(|x| x as f32)),
                        &sensor_positions,
                        &focal_points,
                        sampling_rate,
                        sound_speed,
                        None,
                    ).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_cpu_vs_gpu_beamforming);
criterion_main!(benches);
```

**Performance Targets**:

| Configuration | CPU Time | GPU Time | Speedup | FPS (30 Hz Target) |
|---------------|----------|----------|---------|-------------------|
| 32ch × 100px  | ~10ms    | ~1ms     | 10×     | ✅ 1000 FPS       |
| 64ch × 400px  | ~80ms    | ~4ms     | 20×     | ✅ 250 FPS        |
| 128ch × 1600px| ~640ms   | ~20ms    | 32×     | ✅ 50 FPS         |
| 256ch × 6400px| ~5.1s    | ~100ms   | 51×     | ✅ 10 FPS         |

**Real-Time Criterion**: GPU implementation must achieve ≥30 FPS for clinical imaging (64ch × 400px)

**Deliverables**:
- `gpu_beamforming_benchmark.rs` (200-300 lines)
- Criterion benchmark report
- Performance analysis document

---

#### Task 7: Documentation (1 hour)

**Objective**: Document GPU beamforming architecture, usage, and performance

**Files**:
- `src/analysis/signal_processing/beamforming/gpu/mod.rs` - Rustdoc API documentation
- `docs/sprints/SPRINT_214_SESSION_3_SUMMARY.md` - Session summary
- `README.md` - Update feature list

**Documentation Sections**:

1. **API Documentation** (Rustdoc):
   - Module-level overview with mathematical specification
   - `GpuDasBeamformer` struct and methods
   - Usage examples with complete code
   - Performance characteristics and limitations
   - References to research papers (Van Trees, k-Wave, jwave)

2. **Session Summary**:
   - Implementation details (shader code, buffer layouts)
   - Testing strategy and results
   - Performance benchmarks and analysis
   - Future optimizations (sub-sample interpolation, multi-GPU)

3. **User Guide**:
   - Installation requirements (WGPU, GPU drivers)
   - Feature flag: `--features gpu`
   - Example workflows for clinical imaging
   - Troubleshooting common issues

**Deliverables**:
- Comprehensive Rustdoc comments (inline)
- `SPRINT_214_SESSION_3_SUMMARY.md` (500-800 lines)
- Updated `README.md`

---

### Success Criteria

**P0 (Required)**:
- ✅ GPU beamforming produces mathematically equivalent results to CPU (within f32 precision)
- ✅ All equivalence tests passing (10/10)
- ✅ GPU achieves ≥10× speedup for 128+ channel arrays
- ✅ Real-time capability: ≥30 FPS for 64ch × 400px imaging
- ✅ Zero compilation errors, zero warnings
- ✅ Clean Architecture compliance (no circular dependencies)
- ✅ SSOT enforcement (single interface, dual backends)

**P1 (Nice-to-Have)**:
- 🟡 Sub-sample interpolation (linear or cubic)
- 🟡 Multi-GPU support
- 🟡 Streaming mode (process RF data as it arrives)
- 🟡 Adaptive apodization (optimized per focal point)

**P2 (Future)**:
- 🟢 GPU MUSIC implementation
- 🟢 GPU MVDR (adaptive beamforming)
- 🟢 GPU synthetic aperture imaging

---

## Session 3 Part 2: Benchmark Stub Remediation (2-3 hours)

### Objective

Remove or replace 18+ stub benchmarks in `benches/performance_benchmark.rs` per "no placeholders" principle.

**Strategy**: Remove stub benchmarks, document removal rationale, add production benchmarks for implemented solvers.

### Background

**Audit Findings** (Phase 6):
- 18 stub helper methods in `performance_benchmark.rs`
- Stubs return zero-filled data, clones, or perform no-op operations
- Benchmarks measure placeholder performance, not real physics
- Violates Dev rules: "Absolute Prohibition: TODOs, stubs, dummy data"

**Decision**: Disable all stub benchmarks, implement production benchmarks for real solvers (FDTD, PSTD, k-space).

### Implementation Plan

#### Task 1: Disable Stub Benchmarks (1 hour)

**File**: `benches/performance_benchmark.rs`

**Actions**:

1. **Add Module Documentation**:
```rust
//! # Performance Benchmarks - Production Solvers Only
//!
//! This benchmark suite measures the performance of production-grade solver implementations.
//! All stub benchmarks have been removed per Sprint 209 Phase 2 cleanup.
//!
//! ## Available Benchmarks
//!
//! ### P0 Solvers (Production-Ready)
//! - FDTD acoustic wave propagation (`benchmark_fdtd_production`)
//! - PSTD k-space propagation (`benchmark_pstd_production`)
//! - k-Wave angular spectrum (`benchmark_kwave_production`)
//!
//! ### P1 Solvers (In Development - Sprint 211-212)
//! - Westervelt nonlinear acoustics (awaiting implementation)
//! - Elastic wave equation (awaiting implementation)
//! - Transcranial FUS (awaiting implementation)
//!
//! ## Removed Stub Benchmarks (2025-01-14)
//!
//! The following benchmarks were removed because they measured placeholder
//! operations instead of real physics:
//!
//! - `update_velocity_fdtd()` - Empty stub, no staggered grid
//! - `update_pressure_fdtd()` - Empty stub, no wave equation
//! - `update_westervelt()` - Empty stub, no nonlinear term
//! - `simulate_fft_operations()` - No FFT calls
//! - `simulate_angular_spectrum_propagation()` - No angular spectrum
//! - ... (see BENCHMARK_STUB_REMEDIATION_PLAN.md for complete list)
//!
//! ## Future Benchmarks (Sprint 211-212)
//!
//! When P1 solvers are implemented, add benchmarks for:
//! - Nonlinear acoustics (Westervelt, KZK equations)
//! - Elastic wave propagation (displacement tracking, stiffness estimation)
//! - Multi-physics coupling (cavitation, thermal effects)
//!
//! ## References
//!
//! - BENCHMARK_STUB_REMEDIATION_PLAN.md (Sprint 209 Phase 2)
//! - backlog.md "Sprint 211-212: Advanced Physics Implementation"
```

2. **Rename Stub Methods**:
```rust
/// DISABLED - Stub implementation removed (Sprint 209 Phase 2)
///
/// This method was a placeholder that performed no actual FDTD velocity update.
/// It has been disabled to prevent misleading benchmark data.
///
/// # Implementation Status
///
/// Real FDTD velocity update is implemented in:
/// - `solver::forward::fdtd::acoustic::AcousticFdtdSolver::update_velocity()`
///
/// # Future Work
///
/// When production FDTD benchmarks are needed, use the real solver:
/// ```rust,ignore
/// let mut solver = AcousticFdtdSolver::new(grid, medium)?;
/// b.iter(|| solver.step(dt));
/// ```
///
/// See: backlog.md Sprint 211 "Production FDTD Benchmarks"
#[allow(dead_code)]
fn update_velocity_fdtd_DISABLED(
    velocity: &mut Array3<f64>,
    _pressure: &Array3<f64>,
    _dt: f64,
) {
    // Stub removed - no operation
    let _ = velocity;
}
```

3. **Disable Benchmark Functions**:
```rust
/// DISABLED - Benchmark uses stub implementations (Sprint 209 Phase 2)
///
/// This benchmark has been disabled because it measured placeholder operations
/// instead of real FDTD physics.
///
/// # Removal Rationale
///
/// Per Dev rules:
/// - "Absolute Prohibition: TODOs, stubs, dummy data, zero-filled placeholders"
/// - "Correctness > Functionality: Placeholder benchmarks produce invalid data"
///
/// # Implementation Plan
///
/// To re-enable this benchmark:
/// 1. Implement production FDTD solver with real physics (Sprint 211, 12-16h)
/// 2. Replace stub methods with solver.step() calls
/// 3. Add mathematical validation (compare to analytical solutions)
/// 4. Verify performance meets real-time targets
///
/// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section 2A
#[allow(dead_code)]
fn benchmark_fdtd_wave_DISABLED(&self) -> KwaversResult<BenchmarkResult> {
    panic!("Benchmark disabled - awaiting production FDTD implementation (Sprint 211)");
}
```

4. **Update Criterion Registration**:
```rust
criterion_group!(
    benches,
    // Production benchmarks (enabled)
    benchmark_grid_operations,
    benchmark_memory_allocation,
    
    // Stub benchmarks (disabled - Sprint 209 Phase 2)
    // benchmark_fdtd_wave,              // DISABLED - uses stub updates
    // benchmark_pstd_wave,              // DISABLED - uses stub FFT
    // benchmark_westervelt_wave,        // DISABLED - uses stub nonlinear term
    // benchmark_swe,                    // DISABLED - uses stub elastic equation
    // benchmark_ceus,                   // DISABLED - uses stub cavitation
    // benchmark_transcranial_fus,       // DISABLED - uses stub skull transmission
    // benchmark_uncertainty_quantification, // DISABLED - uses stub statistics
);
```

**Deliverables**:
- Updated `performance_benchmark.rs` (stubs disabled, documented)
- Zero stub benchmarks in active test suite
- Compilation success (`cargo bench --benches`)

---

#### Task 2: Implement Production Benchmarks (1-2 hours)

**Objective**: Add benchmarks for production solvers (FDTD, PSTD, k-space)

**File**: `benches/production_solver_benchmarks.rs` (new)

**Structure**:

```rust
//! Production Solver Benchmarks
//!
//! This benchmark suite measures the performance of production-grade solver
//! implementations with real physics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::solver::forward::fdtd::AcousticFdtdSolver;
use kwavers::solver::forward::pstd::PstdSolver;

fn bench_fdtd_production(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdtd_production");
    
    // Test grid sizes
    let grid_sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
    ];
    
    for (nx, ny, nz) in grid_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", nx, ny, nz)),
            &(nx, ny, nz),
            |b, &(nx, ny, nz)| {
                // Setup
                let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4).unwrap();
                let medium = HomogeneousMedium::new(&grid, 1540.0, 1000.0, 0.0, 0.0);
                let mut solver = AcousticFdtdSolver::new(grid, medium).unwrap();
                let dt = 1e-8;
                
                // Benchmark single time step
                b.iter(|| {
                    solver.step(black_box(dt)).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pstd_production(c: &mut Criterion) {
    let mut group = c.benchmark_group("pstd_production");
    
    // Test grid sizes (PSTD is more efficient for larger grids)
    let grid_sizes = vec![
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];
    
    for (nx, ny, nz) in grid_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", nx, ny, nz)),
            &(nx, ny, nz),
            |b, &(nx, ny, nz)| {
                // Setup
                let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4).unwrap();
                let medium = HomogeneousMedium::new(&grid, 1540.0, 1000.0, 0.0, 0.0);
                let mut solver = PstdSolver::new(grid, medium).unwrap();
                let dt = 1e-8;
                
                // Benchmark single time step (includes FFTs)
                b.iter(|| {
                    solver.step(black_box(dt)).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_fdtd_production, bench_pstd_production);
criterion_main!(benches);
```

**Deliverables**:
- `production_solver_benchmarks.rs` (200-300 lines)
- Benchmarks for FDTD, PSTD, k-space solvers
- Criterion HTML reports

---

### Success Criteria

**P0 (Required)**:
- ✅ All stub benchmarks disabled with documentation
- ✅ Zero stub helper methods in active code
- ✅ Production benchmarks implemented for existing solvers
- ✅ `cargo bench --benches` compiles and runs successfully
- ✅ Benchmark documentation explains removal rationale

**P1 (Nice-to-Have)**:
- 🟡 Comparative analysis (FDTD vs PSTD performance)
- 🟡 Scaling studies (weak scaling, strong scaling)
- 🟡 Memory profiling (heap allocation patterns)

---

## Overall Session 3 Success Criteria

### P0 (Must Complete)

**GPU Beamforming**:
- ✅ WGSL compute shader implemented and validated
- ✅ Rust GPU orchestration with buffer management
- ✅ CPU/GPU equivalence tests (10/10 passing)
- ✅ Performance: ≥10× speedup for 128+ channel arrays
- ✅ Real-time: ≥30 FPS for 64ch × 400px clinical imaging
- ✅ Documentation: Comprehensive API docs and usage examples

**Benchmark Remediation**:
- ✅ All stub benchmarks disabled with documentation
- ✅ Production benchmarks for FDTD, PSTD, k-space
- ✅ Zero compilation errors, zero warnings

**Quality Gates**:
- ✅ Zero circular dependencies (architectural validation)
- ✅ SSOT enforcement (single beamforming interface)
- ✅ Clean Architecture compliance (DDD bounded contexts)
- ✅ Mathematical correctness (CPU/GPU equivalence within f32 precision)
- ✅ Test suite: All tests passing (1969+ tests)

### P1 (Should Complete)

- 🟡 GPU beamforming with linear interpolation (sub-sample accuracy)
- 🟡 Streaming mode (process RF data as it arrives)
- 🟡 Comparative benchmark analysis (CPU vs GPU scaling)

### P2 (Future Sessions)

- 🟢 Multi-GPU beamforming (domain decomposition)
- 🟢 GPU MUSIC implementation
- 🟢 GPU adaptive beamforming (MVDR, ESMV)

---

## Timeline

| Task | Estimated Time | Cumulative |
|------|---------------|------------|
| **Part 1: GPU Beamforming** | | |
| 1. Audit GPU infrastructure | 0.5h | 0.5h |
| 2. Design architecture | 1h | 1.5h |
| 3. Create WGSL shader | 2-3h | 4.5h |
| 4. Implement Rust orchestration | 3-4h | 8.5h |
| 5. CPU/GPU equivalence tests | 2h | 10.5h |
| 6. Performance benchmarks | 2-3h | 13.5h |
| 7. Documentation | 1h | 14.5h |
| **Part 2: Benchmark Remediation** | | |
| 1. Disable stub benchmarks | 1h | 15.5h |
| 2. Implement production benchmarks | 1-2h | 17h |
| **Total** | **14-17h** | |

---

## References

### Research Papers

1. **Van Trees, H. L.** (2002). *Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory*. Wiley-Interscience.
   - Chapter 3: Beamforming Fundamentals
   - Chapter 6: Adaptive Beamforming

2. **Treeby, B. E. & Cox, B. T.** (2010). "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields." *J. Biomed. Opt.* 15(2), 021314.
   - GPU acceleration patterns for k-space methods

3. **Stanziola, A. et al.** (2021). "jwave: An open-source library for the simulation of ultrasound fields in JAX." *arXiv:2106.12292*.
   - JAX-based GPU acceleration with automatic differentiation
   - Differentiable beamforming for optimization

4. **Shen, J. & Ebbini, E. S.** (1996). "A new coded-excitation ultrasound imaging system—Part I: Basic principles." *IEEE Trans. Ultrason., Ferroelect., Freq. Control* 43(1), 131-140.
   - Real-time beamforming requirements for clinical systems

### Software References

- **jwave**: https://github.com/ucl-bug/jwave (JAX GPU acceleration)
- **k-Wave**: https://github.com/ucl-bug/k-wave (MATLAB GPU toolkit)
- **dbua**: https://github.com/waltsims/dbua (Neural beamforming)
- **WGPU**: https://wgpu.rs/ (WebGPU Rust bindings)

### Internal Documentation

- `SPRINT_214_SESSION_1_SUMMARY.md` - Complex eigendecomposition
- `SPRINT_214_SESSION_2_SUMMARY.md` - AIC/MDL and MUSIC
- `BENCHMARK_STUB_REMEDIATION_PLAN.md` - Benchmark cleanup strategy
- `backlog.md` Sprint 211-212 - Advanced physics roadmap

---

## Risk Assessment

### High Risk

1. **GPU Driver Compatibility**
   - **Risk**: WGPU may not work on all systems
   - **Mitigation**: Fallback to CPU implementation, test on multiple platforms
   - **Impact**: Moderate (users without GPU support still have CPU path)

2. **Floating-Point Precision**
   - **Risk**: f32 GPU precision vs f64 CPU precision may cause discrepancies
   - **Mitigation**: Validate equivalence within tolerance (1e-5 relative error)
   - **Impact**: Low (clinical imaging tolerates small precision loss)

### Medium Risk

1. **Performance Targets**
   - **Risk**: GPU may not achieve 10× speedup for small arrays
   - **Mitigation**: Focus on large arrays (128+ channels) where GPU excels
   - **Impact**: Low (CPU sufficient for small arrays)

2. **Memory Limits**
   - **Risk**: Large RF datasets may exceed GPU memory
   - **Mitigation**: Implement streaming mode (process in batches)
   - **Impact**: Moderate (requires additional implementation)

### Low Risk

1. **Shader Compilation**
   - **Risk**: WGSL shader may have compilation errors
   - **Mitigation**: Extensive testing, validate with multiple backends
   - **Impact**: Low (shader code is straightforward)

---

## Next Steps (Post-Session 3)

### Sprint 214 Session 4 (Future)

**P0 Blockers** (4-6 hours):
1. k-Space pseudospectral enhancements (k-Wave integration)
2. Axisymmetric solver (2D rotational symmetry)

**P1 Features** (6-10 hours):
3. Fractional Laplacian absorption model
4. Advanced source modeling (time-varying sources)

**P2 Research Integration** (12-20 hours):
5. Differentiable operators (jwave-inspired)
6. Neural beamforming training pipeline (dbua patterns)

---

## Appendix: WGSL Shader Optimization Strategies

### Strategy 1: Workgroup Shared Memory

Reduce global memory access by using shared memory for sensor data:

```wgsl
var<workgroup> shared_rf_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn das_kernel_optimized(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Load RF data into shared memory
    let sensor_idx = global_id.x;
    if (sensor_idx < params.n_sensors) {
        let rf_idx = sensor_idx * params.n_samples + sample_idx;
        shared_rf_data[local_id.x] = rf_data[rf_idx];
    }
    
    workgroupBarrier();
    
    // Compute with shared memory (faster access)
    // ...
}
```

### Strategy 2: Vectorization

Use `vec4<f32>` to process 4 samples simultaneously:

```wgsl
@group(0) @binding(0) var<storage, read> rf_data: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn das_kernel_vectorized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sensor_idx = global_id.x;
    
    // Process 4 samples per thread
    let vec_idx = sample_idx / 4;
    let rf_vec = rf_data[sensor_idx * n_vec_samples + vec_idx];
    
    // Horizontal sum: rf_vec.x + rf_vec.y + rf_vec.z + rf_vec.w
    let sum = dot(rf_vec, vec4<f32>(1.0, 1.0, 1.0, 1.0));
}
```

### Strategy 3: Multi-Pass Reduction

Use multiple passes for large sensor arrays:

```wgsl
// Pass 1: Each workgroup reduces 256 sensors to 1 value
// Pass 2: Final workgroup reduces 16 partial sums to 1 value
```

---

**End of Sprint 214 Session 3 Plan**