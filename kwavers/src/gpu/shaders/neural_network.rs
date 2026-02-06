//! Neural Network Inference Shaders
//!
//! WGSL compute shaders for real-time PINN inference on GPU.
//! Provides matrix multiplication and activation function kernels.

use crate::core::error::{KwaversError, KwaversResult};
use crate::gpu::device::GpuDevice;
use wgpu::util::DeviceExt;

/// GPU-side parameters for neural network compute shaders.
///
/// Layout must match the WGSL `Params` struct exactly (24 bytes, all 4-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    activation_type: u32,
    weight_scale: f32,
    bias_scale: f32,
}

/// Neural network shader for GPU-accelerated inference
#[derive(Debug)]
pub struct NeuralNetworkShader {
    /// GPU device handle
    device: GpuDevice,
    /// Compute pipeline for matrix multiplication
    matmul_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for activation functions
    activation_pipeline: wgpu::ComputePipeline,
    /// Bind group layouts
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl NeuralNetworkShader {
    /// Create new neural network shader
    pub async fn new(device: &GpuDevice) -> KwaversResult<Self> {
        let shader_module = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Neural Network Shader"),
                source: wgpu::ShaderSource::Wgsl(NEURAL_NETWORK_SHADER.into()),
            });

        // Create bind group layouts
        let bind_group_layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Neural Network Bind Group Layout"),
                    entries: &[
                        // Input buffer
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
                        // Weight buffer
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
                        // Bias buffer
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
                        // Output buffer
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
                        // Params uniform buffer
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

        // Create pipelines
        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Neural Network Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let matmul_pipeline =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Matrix Multiplication Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "matmul_kernel",
                    compilation_options: Default::default(),
                    cache: None,
                });

        let activation_pipeline =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Activation Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "activation_kernel",
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            device: device.clone(),
            matmul_pipeline,
            activation_pipeline,
            bind_group_layouts: vec![bind_group_layout],
        })
    }

    /// Perform matrix multiplication on GPU
    ///
    /// Computes `Y = W·X + b` with INT8 quantized weights and biases.
    /// Falls back to CPU when GPU acceleration is not available.
    ///
    /// The GPU kernel dispatches a 2D grid of (16,16) workgroups where each
    /// thread computes one `(batch, output)` element. Weights and biases are
    /// promoted from `i8` → `i32` for WGSL compatibility and de-quantised
    /// inside the shader using the provided scales.
    pub fn matmul(
        &self,
        input: &[f32],
        weights: &[i8],
        biases: &[i8],
        weight_scale: f32,
        bias_scale: f32,
        batch_size: usize,
        input_size: usize,
        output_size: usize,
    ) -> KwaversResult<Vec<f32>> {
        // Validate dimensions
        if input.len() != batch_size * input_size {
            return Err(KwaversError::DimensionMismatch(format!(
                "input length {} != batch_size({}) * input_size({})",
                input.len(),
                batch_size,
                input_size
            )));
        }
        if weights.len() != output_size * input_size {
            return Err(KwaversError::DimensionMismatch(format!(
                "weights length {} != output_size({}) * input_size({})",
                weights.len(),
                output_size,
                input_size
            )));
        }
        if biases.len() != output_size {
            return Err(KwaversError::DimensionMismatch(format!(
                "biases length {} != output_size({})",
                biases.len(),
                output_size
            )));
        }

        if !self.has_gpu_acceleration() {
            return self.matmul_cpu_quantized(
                input,
                weights,
                biases,
                weight_scale,
                bias_scale,
                batch_size,
                input_size,
                output_size,
            );
        }

        // ── GPU path ──────────────────────────────────────────────────
        let device = self.device.device();
        let queue = self.device.queue();

        // Promote i8 → i32 for WGSL (no native i8 type)
        let weights_i32: Vec<i32> = weights.iter().map(|&w| w as i32).collect();
        let biases_i32: Vec<i32> = biases.iter().map(|&b| b as i32).collect();

        let params = GpuParams {
            batch_size: batch_size as u32,
            input_size: input_size as u32,
            output_size: output_size as u32,
            activation_type: 0,
            weight_scale,
            bias_scale,
        };

        let output_bytes = (batch_size * output_size * std::mem::size_of::<f32>()) as u64;

        // Create GPU buffers
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Weights"),
            contents: bytemuck::cast_slice(&weights_i32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let biases_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Biases"),
            contents: bytemuck::cast_slice(&biases_i32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NN Output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NN Staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NN Bind Group"),
            layout: &self.bind_group_layouts[0],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: biases_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Encode compute pass & copy-back
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NN Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NN MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size is (16,16,1) → ceil-divide
            let wg_x = (batch_size as u32 + 15) / 16;
            let wg_y = (output_size as u32 + 15) / 16;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Synchronous readback (matches codebase pattern in compute_kernels.rs)
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU buffer mapping channel".to_string(),
                })
            })?
            .map_err(|_| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU buffer mapping".to_string(),
                })
            })?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }

    /// CPU fallback for quantized matrix multiplication
    fn matmul_cpu_quantized(
        &self,
        input: &[f32],
        weights: &[i8],
        biases: &[i8],
        weight_scale: f32,
        bias_scale: f32,
        batch_size: usize,
        input_size: usize,
        output_size: usize,
    ) -> KwaversResult<Vec<f32>> {
        let mut output = vec![0.0f32; batch_size * output_size];

        // Perform quantized matrix multiplication: output = input @ weights.T + biases
        for b in 0..batch_size {
            for o in 0..output_size {
                let mut sum = 0.0f32;

                // Matrix multiplication: sum over input features
                for i in 0..input_size {
                    let input_val = input[b * input_size + i];
                    let weight_idx = o * input_size + i;
                    let weight_val = weights[weight_idx] as f32 * weight_scale;
                    sum += input_val * weight_val;
                }

                // Add bias
                let bias_val = biases[o] as f32 * bias_scale;
                output[b * output_size + o] = sum + bias_val;
            }
        }

        Ok(output)
    }

    /// Check if GPU acceleration is available
    fn has_gpu_acceleration(&self) -> bool {
        // If this struct was constructed, a valid GPU device and pipelines exist
        true
    }

    /// Apply activation function (GPU-accelerated when available, CPU fallback otherwise)
    pub fn activate(&self, input: &[f32], activation_type: u32) -> KwaversResult<Vec<f32>> {
        if !self.has_gpu_acceleration() {
            return self.activate_cpu(input, activation_type);
        }

        // ── GPU path ──
        // The activation shader operates in-place on binding 3 (output buffer).
        // We upload the input there, dispatch, and read back.
        let device = self.device.device();
        let queue = self.device.queue();

        let data_bytes = (input.len() * std::mem::size_of::<f32>()) as u64;
        if data_bytes == 0 {
            return Ok(Vec::new());
        }

        let params = GpuParams {
            batch_size: 0,
            input_size: 0,
            output_size: 0,
            activation_type,
            weight_scale: 0.0,
            bias_scale: 0.0,
        };

        // Dummy buffers for unused bindings 0-2 (minimum 4 bytes each)
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NN Dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Activation IO"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NN Act Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NN Act Staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NN Act Bind Group"),
            layout: &self.bind_group_layouts[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NN Act Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NN Activation Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.activation_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (input.len() as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, data_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU activation mapping channel".to_string(),
                })
            })?
            .map_err(|_| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU activation mapping".to_string(),
                })
            })?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }

    /// CPU implementation of activation functions
    fn activate_cpu(&self, input: &[f32], activation_type: u32) -> KwaversResult<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len());

        match activation_type {
            0 => {
                // ReLU activation
                for &x in input {
                    output.push(x.max(0.0));
                }
            }
            1 => {
                // Sigmoid activation
                for &x in input {
                    output.push(1.0 / (1.0 + (-x).exp()));
                }
            }
            2 => {
                // Tanh activation
                for &x in input {
                    output.push(x.tanh());
                }
            }
            3 => {
                // Linear (identity) activation
                output.extend_from_slice(input);
            }
            _ => {
                return Err(KwaversError::InvalidInput(format!(
                    "Unknown activation type: {}",
                    activation_type
                )));
            }
        }

        Ok(output)
    }
}

/// WGSL shader source for neural network operations
pub const NEURAL_NETWORK_SHADER: &str = r#"
// Neural Network Inference Shader
// Performs quantized matrix multiplication and activation functions

struct Params {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    activation_type: u32,
    weight_scale: f32,
    bias_scale: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<i32>; // Quantized weights

@group(0) @binding(2)
var<storage, read> biases: array<i32>; // Quantized biases

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

// Matrix multiplication kernel with quantization
@compute @workgroup_size(16, 16, 1)
fn matmul_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_idx = global_id.y;

    if (batch_idx >= params.batch_size || out_idx >= params.output_size) {
        return;
    }

    // Initialize with bias
    let bias_val = f32(biases[out_idx]) * params.bias_scale;
    var sum = bias_val;

    // Matrix multiplication with quantization
    for (var i = 0u; i < params.input_size; i = i + 1u) {
        let input_val = input[batch_idx * params.input_size + i];
        let weight_val = f32(weights[out_idx * params.input_size + i]) * params.weight_scale;
        sum = sum + input_val * weight_val;
    }

    output[batch_idx * params.output_size + out_idx] = sum;
}

// Activation function kernel
@compute @workgroup_size(256, 1, 1)
fn activation_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&output)) {
        return;
    }

    let x = output[idx];

    // Apply activation based on type
    var result = x;
    if (params.activation_type == 0u) {
        // Tanh
        result = tanh(x);
    } else if (params.activation_type == 1u) {
        // ReLU
        result = max(0.0, x);
    } else if (params.activation_type == 2u) {
        // Linear (no-op)
        result = x;
    }

    output[idx] = result;
}
"#;
