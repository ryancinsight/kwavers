//! Neural Network Inference Shaders
//!
//! WGSL compute shaders for real-time PINN inference on GPU.
//! Provides matrix multiplication and activation function kernels.

use crate::core::error::KwaversResult;
use crate::gpu::device::GpuDevice;
mod activate;
mod matmul;
#[cfg(all(test, feature = "gpu"))]
mod tests;

/// GPU-side parameters for neural network compute shaders.
///
/// Layout must match the WGSL `Params` struct exactly (24 bytes, all 4-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuParams {
    pub batch_size: u32,
    pub input_size: u32,
    pub output_size: u32,
    pub activation_type: u32,
    pub weight_scale: f32,
    pub bias_scale: f32,
}

/// Activation mapping shared by the CPU and GPU neural-network paths.
///
/// Integer contract:
/// - `0` = ReLU, `1` = Sigmoid, `2` = Tanh, `3` = Linear
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum ActivationKind {
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    Linear = 3,
}

impl ActivationKind {
    pub(super) fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Relu),
            1 => Some(Self::Sigmoid),
            2 => Some(Self::Tanh),
            3 => Some(Self::Linear),
            _ => None,
        }
    }
}

/// Neural network shader for GPU-accelerated inference
#[derive(Debug)]
pub struct NeuralNetworkShader {
    pub(super) device: GpuDevice,
    pub(super) matmul_pipeline: wgpu::ComputePipeline,
    pub(super) activation_pipeline: wgpu::ComputePipeline,
    pub(super) bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl NeuralNetworkShader {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub async fn new(device: &GpuDevice) -> KwaversResult<Self> {
        let shader_module = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Neural Network Shader"),
                source: wgpu::ShaderSource::Wgsl(NEURAL_NETWORK_SHADER.into()),
            });

        let bind_group_layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Neural Network Bind Group Layout"),
                    entries: &[
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
                    entry_point: Some("matmul_kernel"),
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
                    entry_point: Some("activation_kernel"),
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

    // Apply activation based on type.
    // Contract:
    //   0 = ReLU, 1 = Sigmoid, 2 = Tanh, 3 = Linear
    var result = x;
    if (params.activation_type == 0u) {
        // ReLU
        result = max(0.0, x);
    } else if (params.activation_type == 1u) {
        // Sigmoid
        result = 1.0 / (1.0 + exp(-x));
    } else if (params.activation_type == 2u) {
        // Tanh
        result = tanh(x);
    } else if (params.activation_type == 3u) {
        // Linear (no-op)
        result = x;
    }

    output[idx] = result;
}
"#;
