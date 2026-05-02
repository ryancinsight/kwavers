use super::{GpuParams, NeuralNetworkShader};
use crate::core::error::KwaversError;
use crate::core::error::KwaversResult;
use wgpu::util::DeviceExt;

impl NeuralNetworkShader {
    /// Perform matrix multiplication on GPU: `Y = W·X + b` with INT8 quantized weights.
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

        let device = self.device.device();
        let queue = self.device.queue();

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
            let wg_x = (batch_size as u32).div_ceil(16);
            let wg_y = (output_size as u32).div_ceil(16);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait);

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

    pub(super) fn matmul_cpu_quantized(
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

        for b in 0..batch_size {
            for o in 0..output_size {
                let mut sum = 0.0f32;

                for i in 0..input_size {
                    let input_val = input[b * input_size + i];
                    let weight_idx = o * input_size + i;
                    let weight_val = weights[weight_idx] as f32 * weight_scale;
                    sum += input_val * weight_val;
                }

                let bias_val = biases[o] as f32 * bias_scale;
                output[b * output_size + o] = sum + bias_val;
            }
        }

        Ok(output)
    }

    pub(super) fn has_gpu_acceleration(&self) -> bool {
        true
    }

    pub(super) fn validate_activation_type(activation_type: u32) -> KwaversResult<()> {
        use super::ActivationKind;
        if ActivationKind::from_u32(activation_type).is_some() {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(format!(
                "Unknown activation type: {}",
                activation_type
            )))
        }
    }
}
