use super::{ActivationKind, GpuParams, NeuralNetworkShader};
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use wgpu::util::DeviceExt;

impl NeuralNetworkShader {
    /// Apply activation function (GPU-accelerated; CPU fallback when GPU unavailable)
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn activate(&self, input: &[f32], activation_type: u32) -> KwaversResult<Vec<f32>> {
        Self::validate_activation_type(activation_type)?;

        if !self.has_gpu_acceleration() {
            return self.activate_cpu(input, activation_type);
        }

        let device = self.device.wgpu_device();
        let queue = self.device.wgpu_queue();

        let data_bytes = std::mem::size_of_val(input) as u64;
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
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy.as_entire_binding(),
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
            label: Some("NN Act Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NN Activation Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.activation_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (input.len() as u32).div_ceil(256);
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, data_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        receiver
            .recv()
            .map_err(|_| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU activation mapping channel".to_string(),
                })
            })?
            .map_err(|_| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU activation mapping".to_string(),
                })
            })?;

        let data = buffer_slice.get_mapped_range().map_err(|error| {
            crate::gpu::map_buffer_range_error("neural activation readback", error)
        })?;
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }
    /// Activate cpu.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn activate_cpu(
        &self,
        input: &[f32],
        activation_type: u32,
    ) -> KwaversResult<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len());

        match ActivationKind::from_u32(activation_type) {
            Some(ActivationKind::Relu) => {
                for &x in input {
                    output.push(x.max(0.0));
                }
            }
            Some(ActivationKind::Sigmoid) => {
                for &x in input {
                    output.push(1.0 / (1.0 + (-x).exp()));
                }
            }
            Some(ActivationKind::Tanh) => {
                for &x in input {
                    output.push(x.tanh());
                }
            }
            Some(ActivationKind::Linear) => {
                output.extend_from_slice(input);
            }
            None => {
                return Err(KwaversError::InvalidInput(format!(
                    "Unknown activation type: {}",
                    activation_type
                )));
            }
        }

        Ok(output)
    }
}