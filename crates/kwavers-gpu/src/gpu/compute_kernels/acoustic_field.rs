//! Provider-generic acoustic-field kernel wrapper.

use crate::backend::{
    init::GpuProviderContext,
    provider::{GpuKernelProvider, GpuProviderBackend},
};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid;
use kwavers_solver::backend::traits::BackendCapabilities;
use leto::Array3 as LetoArray3;
use wgpu::util::DeviceExt;

/// Acoustic-field propagation provider.
///
/// Implementations own real device buffers, pipelines, transfer paths, and
/// dispatch semantics. WGPU is the current implementation because the shipped
/// shader is WGSL; CUDA must implement this trait only when it owns equivalent
/// kernels and value-semantic differential tests.
pub trait AcousticFieldProvider: GpuKernelProvider<Scalar = f32> {
    /// Compute one acoustic-field propagation step.
    ///
    /// # Errors
    ///
    /// Returns a GPU or input error when transfer, dispatch, readback, or
    /// shape validation fails.
    fn compute_propagation(
        &self,
        pressure: &LetoArray3<f32>,
        grid: &kwavers_grid::Grid,
        dt: f32,
        sound_speed: f32,
    ) -> KwaversResult<LetoArray3<f32>>;
}

/// Acoustic field compute kernel.
#[derive(Debug)]
pub struct AcousticFieldKernel<P = WgpuAcousticFieldProvider>
where
    P: AcousticFieldProvider,
{
    provider: P,
}

impl AcousticFieldKernel<WgpuAcousticFieldProvider> {
    /// Create a WGPU-backed acoustic field kernel.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    pub async fn new() -> KwaversResult<Self> {
        Self::try_new()
    }

    /// Create a WGPU-backed acoustic field kernel synchronously.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    pub fn try_new() -> KwaversResult<Self> {
        WgpuAcousticFieldProvider::try_new().map(Self::from_provider)
    }
}

impl<P> AcousticFieldKernel<P>
where
    P: AcousticFieldProvider,
{
    /// Build a kernel wrapper from a concrete provider implementation.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the concrete provider implementation.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Compute acoustic field propagation on GPU.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    pub fn compute_propagation(
        &self,
        pressure: &LetoArray3<f32>,
        grid: &kwavers_grid::Grid,
        dt: f32,
        sound_speed: f32,
    ) -> KwaversResult<LetoArray3<f32>> {
        self.provider
            .compute_propagation(pressure, grid, dt, sound_speed)
    }
}

/// WGPU implementation of the acoustic-field propagation provider.
#[derive(Debug)]
pub struct WgpuAcousticFieldProvider {
    context: GpuProviderContext<WgpuDevice>,
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuAcousticFieldProvider {
    /// Create a WGPU acoustic-field provider.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    pub async fn new() -> KwaversResult<Self> {
        Self::try_new()
    }

    /// Create a WGPU acoustic-field provider synchronously.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    pub fn try_new() -> KwaversResult<Self> {
        let context = GpuProviderContext::<WgpuDevice>::new()?;

        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Acoustic Field Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/acoustic_field.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Acoustic Field Bind Group Layout"),
            entries: &[
                // Input pressure field
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
                // Output pressure field
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
                // Parameters (dt, dx, c, etc.)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Acoustic Field Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Acoustic Field Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }
}

impl GpuProviderBackend for WgpuAcousticFieldProvider {
    type Device = WgpuDevice;

    fn hephaestus_device(&self) -> &Self::Device {
        self.context.hephaestus_device()
    }

    fn device_name(&self) -> &str {
        self.context.device_name()
    }

    fn synchronize(&self) -> KwaversResult<()> {
        self.context.synchronize()
    }
}

impl GpuKernelProvider for WgpuAcousticFieldProvider {
    type Scalar = f32;

    fn capabilities(&self) -> BackendCapabilities {
        let limits = self.context.hephaestus_device().device_limits();
        BackendCapabilities {
            supports_fft: false,
            supports_f64: false,
            supports_f32: true,
            supports_async: true,
            max_parallelism: limits.max_compute_invocations_per_workgroup as usize,
            supports_unified_memory: false,
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn available_memory(&self) -> usize {
        match usize::try_from(
            self.context
                .hephaestus_device()
                .device_limits()
                .max_buffer_size,
        ) {
            Ok(bytes) => bytes,
            Err(_) => usize::MAX,
        }
    }

    fn estimate_peak_performance(&self) -> f64 {
        0.0
    }
}

impl AcousticFieldProvider for WgpuAcousticFieldProvider {
    fn compute_propagation(
        &self,
        pressure: &LetoArray3<f32>,
        grid: &kwavers_grid::Grid,
        dt: f32,
        sound_speed: f32,
    ) -> KwaversResult<LetoArray3<f32>> {
        let device = self.context.device();
        let queue = self.context.queue();
        let [nx, ny, nz] = pressure.shape();
        let total_size = nx * ny * nz;

        let pressure_values = pressure.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Pressure field must be contiguous".to_string())
        })?;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Pressure Buffer"),
            contents: bytemuck::cast_slice(pressure_values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Pressure Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            nx: u32,
            ny: u32,
            nz: u32,
            _padding: u32,
            dt: f32,
            dx: f32,
            dy: f32,
            dz: f32,
            c: f32,
            // WGSL layout rounds the trailing member up to 16 bytes.
            _padding2: [f32; 7],
        }

        let params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            _padding: 0,
            dt,
            dx: grid.dx as f32,
            dy: grid.dy as f32,
            dz: grid.dz as f32,
            c: sound_speed,
            _padding2: [0.0; 7],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Parameters Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Acoustic Field Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Acoustic Field Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Acoustic Field Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 8×8×4 workgroups to fit the 256-invocation limit.
            const WORKGROUP_X: usize = 8;
            const WORKGROUP_Y: usize = 8;
            const WORKGROUP_Z: usize = 4;
            let dispatch_x = nx.div_ceil(WORKGROUP_X);
            let dispatch_y = ny.div_ceil(WORKGROUP_Y);
            let dispatch_z = nz.div_ceil(WORKGROUP_Z);

            compute_pass.dispatch_workgroups(
                dispatch_x as u32,
                dispatch_y as u32,
                dispatch_z as u32,
            );
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_size * std::mem::size_of::<f32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        receiver
            .recv()
            .map_err(|_| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU buffer mapping channel".to_string(),
                })
            })?
            .map_err(|_| {
                KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "GPU buffer mapping".to_string(),
                })
            })?;

        let data = buffer_slice.get_mapped_range().map_err(|error| {
            crate::gpu::map_buffer_range_error("acoustic field readback", error)
        })?;
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result_values = result_f32.to_vec();

        drop(data);
        staging_buffer.unmap();

        LetoArray3::from_shape_vec([nx, ny, nz], result_values)
            .map_err(|e| KwaversError::InvalidInput(format!("Invalid acoustic output shape: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuDeviceProvider;

    #[test]
    fn acoustic_kernel_wrapper_is_generic_over_provider_traits() {
        fn assert_provider<P>()
        where
            P: AcousticFieldProvider,
        {
            let _ = core::mem::size_of::<AcousticFieldKernel<P>>();
            let _ = core::mem::size_of::<<P as GpuProviderBackend>::Device>();
        }

        assert_provider::<WgpuAcousticFieldProvider>();
    }

    #[test]
    fn wgpu_acoustic_provider_declares_native_scalar() {
        fn assert_scalar<P>()
        where
            P: AcousticFieldProvider,
        {
            let _ = core::mem::size_of::<P>();
        }

        assert_scalar::<WgpuAcousticFieldProvider>();
    }

    #[test]
    fn wgpu_acoustic_provider_uses_kwavers_device_contract() {
        assert_eq!(
            <WgpuDevice as GpuDeviceProvider>::acquisition_label(),
            "kwavers-wgpu-device"
        );
        assert_eq!(
            <WgpuDevice as GpuDeviceProvider>::provider_kind(),
            kwavers_solver::backend::traits::GpuProvider::Wgpu
        );
    }
}