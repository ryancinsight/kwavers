/// Uniform buffer matched to `PressureParams` in `fdtd_pressure.wgsl`.
///
/// Layout (std140, all u32/f32 → 4-byte fields, no padding needed):
/// ```text
/// offset 0:  nx    (u32)
/// offset 4:  ny    (u32)
/// offset 8:  nz    (u32)
/// offset 12: coeff (f32)   // (c·dt/dx)²
/// ```
use crate::backend::{
    init::GpuProviderContext,
    provider::{GpuKernelProvider, GpuProviderBackend},
};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::backend::traits::BackendCapabilities;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PressureParams {
    /// Grid size in x
    pub nx: u32,
    /// Grid size in y
    pub ny: u32,
    /// Grid size in z
    pub nz: u32,
    /// CFL² coefficient: (c·dt/dx)²
    pub coeff: f32,
}

/// WGPU dispatcher that loads and dispatches the `fdtd_pressure.wgsl` compute
/// shader for the scalar wave equation pressure update.
///
/// # Algorithm — Yee (1966) scalar wave equation
///
/// ```text
/// p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k]
///                + coeff · ∇²p^n[i,j,k]
/// ```
///
/// where `coeff = (c·dt/dx)²` and the 6-point Laplacian is computed on the GPU
/// using workgroup size 8×8×4 (= 256 threads).
///
/// # Bindings
///
/// - `group(0) binding(0)` — `pressure_curr` (f32 storage, read-only)
/// - `group(0) binding(1)` — `pressure_prev` (f32 storage, read-only)
/// - `group(0) binding(2)` — `pressure_new`  (f32 storage, read-write)
/// - `group(1) binding(0)` — `PressureParams` uniform {nx, ny, nz: u32, coeff: f32}
///
/// # References
///
/// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
/// - Moczo P et al. (2014). The Finite-Difference Modelling of Earthquake
///   Motions. Cambridge Univ. Press. (6-point Laplacian, §3.1)
#[derive(Debug)]
pub struct WgpuFdtdPressureDispatcher {
    context: GpuProviderContext<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
}

impl WgpuFdtdPressureDispatcher {
    /// Create a new dispatcher, compiling `fdtd_pressure.wgsl` and building
    /// the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `ComputeError` if the wgpu device does not support the required
    /// features.
    pub fn new() -> KwaversResult<Self> {
        let context = GpuProviderContext::<WgpuDevice>::new()?;
        let device = context.device();
        let shader_src = include_str!("../shaders/fdtd_pressure.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd_pressure"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd_bgl0_pressure_buffers"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd_bgl1_params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fdtd_pipeline_layout"),
            bind_group_layouts: &[&bgl0, &bgl1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_pressure_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("fdtd_pressure_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            context,
            pipeline,
            bind_group_layout_0: bgl0,
            bind_group_layout_1: bgl1,
        })
    }

    /// Dispatch the FDTD pressure update kernel.
    ///
    /// ## Workgroup layout
    ///
    /// ```text
    /// workgroups_x = ceil(nx / 8)
    /// workgroups_y = ceil(ny / 8)
    /// workgroups_z = ceil(nz / 4)
    /// ```
    ///
    /// ## Errors
    ///
    /// Returns `InvalidInput` if `p_curr.len() != p_prev.len()`.
    pub fn dispatch(
        &self,
        p_curr: &[f32],
        p_prev: &[f32],
        params: PressureParams,
    ) -> KwaversResult<Vec<f32>> {
        let n = p_curr.len();
        if p_prev.len() != n {
            return Err(KwaversError::InvalidInput(
                "WgpuFdtdPressureDispatcher::dispatch: p_curr and p_prev length mismatch".into(),
            ));
        }
        let device = self.context.device();
        let queue = self.context.queue();
        let buf_size = std::mem::size_of_val(p_curr) as u64;
        let usage_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let buf_curr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_curr"),
            size: buf_size,
            usage: usage_src,
            mapped_at_creation: false,
        });
        let buf_prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_prev"),
            size: buf_size,
            usage: usage_src,
            mapped_at_creation: false,
        });
        let buf_new = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_new"),
            size: buf_size,
            usage: usage_dst,
            mapped_at_creation: false,
        });
        let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_params"),
            size: std::mem::size_of::<PressureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buf_curr, 0, bytemuck::cast_slice(p_curr));
        queue.write_buffer(&buf_prev, 0, bytemuck::cast_slice(p_prev));
        queue.write_buffer(&buf_params, 0, bytemuck::bytes_of(&params));

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd_bg0"),
            layout: &self.bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_new.as_entire_binding(),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd_bg1"),
            layout: &self.bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fdtd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fdtd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            let wx = params.nx.div_ceil(8);
            let wy = params.ny.div_ceil(8);
            let wz = params.nz.div_ceil(4);
            pass.dispatch_workgroups(wx, wy, wz);
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_new, 0, &staging, 0, buf_size);

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|e| KwaversError::GpuError(format!("GPU map_async failed: {e}")))?
            .map_err(|e| crate::gpu::map_buffer_async_error("FDTD readback", e))?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

impl GpuProviderBackend for WgpuFdtdPressureDispatcher {
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

impl GpuKernelProvider for WgpuFdtdPressureDispatcher {
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
