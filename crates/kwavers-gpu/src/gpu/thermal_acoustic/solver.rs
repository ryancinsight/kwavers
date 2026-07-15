//! Provider-generic thermal-acoustic solver wrapper and WGPU implementation.
//!
//! SRP: changes when the pipeline layout, workgroup dispatch, or time-step
//! protocol changes.

use super::buffers::{ThermalAcousticBufferProvider, WgpuThermalAcousticBuffers};
use super::config::GpuThermalAcousticConfig;
use super::shader::thermal_acoustic_wgsl;
use crate::backend::{
    init::GpuProviderContext,
    provider::{GpuKernelProvider, GpuProviderBackend},
};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::KwaversResult;
use kwavers_solver::backend::traits::BackendCapabilities;

/// Provider contract for thermal-acoustic solver execution.
pub trait ThermalAcousticSolverProvider: GpuKernelProvider<Scalar = f32> {
    /// Buffer provider used by this solver provider.
    type Buffers: ThermalAcousticBufferProvider<Scalar = f32>;

    /// Execute one time step of the coupled thermal-acoustic simulation.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when command encoding or submission fails.
    fn step(&self) -> KwaversResult<()>;

    /// Return the thermal-acoustic configuration.
    fn config(&self) -> GpuThermalAcousticConfig;

    /// Borrow the provider-owned buffers.
    fn buffers(&self) -> &Self::Buffers;
}

/// Provider-generic GPU-accelerated thermal-acoustic coupling solver.
#[derive(Debug)]
pub struct GpuThermalAcousticSolver<P = WgpuThermalAcousticSolverProvider>
where
    P: ThermalAcousticSolverProvider,
{
    provider: P,
}

impl GpuThermalAcousticSolver<WgpuThermalAcousticSolverProvider> {
    /// Create a WGPU-backed thermal-acoustic solver.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn new(config: GpuThermalAcousticConfig) -> KwaversResult<Self> {
        WgpuThermalAcousticSolverProvider::new(config).map(Self::from_provider)
    }
}

impl<P> GpuThermalAcousticSolver<P>
where
    P: ThermalAcousticSolverProvider,
{
    /// Build a solver wrapper from a concrete provider implementation.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the concrete solver provider.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Execute one time step of the coupled thermal-acoustic simulation.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn step(&self) -> KwaversResult<()> {
        self.provider.step()
    }

    /// Return the thermal-acoustic configuration.
    #[must_use]
    pub fn config(&self) -> GpuThermalAcousticConfig {
        self.provider.config()
    }

    /// Borrow the provider-owned buffers.
    #[must_use]
    pub fn buffers(&self) -> &P::Buffers {
        self.provider.buffers()
    }
}

/// WGPU implementation of the thermal-acoustic solver provider.
///
/// Three compute pipelines are held — one per pass — so that wgpu can insert
/// correct pipeline barriers between them via separate `ComputePass` objects.
///
/// | Field                    | Entry point        | Pass |
/// |--------------------------|--------------------|------|
/// | `pressure_pipeline`      | `update_pressure`  | 1    |
/// | `velocity_pipeline`      | `update_velocity`  | 2    |
/// | `thermal_pipeline`       | `update_thermal`   | 3    |
#[derive(Debug)]
pub struct WgpuThermalAcousticSolverProvider {
    pub(super) context: GpuProviderContext<WgpuDevice>,
    pub(super) config: GpuThermalAcousticConfig,
    pub(super) buffers: WgpuThermalAcousticBuffers,
    /// Pass 1: p_curr and Q_ac from p_prev + ux/uy/uz_prev.
    pub(super) pressure_pipeline: wgpu::ComputePipeline,
    /// Pass 2: ux/uy/uz_curr from p_curr (committed by Pass 1) + _prev.
    pub(super) velocity_pipeline: wgpu::ComputePipeline,
    /// Pass 3: T_curr from T_prev Laplacian + Q_ac (committed by Pass 1).
    pub(super) thermal_pipeline: wgpu::ComputePipeline,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) workgroup_size: [u32; 3],
}

impl WgpuThermalAcousticSolverProvider {
    /// Create a WGPU thermal-acoustic solver provider.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn new(config: GpuThermalAcousticConfig) -> KwaversResult<Self> {
        config.validate()?;
        let context = GpuProviderContext::<WgpuDevice>::new()?;
        let device = context.device();
        let queue = context.queue();
        let buffers = WgpuThermalAcousticBuffers::new(device, queue, &config)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thermal-Acoustic Fused Kernel"),
            source: wgpu::ShaderSource::Wgsl(thermal_acoustic_wgsl().into()),
        });

        let bgl = build_bgl(device);
        let bind_group = build_bind_group(device, &bgl, &buffers);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Thermal-Acoustic Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let make_pipe = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pressure_pipeline = make_pipe("update_pressure");
        let velocity_pipeline = make_pipe("update_velocity");
        let thermal_pipeline = make_pipe("update_thermal");

        Ok(Self {
            context,
            config,
            buffers,
            pressure_pipeline,
            velocity_pipeline,
            thermal_pipeline,
            bind_group,
            workgroup_size: [8, 8, 4],
        })
    }
}

impl GpuProviderBackend for WgpuThermalAcousticSolverProvider {
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

impl GpuKernelProvider for WgpuThermalAcousticSolverProvider {
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

impl ThermalAcousticSolverProvider for WgpuThermalAcousticSolverProvider {
    type Buffers = WgpuThermalAcousticBuffers;

    fn step(&self) -> KwaversResult<()> {
        let device = self.context.device();
        let queue = self.context.queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Thermal-Acoustic Step Encoder"),
        });

        let wg_x = self.config.nx.div_ceil(self.workgroup_size[0]);
        let wg_y = self.config.ny.div_ceil(self.workgroup_size[1]);
        let wg_z = self.config.nz.div_ceil(self.workgroup_size[2]);

        // Pass 1: pressure + Q_ac (reads _prev only)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pressure_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // Pass 2: velocity (reads p_curr committed by Pass 1 barrier)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.velocity_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // Pass 3: thermal (reads T_prev Laplacian + Q_ac committed by Pass 1 barrier)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_thermal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.thermal_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn config(&self) -> GpuThermalAcousticConfig {
        self.config
    }

    fn buffers(&self) -> &Self::Buffers {
        &self.buffers
    }
}

fn build_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let rw = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Thermal-Acoustic Bind Group Layout"),
        entries: &[
            rw(0),
            ro(1), // pressure curr/prev
            rw(2),
            ro(3), // velocity_x curr/prev
            rw(4),
            ro(5), // velocity_y curr/prev
            rw(6),
            ro(7), // velocity_z curr/prev
            rw(8),
            ro(9),       // temperature curr/prev
            rw(10),      // q_ac
            uniform(11), // config
        ],
    })
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf: &WgpuThermalAcousticBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Thermal-Acoustic Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.pressure_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf.pressure_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf.velocity_x_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf.velocity_x_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf.velocity_y_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf.velocity_y_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf.velocity_z_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf.velocity_z_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: buf.temperature_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: buf.temperature_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: buf.q_ac.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: buf.config_buffer.as_entire_binding(),
            },
        ],
    })
}
