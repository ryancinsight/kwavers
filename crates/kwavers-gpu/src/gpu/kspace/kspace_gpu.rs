//! `KSpaceGpu` — GPU-accelerated k-space solver using wgpu.
//!
//! ## Architecture
//!
//! Two WGSL shaders cooperate to implement a full k-space propagation step:
//!
//! | Shader               | Entry points                              | Binding layout                     |
//! |----------------------|-------------------------------------------|------------------------------------|
//! | `fft.wgsl`           | `fft_bitrev`, `fft_forward`, `fft_scale`  | group(0)={data_re, data_im}; push_constant=FftParams |
//! | `kspace_propagate`   | `propagate`                               | group(0)={spectrum_re, spectrum_im, kspace}; push_constant=GridParams |
//!
//! Both shaders use the **split complex layout**: real parts in one `array<f32>`,
//! imaginary parts in a second `array<f32>`.  The `kspace` buffer stores
//! `[kx, ky, kz]` per voxel as packed `f32` triples (12 bytes/voxel, matching
//! `upload_kspace`'s interleaved output).
//!
//! ## Dispatch sequence (`propagate`)
//!
//! 1. **Bit-reversal** (`fft_bitrev`): reorder data into bit-reversed order.
//! 2. **Forward FFT**: `log₂(N)` butterfly passes (`fft_forward`, stage=0..log₂N-1).
//! 3. **Propagation** (`propagate`): multiply each k-space mode by `exp(−iωΔt)`.
//! 4. **Bit-reversal** (`fft_bitrev`): reorder for inverse FFT.
//! 5. **Inverse FFT**: `log₂(N)` butterfly passes (`fft_forward`, inverse=1).
//! 6. **Normalisation** (`fft_scale`): divide by N.
//!
//! ## Invariants
//!
//! - N = nx·ny·nz must be a power of 2.
//! - The kspace buffer has size 3·N·4 bytes (packed f32 triples).
//! - FFT push_constants occupy bytes 0..16 (FftParams: n, stage, inverse, _pad).
//! - Propagate push_constants occupy bytes 0..20 (GridParams: nx, ny, nz, dt, c0).

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::grid::Grid;
use ndarray::Array3;

/// GPU-accelerated k-space solver.
///
/// Holds pre-built pipelines, bind groups, and buffers for one grid shape.
/// Call [`upload_kspace`] once to upload k-vectors, then [`upload_field`] +
/// [`propagate`] per time step.
#[derive(Debug)]
pub struct KSpaceGpu {
    // FFT pipelines (fft.wgsl, three entry points)
    fft_bitrev_pipeline: wgpu::ComputePipeline,
    fft_butterfly_pipeline: wgpu::ComputePipeline,
    fft_scale_pipeline: wgpu::ComputePipeline,
    // Propagation pipeline (kspace_propagate.wgsl)
    propagate_pipeline: wgpu::ComputePipeline,
    // Bind group for FFT passes: {re_buffer, im_buffer}
    fft_bind_group: wgpu::BindGroup,
    // Bind group for propagation pass: {re_buffer, im_buffer, kspace_buffer}
    propagate_bind_group: wgpu::BindGroup,
    // CPU-writable GPU buffers
    re_buffer: wgpu::Buffer,
    im_buffer: wgpu::Buffer,
    kspace_buffer: wgpu::Buffer,
    // Grid parameters cached for dispatch computation
    n: u32, // nx·ny·nz (total voxel count)
    log2_n: u32,
    nx: u32,
    ny: u32,
    nz: u32,
}

impl KSpaceGpu {
    /// Create a new k-space GPU solver for the given grid.
    ///
    /// N = `grid.nx · grid.ny · grid.nz` must be a power of two.
    ///
    /// # Errors
    ///
    /// Returns `Err` if N is not a power of two or if wgpu pipeline creation fails.
    pub fn new(device: &wgpu::Device, grid: &Grid) -> KwaversResult<Self> {
        let n_usize = grid.nx * grid.ny * grid.nz;
        if !n_usize.is_power_of_two() {
            return Err(KwaversError::InvalidInput(format!(
                "KSpaceGpu requires N = nx·ny·nz to be a power of two; got {}·{}·{} = {}",
                grid.nx, grid.ny, grid.nz, n_usize
            )));
        }
        let n = n_usize as u32;
        let log2_n = n.trailing_zeros(); // exact because n is a power of 2

        // ── Buffers ───────────────────────────────────────────────────────────
        let float_bytes = std::mem::size_of::<f32>() as u64;
        let field_size = n as u64 * float_bytes; // N × f32 per component
        let kspace_size = n as u64 * 3 * float_bytes; // 3N × f32 (packed triples)

        let re_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KSpaceGpu re_buffer"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let im_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KSpaceGpu im_buffer"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let kspace_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KSpaceGpu kspace_buffer"),
            // Packed f32 triples: 12 bytes/voxel.
            // NOT array<vec3<f32>> which has std430 stride of 16 bytes.
            size: kspace_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        let storage_rw_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // FFT bind group layout: {binding(0)=data_re rw, binding(1)=data_im rw}
        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("KSpaceGpu fft_bgl"),
            entries: &[storage_rw_entry(0), storage_rw_entry(1)],
        });

        // Propagate bind group layout: {re rw, im rw, kspace ro}
        let propagate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("KSpaceGpu propagate_bgl"),
            entries: &[
                storage_rw_entry(0),
                storage_rw_entry(1),
                storage_ro_entry(2),
            ],
        });

        // ── Pipeline layouts ──────────────────────────────────────────────────
        // FftParams: {n: u32, stage: u32, inverse: u32, _pad: u32} = 16 bytes.
        let fft_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("KSpaceGpu fft_layout"),
            bind_group_layouts: &[&fft_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        // GridParams: {nx: u32, ny: u32, nz: u32, dt: f32, c0: f32} = 20 bytes.
        let propagate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("KSpaceGpu propagate_layout"),
            bind_group_layouts: &[&propagate_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..20,
            }],
        });

        // ── Shader modules ────────────────────────────────────────────────────
        let fft_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("KSpaceGpu fft.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fft.wgsl").into()),
        });
        let propagate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("KSpaceGpu kspace_propagate.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/kspace_propagate.wgsl").into(),
            ),
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let make_fft_pipe = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&fft_layout),
                module: &fft_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fft_bitrev_pipeline = make_fft_pipe("fft_bitrev");
        let fft_butterfly_pipeline = make_fft_pipe("fft_forward");
        let fft_scale_pipeline = make_fft_pipe("fft_scale");

        let propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kspace_propagate"),
            layout: Some(&propagate_layout),
            module: &propagate_module,
            entry_point: Some("propagate"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Bind groups ───────────────────────────────────────────────────────
        let fft_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KSpaceGpu fft_bind_group"),
            layout: &fft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buffer.as_entire_binding(),
                },
            ],
        });
        let propagate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KSpaceGpu propagate_bind_group"),
            layout: &propagate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kspace_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            fft_bitrev_pipeline,
            fft_butterfly_pipeline,
            fft_scale_pipeline,
            propagate_pipeline,
            fft_bind_group,
            propagate_bind_group,
            re_buffer,
            im_buffer,
            kspace_buffer,
            n,
            log2_n,
            nx: grid.nx as u32,
            ny: grid.ny as u32,
            nz: grid.nz as u32,
        })
    }

    /// Upload the complex pressure field to GPU (split re/im layout).
    ///
    /// `real` and `imag` must each have shape `(nx, ny, nz)` matching the grid.
    pub fn upload_field(&self, queue: &wgpu::Queue, real: &Array3<f64>, imag: &Array3<f64>) {
        let re_f32: Vec<f32> = real.iter().map(|&v| v as f32).collect();
        let im_f32: Vec<f32> = imag.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&self.re_buffer, 0, bytemuck::cast_slice(&re_f32));
        queue.write_buffer(&self.im_buffer, 0, bytemuck::cast_slice(&im_f32));
    }

    /// Upload k-space vectors to GPU.
    ///
    /// Packs `[kx, ky, kz]` per voxel as consecutive f32 triples (12 bytes/voxel).
    /// This matches the `array<f32>` buffer in `kspace_propagate.wgsl`, which uses
    /// indices `kspace[3*vox]`, `kspace[3*vox+1]`, `kspace[3*vox+2]`.
    pub fn upload_kspace(
        &self,
        queue: &wgpu::Queue,
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
    ) {
        let mut k_data: Vec<f32> = Vec::with_capacity(3 * self.n as usize);
        for ((kx_val, ky_val), kz_val) in kx.iter().zip(ky.iter()).zip(kz.iter()) {
            k_data.push(*kx_val as f32);
            k_data.push(*ky_val as f32);
            k_data.push(*kz_val as f32);
        }
        queue.write_buffer(&self.kspace_buffer, 0, bytemuck::cast_slice(&k_data));
    }

    /// Encode a full k-space propagation step into `encoder`.
    ///
    /// ## Dispatch sequence
    ///
    /// 1. Bit-reversal permutation (1 pass, `N/256` workgroups).
    /// 2. Forward FFT: `log₂(N)` butterfly passes, each `N/2/256` workgroups.
    /// 3. K-space propagation: 1 pass, `⌈nx/8⌉ × ⌈ny/8⌉ × ⌈nz/8⌉` workgroups.
    /// 4. Bit-reversal permutation (1 pass, identical to step 1).
    /// 5. Inverse FFT: `log₂(N)` butterfly passes (inverse=1).
    /// 6. Normalisation: 1 pass (`fft_scale`), `N/256` workgroups.
    ///
    /// Each pass is a separate `ComputePass`, ensuring wgpu inserts correct
    /// pipeline barriers between passes.
    pub fn propagate(&self, encoder: &mut wgpu::CommandEncoder, dt: f32, c0: f32) {
        let n = self.n;
        let log2_n = self.log2_n;

        // FftParams layout: [n: u32, stage: u32, inverse: u32, _pad: u32]
        let fft_pc = |stage: u32, inverse: u32| -> [u8; 16] {
            let mut buf = [0u8; 16];
            buf[0..4].copy_from_slice(&n.to_le_bytes());
            buf[4..8].copy_from_slice(&stage.to_le_bytes());
            buf[8..12].copy_from_slice(&inverse.to_le_bytes());
            // bytes 12..16 = _pad = 0
            buf
        };

        // GridParams layout: [nx: u32, ny: u32, nz: u32, dt: f32, c0: f32]
        let grid_pc = {
            let mut buf = [0u8; 20];
            buf[0..4].copy_from_slice(&self.nx.to_le_bytes());
            buf[4..8].copy_from_slice(&self.ny.to_le_bytes());
            buf[8..12].copy_from_slice(&self.nz.to_le_bytes());
            buf[12..16].copy_from_slice(&dt.to_le_bytes());
            buf[16..20].copy_from_slice(&c0.to_le_bytes());
            buf
        };

        // Dispatch helpers (workgroup counts, clamped to ≥ 1)
        let wg_n_256 = n.div_ceil(256).max(1); // for N-thread passes
        let wg_n2_256 = (n / 2).div_ceil(256).max(1); // for N/2-thread passes
        let wg_x = self.nx.div_ceil(8).max(1);
        let wg_y = self.ny.div_ceil(8).max(1);
        let wg_z = self.nz.div_ceil(8).max(1);

        // ── Step 1: bit-reversal (forward) ───────────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_bitrev (forward)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_bitrev_pipeline);
            pass.set_bind_group(0, &self.fft_bind_group, &[]);
            pass.set_push_constants(0, &fft_pc(0, 0));
            pass.dispatch_workgroups(wg_n_256, 1, 1);
        }

        // ── Steps 2: forward butterfly passes ────────────────────────────────
        for stage in 0..log2_n {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_butterfly_pipeline);
            pass.set_bind_group(0, &self.fft_bind_group, &[]);
            pass.set_push_constants(0, &fft_pc(stage, 0));
            pass.dispatch_workgroups(wg_n2_256, 1, 1);
        }

        // ── Step 3: k-space propagation ───────────────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kspace_propagate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.propagate_pipeline);
            pass.set_bind_group(0, &self.propagate_bind_group, &[]);
            pass.set_push_constants(0, &grid_pc);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // ── Step 4: bit-reversal (inverse) ───────────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_bitrev (inverse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_bitrev_pipeline);
            pass.set_bind_group(0, &self.fft_bind_group, &[]);
            pass.set_push_constants(0, &fft_pc(0, 0));
            pass.dispatch_workgroups(wg_n_256, 1, 1);
        }

        // ── Step 5: inverse butterfly passes ─────────────────────────────────
        for stage in 0..log2_n {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_forward (inverse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_butterfly_pipeline);
            pass.set_bind_group(0, &self.fft_bind_group, &[]);
            pass.set_push_constants(0, &fft_pc(stage, 1));
            pass.dispatch_workgroups(wg_n2_256, 1, 1);
        }

        // ── Step 6: normalise (divide by N) ──────────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_scale_pipeline);
            pass.set_bind_group(0, &self.fft_bind_group, &[]);
            pass.set_push_constants(0, &fft_pc(0, 0));
            pass.dispatch_workgroups(wg_n_256, 1, 1);
        }
    }

    /// Return read-only references to the GPU buffers for COPY_SRC operations.
    pub fn re_buffer(&self) -> &wgpu::Buffer {
        &self.re_buffer
    }
    /// Return read-only references to the GPU buffers for COPY_SRC operations.
    pub fn im_buffer(&self) -> &wgpu::Buffer {
        &self.im_buffer
    }

    /// Total voxel count N = nx·ny·nz.
    pub fn n(&self) -> u32 {
        self.n
    }
}
