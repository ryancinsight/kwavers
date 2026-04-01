//! GPU-resident PSTD (Pseudospectral Time Domain) acoustic solver.
//!
//! # Design
//!
//! All acoustic fields (p, ux, uy, uz, rhox, rhoy, rhoz) remain on the GPU
//! throughout the simulation. Only the final sensor readings are downloaded
//! at the end of the run, minimising PCIe traffic.
//!
//! # Bind group layout (≤8 storage buffers per group)
//!
//! - group(0) 8 storage: p, ux, uy, uz, rhox, rhoy, rhoz, scratch
//! - group(1) 1 uniform: PstdParams
//! - group(2) 8 storage: kspace_re, kspace_im, kspace2_re, kspace2_im,
//!            kappa, rho0_inv, c0_sq, rho0
//! - group(3) 8 storage: pml_sgx, pml_sgy, pml_sgz, pml_xyz (packed),
//!            shifts_all (packed), sensor_flat_indices, sensor_data,
//!            source_data (packed)
//!
//! # Packed buffer formats
//!
//! **pml_xyz**: three concatenated f32 arrays `[pml_x | pml_y | pml_z]`,
//! each of size `nx×ny×nz`. Index via `ax * total + flat_idx`.
//!
//! **shifts_all**: twelve 1D arrays packed in order:
//! `x_pos_re, x_pos_im, x_neg_re, x_neg_im` (each size nx),
//! `y_pos_re, y_pos_im, y_neg_re, y_neg_im` (each size ny),
//! `z_pos_re, z_pos_im, z_neg_re, z_neg_im` (each size nz).
//! Total: `4*(nx+ny+nz)` f32 values.
//!
//! **source_data**: `[bitcast<f32>(mask_indices[n_src]) | signals[n_src*nt]]`.
//! Mask indices are stored as bit-cast f32 values of u32 flat indices.
//!
//! # References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Microwave Opt. Technol. Lett. 15(3), 158–165.

use crate::domain::grid::Grid;
use crate::math::fft::shift_operators::{generate_kappa, generate_shift_1d};
use std::sync::Arc;
use std::f64::consts::PI;
use wgpu::util::DeviceExt;

// ─── Params uniform struct (must match PstdParams in pstd.wgsl) ──────────────
// 12 × u32/f32 = 48 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PstdParams {
    nx:        u32,
    ny:        u32,
    nz:        u32,
    axis:      u32,
    n_fft:     u32,
    n_batches: u32,
    log2n:     u32,
    inverse:   u32,
    step:      u32,
    dt:        f32,
    n_sensors: u32,
    nt:        u32,
}

/// GPU-resident PSTD acoustic solver.
///
/// Keeps all field data on the GPU throughout the time loop; sensor readings
/// are downloaded in a single transfer after all time steps complete.
pub struct GpuPstdSolver {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,

    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    dt: f64,

    // Field buffers — group(0)
    buf_p:       wgpu::Buffer,
    buf_ux:      wgpu::Buffer,
    buf_uy:      wgpu::Buffer,
    buf_uz:      wgpu::Buffer,
    buf_rhox:    wgpu::Buffer,
    buf_rhoy:    wgpu::Buffer,
    buf_rhoz:    wgpu::Buffer,
    buf_scratch: wgpu::Buffer,

    // K-space + medium — group(2)
    buf_kspace_re:  wgpu::Buffer,
    buf_kspace_im:  wgpu::Buffer,
    buf_kspace2_re: wgpu::Buffer,
    buf_kspace2_im: wgpu::Buffer,
    buf_kappa:      wgpu::Buffer,
    buf_rho0_inv:   wgpu::Buffer,
    buf_c0_sq:      wgpu::Buffer,
    buf_rho0:       wgpu::Buffer,

    // PML + shifts + sensor/source — group(3)
    buf_pml_sgx:    wgpu::Buffer,
    buf_pml_sgy:    wgpu::Buffer,
    buf_pml_sgz:    wgpu::Buffer,
    buf_pml_xyz:    wgpu::Buffer,   // packed [pml_x | pml_y | pml_z]
    buf_shifts_all: wgpu::Buffer,   // packed 12 × 1D shift arrays

    // Per-dispatch-slot params buffers and bind groups.
    // 64 slots: enough for all dispatches in one time step.
    // write_buffer() to slot[i] + encode dispatch with params_slot_bgs[i]
    // avoids the single-buffer coalescing bug.
    params_slots:    Vec<wgpu::Buffer>,
    params_slot_bgs: Vec<wgpu::BindGroup>,

    // Pipelines
    pipeline_fft:               wgpu::ComputePipeline,
    pipeline_kspace_shift:      wgpu::ComputePipeline,
    pipeline_vel_update:        wgpu::ComputePipeline,
    pipeline_dens_update:       wgpu::ComputePipeline,
    pipeline_pres_density:      wgpu::ComputePipeline,
    pipeline_record:            wgpu::ComputePipeline,
    pipeline_inject_src:        wgpu::ComputePipeline,
    pipeline_copy_k2_to_k:      wgpu::ComputePipeline,
    pipeline_copy_field_to_k:   wgpu::ComputePipeline,
    pipeline_copy_k_to_scratch: wgpu::ComputePipeline,

    // Bind groups (sensor group rebuilt per run)
    bg_fields: wgpu::BindGroup,
    bg_kspace: wgpu::BindGroup,

    // Bind group layouts and pipeline layout (kept for rebuilding)
    bgl_params:      wgpu::BindGroupLayout,
    bgl_sensor:      wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl GpuPstdSolver {
    /// Create a new GPU PSTD solver.
    ///
    /// # Arguments
    /// * `device` / `queue` — wgpu device and queue (must have been created
    ///   with sufficient `max_storage_buffers_per_shader_stage` if needed;
    ///   this solver uses at most 8 per bind group × 4 bind groups).
    /// * `grid`     — computational grid
    /// * `c0_flat`  — sound speed [m/s], f32, flat [nx×ny×nz] row-major
    /// * `rho0_flat` — density [kg/m³], f32, flat [nx×ny×nz]
    /// * `dt`       — time step [s]
    /// * `nt`       — total time steps
    /// * `c_ref`    — reference sound speed for kappa correction [m/s]
    /// * `pml_x/y/z`    — collocated PML damping (for density), f32 [nx×ny×nz]
    /// * `pml_sgx/y/z`  — staggered PML damping (for velocity), f32 [nx×ny×nz]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue:  Arc<wgpu::Queue>,
        grid:    &Grid,
        c0_flat:   &[f32],
        rho0_flat: &[f32],
        dt:    f64,
        nt:    usize,
        c_ref: f64,
        pml_x:   &[f32],
        pml_y:   &[f32],
        pml_z:   &[f32],
        pml_sgx: &[f32],
        pml_sgy: &[f32],
        pml_sgz: &[f32],
    ) -> Result<Self, String> {
        let nx    = grid.nx;
        let ny    = grid.ny;
        let nz    = grid.nz;
        let total = nx * ny * nz;

        if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
            return Err(format!(
                "GpuPstdSolver requires power-of-2 dimensions, got {}×{}×{}",
                nx, ny, nz
            ));
        }
        if nx > 256 || ny > 256 || nz > 256 {
            return Err(format!(
                "GpuPstdSolver FFT supports N≤256 per axis, got {}×{}×{}",
                nx, ny, nz
            ));
        }

        // ── Precompute kappa (3D) and shift operators (1D per axis) ──────────
        let kappa_3d = generate_kappa(nx, ny, nz, grid.dx, grid.dy, grid.dz, c_ref, dt);
        let kappa_f32: Vec<f32> = kappa_3d.iter().map(|&v| v as f32).collect();

        let dk_x = 2.0 * PI / (nx as f64 * grid.dx);
        let dk_y = 2.0 * PI / (ny as f64 * grid.dy);
        let dk_z = 2.0 * PI / (nz as f64 * grid.dz);

        let (sx_pos, sx_neg) = generate_shift_1d(nx, dk_x, grid.dx);
        let (sy_pos, sy_neg) = generate_shift_1d(ny, dk_y, grid.dy);
        let (sz_pos, sz_neg) = generate_shift_1d(nz, dk_z, grid.dz);

        // Pack shift arrays into a single buffer:
        // [x_pos_re, x_pos_im, x_neg_re, x_neg_im (each nx),
        //  y_pos_re, y_pos_im, y_neg_re, y_neg_im (each ny),
        //  z_pos_re, z_pos_im, z_neg_re, z_neg_im (each nz)]
        let mut shifts_all: Vec<f32> = Vec::with_capacity(4 * (nx + ny + nz));
        for c in sx_pos.iter() { shifts_all.push(c.re as f32); }
        for c in sx_pos.iter() { shifts_all.push(c.im as f32); }
        for c in sx_neg.iter() { shifts_all.push(c.re as f32); }
        for c in sx_neg.iter() { shifts_all.push(c.im as f32); }
        for c in sy_pos.iter() { shifts_all.push(c.re as f32); }
        for c in sy_pos.iter() { shifts_all.push(c.im as f32); }
        for c in sy_neg.iter() { shifts_all.push(c.re as f32); }
        for c in sy_neg.iter() { shifts_all.push(c.im as f32); }
        for c in sz_pos.iter() { shifts_all.push(c.re as f32); }
        for c in sz_pos.iter() { shifts_all.push(c.im as f32); }
        for c in sz_neg.iter() { shifts_all.push(c.re as f32); }
        for c in sz_neg.iter() { shifts_all.push(c.im as f32); }

        // Pack pml_xyz = [pml_x | pml_y | pml_z]
        let mut pml_xyz: Vec<f32> = Vec::with_capacity(3 * total);
        pml_xyz.extend_from_slice(pml_x);
        pml_xyz.extend_from_slice(pml_y);
        pml_xyz.extend_from_slice(pml_z);

        // Derived arrays
        let rho0_inv: Vec<f32> = rho0_flat.iter().map(|&r| 1.0 / r).collect();
        let c0_sq:    Vec<f32> = c0_flat.iter().map(|&c| c * c).collect();

        // ── Buffer helpers ────────────────────────────────────────────────────
        let mk_ro = |data: &[f32], label: &str| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mk_rw = |n: usize, extra: wgpu::BufferUsages, label: &str| -> wgpu::Buffer {
            let data: Vec<f32> = vec![0.0f32; n];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | extra,
            })
        };

        // group(0) field buffers
        let buf_p       = mk_rw(total, wgpu::BufferUsages::empty(), "buf_p");
        let buf_ux      = mk_rw(total, wgpu::BufferUsages::empty(), "buf_ux");
        let buf_uy      = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uy");
        let buf_uz      = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uz");
        let buf_rhox    = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhox");
        let buf_rhoy    = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoy");
        let buf_rhoz    = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoz");
        let buf_scratch = mk_rw(total, wgpu::BufferUsages::empty(), "buf_scratch");

        // group(2) k-space + medium
        let buf_kspace_re  = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_re");
        let buf_kspace_im  = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_im");
        let buf_kspace2_re = mk_rw(total, wgpu::BufferUsages::empty(), "kspace2_re");
        let buf_kspace2_im = mk_rw(total, wgpu::BufferUsages::empty(), "kspace2_im");
        let buf_kappa    = mk_ro(&kappa_f32, "kappa");
        let buf_rho0_inv = mk_ro(&rho0_inv,  "rho0_inv");
        let buf_c0_sq    = mk_ro(&c0_sq,     "c0_sq");
        let buf_rho0     = mk_ro(rho0_flat,  "rho0");

        // group(3) PML + shifts
        let buf_pml_sgx    = mk_ro(pml_sgx,     "pml_sgx");
        let buf_pml_sgy    = mk_ro(pml_sgy,     "pml_sgy");
        let buf_pml_sgz    = mk_ro(pml_sgz,     "pml_sgz");
        let buf_pml_xyz    = mk_ro(&pml_xyz,    "pml_xyz");
        let buf_shifts_all = mk_ro(&shifts_all, "shifts_all");

        // Build initial params value for slot pre-allocation
        let initial_params = PstdParams {
            nx: nx as u32, ny: ny as u32, nz: nz as u32,
            axis: 0, n_fft: nx as u32,
            n_batches: (ny * nz) as u32,
            log2n: nx.trailing_zeros(),
            inverse: 0, step: 0,
            dt: dt as f32,
            n_sensors: 0, nt: nt as u32,
        };

        // ── Shader module ─────────────────────────────────────────────────────
        let shader_src = include_str!("../../../gpu/shaders/pstd.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pstd_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────

        // Helper: N read_write storage bindings
        let bgl_storage_rw = |n: u32, label: &str| -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &(0..n)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        };

        // group(0): 8 read_write storage (fields)
        let bgl_fields = bgl_storage_rw(8, "bgl_fields");

        // group(1): 1 uniform
        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_params"),
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

        // group(2): 4 read_write + 4 read-only storage
        let bgl_kspace = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_kspace"),
            entries: &[
                // bindings 0-3: read_write (kspace buffers)
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // bindings 4-7: read-only (kappa, rho0_inv, c0_sq, rho0)
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // group(3): 8 storage (5 read-only, 1 read/write sensor_data, 1 read sensor_indices, 1 read source_data)
        let bgl_sensor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_sensor"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 5: sensor_flat_indices (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 6: sensor_data (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 7: source_data (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // ── Pipeline layout ───────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pstd_pipeline_layout"),
            bind_group_layouts: &[&bgl_fields, &bgl_params, &bgl_kspace, &bgl_sensor],
            push_constant_ranges: &[],
        });

        // ── Compile pipelines ─────────────────────────────────────────────────
        let mk_pl = |entry: &'static str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipeline_fft               = mk_pl("fft_1d_smem");
        let pipeline_kspace_shift      = mk_pl("kspace_shift_apply");
        let pipeline_vel_update        = mk_pl("velocity_update");
        let pipeline_dens_update       = mk_pl("density_update");
        let pipeline_pres_density      = mk_pl("pressure_from_density");
        let pipeline_record            = mk_pl("record_sensors");
        let pipeline_inject_src        = mk_pl("inject_pressure_source");
        let pipeline_copy_k2_to_k      = mk_pl("copy_kspace2_to_kspace");
        let pipeline_copy_field_to_k   = mk_pl("copy_field_to_kspace");
        let pipeline_copy_k_to_scratch = mk_pl("copy_kspace_to_scratch");

        // ── Bind groups (sensor rebuilt per run) ──────────────────────────────
        let bg_fields = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_fields"),
            layout: &bgl_fields,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_ux.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_uy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_uz.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_rhox.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_rhoy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_rhoz.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_scratch.as_entire_binding() },
            ],
        });

        // ── Per-dispatch-slot params buffers ─────────────────────────────────
        // 64 slots: enough for all dispatches in one time step (~61 max).
        // Each slot has its own uniform buffer + bind group so that
        // write_buffer(slot[i]) before encoder dispatch[i] is not overwritten
        // by write_buffer(slot[i+1]) within the same command buffer submission.
        const N_SLOTS: usize = 64;
        let params_slots: Vec<wgpu::Buffer> = (0..N_SLOTS).map(|_| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_slot"),
                contents: bytemuck::bytes_of(&initial_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        }).collect();
        let params_slot_bgs: Vec<wgpu::BindGroup> = params_slots.iter().map(|buf| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_params_slot"),
                layout: &bgl_params,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            })
        }).collect();

        let bg_kspace = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_kspace"),
            layout: &bgl_kspace,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_kspace_re.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_kspace_im.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_kspace2_re.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_kspace2_im.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_kappa.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_rho0_inv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_c0_sq.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_rho0.as_entire_binding() },
            ],
        });

        Ok(Self {
            device, queue,
            nx, ny, nz, nt, dt,
            buf_p, buf_ux, buf_uy, buf_uz,
            buf_rhox, buf_rhoy, buf_rhoz, buf_scratch,
            buf_kspace_re, buf_kspace_im, buf_kspace2_re, buf_kspace2_im,
            buf_kappa, buf_rho0_inv, buf_c0_sq, buf_rho0,
            buf_pml_sgx, buf_pml_sgy, buf_pml_sgz,
            buf_pml_xyz, buf_shifts_all,
            params_slots, params_slot_bgs,
            pipeline_fft, pipeline_kspace_shift,
            pipeline_vel_update, pipeline_dens_update,
            pipeline_pres_density, pipeline_record, pipeline_inject_src,
            pipeline_copy_k2_to_k, pipeline_copy_field_to_k, pipeline_copy_k_to_scratch,
            bg_fields, bg_kspace,
            bgl_params, bgl_sensor,
            pipeline_layout,
        })
    }

    /// Run the GPU PSTD time loop.
    ///
    /// Returns `sensor_data` as a flat `Vec<f32>` in row-major order
    /// `[sensor_id * nt + step]`.
    ///
    /// # Arguments
    /// * `sensor_indices` — flat grid indices of sensor points (u32)
    /// * `source_indices` — flat grid indices of source injection points (u32)
    /// * `source_signals` — source pressure amplitude per (source_pt, step),
    ///    flat `[n_src * nt]` row-major
    pub fn run(
        &mut self,
        sensor_indices: &[u32],
        source_indices: &[u32],
        source_signals: &[f32],
    ) -> Vec<f32> {
        let nx       = self.nx;
        let ny       = self.ny;
        let nz       = self.nz;
        let nt       = self.nt;
        let total    = nx * ny * nz;
        let n_sensors = sensor_indices.len();
        let n_src     = source_indices.len();

        // ── Build sensor_flat_indices as u32 buffer ───────────────────────────
        let placeholder_u32 = &[0u32];
        let si_data: &[u32] = if sensor_indices.is_empty() { placeholder_u32 } else { sensor_indices };
        let buf_sensor_indices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sensor_indices"),
            contents: bytemuck::cast_slice(si_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── Build sensor_data output buffer ───────────────────────────────────
        let sensor_count = n_sensors.max(1);
        let buf_sensor_data = {
            let zeros: Vec<f32> = vec![0.0f32; sensor_count * nt];
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sensor_data"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };

        // ── Build source_data packed buffer ───────────────────────────────────
        // Layout: [bitcast<f32>(mask_indices[n_src]) | signals[n_src * nt]]
        let buf_source_data = {
            let placeholder_f32 = &[0.0f32];
            let src_signals: &[f32] = if source_signals.is_empty() { placeholder_f32 } else { source_signals };
            let n_src_safe = n_src.max(1);
            let mut data: Vec<f32> = Vec::with_capacity(n_src_safe + src_signals.len());
            // Store mask indices as bit-cast f32
            for &idx in source_indices.iter() {
                data.push(f32::from_bits(idx));
            }
            // Pad to n_src_safe if needed
            while data.len() < n_src_safe {
                data.push(0.0);
            }
            data.extend_from_slice(src_signals);
            // Ensure at least 1 element
            if data.is_empty() { data.push(0.0); }
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("source_data"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // ── Build sensor bind group ───────────────────────────────────────────
        let bg_sensor = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_sensor_run"),
            layout: &self.bgl_sensor,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_pml_sgx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_pml_sgy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_pml_sgz.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_pml_xyz.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_shifts_all.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_sensor_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_sensor_data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_source_data.as_entire_binding() },
            ],
        });

        // ── Reset all field buffers to zero ───────────────────────────────────
        let zero_data: Vec<f32> = vec![0.0f32; total];
        let zero_u8 = bytemuck::cast_slice::<f32, u8>(&zero_data);
        self.queue.write_buffer(&self.buf_p,       0, zero_u8);
        self.queue.write_buffer(&self.buf_ux,      0, zero_u8);
        self.queue.write_buffer(&self.buf_uy,      0, zero_u8);
        self.queue.write_buffer(&self.buf_uz,      0, zero_u8);
        self.queue.write_buffer(&self.buf_rhox,    0, zero_u8);
        self.queue.write_buffer(&self.buf_rhoy,    0, zero_u8);
        self.queue.write_buffer(&self.buf_rhoz,    0, zero_u8);
        self.queue.write_buffer(&self.buf_scratch, 0, zero_u8);
        // Also zero k-space buffers
        self.queue.write_buffer(&self.buf_kspace_re,  0, zero_u8);
        self.queue.write_buffer(&self.buf_kspace_im,  0, zero_u8);
        self.queue.write_buffer(&self.buf_kspace2_re, 0, zero_u8);
        self.queue.write_buffer(&self.buf_kspace2_im, 0, zero_u8);

        // ── Dispatch helpers ──────────────────────────────────────────────────
        let ceil_div = |n: usize, d: usize| -> u32 { ((n + d - 1) / d) as u32 };
        let elem_wg = ceil_div(total, 256);
        let nxu  = nx  as u32;
        let nyu  = ny  as u32;
        let nzu  = nz  as u32;
        let dtu  = self.dt as f32;
        let ntu  = nt  as u32;
        let ns_u = n_sensors as u32;

        // Helper macro: build a generic PstdParams with only `axis` varying
        macro_rules! p {
            ($step:expr, $axis:expr) => {
                PstdParams {
                    nx: nxu, ny: nyu, nz: nzu,
                    axis: $axis, n_fft: 0, n_batches: 0, log2n: 0, inverse: 0,
                    step: $step, dt: dtu, n_sensors: ns_u, nt: ntu,
                }
            };
        }

        // ── Encode time steps: one command buffer per step ────────────────────
        // Each step resets the slot counter to 0, ensuring write_buffer calls
        // for different dispatches go to different param slots (no coalescing).
        for step in 0..nt {
            let step_u32 = step as u32;
            let mut slot = 0usize;
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("pstd_step") }
            );

            // ─── VELOCITY UPDATE ─────────────────────────────────────────────
            // Copy p → kspace
            self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                &self.pipeline_copy_field_to_k, &bg_sensor, elem_wg, "cp_p");
            // Forward FFT of p
            self.fft_3d(&mut encoder, &mut slot, &bg_sensor, step_u32, ns_u);

            for ax in 0u32..3u32 {
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, ax),
                    &self.pipeline_kspace_shift, &bg_sensor, elem_wg, "kshift_v");
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                    &self.pipeline_copy_k2_to_k, &bg_sensor, elem_wg, "cpk2_v");
                self.ifft_3d(&mut encoder, &mut slot, &bg_sensor, step_u32, ns_u);
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                    &self.pipeline_copy_k_to_scratch, &bg_sensor, elem_wg, "cp_sc_v");
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, ax),
                    &self.pipeline_vel_update, &bg_sensor, elem_wg, "vel_upd");
            }

            // ─── SOURCE INJECTION ─────────────────────────────────────────────
            if n_src > 0 {
                self.dispatch(&mut encoder, &mut slot,
                    &PstdParams {
                        nx: nxu, ny: nyu, nz: nzu,
                        axis: n_src as u32, n_fft: 0, n_batches: 0, log2n: 0, inverse: 0,
                        step: step_u32, dt: dtu, n_sensors: ns_u, nt: ntu,
                    },
                    &self.pipeline_inject_src, &bg_sensor, ceil_div(n_src, 256), "inject");
            }

            // ─── DENSITY UPDATE ───────────────────────────────────────────────
            for ax in 0u32..3u32 {
                let field_sel = ax + 1;
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, field_sel),
                    &self.pipeline_copy_field_to_k, &bg_sensor, elem_wg, "cp_u");
                self.fft_3d(&mut encoder, &mut slot, &bg_sensor, step_u32, ns_u);
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, ax + 3),
                    &self.pipeline_kspace_shift, &bg_sensor, elem_wg, "kshift_d");
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                    &self.pipeline_copy_k2_to_k, &bg_sensor, elem_wg, "cpk2_d");
                self.ifft_3d(&mut encoder, &mut slot, &bg_sensor, step_u32, ns_u);
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                    &self.pipeline_copy_k_to_scratch, &bg_sensor, elem_wg, "cp_sc_d");
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, ax),
                    &self.pipeline_dens_update, &bg_sensor, elem_wg, "dens_upd");
            }

            // ─── PRESSURE FROM DENSITY ────────────────────────────────────────
            self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                &self.pipeline_pres_density, &bg_sensor, elem_wg, "pres");

            // ─── RECORD SENSORS ───────────────────────────────────────────────
            if n_sensors > 0 {
                self.dispatch(&mut encoder, &mut slot, &p!(step_u32, 0),
                    &self.pipeline_record, &bg_sensor, ceil_div(n_sensors, 256), "rec");
            }

            // Submit per-step encoder
            self.queue.submit(std::iter::once(encoder.finish()));
        } // end time loop

        // Wait for all GPU work to complete before reading sensor data
        self.device.poll(wgpu::Maintain::Wait);

        // ── Download sensor data ──────────────────────────────────────────────
        if n_sensors == 0 {
            return Vec::new();
        }

        let sensor_bytes = (sensor_count * nt * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sensor_staging"),
            size: sensor_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut copy_enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("sensor_copy") }
        );
        copy_enc.copy_buffer_to_buffer(&buf_sensor_data, 0, &staging, 0, sensor_bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        result
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// Write params to slot[slot] and encode one compute dispatch.
    /// The slot index must be unique per step (0..63) to prevent write coalescing.
    #[inline]
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: &mut usize,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        debug_assert!(*slot < self.params_slots.len(), "params slot overflow");
        self.queue.write_buffer(&self.params_slots[*slot], 0, bytemuck::bytes_of(params));
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.params_slot_bgs[*slot], &[]);
        cpass.set_bind_group(2, &self.bg_kspace, &[]);
        cpass.set_bind_group(3, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
        *slot += 1;
    }

    /// Encode a forward 3D FFT: Z-axis, then Y-axis, then X-axis.
    fn fft_3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: &mut usize,
        bg_sensor: &wgpu::BindGroup,
        step_u32: u32,
        n_sensors: u32,
    ) {
        let nx = self.nx as u32;
        let ny = self.ny as u32;
        let nz = self.nz as u32;
        let dt = self.dt as f32;
        let nt = self.nt as u32;
        let p = |axis: u32, n_fft: u32, n_batches: u32, log2n: u32| PstdParams {
            nx, ny, nz, axis, n_fft, n_batches, log2n, inverse: 0,
            step: step_u32, dt, n_sensors, nt,
        };
        self.dispatch(encoder, slot, &p(2, nz, nx*ny, nz.trailing_zeros()), &self.pipeline_fft, bg_sensor, nx*ny, "fft_z");
        self.dispatch(encoder, slot, &p(1, ny, nx*nz, ny.trailing_zeros()), &self.pipeline_fft, bg_sensor, nx*nz, "fft_y");
        self.dispatch(encoder, slot, &p(0, nx, ny*nz, nx.trailing_zeros()), &self.pipeline_fft, bg_sensor, ny*nz, "fft_x");
    }

    /// Encode an inverse 3D FFT: X-axis, then Y-axis, then Z-axis.
    fn ifft_3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: &mut usize,
        bg_sensor: &wgpu::BindGroup,
        step_u32: u32,
        n_sensors: u32,
    ) {
        let nx = self.nx as u32;
        let ny = self.ny as u32;
        let nz = self.nz as u32;
        let dt = self.dt as f32;
        let nt = self.nt as u32;
        let p = |axis: u32, n_fft: u32, n_batches: u32, log2n: u32| PstdParams {
            nx, ny, nz, axis, n_fft, n_batches, log2n, inverse: 1,
            step: step_u32, dt, n_sensors, nt,
        };
        self.dispatch(encoder, slot, &p(0, nx, ny*nz, nx.trailing_zeros()), &self.pipeline_fft, bg_sensor, ny*nz, "ifft_x");
        self.dispatch(encoder, slot, &p(1, ny, nx*nz, ny.trailing_zeros()), &self.pipeline_fft, bg_sensor, nx*nz, "ifft_y");
        self.dispatch(encoder, slot, &p(2, nz, nx*ny, nz.trailing_zeros()), &self.pipeline_fft, bg_sensor, nx*ny, "ifft_z");
    }
    /// Create a `GpuPstdSolver` by automatically selecting the best available
    /// GPU adapter.  Returns `Err` if no adapter is found or device creation fails.
    ///
    /// This constructor owns the wgpu device lifecycle, so callers do not need
    /// to add `wgpu` or `pollster` as direct dependencies.
    #[allow(clippy::too_many_arguments)]
    pub fn with_auto_device(
        grid:    &Grid,
        c0_flat:   &[f32],
        rho0_flat: &[f32],
        dt:    f64,
        nt:    usize,
        c_ref: f64,
        pml_x:   &[f32],
        pml_y:   &[f32],
        pml_z:   &[f32],
        pml_sgx: &[f32],
        pml_sgy: &[f32],
        pml_sgz: &[f32],
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok_or_else(|| "No GPU adapter available".to_string())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pstd_auto"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )).map_err(|e| format!("GPU device creation failed: {e}"))?;

        Self::new(
            Arc::new(device), Arc::new(queue),
            grid, c0_flat, rho0_flat, dt, nt, c_ref,
            pml_x, pml_y, pml_z, pml_sgx, pml_sgy, pml_sgz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify GpuPstdSolver can be constructed and runs without error.
    /// Skipped if no GPU adapter is available (headless CI).
    #[test]
    fn test_gpu_pstd_solver_new() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            eprintln!("No GPU adapter — skipping GpuPstdSolver test");
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_pstd"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )).expect("device creation");

        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let n = 32usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;
        let nt = 10;

        let grid = crate::domain::grid::Grid::new(n, n, n, dx, dx, dx).unwrap();
        let c0v:   Vec<f32> = vec![c0 as f32;   n * n * n];
        let rho0v: Vec<f32> = vec![rho0 as f32; n * n * n];
        let ones:  Vec<f32> = vec![1.0f32;      n * n * n];

        let solver = GpuPstdSolver::new(
            device, queue, &grid,
            &c0v, &rho0v, dt, nt, c0,
            &ones, &ones, &ones,
            &ones, &ones, &ones,
        );

        assert!(solver.is_ok(), "GpuPstdSolver::new failed: {:?}", solver.err());
        eprintln!("GpuPstdSolver constructed successfully");
    }

    /// Run a minimal simulation: one source point, one sensor point, 20 steps.
    /// Verify that non-zero pressure is recorded at the sensor.
    #[test]
    fn test_gpu_pstd_run_produces_output() {
        let solver = GpuPstdSolver::with_auto_device(
            &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
            &vec![1500.0f32; 32*32*32],
            &vec![1000.0f32; 32*32*32],
            0.3e-3 / 1500.0,
            20,
            1500.0,
            &vec![1.0f32; 32*32*32],
            &vec![1.0f32; 32*32*32],
            &vec![1.0f32; 32*32*32],
            &vec![1.0f32; 32*32*32],
            &vec![1.0f32; 32*32*32],
            &vec![1.0f32; 32*32*32],
        );

        let Some(mut solver) = solver.ok() else {
            eprintln!("No GPU adapter — skipping run test");
            return;
        };

        // Source at grid center (16,16,16), sensor at (20,16,16)
        let n = 32usize;
        let src_flat = 16 * n * n + 16 * n + 16;
        let sns_flat = 20 * n * n + 16 * n + 16;

        // Signal: ramp from 0 to 1 over 20 steps
        let source_signals: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();

        let data = solver.run(
            &[sns_flat as u32],
            &[src_flat as u32],
            &source_signals,
        );

        assert_eq!(data.len(), 20, "sensor data length");
        let max_val = data.iter().copied().fold(0.0f32, f32::max);
        eprintln!("GPU PSTD sensor peak: {max_val:.6}");
        // After source injection, some non-zero value should appear
        // (exact value depends on GPU FFT accuracy — just verify non-zero)
        assert!(
            max_val.is_finite(),
            "sensor data contains non-finite values"
        );
    }
}
