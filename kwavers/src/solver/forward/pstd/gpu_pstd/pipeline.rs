//! `GpuPstdSolver` constructor (`new`) and auto-device factory (`with_auto_device`).
//!
//! **SRP boundary**: this module changes when the wgpu API changes, when the
//! bind-group layout changes, or when the shader compilation strategy changes.
//! It does *not* change when time-stepping physics algorithms change.

use super::{GpuPstdSolver, PstdParams};
use crate::domain::grid::Grid;
use crate::math::fft::shift_operators::{generate_kappa, generate_shift_1d, generate_source_kappa};
use std::f64::consts::PI;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    /// * `pml_x/y/z`       — collocated PML damping (for density), f32 [nx×ny×nz]
    /// * `pml_sgx/y/z`     — staggered PML damping (for velocity), f32 [nx×ny×nz]
    /// * `bon_a_flat`    — B/(2A) per voxel; pass all-zeros for linear simulation
    /// * `absorb_nabla1` — |k|^(y−2) in FFT order; pass all-zeros for lossless
    /// * `absorb_nabla2` — |k|^(y−1) in FFT order; pass all-zeros for lossless
    /// * `absorb_tau`    — −2α₀c₀^(y−1) per voxel; pass all-zeros for lossless
    /// * `absorb_eta`    — 2α₀c₀^y·tan(πy/2) per voxel; pass all-zeros for lossless
    /// * `nonlinear`     — enable Westervelt BonA pressure correction
    /// * `absorbing`     — enable fractional-Laplacian absorption (Treeby & Cox 2010)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        grid: &Grid,
        c0_flat: &[f32],
        rho0_flat: &[f32],
        dt: f64,
        nt: usize,
        c_ref: f64,
        pml_x: &[f32],
        pml_y: &[f32],
        pml_z: &[f32],
        pml_sgx: &[f32],
        pml_sgy: &[f32],
        pml_sgz: &[f32],
        bon_a_flat: &[f32],
        absorb_nabla1: &[f32],
        absorb_nabla2: &[f32],
        absorb_tau: &[f32],
        absorb_eta: &[f32],
        nonlinear: bool,
        absorbing: bool,
    ) -> Result<Self, String> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
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
        let source_kappa_3d =
            generate_source_kappa(nx, ny, nz, grid.dx, grid.dy, grid.dz, c_ref, dt);
        let source_kappa_f32: Vec<f32> = source_kappa_3d.iter().map(|&v| v as f32).collect();

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
        for c in sx_pos.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sx_pos.iter() {
            shifts_all.push(c.im as f32);
        }
        for c in sx_neg.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sx_neg.iter() {
            shifts_all.push(c.im as f32);
        }
        for c in sy_pos.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sy_pos.iter() {
            shifts_all.push(c.im as f32);
        }
        for c in sy_neg.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sy_neg.iter() {
            shifts_all.push(c.im as f32);
        }
        for c in sz_pos.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sz_pos.iter() {
            shifts_all.push(c.im as f32);
        }
        for c in sz_neg.iter() {
            shifts_all.push(c.re as f32);
        }
        for c in sz_neg.iter() {
            shifts_all.push(c.im as f32);
        }

        // Pack pml_xyz = [pml_x | pml_y | pml_z]
        let mut pml_xyz: Vec<f32> = Vec::with_capacity(3 * total);
        pml_xyz.extend_from_slice(pml_x);
        pml_xyz.extend_from_slice(pml_y);
        pml_xyz.extend_from_slice(pml_z);

        // Derived arrays
        let rho0_inv: Vec<f32> = rho0_flat.iter().map(|&r| 1.0 / r).collect();
        let c0_sq: Vec<f32> = c0_flat.iter().map(|&c| c * c).collect();

        // ── Buffer helpers ────────────────────────────────────────────────────
        let mk_ro = |data: &[f32], label: &str| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        // These storage buffers are cleared or overwritten by the first GPU
        // dispatch before any readback occurs. Allocating a zero-filled host
        // buffer here would double initialization bandwidth and spike peak RSS.
        let mk_rw = |n: usize, extra: wgpu::BufferUsages, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | extra,
                mapped_at_creation: false,
            })
        };

        // group(0) field buffers
        let buf_p = mk_rw(total, wgpu::BufferUsages::empty(), "buf_p");
        let buf_ux = mk_rw(total, wgpu::BufferUsages::empty(), "buf_ux");
        let buf_uy = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uy");
        let buf_uz = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uz");
        let buf_rhox = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhox");
        let buf_rhoy = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoy");
        let buf_rhoz = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoz");
        let buf_source_kappa = mk_ro(&source_kappa_f32, "source_kappa");
        // group(1) k-space + medium (kspace2_re/im and field_scratch removed)
        let buf_kspace_re = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_re");
        let buf_kspace_im = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_im");
        let buf_kappa = mk_ro(&kappa_f32, "kappa");
        let buf_rho0_inv = mk_ro(&rho0_inv, "rho0_inv");
        let buf_c0_sq = mk_ro(&c0_sq, "c0_sq");
        let buf_rho0 = mk_ro(rho0_flat, "rho0");
        let buf_bon_a = mk_ro(bon_a_flat, "bon_a");
        // Repurpose buf_alpha_decay as the FFT twiddle-factor buffer.
        // apply_absorption (broadband multiplicative decay) is never dispatched;
        // the fractional-Laplacian path is used instead.  The first 480 elements
        // store precomputed cos/sin tables for the FFT kernel (eliminating runtime
        // trig from every butterfly stage):
        //   [0..128)   cos(-2πk/256) k=0..127  (n=256 re)
        //   [128..256) sin(-2πk/256) k=0..127  (n=256 im)
        //   [256..320) cos(-2πk/128) k=0..63   (n=128 re)
        //   [320..384) sin(-2πk/128) k=0..63   (n=128 im)
        //   [384..416) cos(-2πk/64)  k=0..31   (n=64  re)
        //   [416..448) sin(-2πk/64)  k=0..31   (n=64  im)
        //   [448..464) cos(-2πk/32)  k=0..15   (n=32  re)
        //   [464..480) sin(-2πk/32)  k=0..15   (n=32  im)
        // Remaining elements are unused padding (zero-filled).
        let mut twiddle_data: Vec<f32> = vec![0.0f32; total];
        // n=256
        for k in 0usize..128 {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 256.0;
            twiddle_data[k] = angle.cos() as f32;
            twiddle_data[128 + k] = angle.sin() as f32;
        }
        // n=128
        for k in 0usize..64 {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 128.0;
            twiddle_data[256 + k] = angle.cos() as f32;
            twiddle_data[320 + k] = angle.sin() as f32;
        }
        // n=64
        for k in 0usize..32 {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 64.0;
            twiddle_data[384 + k] = angle.cos() as f32;
            twiddle_data[416 + k] = angle.sin() as f32;
        }
        // n=32
        for k in 0usize..16 {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / 32.0;
            twiddle_data[448 + k] = angle.cos() as f32;
            twiddle_data[464 + k] = angle.sin() as f32;
        }
        let buf_alpha_decay = mk_ro(&twiddle_data, "twiddle_fft");
        // Fractional-Laplacian absorption buffers (group 3)
        let buf_absorb_nabla1 = mk_ro(absorb_nabla1, "absorb_nabla1");
        let buf_absorb_nabla2 = mk_ro(absorb_nabla2, "absorb_nabla2");
        let buf_absorb_tau = mk_ro(absorb_tau, "absorb_tau");
        let buf_absorb_eta = mk_ro(absorb_eta, "absorb_eta");
        let buf_absorb_scratch_kre = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_kre");
        let buf_absorb_scratch_kim = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_kim");
        let buf_absorb_scratch_l1 = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_l1");
        let buf_absorb_scratch_l2 = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_l2");

        // group(3) PML + shifts
        let buf_pml_sgx = mk_ro(pml_sgx, "pml_sgx");
        let buf_pml_sgy = mk_ro(pml_sgy, "pml_sgy");
        let buf_pml_sgz = mk_ro(pml_sgz, "pml_sgz");
        let buf_pml_xyz = mk_ro(&pml_xyz, "pml_xyz");
        let buf_shifts_all = mk_ro(&shifts_all, "shifts_all");

        // ── Shader module ─────────────────────────────────────────────────────
        let shader_src = include_str!("../../../../gpu/shaders/pstd.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pstd_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────

        // Helper: N read_write storage bindings
        let _bgl_storage_rw = |n: u32, label: &str| -> wgpu::BindGroupLayout {
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

        // group(0): 7 read_write field buffers + 1 read-only source-kappa buffer
        let bgl_fields = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_fields"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // group(1): 2 read_write + 6 read-only storage (kspace2_re/im removed)
        // bindings 0-1: kspace_re/im (read_write)
        // bindings 2-7: kappa, rho0_inv, c0_sq, rho0, bon_a, alpha_decay (read-only)
        let bgl_kspace = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_kspace"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // group(3): 8 storage (5 read-only, 1 read/write sensor_data, 1 read sensor_indices, 1 read source_data)
        let bgl_sensor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_sensor"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: sensor_flat_indices (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: sensor_data (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: source_data (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // group(3): 4 read-only + 4 read_write storage buffers for fractional-Laplacian absorption
        let bgl_absorb = {
            let ro_entry = |b: u32| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let rw_entry = |b: u32| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bgl_absorb"),
                entries: &[
                    ro_entry(0),
                    ro_entry(1),
                    ro_entry(2),
                    ro_entry(3), // nabla1, nabla2, tau, eta
                    rw_entry(4),
                    rw_entry(5),
                    rw_entry(6),
                    rw_entry(7), // scratch_kre/kim, l1, l2
                ],
            })
        };

        // ── Pipeline layouts ──────────────────────────────────────────────────
        // Standard layout: groups 0–2 (fields, kspace, sensor)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pstd_pipeline_layout"),
            bind_group_layouts: &[&bgl_fields, &bgl_kspace, &bgl_sensor],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PstdParams>() as u32,
            }],
        });

        // Absorption layout: groups 0–3 (fields, kspace, sensor[placeholder], absorb)
        // Absorption shaders access group(0) (rho fields), group(1) (kspace), group(3) (absorption).
        // group(2) (bgl_sensor) is present in the layout for API compatibility but unused by these shaders.
        let pipeline_layout_absorb =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pstd_pipeline_layout_absorb"),
                bind_group_layouts: &[&bgl_fields, &bgl_kspace, &bgl_sensor, &bgl_absorb],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<PstdParams>() as u32,
                }],
            });

        // ── Compile pipelines ─────────────────────────────────────────────────
        let mk_pl = |entry: &'static str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipeline_zero_fields = mk_pl("zero_acoustic_fields");
        let pipeline_zero_kspace = mk_pl("zero_kspace");
        let pipeline_fft = mk_pl("fft_1d_smem");
        let pipeline_kspace_shift = mk_pl("kspace_shift_apply");
        let pipeline_vel_update = mk_pl("velocity_update");
        let pipeline_dens_update = mk_pl("density_update");
        let pipeline_snapshot_rho0_plus_rho = mk_pl("snapshot_rho0_plus_rho");
        let pipeline_absorption = mk_pl("apply_absorption");
        let pipeline_pres_density = mk_pl("pressure_from_density");
        let pipeline_record = mk_pl("record_sensors");
        let pipeline_inject_src = mk_pl("inject_pressure_source");
        let pipeline_inject_vel_x = mk_pl("inject_velocity_x_source");
        let pipeline_apply_source_kappa = mk_pl("apply_source_kappa");
        let pipeline_add_kspace_to_field_ux = mk_pl("add_kspace_to_field_ux");
        let pipeline_copy_field_to_k = mk_pl("copy_field_to_kspace");

        // Fractional-Laplacian absorption pipelines (use 4-group layout)
        let mk_pl_absorb = |entry: &'static str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout_absorb),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        // Correct k-Wave pressure-based absorption pipelines
        let pipeline_absorb_mul_nabla = mk_pl_absorb("absorb_mul_nabla");
        let pipeline_absorb_copy_to_scratch = mk_pl_absorb("absorb_copy_to_scratch");
        let pipeline_absorb_accum_div_u = mk_pl_absorb("absorb_accum_div_u");
        let pipeline_absorb_prep_l1_kspace = mk_pl_absorb("absorb_prep_l1_kspace");
        let pipeline_absorb_prep_l2_kspace = mk_pl_absorb("absorb_prep_l2_kspace");
        let pipeline_absorb_pressure_correction = mk_pl_absorb("absorb_pressure_correction");
        // kspace save/restore — used to cache FFT(p) during the velocity loop so
        // the forward FFT of pressure is only computed once per time step (not 3×).
        // absorb_scratch_kre/kim are idle during the velocity section and safe to reuse.
        let pipeline_absorb_save_kspace = mk_pl_absorb("absorb_save_kspace");
        let pipeline_absorb_restore_kspace = mk_pl_absorb("absorb_restore_kspace");

        // ── Bind groups (sensor rebuilt per run) ──────────────────────────────
        let bg_fields = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_fields"),
            layout: &bgl_fields,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_ux.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_uy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_uz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_rhox.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_rhoy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_rhoz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_source_kappa.as_entire_binding(),
                },
            ],
        });

        let bg_kspace = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_kspace"),
            layout: &bgl_kspace,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_kspace_re.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_kspace_im.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_kappa.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_rho0_inv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_c0_sq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_rho0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_bon_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_alpha_decay.as_entire_binding(),
                },
            ],
        });

        let bg_absorb = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_absorb"),
            layout: &bgl_absorb,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_absorb_nabla1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_absorb_nabla2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_absorb_tau.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_absorb_eta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_absorb_scratch_kre.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_absorb_scratch_kim.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_absorb_scratch_l1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_absorb_scratch_l2.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            nx,
            ny,
            nz,
            nt,
            dt,
            buf_p,
            buf_ux,
            buf_uy,
            buf_uz,
            buf_rhox,
            buf_rhoy,
            buf_rhoz,
            buf_kspace_re,
            buf_kspace_im,
            buf_kappa,
            buf_rho0_inv,
            buf_c0_sq,
            buf_rho0,
            buf_bon_a,
            buf_alpha_decay,
            nonlinear,
            absorbing,
            buf_pml_sgx,
            buf_pml_sgy,
            buf_pml_sgz,
            buf_pml_xyz,
            buf_shifts_all,
            pipeline_zero_fields,
            pipeline_zero_kspace,
            pipeline_fft,
            pipeline_kspace_shift,
            pipeline_vel_update,
            pipeline_dens_update,
            pipeline_snapshot_rho0_plus_rho,
            pipeline_absorption,
            pipeline_pres_density,
            pipeline_record,
            pipeline_inject_src,
            pipeline_inject_vel_x,
            pipeline_apply_source_kappa,
            pipeline_add_kspace_to_field_ux,
            pipeline_copy_field_to_k,
            pipeline_absorb_mul_nabla,
            pipeline_absorb_copy_to_scratch,
            pipeline_absorb_accum_div_u,
            pipeline_absorb_prep_l1_kspace,
            pipeline_absorb_prep_l2_kspace,
            pipeline_absorb_pressure_correction,
            pipeline_absorb_save_kspace,
            pipeline_absorb_restore_kspace,
            buf_absorb_nabla1,
            buf_absorb_nabla2,
            buf_absorb_tau,
            buf_absorb_eta,
            buf_absorb_scratch_kre,
            buf_absorb_scratch_kim,
            buf_absorb_scratch_l1,
            buf_absorb_scratch_l2,
            bg_fields,
            bg_kspace,
            bg_absorb,
            bgl_sensor,
            pipeline_layout,
            // CPU scratch for update_medium() — preallocated to avoid per-scan-line alloc.
            scratch_c0_sq: vec![0.0f32; total],
            scratch_rho0_inv: vec![0.0f32; total],
            scratch_source_data: Vec::new(),
            scratch_vel_x_data: Vec::new(),
            // Cached run() buffers — allocated lazily on first run().
            cache_sensor_indices_buf: None,
            cache_sensor_data_buf: None,
            cache_source_data_buf: None,
            cache_vel_x_data_buf: None,
            cache_staging_buf: None,
            cache_bg_sensor: None,
            cache_bg_sensor_vel: None,
            cache_n_sensors: 0,
            cache_n_src: 0,
            cache_n_vel_x: 0,
            buf_source_kappa,
        })
    }

    /// Create a `GpuPstdSolver` by automatically selecting the best available
    /// GPU adapter.  Returns `Err` if no adapter is found or device creation fails.
    ///
    /// This constructor owns the wgpu device lifecycle, so callers do not need
    /// to add `wgpu` or `pollster` as direct dependencies.
    #[allow(clippy::too_many_arguments)]
    pub fn with_auto_device(
        grid: &Grid,
        c0_flat: &[f32],
        rho0_flat: &[f32],
        dt: f64,
        nt: usize,
        c_ref: f64,
        pml_x: &[f32],
        pml_y: &[f32],
        pml_z: &[f32],
        pml_sgx: &[f32],
        pml_sgy: &[f32],
        pml_sgz: &[f32],
        bon_a_flat: &[f32],
        absorb_nabla1: &[f32],
        absorb_nabla2: &[f32],
        absorb_tau: &[f32],
        absorb_eta: &[f32],
        nonlinear: bool,
        absorbing: bool,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter available".to_string())?;

        // Query adapter's native limits; use them as a ceiling so we don't
        // exceed what the hardware supports but do raise the WebGPU defaults
        // (e.g. max_storage_buffers_per_shader_stage is 8 by default but the
        // PSTD kspace bind group uses 10 storage buffers).
        let native_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pstd_auto"),
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                // Allow up to the adapter maximum so we don't hit the
                // WebGPU default limit of 8 storage buffers / stage.
                max_storage_buffers_per_shader_stage:
                    native_limits.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("GPU device creation failed: {e}"))?;

        Self::new(
            Arc::new(device),
            Arc::new(queue),
            grid,
            c0_flat,
            rho0_flat,
            dt,
            nt,
            c_ref,
            pml_x,
            pml_y,
            pml_z,
            pml_sgx,
            pml_sgy,
            pml_sgz,
            bon_a_flat,
            absorb_nabla1,
            absorb_nabla2,
            absorb_tau,
            absorb_eta,
            nonlinear,
            absorbing,
        )
    }
}
