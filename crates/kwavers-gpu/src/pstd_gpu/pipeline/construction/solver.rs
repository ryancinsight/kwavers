use super::super::super::{GpuPstdSolver, PstdParams};
use super::super::bgl::{build_bgl_absorb, build_bgl_fields, build_bgl_kspace, build_bgl_sensor};
use super::super::bind_groups::{
    build_bg_absorb, build_bg_fields, build_bg_kspace, AbsorbBuffers, FieldBuffers, KspaceBuffers,
};
use super::super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use super::kspace::{precompute_kspace_shifts, KSpaceGridParams};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use std::sync::Arc;
use wgpu::util::DeviceExt;

impl GpuPstdSolver {
    /// Create a new GPU PSTD solver.
    ///
    /// See `mod.rs` for argument documentation and bind-group layout.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, String> {
        let MediumArrays { c0_flat, rho0_flat } = medium;
        let SolverParams {
            dt,
            nt,
            c_ref,
            nonlinear,
            absorbing,
        } = solver;
        let PmlArrays {
            x: pml_x,
            y: pml_y,
            z: pml_z,
            sgx: pml_sgx,
            sgy: pml_sgy,
            sgz: pml_sgz,
        } = pml;
        let AbsorptionArrays {
            bon_a_flat,
            nabla1: absorb_nabla1,
            nabla2: absorb_nabla2,
            tau: absorb_tau,
            eta: absorb_eta,
        } = absorption;
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

        // ── Precompute kappa (3-D) and 1-D shift operators ───────────────────
        let (kappa_f32, source_kappa_f32, shifts_all) =
            precompute_kspace_shifts(KSpaceGridParams {
                nx,
                ny,
                nz,
                dx: grid.dx,
                dy: grid.dy,
                dz: grid.dz,
                c_ref,
                dt,
            });

        // Pack pml_xyz = [pml_x | pml_y | pml_z]
        let mut pml_xyz: Vec<f32> = Vec::with_capacity(3 * total);
        pml_xyz.extend_from_slice(pml_x);
        pml_xyz.extend_from_slice(pml_y);
        pml_xyz.extend_from_slice(pml_z);

        let rho0_inv: Vec<f32> = rho0_flat.iter().map(|&r| 1.0 / r).collect();
        let c0_sq: Vec<f32> = c0_flat.iter().map(|&c| c * c).collect();

        // ── Buffer helpers ────────────────────────────────────────────────────
        let mk_ro = |data: &[f32], label: &str| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        // Allocated uninitialised — GPU zero-fills before first read via zero_acoustic_fields.
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

        // ── Allocate GPU buffers ──────────────────────────────────────────────
        let buf_p = mk_rw(total, wgpu::BufferUsages::empty(), "buf_p");
        let buf_ux = mk_rw(total, wgpu::BufferUsages::empty(), "buf_ux");
        let buf_uy = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uy");
        let buf_uz = mk_rw(total, wgpu::BufferUsages::empty(), "buf_uz");
        let buf_rhox = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhox");
        let buf_rhoy = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoy");
        let buf_rhoz = mk_rw(total, wgpu::BufferUsages::empty(), "buf_rhoz");
        let buf_source_kappa = mk_ro(&source_kappa_f32, "source_kappa");

        let buf_kspace_re = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_re");
        let buf_kspace_im = mk_rw(total, wgpu::BufferUsages::empty(), "kspace_im");
        let buf_kappa = mk_ro(&kappa_f32, "kappa");
        let buf_rho0_inv = mk_ro(&rho0_inv, "rho0_inv");
        let buf_c0_sq = mk_ro(&c0_sq, "c0_sq");
        let buf_rho0 = mk_ro(rho0_flat, "rho0");
        let buf_bon_a = mk_ro(bon_a_flat, "bon_a");

        // FFT twiddle factors: precomputed cos/sin tables for n=256,128,64,32.
        // Stored in buf_alpha_decay (repurposed as twiddle table; absorption uses fractional-Laplacian).
        let mut twiddle_data: Vec<f32> = vec![0.0f32; total];
        for k in 0usize..128 {
            let a = -TWO_PI * k as f64 / 256.0;
            twiddle_data[k] = a.cos() as f32;
            twiddle_data[128 + k] = a.sin() as f32;
        }
        for k in 0usize..64 {
            let a = -TWO_PI * k as f64 / 128.0;
            twiddle_data[256 + k] = a.cos() as f32;
            twiddle_data[320 + k] = a.sin() as f32;
        }
        for k in 0usize..32 {
            let a = -TWO_PI * k as f64 / 64.0;
            twiddle_data[384 + k] = a.cos() as f32;
            twiddle_data[416 + k] = a.sin() as f32;
        }
        for k in 0usize..16 {
            let a = -TWO_PI * k as f64 / 32.0;
            twiddle_data[448 + k] = a.cos() as f32;
            twiddle_data[464 + k] = a.sin() as f32;
        }
        let buf_alpha_decay = mk_ro(&twiddle_data, "twiddle_fft");

        let buf_absorb_nabla1 = mk_ro(absorb_nabla1, "absorb_nabla1");
        let buf_absorb_nabla2 = mk_ro(absorb_nabla2, "absorb_nabla2");
        let buf_absorb_tau = mk_ro(absorb_tau, "absorb_tau");
        let buf_absorb_eta = mk_ro(absorb_eta, "absorb_eta");
        let buf_absorb_scratch_kre = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_kre");
        let buf_absorb_scratch_kim = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_kim");
        let buf_absorb_scratch_l1 = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_l1");
        let buf_absorb_scratch_l2 = mk_rw(total, wgpu::BufferUsages::empty(), "absorb_l2");

        let buf_pml_sgx = mk_ro(pml_sgx, "pml_sgx");
        let buf_pml_sgy = mk_ro(pml_sgy, "pml_sgy");
        let buf_pml_sgz = mk_ro(pml_sgz, "pml_sgz");
        let buf_pml_xyz = mk_ro(&pml_xyz, "pml_xyz");
        let buf_shifts_all = mk_ro(&shifts_all, "shifts_all");

        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pstd_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pstd.wgsl").into()),
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        let bgl_fields = build_bgl_fields(&device);
        let bgl_kspace = build_bgl_kspace(&device);
        let bgl_sensor = build_bgl_sensor(&device);
        let bgl_absorb = build_bgl_absorb(&device);

        // ── Pipeline layouts ──────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pstd_pipeline_layout"),
            bind_group_layouts: &[&bgl_fields, &bgl_kspace, &bgl_sensor],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PstdParams>() as u32,
            }],
        });
        let pipeline_layout_absorb =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pstd_pipeline_layout_absorb"),
                bind_group_layouts: &[&bgl_fields, &bgl_kspace, &bgl_sensor, &bgl_absorb],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<PstdParams>() as u32,
                }],
            });

        // ── Compile compute pipelines ─────────────────────────────────────────
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

        let pipeline_zero_fields = mk_pl("zero_acoustic_fields");
        let pipeline_zero_kspace = mk_pl("zero_kspace");
        let pipeline_fft = mk_pl("fft_1d_smem");
        let pipeline_kspace_shift = mk_pl("kspace_shift_apply");
        let pipeline_vel_update = mk_pl("velocity_update");
        let pipeline_dens_update = mk_pl("density_update");
        let pipeline_snapshot_rho0_plus_rho = mk_pl("snapshot_rho0_plus_rho");
        let pipeline_pres_density = mk_pl("pressure_from_density");
        let pipeline_record = mk_pl("record_sensors");
        let pipeline_inject_src = mk_pl("inject_pressure_source");
        let pipeline_inject_vel_x = mk_pl("inject_velocity_x_source");
        let pipeline_apply_source_kappa = mk_pl("apply_source_kappa");
        let pipeline_add_kspace_to_field_ux = mk_pl("add_kspace_to_field_ux");
        let pipeline_copy_field_to_k = mk_pl("copy_field_to_kspace");

        let pipeline_absorb_mul_nabla = mk_pl_absorb("absorb_mul_nabla");
        let pipeline_absorb_copy_to_scratch = mk_pl_absorb("absorb_copy_to_scratch");
        let pipeline_absorb_accum_div_u = mk_pl_absorb("absorb_accum_div_u");
        let pipeline_absorb_prep_l1_kspace = mk_pl_absorb("absorb_prep_l1_kspace");
        let pipeline_absorb_prep_l2_kspace = mk_pl_absorb("absorb_prep_l2_kspace");
        let pipeline_absorb_pressure_correction = mk_pl_absorb("absorb_pressure_correction");
        let pipeline_absorb_save_kspace = mk_pl_absorb("absorb_save_kspace");
        let pipeline_restore_and_shift = mk_pl_absorb("restore_and_shift_apply");

        // ── Build permanent bind groups ───────────────────────────────────────
        let bg_fields = build_bg_fields(
            &device,
            &bgl_fields,
            &FieldBuffers {
                p: &buf_p,
                ux: &buf_ux,
                uy: &buf_uy,
                uz: &buf_uz,
                rhox: &buf_rhox,
                rhoy: &buf_rhoy,
                rhoz: &buf_rhoz,
                source_kappa: &buf_source_kappa,
            },
        );
        let bg_kspace = build_bg_kspace(
            &device,
            &bgl_kspace,
            &KspaceBuffers {
                kspace_re: &buf_kspace_re,
                kspace_im: &buf_kspace_im,
                kappa: &buf_kappa,
                rho0_inv: &buf_rho0_inv,
                c0_sq: &buf_c0_sq,
                rho0: &buf_rho0,
                bon_a: &buf_bon_a,
                alpha_decay: &buf_alpha_decay,
            },
        );
        let bg_absorb = build_bg_absorb(
            &device,
            &bgl_absorb,
            &AbsorbBuffers {
                nabla1: &buf_absorb_nabla1,
                nabla2: &buf_absorb_nabla2,
                tau: &buf_absorb_tau,
                eta: &buf_absorb_eta,
                scratch_kre: &buf_absorb_scratch_kre,
                scratch_kim: &buf_absorb_scratch_kim,
                scratch_l1: &buf_absorb_scratch_l1,
                scratch_l2: &buf_absorb_scratch_l2,
            },
        );

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
            pipeline_restore_and_shift,
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
            scratch_c0_sq: vec![0.0f32; total],
            scratch_rho0_inv: vec![0.0f32; total],
            scratch_rho0_flat: vec![0.0f32; total],
            scratch_source_kappa_ones: vec![1.0f32; total],
            scratch_source_data: Vec::new(),
            scratch_vel_x_data: Vec::new(),
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
}
