use super::super::super::{GpuPstdSolver, PstdParams};
use super::super::bind_groups::{
    build_bg_absorb, build_bg_fields, build_bg_kspace, AbsorbBuffers, FieldBuffers, KspaceBuffers,
};
use super::super::{
    AbsorptionArrays, MediumArrays, PmlArrays, PstdBindGroupLayoutProvider, PstdBufferProvider,
    PstdPipelineProvider, SolverParams, WgpuPstdBindGroupFactory, WgpuPstdBindGroupLayoutFactory,
    WgpuPstdBufferFactory, WgpuPstdPipelineFactory,
};
use super::kspace::{precompute_kspace_shifts, KSpaceGridParams};
use crate::backend::init::GpuProviderContext;
use crate::pstd_gpu::state::{
    PstdStateBuilder, WgpuPstdAbsorptionBuffers, WgpuPstdFieldBuffers, WgpuPstdKspaceBuffers,
    WgpuPstdLayouts, WgpuPstdMediumBuffers, WgpuPstdPermanentBindGroups, WgpuPstdPipelines,
    WgpuPstdPmlShiftBuffers, WgpuPstdRunCache, WgpuPstdState, WgpuPstdStateProvider,
};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;

impl PstdStateBuilder for WgpuPstdStateProvider {
    type Context = GpuProviderContext<WgpuDevice>;

    fn build_state(
        context: Self::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<WgpuPstdState, String> {
        let MediumArrays { c0_flat, rho0_flat } = medium;
        let SolverParams { dt, c_ref, .. } = solver;
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
        let device = context.device();
        let buffers = WgpuPstdBufferFactory::new(device);

        // ── Buffer helpers ────────────────────────────────────────────────────
        let mk_ro = |data: &[f32], label: &'static str| -> wgpu::Buffer {
            buffers.read_only_storage(data, label)
        };
        // Allocated uninitialised — GPU zero-fills before first read via zero_acoustic_fields.
        let mk_rw = |n: usize, label: &'static str| -> wgpu::Buffer {
            buffers.read_write_storage::<f32>(n, label)
        };

        // ── Allocate GPU buffers ──────────────────────────────────────────────
        let buf_p = mk_rw(total, "buf_p");
        let buf_ux = mk_rw(total, "buf_ux");
        let buf_uy = mk_rw(total, "buf_uy");
        let buf_uz = mk_rw(total, "buf_uz");
        let buf_rhox = mk_rw(total, "buf_rhox");
        let buf_rhoy = mk_rw(total, "buf_rhoy");
        let buf_rhoz = mk_rw(total, "buf_rhoz");
        let field_buffers = WgpuPstdFieldBuffers {
            p: buf_p,
            ux: buf_ux,
            uy: buf_uy,
            uz: buf_uz,
            rhox: buf_rhox,
            rhoy: buf_rhoy,
            rhoz: buf_rhoz,
        };
        let buf_source_kappa = mk_ro(&source_kappa_f32, "source_kappa");

        let buf_kspace_re = mk_rw(total, "kspace_re");
        let buf_kspace_im = mk_rw(total, "kspace_im");
        let kspace_buffers = WgpuPstdKspaceBuffers {
            re: buf_kspace_re,
            im: buf_kspace_im,
        };
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
        let medium_buffers = WgpuPstdMediumBuffers {
            kappa: buf_kappa,
            rho0_inv: buf_rho0_inv,
            c0_sq: buf_c0_sq,
            rho0: buf_rho0,
            bon_a: buf_bon_a,
            alpha_decay: buf_alpha_decay,
            source_kappa: buf_source_kappa,
        };

        let buf_absorb_nabla1 = mk_ro(absorb_nabla1, "absorb_nabla1");
        let buf_absorb_nabla2 = mk_ro(absorb_nabla2, "absorb_nabla2");
        let buf_absorb_tau = mk_ro(absorb_tau, "absorb_tau");
        let buf_absorb_eta = mk_ro(absorb_eta, "absorb_eta");
        let buf_absorb_scratch_kre = mk_rw(total, "absorb_kre");
        let buf_absorb_scratch_kim = mk_rw(total, "absorb_kim");
        let buf_absorb_scratch_l1 = mk_rw(total, "absorb_l1");
        let buf_absorb_scratch_l2 = mk_rw(total, "absorb_l2");
        let absorption_buffers = WgpuPstdAbsorptionBuffers {
            nabla1: buf_absorb_nabla1,
            nabla2: buf_absorb_nabla2,
            tau: buf_absorb_tau,
            eta: buf_absorb_eta,
            scratch_kre: buf_absorb_scratch_kre,
            scratch_kim: buf_absorb_scratch_kim,
            scratch_l1: buf_absorb_scratch_l1,
            scratch_l2: buf_absorb_scratch_l2,
        };

        let buf_pml_sgx = mk_ro(pml_sgx, "pml_sgx");
        let buf_pml_sgy = mk_ro(pml_sgy, "pml_sgy");
        let buf_pml_sgz = mk_ro(pml_sgz, "pml_sgz");
        let buf_pml_xyz = mk_ro(&pml_xyz, "pml_xyz");
        let buf_shifts_all = mk_ro(&shifts_all, "shifts_all");
        let pml_shift_buffers = WgpuPstdPmlShiftBuffers {
            pml_sgx: buf_pml_sgx,
            pml_sgy: buf_pml_sgy,
            pml_sgz: buf_pml_sgz,
            pml_xyz: buf_pml_xyz,
            shifts_all: buf_shifts_all,
        };

        let pipelines = WgpuPstdPipelineFactory::new(device);

        // ── Shader ────────────────────────────────────────────────────────────
        let shader =
            pipelines.shader_module(include_str!("../../shaders/pstd.wgsl"), "pstd_shader");

        // ── Bind group layouts ────────────────────────────────────────────────
        let bind_group_layouts = WgpuPstdBindGroupLayoutFactory::new(device);
        let bgl_fields = bind_group_layouts.fields_layout();
        let bgl_kspace = bind_group_layouts.kspace_layout();
        let bgl_sensor = bind_group_layouts.sensor_layout();
        let bgl_absorb = bind_group_layouts.absorb_layout();

        // ── Pipeline layouts ──────────────────────────────────────────────────
        let immediate_data_bytes = std::mem::size_of::<PstdParams>();
        let pipeline_layout = pipelines.pipeline_layout(
            &[Some(&bgl_fields), Some(&bgl_kspace), Some(&bgl_sensor)],
            immediate_data_bytes,
            "pstd_pipeline_layout",
        );
        let pipeline_layout_absorb = pipelines.pipeline_layout(
            &[
                Some(&bgl_fields),
                Some(&bgl_kspace),
                Some(&bgl_sensor),
                Some(&bgl_absorb),
            ],
            immediate_data_bytes,
            "pstd_pipeline_layout_absorb",
        );

        // ── Compile compute pipelines ─────────────────────────────────────────
        let mk_pl = |entry: &'static str| -> wgpu::ComputePipeline {
            pipelines.compute_pipeline(&pipeline_layout, &shader, entry)
        };
        let mk_pl_absorb = |entry: &'static str| -> wgpu::ComputePipeline {
            pipelines.compute_pipeline(&pipeline_layout_absorb, &shader, entry)
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
        let pstd_pipelines = WgpuPstdPipelines {
            zero_fields: pipeline_zero_fields,
            zero_kspace: pipeline_zero_kspace,
            fft: pipeline_fft,
            kspace_shift: pipeline_kspace_shift,
            vel_update: pipeline_vel_update,
            dens_update: pipeline_dens_update,
            snapshot_rho0_plus_rho: pipeline_snapshot_rho0_plus_rho,
            pres_density: pipeline_pres_density,
            record: pipeline_record,
            inject_src: pipeline_inject_src,
            inject_vel_x: pipeline_inject_vel_x,
            apply_source_kappa: pipeline_apply_source_kappa,
            add_kspace_to_field_ux: pipeline_add_kspace_to_field_ux,
            copy_field_to_k: pipeline_copy_field_to_k,
            absorb_mul_nabla: pipeline_absorb_mul_nabla,
            absorb_copy_to_scratch: pipeline_absorb_copy_to_scratch,
            absorb_accum_div_u: pipeline_absorb_accum_div_u,
            absorb_prep_l1_kspace: pipeline_absorb_prep_l1_kspace,
            absorb_prep_l2_kspace: pipeline_absorb_prep_l2_kspace,
            absorb_pressure_correction: pipeline_absorb_pressure_correction,
            absorb_save_kspace: pipeline_absorb_save_kspace,
            restore_and_shift: pipeline_restore_and_shift,
        };

        // ── Build permanent bind groups ───────────────────────────────────────
        let bind_groups = WgpuPstdBindGroupFactory::new(device);
        let bg_fields = build_bg_fields(
            &bind_groups,
            &bgl_fields,
            &FieldBuffers {
                p: &field_buffers.p,
                ux: &field_buffers.ux,
                uy: &field_buffers.uy,
                uz: &field_buffers.uz,
                rhox: &field_buffers.rhox,
                rhoy: &field_buffers.rhoy,
                rhoz: &field_buffers.rhoz,
                source_kappa: &medium_buffers.source_kappa,
            },
        );
        let bg_kspace = build_bg_kspace(
            &bind_groups,
            &bgl_kspace,
            &KspaceBuffers {
                kspace_re: &kspace_buffers.re,
                kspace_im: &kspace_buffers.im,
                kappa: &medium_buffers.kappa,
                rho0_inv: &medium_buffers.rho0_inv,
                c0_sq: &medium_buffers.c0_sq,
                rho0: &medium_buffers.rho0,
                bon_a: &medium_buffers.bon_a,
                alpha_decay: &medium_buffers.alpha_decay,
            },
        );
        let bg_absorb = build_bg_absorb(
            &bind_groups,
            &bgl_absorb,
            &AbsorbBuffers {
                nabla1: &absorption_buffers.nabla1,
                nabla2: &absorption_buffers.nabla2,
                tau: &absorption_buffers.tau,
                eta: &absorption_buffers.eta,
                scratch_kre: &absorption_buffers.scratch_kre,
                scratch_kim: &absorption_buffers.scratch_kim,
                scratch_l1: &absorption_buffers.scratch_l1,
                scratch_l2: &absorption_buffers.scratch_l2,
            },
        );
        let permanent_bind_groups = WgpuPstdPermanentBindGroups {
            fields: bg_fields,
            kspace: bg_kspace,
            absorb: bg_absorb,
        };
        let layouts = WgpuPstdLayouts { sensor: bgl_sensor };
        let state = WgpuPstdState {
            context,
            field_buffers,
            kspace_buffers,
            medium_buffers,
            absorption_buffers,
            pml_shift_buffers,
            pipelines: pstd_pipelines,
            permanent_bind_groups,
            layouts,
            run_cache: WgpuPstdRunCache::default(),
            scratch_c0_sq: vec![0.0f32; total],
            scratch_rho0_inv: vec![0.0f32; total],
            scratch_rho0_flat: vec![0.0f32; total],
            scratch_source_kappa_ones: vec![1.0f32; total],
            scratch_source_data: Vec::new(),
            scratch_vel_x_data: Vec::new(),
        };

        Ok(state)
    }
}

impl<P> GpuPstdSolver<P>
where
    P: PstdStateBuilder,
{
    /// Create a new GPU PSTD solver.
    ///
    /// See `mod.rs` for argument documentation and bind-group layout.
    pub fn new(
        context: P::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, String> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let SolverParams {
            dt,
            nt,
            nonlinear,
            absorbing,
            ..
        } = solver;
        let state = P::build_state(context, grid, medium, solver, pml, absorption)?;

        Ok(Self {
            nx,
            ny,
            nz,
            nt,
            dt,
            state,
            _provider: std::marker::PhantomData,
            nonlinear,
            absorbing,
        })
    }
}
