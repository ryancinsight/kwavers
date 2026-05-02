//! Per-step compute-pass encoding: velocity, source, density, pressure, record.
//!
//! Each `encode_*` method appends GPU dispatches to an already-open `ComputePass`.
//! The caller creates one pass per time step and drops it after all phases complete,
//! keeping all dispatches inside one uninterrupted compute pass (no UAV barriers).

use super::super::{GpuPstdSolver, PstdParams};

/// Scalar constants and workgroup sizes extracted from `GpuPstdSolver` once per
/// batch and threaded to all per-step encoders.
pub(super) struct StepCtx {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dt: f32,
    pub n_sensors: u32,
    pub nt: u32,
    pub nonlinear: u32,
    pub absorbing: u32,
    pub n_src: usize,
    pub n_vel_x: usize,
    pub elem_wg: u32,
}

impl StepCtx {
    /// Build a `PstdParams` with `n_fft/n_batches/log2n = 0` (physics dispatch).
    pub(super) fn params(&self, step: u32, axis: u32) -> PstdParams {
        PstdParams {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            axis,
            n_fft: 0,
            n_batches: 0,
            log2n: 0,
            inverse: 0,
            step,
            dt: self.dt,
            n_sensors: self.n_sensors,
            nt: self.nt,
            nonlinear: self.nonlinear,
            absorbing: self.absorbing,
        }
    }

    #[inline]
    pub(super) fn ceil_div(n: usize, d: usize) -> u32 {
        ((n + d - 1) / d) as u32
    }
}

impl GpuPstdSolver {
    /// Encode the velocity-update phase.
    ///
    /// ## Algorithm
    ///
    /// 1. Copy `field_p` → `kspace_re/im` and compute one forward FFT(p).
    /// 2. Save FFT(p) to `absorb_scratch_kre/kim` (reused as a cache register).
    /// 3. For each axis `ax ∈ {0,1,2}`:
    ///    - Restore FFT(p) from cache (skipped for `ax=0` since kspace already holds it).
    ///    - Apply k-space staggered shift in-place on `kspace_re/im`.
    ///    - IFFT → write gradient into `kspace_re`.
    ///    - `vel_update`: accumulate into `ux/uy/uz`.
    ///
    /// Net saving over the naive approach: −2 full 3-D forward FFTs and −8 GPU dispatches
    /// per time step (shared FFT(p) across all three axes).
    pub(super) fn encode_velocity_update(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        let ew = ctx.elem_wg;
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_copy_field_to_k,
            bg,
            ew,
            "cp_p",
        );
        self.fft_3d(cpass, bg, step, ctx.n_sensors);
        self.dispatch_absorb(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_absorb_save_kspace,
            bg,
            ew,
            "save_fftp",
        );

        for ax in 0u32..3u32 {
            if ax > 0 {
                self.dispatch_absorb(
                    cpass,
                    &ctx.params(step, ax),
                    &self.pipeline_absorb_restore_kspace,
                    bg,
                    ew,
                    "rest_fftp",
                );
            }
            self.dispatch(
                cpass,
                &ctx.params(step, ax),
                &self.pipeline_kspace_shift,
                bg,
                ew,
                "kshift_v",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, ax),
                &self.pipeline_vel_update,
                bg,
                ew,
                "vel_upd",
            );
        }
    }

    /// Encode pressure-source and velocity-source injection.
    ///
    /// Pressure injection: scatter `source_signals[step]` to `field_p[source_indices]`.
    ///
    /// Velocity-source injection (when `ctx.n_vel_x > 0`):
    /// zeros `kspace_re/im`, scatters the velocity signal to kspace, applies
    /// `source_kappa` k-correction via FFT/IFFT, then adds the corrected field
    /// back into `ux`.  This avoids the PCIe-side `clear_buffer` which would end
    /// the compute pass; the GPU `pipeline_zero_kspace` shader clears in-pass.
    pub(super) fn encode_source_injection(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        bg_vel: &wgpu::BindGroup,
        step: u32,
    ) {
        if ctx.n_src > 0 {
            let src_params = PstdParams {
                axis: ctx.n_src as u32,
                ..ctx.params(step, 0)
            };
            self.dispatch(
                cpass,
                &src_params,
                &self.pipeline_inject_src,
                bg,
                StepCtx::ceil_div(ctx.n_src, 256),
                "inject",
            );
        }

        if ctx.n_vel_x > 0 {
            let ew = ctx.elem_wg;
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_zero_kspace,
                bg,
                ew,
                "zero_ksp",
            );
            let vel_params = PstdParams {
                axis: ctx.n_vel_x as u32,
                ..ctx.params(step, 0)
            };
            self.dispatch(
                cpass,
                &vel_params,
                &self.pipeline_inject_vel_x,
                bg_vel,
                StepCtx::ceil_div(ctx.n_vel_x, 256),
                "inject_vx",
            );
            self.fft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_apply_source_kappa,
                bg,
                ew,
                "src_kappa",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_add_kspace_to_field_ux,
                bg,
                ew,
                "add_ux",
            );
        }
    }

    /// Encode the nonlinear mass-conservation density snapshot (no-op when linear).
    ///
    /// Pre-computes `2*(rhox+rhoy+rhoz)+rho0` into `field_p` before the density
    /// loop.  `density_update` reads this as the mass-conservation coefficient,
    /// matching k-Wave C++ `computeDensityNonlinear` (Treeby & Cox 2010, Eq. A.3).
    ///
    /// `field_p` is safe to use as scratch here: the previous step's pressure has
    /// already been recorded by sensors; it is overwritten by `pressure_from_density`
    /// after the density loop.
    pub(super) fn encode_nonlinear_snapshot(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        if ctx.nonlinear != 0 {
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_snapshot_rho0_plus_rho,
                bg,
                ctx.elem_wg,
                "snap_rho",
            );
        }
    }

    /// Encode the density-update phase and fractional-Laplacian absorption operators.
    ///
    /// ## Density loop (axes 0–2)
    ///
    /// For each axis `ax`:
    /// 1. `copy_field_to_k(field_sel = ax+1)` — copy `ux/uy/uz` to kspace.
    /// 2. Forward FFT, k-space shift (density staggered: `kshift_d`), inverse FFT.
    /// 3. `dens_update(ax)` — accumulate into `rhox/rhoy/rhoz`.
    /// 4. When absorbing: `absorb_accum_div_u(ax)` — accumulate `div_u` into `scratch_kre`.
    ///    `ax=0` initialises `scratch_kre` (no separate clear needed).
    ///
    /// ## Fractional-Laplacian absorption (Treeby & Cox 2010 Eqs. 19–21)
    ///
    /// After the density loop `scratch_kre` holds `div_u_total`.
    ///
    /// ```text
    /// L1 = IFFT(nabla1 · FFT(ρ₀ · div_u_total))   [τ term]
    /// L2 = IFFT(nabla2 · FFT(ρ_total))              [η term]
    /// p  += c₀² · (τ · L1 − η · L2)
    /// ```
    pub(super) fn encode_density_update(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        let ew = ctx.elem_wg;
        for ax in 0u32..3u32 {
            self.dispatch(
                cpass,
                &ctx.params(step, ax + 1),
                &self.pipeline_copy_field_to_k,
                bg,
                ew,
                "cp_u",
            );
            self.fft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, ax + 3),
                &self.pipeline_kspace_shift,
                bg,
                ew,
                "kshift_d",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, ax),
                &self.pipeline_dens_update,
                bg,
                ew,
                "dens_upd",
            );
            if ctx.absorbing != 0 {
                self.dispatch_absorb(
                    cpass,
                    &ctx.params(step, ax),
                    &self.pipeline_absorb_accum_div_u,
                    bg,
                    ew,
                    "abs_accum",
                );
            }
        }

        if ctx.absorbing != 0 {
            // L1: IFFT(nabla1 · FFT(ρ₀ · div_u_total))
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_absorb_prep_l1_kspace,
                bg,
                ew,
                "abs_prep_l1",
            );
            self.fft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_absorb_mul_nabla,
                bg,
                ew,
                "abs_n1",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_absorb_copy_to_scratch,
                bg,
                ew,
                "abs_cp_l1",
            );

            // L2: IFFT(nabla2 · FFT(ρ_total))
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 1),
                &self.pipeline_absorb_prep_l2_kspace,
                bg,
                ew,
                "abs_prep_l2",
            );
            self.fft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 1),
                &self.pipeline_absorb_mul_nabla,
                bg,
                ew,
                "abs_n2",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 1),
                &self.pipeline_absorb_copy_to_scratch,
                bg,
                ew,
                "abs_cp_l2",
            );
        }
    }

    /// Encode pressure-from-density, absorption correction, and sensor recording.
    ///
    /// `pres_density` accumulates `rhox + rhoy + rhoz` into `field_p`.  When
    /// absorbing, `absorb_pressure_correction` adds `c₀² · (τ·L1 − η·L2)`.
    /// `record` scatters `field_p[sensor_indices]` into the sensor_data buffer.
    pub(super) fn encode_pressure_record(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        let ew = ctx.elem_wg;
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_pres_density,
            bg,
            ew,
            "pres",
        );
        if ctx.absorbing != 0 {
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_absorb_pressure_correction,
                bg,
                ew,
                "abs_pres_corr",
            );
        }
        if ctx.n_sensors > 0 {
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_record,
                bg,
                StepCtx::ceil_div(ctx.n_sensors as usize, 256),
                "rec",
            );
        }
    }
}
