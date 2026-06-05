//! `encode_nonlinear_snapshot` and `encode_density_update`: density phase dispatches.

use super::super::super::GpuPstdSolver;
use super::StepCtx;

impl GpuPstdSolver {
    /// Encode the nonlinear mass-conservation density snapshot (no-op when linear).
    ///
    /// Pre-computes `2*(rhox+rhoy+rhoz)+rho0` into `field_p` before the density
    /// loop.  `density_update` reads this as the mass-conservation coefficient,
    /// matching k-Wave C++ `computeDensityNonlinear` (Treeby & Cox 2010, Eq. A.3).
    ///
    /// `field_p` is safe to use as scratch here: the previous step's pressure has
    /// already been recorded by sensors; it is overwritten by `pressure_from_density`
    /// after the density loop.
    pub(in crate::pstd_gpu) fn encode_nonlinear_snapshot(
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
    pub(in crate::pstd_gpu) fn encode_density_update(
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
}
