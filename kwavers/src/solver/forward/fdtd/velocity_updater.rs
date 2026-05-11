//! FDTD velocity field update — extracted from solver.rs for SRP compliance.
//!
//! Contains velocity-related update methods as an `impl FdtdSolver` block:
//! - `update_velocity` (dispatch to staggered or collocated)
//! - `update_velocity_staggered`

use crate::core::error::KwaversResult;
use ndarray::{s, Zip};

use super::solver::FdtdSolver;

impl FdtdSolver {
    /// K-space corrected velocity update (spectral gradient, dispersion-free).
    ///
    /// Replaces finite-difference staggered gradient with:
    ///   `u -= (dt/rho) * IFFT( ddx_k_shift_pos * kappa * FFT(p) )`
    ///
    /// CPML gradient corrections are NOT applied in this path; spectral gradients
    /// are incompatible with CPML's finite-difference convolutional memory update.
    ///
    /// # Reference
    /// Treeby & Cox (2010), §II.A (k-space corrected FDTD velocity update).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn update_velocity_kspace(&mut self, dt: f64) -> KwaversResult<()> {
        // Compute spectral gradients of pressure (fills kops.grad_x/y/z)
        {
            let kops = self.kspace_ops.as_mut().unwrap();
            kops.compute_grad_pos(&self.fields.p);
        }

        // v -= (dt / rho0) * grad_p
        // Borrows: kspace_ops (kops.grad_*) and fields.u*/materials.rho0 are disjoint fields.
        {
            let (nx, _ny, _nz) = self.fields.p.dim();
            let kops = self.kspace_ops.as_ref().unwrap();

            // ux
            Zip::from(&mut self.fields.ux)
                .and(&kops.grad_x)
                .and(&self.materials.rho0)
                .par_for_each(|u, &dp, &rho| {
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            // Zero edge layer (Dirichlet at domain boundary, matching staggered path)
            self.fields
                .ux
                .slice_mut(ndarray::s![nx - 1, .., ..])
                .fill(0.0);

            // uy
            let (_nx, ny, _nz) = self.fields.p.dim();
            Zip::from(&mut self.fields.uy)
                .and(&kops.grad_y)
                .and(&self.materials.rho0)
                .par_for_each(|u, &dp, &rho| {
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            self.fields
                .uy
                .slice_mut(ndarray::s![.., ny - 1, ..])
                .fill(0.0);

            // uz
            let (_nx, _ny, nz) = self.fields.p.dim();
            Zip::from(&mut self.fields.uz)
                .and(&kops.grad_z)
                .and(&self.materials.rho0)
                .par_for_each(|u, &dp, &rho| {
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            self.fields
                .uz
                .slice_mut(ndarray::s![.., .., nz - 1])
                .fill(0.0);
        }

        Ok(())
    }
}

impl FdtdSolver {
    /// Update velocity field using pressure gradient.
    ///
    /// Dispatch order:
    /// 1. **K-space** (`kspace_correction = Spectral`): spectral FFT gradient,
    ///    dispersion-free, CPML not applied.
    /// 2. **Staggered FD** (`staggered_grid = true`): Yee-cell forward-difference
    ///    gradient, CPML applied if present.
    /// 3. **Collocated FD** (`staggered_grid = false`): central-difference gradient,
    ///    CPML applied if present.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        if self.kspace_ops.is_some() {
            return self.update_velocity_kspace(dt);
        }
        if self.config.staggered_grid {
            return self.update_velocity_staggered(dt);
        }

        // Collocated (non-staggered) update
        let (mut grad_x, mut grad_y, mut grad_z) =
            self.central_operator.gradient(self.fields.p.view())?;

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_p_gradient_correction(&mut grad_x, 0);
            cpml.update_and_apply_p_gradient_correction(&mut grad_y, 1);
            cpml.update_and_apply_p_gradient_correction(&mut grad_z, 2);
        }

        // v^{n+1/2} = v^{n-1/2} - dt/ρ * grad(p)
        let velocity_components = [
            &mut self.fields.ux,
            &mut self.fields.uy,
            &mut self.fields.uz,
        ];
        let gradients = [&grad_x, &grad_y, &grad_z];

        for (vel_component, grad_component) in velocity_components.into_iter().zip(gradients) {
            Zip::from(vel_component)
                .and(grad_component)
                .and(&self.materials.rho0)
                .par_for_each(|v, &grad, &rho| {
                    if rho > 1e-9 {
                        *v -= dt / rho * grad;
                    }
                });
        }

        Ok(())
    }

    /// Staggered-grid velocity update.
    ///
    /// ## Algorithm (Yee leapfrog, Virieux 1986)
    ///
    /// ```text
    /// u^{n+½}[i+½] = u^{n-½}[i+½] − (Δt / ρ_avg) · (p^n[i+1] − p^n[i]) / Δx
    /// ```
    ///
    /// Density is linearly averaged at the half-cell interface: `ρ_avg = (ρ[i] + ρ[i+1]) / 2`.
    /// The last row/column/layer of each velocity component is zeroed (Dirichlet at domain edge).
    ///
    /// ## Memory layout
    ///
    /// Phase 1 fills pre-allocated scratch buffers `dp_dx_scratch`, `dp_dy_scratch`,
    /// `dp_dz_scratch` using a vectorizable Zip slice-pair pattern — zero heap allocation
    /// per step. Phase 2 applies CPML corrections in-place. Phase 3 reads the scratch
    /// gradients and updates the velocity components.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update_velocity_staggered(&mut self, dt: f64) -> KwaversResult<()> {
        let (nx, ny, nz) = self.fields.p.dim();
        // Extract grid spacings as Copy values — avoids re-borrowing self inside closures.
        let dx = self.staggered_operator.dx;
        let dy = self.staggered_operator.dy;
        let dz = self.staggered_operator.dz;

        // Phase 1: fill staggered gradient scratch buffers.
        // Vectorizable Zip slice-pair: dst[i] = (p[i+1] - p[i]) / Δ
        // No heap allocation — scratch was pre-allocated at solver construction.
        if nx > 1 {
            if let Some(ref mut dp_dx) = self.dp_dx_scratch {
                let hi = self.fields.p.slice(s![1.., .., ..]);
                let lo = self.fields.p.slice(s![..nx - 1, .., ..]);
                Zip::from(&mut *dp_dx)
                    .and(hi)
                    .and(lo)
                    .par_for_each(|r, &h, &l| *r = (h - l) / dx);
            }
        }
        if ny > 1 {
            if let Some(ref mut dp_dy) = self.dp_dy_scratch {
                let hi = self.fields.p.slice(s![.., 1.., ..]);
                let lo = self.fields.p.slice(s![.., ..ny - 1, ..]);
                Zip::from(&mut *dp_dy)
                    .and(hi)
                    .and(lo)
                    .par_for_each(|r, &h, &l| *r = (h - l) / dy);
            }
        }
        if nz > 1 {
            if let Some(ref mut dp_dz) = self.dp_dz_scratch {
                let hi = self.fields.p.slice(s![.., .., 1..]);
                let lo = self.fields.p.slice(s![.., .., ..nz - 1]);
                Zip::from(&mut *dp_dz)
                    .and(hi)
                    .and(lo)
                    .par_for_each(|r, &h, &l| *r = (h - l) / dz);
            }
        }

        // Phase 2: CPML gradient corrections (convolutional memory update, Roden & Gedney 2000).
        // Borrows: self.cpml_boundary (mutable) and self.dp_*_scratch (mutable) — disjoint fields.
        if let Some(ref mut cpml) = self.cpml_boundary {
            if let Some(ref mut dp_dx) = self.dp_dx_scratch {
                cpml.update_and_apply_p_gradient_correction(dp_dx, 0);
            }
            if let Some(ref mut dp_dy) = self.dp_dy_scratch {
                cpml.update_and_apply_p_gradient_correction(dp_dy, 1);
            }
            if let Some(ref mut dp_dz) = self.dp_dz_scratch {
                cpml.update_and_apply_p_gradient_correction(dp_dz, 2);
            }
        }

        // Phase 3: apply gradient to velocity components.
        // Density averaged at staggered half-cell interface: ρ_avg = (ρ[i] + ρ[i+1]) / 2.
        // Borrows: self.dp_*_scratch (immutable), self.materials.rho0 (immutable),
        //          self.fields.u* (mutable) — all disjoint fields.
        if let Some(ref dp_dx) = self.dp_dx_scratch {
            let rho_left = self.materials.rho0.slice(s![..nx - 1, .., ..]);
            let rho_right = self.materials.rho0.slice(s![1..nx, .., ..]);
            Zip::from(self.fields.ux.slice_mut(s![..nx - 1, .., ..]))
                .and(rho_left)
                .and(rho_right)
                .and(dp_dx.view())
                .par_for_each(|u, &rl, &rr, &dp| {
                    let rho = 0.5 * (rl + rr);
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            self.fields.ux.slice_mut(s![nx - 1, .., ..]).fill(0.0);
        }

        if let Some(ref dp_dy) = self.dp_dy_scratch {
            let rho_front = self.materials.rho0.slice(s![.., ..ny - 1, ..]);
            let rho_back = self.materials.rho0.slice(s![.., 1..ny, ..]);
            Zip::from(self.fields.uy.slice_mut(s![.., ..ny - 1, ..]))
                .and(rho_front)
                .and(rho_back)
                .and(dp_dy.view())
                .par_for_each(|u, &rf, &rb, &dp| {
                    let rho = 0.5 * (rf + rb);
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            self.fields.uy.slice_mut(s![.., ny - 1, ..]).fill(0.0);
        }

        if let Some(ref dp_dz) = self.dp_dz_scratch {
            let rho_near = self.materials.rho0.slice(s![.., .., ..nz - 1]);
            let rho_far = self.materials.rho0.slice(s![.., .., 1..nz]);
            Zip::from(self.fields.uz.slice_mut(s![.., .., ..nz - 1]))
                .and(rho_near)
                .and(rho_far)
                .and(dp_dz.view())
                .par_for_each(|u, &rn, &rf, &dp| {
                    let rho = 0.5 * (rn + rf);
                    if rho > 1e-9 {
                        *u -= dt / rho * dp;
                    }
                });
            self.fields.uz.slice_mut(s![.., .., nz - 1]).fill(0.0);
        }

        Ok(())
    }
}
