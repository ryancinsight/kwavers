//! Velocity field updates for spectral solver
//!
//! # Momentum Conservation with Split-Field PML
//!
//! ## Background
//! The linearized Euler momentum equation in a heterogeneous medium is:
//! ```text
//!   ∂u/∂t = −(1/ρ₀) ∇p
//! ```
//! On a staggered grid (k-Wave convention), pressure lives at cell centers and velocity
//! at cell edges shifted by ½ cell in each respective axis.
//!
//! ## Theorem: Spectral Gradient with Staggered Shift
//! Let p̂ = FFT(p) be the 3-D DFT of the pressure field. The spectral derivative
//! with a positive half-grid-point shift is:
//! ```text
//!   ∂p/∂x |ₓ₊Δₓ/₂ = IFFT( iκₓ · exp(+iκₓ Δx/2) · κ(k) · p̂ )
//! ```
//! where κₓ = 2π n / (Nₓ Δx) is the wavenumber, and κ(k) = sinc(c_ref Δt |k|/2) is
//! the k-space correction factor that improves temporal accuracy to spectral order.
//! The operator `iκₓ · exp(+iκₓ Δx/2)` is stored in `ddx_k_shift_pos`.
//!
//! *Proof:* By the shift theorem of the DFT, shifting by Δx/2 multiplies each mode
//! by exp(+iκₓ Δx/2). Multiplication by iκₓ implements the spectral x-derivative.
//! The k-space correction κ(k) = sinc(c_ref Δt |k|/2) reduces temporal phase
//! error to O(Δt²) for all spatial frequencies simultaneously (Liu 1998, §3).
//!
//! ## Split-Field PML Update Order
//! K-Wave's multiplicative split-field PML (Treeby & Cox 2010, Eq. 17) applies
//! the PML factor **twice** per time step for each velocity component:
//! ```text
//!   u_x^{n+1} = pml_x_sgx · (pml_x_sgx · u_x^n  −  Δt/ρ₀ · ∂p/∂x^{n+½})
//! ```
//! where `pml_x_sgx = exp(−σₓ_sg · Δt/2)` uses the **staggered-grid sigma** evaluated
//! at the half-cell-shifted position. The double application means:
//! - u_x^n is damped by `pml_x_sgx²  = exp(−σₓ_sg · Δt)` per step.
//! - The gradient term is damped by `pml_x_sgx = exp(−σₓ_sg · Δt/2)`.
//!
//! *Why staggered sigma?* The velocity u_x lives at position i+½, so the PML must
//! be evaluated there. Using the collocated sigma (at position i) over-damps velocity
//! by ≈20% at the deepest PML cell, where the staggered sigma is only
//! `(pml_size − 0.5)^4 / pml_size^4 ≈ 0.71 × σ_max` rather than σ_max.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Geophysics 63(6), 2082–2089. (k-space PSTD method)
//! - Berenger (1994). J. Comput. Phys. 114(2), 185–200. (split-field PML)

use crate::core::error::KwaversResult;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::solver::geometry::Geometry;
use ndarray::{s, Zip};

impl PSTDSolver {
    /// Update velocity fields based on pressure gradients (Momentum Conservation).
    ///
    /// Dispatches to [`update_velocity_as`] when `config.geometry == CylindricalAS`,
    /// otherwise uses the standard 3-D spectral path.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[inline]
    pub(crate) fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.geometry == Geometry::CylindricalAS {
            return self.update_velocity_as(dt);
        }
        self.update_velocity_cartesian(dt)
    }

    /// Standard 3-D Cartesian velocity update via spectral FFT gradient operators.
    ///
    /// Uses staggered grid shift operators matching the C++ k-wave binary:
    ///   grad_x(p) = IFFT( ddx_k_shift_pos[x] * kappa[i,j,k] * FFT(p)[i,j,k] )
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_velocity_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // k-Wave split-field PML for velocity (Treeby & Cox 2010, Eq. 17):
        //   u_new = pml * (pml * u_old - dt/rho * grad_p)
        //
        // The PML multiplier `pml = exp(-sigma * dt/2)` is applied TWICE per step:
        //   1. Pre-update: attenuate stored velocity by pml  → pml * u_old
        //   2. Post-update: attenuate the complete result    → pml * (pml*u_old - dt/rho*grad_p)
        //
        // This double application means u_old decays by pml^2 = exp(-sigma*dt) per step,
        // while the injected gradient term decays by only pml = exp(-sigma*dt/2).
        // Using only one application (pre or post) produces incorrect amplitude.
        self.apply_pml_to_velocity()?; // pre: pml * u_old

        // nz_c: half-spectrum z-length (nz/2+1); p_k/grad_k have shape (nx, ny, nz_c).
        let nz_c = self.p_k.dim().2;

        // R2C forward: real pressure (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
        self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);

        // Compute pressure gradients in k-space with staggered grid shifts.
        // k-wave uses ddx_k_shift_pos for pressure→velocity (positive shift).
        // grad_k is reused sequentially for each axis; p_k is read-only throughout.
        // kappa is sliced to (nx, ny, nz_c) — only non-negative kz values appear in
        // the r2c half-spectrum; kappa is symmetric under kz sign flip (|k|² invariant).

        // X-direction
        {
            let ddx = self.ddx_k_shift_pos.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.p_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(i, _j, _k), gk, &p_val, &kap| {
                    *gk = (ddx[i] * p_val) * kap;
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
        Zip::from(&mut self.fields.ux)
            .and(&self.dpx)
            .and(&self.materials.rho0)
            .par_for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Y-direction. For singleton embedding axes, all admissible modes have
        // k_y = 0, so the derivative is exactly zero and the FFT pass is redundant.
        if has_y {
            let ddy = self.ddy_k_shift_pos.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.p_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, j, _k), gk, &p_val, &kap| {
                    *gk = (ddy[j] * p_val) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.uy_k);
            Zip::from(&mut self.fields.uy)
                .and(&self.dpy)
                .and(&self.materials.rho0)
                .par_for_each(|u, &dp, &rho| {
                    *u -= (dt / rho) * dp;
                });
        }

        // Z-direction. A singleton z-axis has only the zero wavenumber, hence
        // dp/dz is identically zero under the periodic spectral derivative.
        // ddz has length nz_c (truncated in construction); k ∈ [0, nz_c) from grad_k.
        if has_z {
            let ddz = self.ddz_k_shift_pos.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.p_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, _j, k), gk, &p_val, &kap| {
                    *gk = (ddz[k] * p_val) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpz, &mut self.uz_k);
            Zip::from(&mut self.fields.uz)
                .and(&self.dpz)
                .and(&self.materials.rho0)
                .par_for_each(|u, &dp, &rho| {
                    *u -= (dt / rho) * dp;
                });
        }

        // NOTE: Velocity source injection is NOT performed here.
        // It happens in step_forward() after update_velocity() returns,
        // matching the C++ k-wave binary time loop order (Step 2: addVelocitySource).

        self.apply_pml_to_velocity()?; // post: pml * (pml*u_old - dt/rho*grad_p)

        Ok(())
    }

    /// Axisymmetric WSWA-FFT velocity update.
    ///
    /// Updates axial velocity `ux` and radial velocity `uz` (= `u_r` in cylindrical coordinates).
    /// `uy` is not updated (ny = 1 in axisymmetric mode).
    ///
    /// # Equations
    /// ```text
    /// ux -= dt / rho0 * dp/dx          (axial momentum)
    /// uz -= dt / rho0 * dp/dr          (radial momentum, staggered at r_{m+1/2})
    /// ```
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `AsContext must be Some for CylindricalAS`.
    ///
    pub(crate) fn update_velocity_as(&mut self, dt: f64) -> KwaversResult<()> {
        self.apply_pml_to_velocity()?; // pre-step PML

        // Take AsContext out of the Option so we hold an owned value while
        // also mutably borrowing self.fields / self.materials (disjoint fields).
        // No heap allocation: take/replace are pointer moves only.
        let mut ctx = self
            .as_ctx
            .take()
            .expect("AsContext must be Some for CylindricalAS");

        ctx.compute_vel_grads(self.fields.p.slice(s![.., 0, ..]));

        Zip::from(self.fields.ux.slice_mut(s![.., 0, ..]))
            .and(self.materials.rho0.slice(s![.., 0, ..]))
            .and(&ctx.dpdx)
            .par_for_each(|u, &rho, &dp| {
                *u -= (dt / rho) * dp;
            });

        Zip::from(self.fields.uz.slice_mut(s![.., 0, ..]))
            .and(self.materials.rho0.slice(s![.., 0, ..]))
            .and(&ctx.dpdr)
            .par_for_each(|u, &rho, &dp| {
                *u -= (dt / rho) * dp;
            });

        self.as_ctx = Some(ctx);

        self.apply_pml_to_velocity()?; // post-step PML
        Ok(())
    }

    /// Apply split-field directional PML damping to velocity components.
    ///
    /// Each velocity component is damped only by its corresponding directional sigma,
    /// matching k-Wave's formulation: `ux *= pml_x`, `uy *= pml_y`, `uz *= pml_z`.
    /// Apply split-field directional PML to velocity components using staggered-grid sigma.
    ///
    /// Velocity fields are staggered at half-cell positions relative to pressure/density.
    /// K-Wave therefore uses `pml_x_sgx` / `pml_y_sgy` / `pml_z_sgz` (computed at i+0.5)
    /// rather than the collocated `pml_x` / `pml_y` / `pml_z` used for density.
    ///
    /// The staggered sigma is smaller at PML boundary cells (~70% of σ_max at deepest cell),
    /// so using non-staggered sigma for velocity over-damps it by ≈ 20%.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_pml_to_velocity(&mut self) -> KwaversResult<()> {
        let Some(mut boundary) = self.boundary.take() else {
            return Ok(());
        };

        let result = (|| -> KwaversResult<()> {
            if self.dirichlet_pml_bypass_x.is_empty() {
                boundary.apply_velocity_pml_directional(
                    self.fields.ux.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    0,
                )?;
                boundary.apply_velocity_pml_directional(
                    self.fields.uy.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    1,
                )?;
                boundary.apply_velocity_pml_directional(
                    self.fields.uz.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    2,
                )?;
            } else {
                self.resize_pml_bypass_scratch();
                let rows = self.dirichlet_pml_bypass_x.as_slice();
                let grid = self.grid.as_ref();
                let step = self.time_step_index;

                Self::apply_x_plane_pml_bypass(
                    &mut self.fields.ux,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_velocity_pml_directional(field, grid, step, 0),
                )?;
                Self::apply_x_plane_pml_bypass(
                    &mut self.fields.uy,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_velocity_pml_directional(field, grid, step, 1),
                )?;
                Self::apply_x_plane_pml_bypass(
                    &mut self.fields.uz,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_velocity_pml_directional(field, grid, step, 2),
                )?;
            }
            Ok(())
        })();

        self.boundary = Some(boundary);
        result
    }
}
