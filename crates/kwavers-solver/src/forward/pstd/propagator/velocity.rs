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

use kwavers_math::fft::Fft3dInOutExt;
use kwavers_core::error::{KwaversError, KwaversResult};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::geometry::SolverGeometry;
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
        if self.config.geometry == SolverGeometry::CylindricalAS {
            return self.update_velocity_as(dt);
        }
        self.update_velocity_cartesian(dt)
    }

    /// Standard 3-D Cartesian velocity update via spectral FFT gradient operators.
    ///
    /// Uses staggered grid shift operators matching the C++ k-wave binary:
    ///   grad_x(p) = IFFT( ddx_k_shift_pos[x] * kappa[i,j,k] * FFT(p)[i,j,k] )
    ///
    /// ## Split-field PML — fused vs. fallback paths
    ///
    /// When `self.pml_exp` is populated (CPML boundary, no Dirichlet bypass), the
    /// update is **fused** into a single Zip pass per axis:
    /// ```text
    ///   u_x^{n+1}[i,j,k] = p[i] · (p[i] · u_x^n[i,j,k] − (Δt/ρ₀) · ∂p/∂x)
    /// ```
    /// where `p[i] = pml_vel_x[i] = exp(-σ_x_sg[i]·Δt/2)` is precomputed at
    /// construction (Treeby & Cox 2010, Eq. 17).  This replaces the previous
    /// three-pass sequence (pre-PML → gradient update → post-PML) with one pass,
    /// saving 2 × N element reads/writes per velocity axis per step and eliminating
    /// O(N) transcendental evaluations in favour of O(N) multiplications.
    ///
    /// The fallback path (Dirichlet bypass or non-CPML boundary) preserves the
    /// original `apply_pml_to_velocity()` call structure for correctness.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_velocity_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // R2C forward: real pressure (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
        // kappa is pre-truncated to (nx,ny,nz_c) — no slice needed at each use (Opt-10).
        // Shared across all three gradient axes — p_k is read-only during kspace multiply.
        self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);

        // Extract precomputed PML factors (or fall back if unavailable / Dirichlet bypass).
        // Taking the slice references here avoids borrow-checker conflicts with the
        // mutable field borrows in the Zip loops below (disjoint struct fields).
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if use_fused {
            // ── Fused path: no separate pre/post PML passes ───────────────────────
            // SAFETY of disjoint borrows: `pml_exp` is a separate field from
            // `fields`, `dpx`, `materials` — Rust's field-granular borrow rules allow
            // simultaneous `&self.pml_exp` and `&mut self.fields.ux` / `&self.dpx`.
            //
            // X-direction — PML factor indexed by i (row-major outer index).
            {
                let ddx = self.ddx_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
                    .par_for_each(|(i, _j, _k), gk, &p_val, &kap| {
                        *gk = (ddx[i] * p_val) * kap;
                    });
            }
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
            // Fused: u = pml * (pml * u - (dt/rho) * dp)
            // pml_vel_x[i] = exp(-sigma_x_sgx[i] * dt/2)
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError("pml_exp unexpectedly None in fused velocity path".into())
            })?;
            let pml_vx = pml_exp.vel_x.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_vel_x must be contiguous".into())
            })?;
            Zip::indexed(self.fields.ux.view_mut())
                .and(&self.dpx)
                .and(&self.materials.rho0)
                .par_for_each(|(i, _j, _k), u, &dp, &rho| {
                    let p = pml_vx[i];
                    *u = p * (p * *u - (dt / rho) * dp);
                });

            // Y-direction — PML factor indexed by j (middle index).
            if has_y {
                let ddy = self.ddy_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
                    .par_for_each(|(_i, j, _k), gk, &p_val, &kap| {
                        *gk = (ddy[j] * p_val) * kap;
                    });
                // Reuse dpx for y-gradient IFFT (Opt-12): x-axis Zip has completed;
                // dpx is free to overwrite before y-axis Zip reads it.
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
                let pml_vy = pml_exp.vel_y.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_vel_y must be contiguous".into())
                })?;
                Zip::indexed(self.fields.uy.view_mut())
                    .and(&self.dpx)
                    .and(&self.materials.rho0)
                    .par_for_each(|(_i, j, _k), u, &dp, &rho| {
                        let p = pml_vy[j];
                        *u = p * (p * *u - (dt / rho) * dp);
                    });
            }

            // Z-direction — PML factor indexed by k (innermost index).
            // ddz has length nz_c (truncated in construction).
            if has_z {
                let ddz = self.ddz_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
                    .par_for_each(|(_i, _j, k), gk, &p_val, &kap| {
                        *gk = (ddz[k] * p_val) * kap;
                    });
                // Reuse dpx for z-gradient IFFT (Opt-12): y-axis Zip has completed.
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
                let pml_vz = pml_exp.vel_z.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_vel_z must be contiguous".into())
                })?;
                Zip::indexed(self.fields.uz.view_mut())
                    .and(&self.dpx)
                    .and(&self.materials.rho0)
                    .par_for_each(|(_i, _j, k), u, &dp, &rho| {
                        let p = pml_vz[k];
                        *u = p * (p * *u - (dt / rho) * dp);
                    });
            }
        } else {
            // ── Fallback path: explicit pre/post PML passes (Dirichlet bypass or
            //   non-CPML boundary). Semantics identical to the pre-optimisation code.
            self.apply_pml_to_velocity()?; // pre: pml * u_old

            // X-direction
            {
                let ddx = self.ddx_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
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

            // Y-direction
            if has_y {
                let ddy = self.ddy_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
                    .par_for_each(|(_i, j, _k), gk, &p_val, &kap| {
                        *gk = (ddy[j] * p_val) * kap;
                    });
                // Reuse dpx for y-gradient IFFT (Opt-12): x-axis Zip has completed.
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
                Zip::from(&mut self.fields.uy)
                    .and(&self.dpx)
                    .and(&self.materials.rho0)
                    .par_for_each(|u, &dp, &rho| {
                        *u -= (dt / rho) * dp;
                    });
            }

            // Z-direction
            if has_z {
                let ddz = self.ddz_k_shift_pos.view();
                Zip::indexed(self.grad_k.view_mut())
                    .and(self.p_k.view())
                    .and(self.kappa.view())
                    .par_for_each(|(_i, _j, k), gk, &p_val, &kap| {
                        *gk = (ddz[k] * p_val) * kap;
                    });
                // Reuse dpx for z-gradient IFFT (Opt-12): y-axis Zip has completed.
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
                Zip::from(&mut self.fields.uz)
                    .and(&self.dpx)
                    .and(&self.materials.rho0)
                    .par_for_each(|u, &dp, &rho| {
                        *u -= (dt / rho) * dp;
                    });
            }

            self.apply_pml_to_velocity()?; // post: pml * (pml*u_old - dt/rho*grad_p)
        }

        // NOTE: Velocity source injection is NOT performed here.
        // It happens in step_forward() after update_velocity() returns,
        // matching the C++ k-wave binary time loop order (Step 2: addVelocitySource).

        Ok(())
    }

    /// Axisymmetric WSWA-FFT velocity update.
    ///
    /// Updates axial velocity `ux` and radial velocity `uz` (= `u_r` in cylindrical coordinates).
    /// `uy` is not updated (ny = 1 in axisymmetric mode).
    ///
    /// # Equations (split-field PML, Treeby & Cox 2010 Eq. 17)
    /// ```text
    /// ux^{n+1}[i,k] = pml_x[i] · (pml_x[i] · ux^n − (dt/ρ₀) · ∂p/∂x)
    /// uz^{n+1}[i,k] = pml_z[k] · (pml_z[k] · uz^n − (dt/ρ₀) · ∂p/∂r)
    /// ```
    /// where `pml_x[i] = exp(-σ_x_sgx[i]·Δt/2)` (staggered-grid sigma, x-axis)
    /// and `pml_z[k] = exp(-σ_z_sgz[k]·Δt/2)` (staggered-grid sigma, r-axis mapped to z).
    ///
    /// **Fused path** (CPML, no Dirichlet bypass): pre-computed `pml_vel_x/z` arrays from
    /// `self.pml_exp` are applied inline — eliminates 2 `apply_pml_to_velocity()` calls
    /// (each scanning all AS cells with per-element `exp()` evaluations).
    ///
    /// **Fallback path**: original pre-PML → update → post-PML call structure preserved.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    /// - Returns [`KwaversError::InternalError`] if `AsContext` is unexpectedly `None`
    ///   for `CylindricalAS` geometry.
    ///
    pub(crate) fn update_velocity_as(&mut self, dt: f64) -> KwaversResult<()> {
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if !use_fused {
            self.apply_pml_to_velocity()?; // pre-step PML (fallback only)
        }

        // Take AsContext out of the Option so we hold an owned value while
        // also mutably borrowing self.fields / self.materials (disjoint fields).
        // No heap allocation: take/replace are pointer moves only.
        let mut ctx = self.as_ctx.take().ok_or_else(|| {
            KwaversError::InternalError("AsContext unexpectedly None for CylindricalAS".into())
        })?;

        ctx.compute_vel_grads(self.fields.p.slice(s![.., 0, ..]));

        if use_fused {
            // Fused: ux = pml_x[i] · (pml_x[i] · ux − (dt/ρ₀) · ∂p/∂x)
            // In the 2-D slice (nx, nr), indexed returns (i, k).
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError("pml_exp unexpectedly None in fused AS velocity path".into())
            })?;
            let pml_vx = pml_exp.vel_x.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_vel_x contiguous".into())
            })?;
            let pml_vz = pml_exp.vel_z.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_vel_z contiguous".into())
            })?;

            Zip::indexed(self.fields.ux.slice_mut(s![.., 0, ..]))
                .and(self.materials.rho0.slice(s![.., 0, ..]))
                .and(&ctx.dpdx)
                .par_for_each(|(i, _k), u, &rho, &dp| {
                    let p = pml_vx[i];
                    *u = p * (p * *u - (dt / rho) * dp);
                });

            Zip::indexed(self.fields.uz.slice_mut(s![.., 0, ..]))
                .and(self.materials.rho0.slice(s![.., 0, ..]))
                .and(&ctx.dpdr)
                .par_for_each(|(_i, k), u, &rho, &dp| {
                    let p = pml_vz[k];
                    *u = p * (p * *u - (dt / rho) * dp);
                });
        } else {
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

            self.apply_pml_to_velocity()?; // post-step PML (fallback only)
        }

        self.as_ctx = Some(ctx);
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
