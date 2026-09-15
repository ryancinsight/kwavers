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

use super::lanes::{axis_index, for_each_z_lane, LaneAxis};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::geometry::SolverGeometry;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Fft3dInOutExt};
use leto::Array3;
use leto::{Array1, Array2, ArrayView2, ArrayView3, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

#[derive(Clone, Copy)]
enum AsLaneAxis {
    X,
    R,
}

#[inline]
fn dense_indices_2(index: usize, nr: usize) -> (usize, usize) {
    (index / nr, index % nr)
}

#[inline]
fn as_pml_index(axis: AsLaneAxis, i: usize, k: usize) -> usize {
    match axis {
        AsLaneAxis::X => i,
        AsLaneAxis::R => k,
    }
}

fn apply_shifted_kappa(
    grad_k: &mut Array3<Complex64>,
    spectrum: &Array3<Complex64>,
    kappa: &Array3<f64>,
    shift: &Array1<Complex64>,
    axis: LaneAxis,
) {
    assert_eq!(
        grad_k.shape(),
        spectrum.shape(),
        "invariant: PSTD velocity gradient spectrum shape matches pressure spectrum"
    );
    assert_eq!(
        grad_k.shape(),
        kappa.shape(),
        "invariant: PSTD velocity gradient spectrum shape matches kappa"
    );

    let [_nx, ny, nz] = grad_k.shape();
    if let (Some(grad_values), Some(spectrum_values), Some(kappa_values), Some(shift_values)) = (
        grad_k.as_slice_mut(),
        spectrum.as_slice(),
        kappa.as_slice(),
        shift.as_slice(),
    ) {
        let element_bytes = 2 * size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(grad_values, [ny, nz], element_bytes, |start, i, j, grad| {
            let spectrum = &spectrum_values[start..start + nz];
            let kappa = &kappa_values[start..start + nz];
            let lane = grad.iter_mut().zip(spectrum).zip(kappa);
            match axis {
                LaneAxis::X | LaneAxis::Y => {
                    let shift = shift_values[axis_index(axis, i, j, 0)];
                    for ((grad, &spectrum), &kappa) in lane {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
                LaneAxis::Z => {
                    for (((grad, &spectrum), &kappa), &shift) in lane.zip(&shift_values[..nz]) {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
            }
        });
        return;
    }

    let [nx, ny, nz] = grad_k.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                grad_k[[i, j, k]] =
                    (shift[axis_index(axis, i, j, k)] * spectrum[[i, j, k]]) * kappa[[i, j, k]];
            }
        }
    }
}

fn update_velocity_fused(
    velocity: &mut Array3<f64>,
    gradient: &Array3<f64>,
    rho0: ArrayView3<'_, f64>,
    pml: &[f64],
    axis: LaneAxis,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        gradient.shape(),
        "invariant: PSTD velocity shape matches pressure gradient"
    );
    let [rho_nx, rho_ny, rho_nz] = rho0.shape();
    assert_eq!(
        velocity.shape(),
        [rho_nx, rho_ny, rho_nz],
        "invariant: PSTD velocity shape matches rho0"
    );

    let shape = velocity.shape();
    let (_nx, ny, nz) = (shape[0], shape[1], shape[2]);
    if let (Some(velocity_values), Some(gradient_values), Some(rho_values)) = (
        velocity.as_slice_mut(),
        gradient.as_slice(),
        rho0.as_slice(),
    ) {
        for_each_z_lane(
            velocity_values,
            [ny, nz],
            3 * size_of::<f64>(),
            |start, i, j, velocity| {
                let gradient = &gradient_values[start..start + nz];
                let rho = &rho_values[start..start + nz];
                let lane = velocity.iter_mut().zip(gradient).zip(rho);
                match axis {
                    LaneAxis::X | LaneAxis::Y => {
                        let p = pml[axis_index(axis, i, j, 0)];
                        for ((velocity, &gradient), &rho) in lane {
                            *velocity = p * (p * *velocity - (dt / rho) * gradient);
                        }
                    }
                    LaneAxis::Z => {
                        for (((velocity, &gradient), &rho), &p) in lane.zip(&pml[..nz]) {
                            *velocity = p * (p * *velocity - (dt / rho) * gradient);
                        }
                    }
                }
            },
        );
        return;
    }

    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = pml[axis_index(axis, i, j, k)];
                velocity[[i, j, k]] =
                    p * (p * velocity[[i, j, k]] - (dt / rho0[[i, j, k]]) * gradient[[i, j, k]]);
            }
        }
    }
}

fn update_velocity_unfused(
    velocity: &mut Array3<f64>,
    gradient: &Array3<f64>,
    rho0: ArrayView3<'_, f64>,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        gradient.shape(),
        "invariant: PSTD velocity shape matches pressure gradient"
    );
    let [rho_nx, rho_ny, rho_nz] = rho0.shape();
    assert_eq!(
        velocity.shape(),
        [rho_nx, rho_ny, rho_nz],
        "invariant: PSTD velocity shape matches rho0"
    );

    if let (Some(velocity_values), Some(gradient_values), Some(rho_values)) = (
        velocity.as_slice_mut(),
        gradient.as_slice(),
        rho0.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(velocity_values, |index, velocity| {
            *velocity -= (dt / rho_values[index]) * gradient_values[index];
        });
        return;
    }

    let shape = velocity.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                velocity[[i, j, k]] -= (dt / rho0[[i, j, k]]) * gradient[[i, j, k]];
            }
        }
    }
}

fn update_axisymmetric_velocity_fused(
    mut velocity: ArrayViewMut2<'_, f64>,
    gradient: &Array2<f64>,
    rho0: ArrayView2<'_, f64>,
    pml: &[f64],
    axis: AsLaneAxis,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        gradient.shape(),
        "invariant: AS velocity shape matches pressure gradient"
    );
    assert_eq!(
        velocity.shape(),
        rho0.shape(),
        "invariant: AS velocity shape matches rho0"
    );

    let [_nx, nr] = velocity.shape();
    if let (Some(velocity_values), Some(gradient_values), Some(rho_values)) = (
        velocity.as_mut_slice(),
        gradient.as_slice(),
        rho0.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(velocity_values, |index, velocity| {
            let (i, k) = dense_indices_2(index, nr);
            let p = pml[as_pml_index(axis, i, k)];
            *velocity = p * (p * *velocity - (dt / rho_values[index]) * gradient_values[index]);
        });
        return;
    }

    let [nx, nr] = velocity.shape();
    for k in 0..nr {
        for i in 0..nx {
            let p = pml[as_pml_index(axis, i, k)];
            velocity[[i, k]] = p * (p * velocity[[i, k]] - (dt / rho0[[i, k]]) * gradient[[i, k]]);
        }
    }
}

fn update_axisymmetric_velocity_unfused(
    mut velocity: ArrayViewMut2<'_, f64>,
    gradient: &Array2<f64>,
    rho0: ArrayView2<'_, f64>,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        gradient.shape(),
        "invariant: AS velocity shape matches pressure gradient"
    );
    assert_eq!(
        velocity.shape(),
        rho0.shape(),
        "invariant: AS velocity shape matches rho0"
    );

    if let (Some(velocity_values), Some(gradient_values), Some(rho_values)) = (
        velocity.as_mut_slice(),
        gradient.as_slice(),
        rho0.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(velocity_values, |index, velocity| {
            *velocity -= (dt / rho_values[index]) * gradient_values[index];
        });
        return;
    }

    let [nx, nr] = velocity.shape();
    for k in 0..nr {
        for i in 0..nx {
            velocity[[i, k]] -= (dt / rho0[[i, k]]) * gradient[[i, k]];
        }
    }
}

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
    /// update is **fused** into a single dense pass per axis:
    /// ```text
    ///   u_x^{n+1}[i,j,k] = p`i` · (p`i` · u_x^n[i,j,k] − (Δt/ρ₀) · ∂p/∂x)
    /// ```
    /// where `p`i` = pml_vel_x`i` = exp(-σ_x_sg`i`·Δt/2)` is precomputed at
    /// construction (Treeby & Cox 2010, Eq. 17).  This replaces the previous
    /// three-pass sequence (pre-PML → gradient update → post-PML) with one pass,
    /// saving 2 × N element reads/writes per velocity axis per step and eliminating
    /// O(N) transcendental evaluations in favour of O(N) multiplications.
    ///
    /// The fallback path (Dirichlet bypass or non-CPML boundary) preserves the
    /// original `apply_pml_to_velocity()` call structure for correctness.
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
        // mutable field borrows in the dense loops below (disjoint struct fields).
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if use_fused {
            let rho0 = self.materials.rho0.view();
            // ── Fused path: no separate pre/post PML passes ───────────────────────
            // SAFETY of disjoint borrows: `pml_exp` is a separate field from
            // `fields`, `dpx`, `materials` — Rust's field-granular borrow rules allow
            // simultaneous `&self.pml_exp` and `&mut self.fields.ux` / `&self.dpx`.
            //
            // X-direction — PML factor indexed by i (row-major outer index).
            apply_shifted_kappa(
                &mut self.grad_k,
                &self.p_k,
                &self.kappa,
                &self.ddx_k_shift_pos,
                LaneAxis::X,
            );
            self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
            // Fused: u = pml * (pml * u - (dt/rho) * dp)
            // pml_vel_x[i] = exp(-sigma_x_sgx[i] * dt/2)
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError(
                    "pml_exp unexpectedly None in fused velocity path".into(),
                )
            })?;
            let pml_vx = pml_exp.vel_x.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_vel_x must be contiguous".into())
            })?;
            update_velocity_fused(
                &mut self.fields.ux,
                &self.dpx,
                rho0,
                pml_vx,
                LaneAxis::X,
                dt,
            );

            // Y-direction — PML factor indexed by j (middle index).
            if has_y {
                apply_shifted_kappa(
                    &mut self.grad_k,
                    &self.p_k,
                    &self.kappa,
                    &self.ddy_k_shift_pos,
                    LaneAxis::Y,
                );
                // Reuse dpx for y-gradient IFFT (Opt-12): x-axis update has completed;
                // dpx is free to overwrite before y-axis update reads it.
                self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
                let pml_vy = pml_exp.vel_y.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_vel_y must be contiguous".into())
                })?;
                update_velocity_fused(
                    &mut self.fields.uy,
                    &self.dpx,
                    rho0,
                    pml_vy,
                    LaneAxis::Y,
                    dt,
                );
            }

            // Z-direction — PML factor indexed by k (innermost index).
            // ddz has length nz_c (truncated in construction).
            if has_z {
                apply_shifted_kappa(
                    &mut self.grad_k,
                    &self.p_k,
                    &self.kappa,
                    &self.ddz_k_shift_pos,
                    LaneAxis::Z,
                );
                // Reuse dpx for z-gradient IFFT (Opt-12): y-axis update has completed.
                self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
                let pml_vz = pml_exp.vel_z.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_vel_z must be contiguous".into())
                })?;
                update_velocity_fused(
                    &mut self.fields.uz,
                    &self.dpx,
                    rho0,
                    pml_vz,
                    LaneAxis::Z,
                    dt,
                );
            }
        } else {
            // ── Fallback path: explicit pre/post PML passes (Dirichlet bypass or
            //   non-CPML boundary). Semantics identical to the pre-optimisation code.
            self.apply_pml_to_velocity()?; // pre: pml * u_old

            // X-direction
            apply_shifted_kappa(
                &mut self.grad_k,
                &self.p_k,
                &self.kappa,
                &self.ddx_k_shift_pos,
                LaneAxis::X,
            );
            self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
            update_velocity_unfused(
                &mut self.fields.ux,
                &self.dpx,
                self.materials.rho0.view(),
                dt,
            );

            // Y-direction
            if has_y {
                apply_shifted_kappa(
                    &mut self.grad_k,
                    &self.p_k,
                    &self.kappa,
                    &self.ddy_k_shift_pos,
                    LaneAxis::Y,
                );
                // Reuse dpx for y-gradient IFFT (Opt-12): x-axis update has completed.
                self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
                update_velocity_unfused(
                    &mut self.fields.uy,
                    &self.dpx,
                    self.materials.rho0.view(),
                    dt,
                );
            }

            // Z-direction
            if has_z {
                apply_shifted_kappa(
                    &mut self.grad_k,
                    &self.p_k,
                    &self.kappa,
                    &self.ddz_k_shift_pos,
                    LaneAxis::Z,
                );
                // Reuse dpx for z-gradient IFFT (Opt-12): y-axis update has completed.
                self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
                update_velocity_unfused(
                    &mut self.fields.uz,
                    &self.dpx,
                    self.materials.rho0.view(),
                    dt,
                );
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
    /// ux^{n+1}[i,k] = pml_x`i` · (pml_x`i` · ux^n − (dt/ρ₀) · ∂p/∂x)
    /// uz^{n+1}[i,k] = pml_z`K` · (pml_z`K` · uz^n − (dt/ρ₀) · ∂p/∂r)
    /// ```
    /// where `pml_x`i` = exp(-σ_x_sgx`i`·Δt/2)` (staggered-grid sigma, x-axis)
    /// and `pml_z`K` = exp(-σ_z_sgz`K`·Δt/2)` (staggered-grid sigma, r-axis mapped to z).
    ///
    /// **Fused path** (CPML, no Dirichlet bypass): pre-computed `pml_vel_x/z` arrays from
    /// `self.pml_exp` are applied inline — eliminates 2 `apply_pml_to_velocity()` calls
    /// (each scanning all AS cells with per-element `exp()` evaluations).
    ///
    /// **Fallback path**: original pre-PML → update → post-PML call structure preserved.
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    /// - Returns [`crate::KwaversError::InternalError`] if `AsContext` is unexpectedly `None`
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

        let pressure = self.fields.p.view();
        let rho0 = self.materials.rho0.view();
        ctx.compute_vel_grads(
            pressure
                .slice_with::<2>(&s![.., 0, ..])
                .expect("invariant: axisymmetric r=0 plane within pressure bounds"),
        );

        if use_fused {
            // Fused: ux = pml_x[i] · (pml_x[i] · ux − (dt/ρ₀) · ∂p/∂x)
            // In the 2-D slice (nx, nr), dense row-major indices map to (i, k).
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError(
                    "pml_exp unexpectedly None in fused AS velocity path".into(),
                )
            })?;
            let pml_vx = pml_exp
                .vel_x
                .as_slice()
                .ok_or_else(|| KwaversError::InternalError("pml_vel_x contiguous".into()))?;
            let pml_vz = pml_exp
                .vel_z
                .as_slice()
                .ok_or_else(|| KwaversError::InternalError("pml_vel_z contiguous".into()))?;

            let ux = self.fields.ux.view_mut();
            update_axisymmetric_velocity_fused(
                ux.slice_with_mut::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within ux bounds"),
                &ctx.dpdx,
                rho0.slice_with::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within rho0 bounds"),
                pml_vx,
                AsLaneAxis::X,
                dt,
            );

            let uz = self.fields.uz.view_mut();
            update_axisymmetric_velocity_fused(
                uz.slice_with_mut::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within uz bounds"),
                &ctx.dpdr,
                rho0.slice_with::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within rho0 bounds"),
                pml_vz,
                AsLaneAxis::R,
                dt,
            );
        } else {
            let ux = self.fields.ux.view_mut();
            update_axisymmetric_velocity_unfused(
                ux.slice_with_mut::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within ux bounds"),
                &ctx.dpdx,
                rho0.slice_with::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within rho0 bounds"),
                dt,
            );

            let uz = self.fields.uz.view_mut();
            update_axisymmetric_velocity_unfused(
                uz.slice_with_mut::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within uz bounds"),
                &ctx.dpdr,
                rho0.slice_with::<2>(&s![.., 0, ..])
                    .expect("invariant: axisymmetric r=0 plane within rho0 bounds"),
                dt,
            );

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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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

                Self::apply_x_plane_pml_bypass_leto(
                    &mut self.fields.ux,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_velocity_pml_directional(field, grid, step, 0),
                )?;
                Self::apply_x_plane_pml_bypass_leto(
                    &mut self.fields.uy,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_velocity_pml_directional(field, grid, step, 1),
                )?;
                Self::apply_x_plane_pml_bypass_leto(
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

#[cfg(test)]
mod tests {
    use super::{apply_shifted_kappa, axis_index, update_velocity_fused, LaneAxis};
    use kwavers_math::fft::Complex64;
    use leto::{Array1, Array3};

    /// One volume below the lane pass's parallel floor and one above it.
    const SHAPES: [[usize; 3]; 2] = [[5, 3, 7], [37, 29, 31]];
    const AXES: [LaneAxis; 3] = [LaneAxis::X, LaneAxis::Y, LaneAxis::Z];

    fn real_field(shape: [usize; 3], seed: f64) -> Array3<f64> {
        let values = (0..shape.iter().product::<usize>())
            .map(|index| (index as f64).mul_add(0.754_8, seed).sin())
            .collect();
        Array3::from_shape_vec(shape, values).expect("values match the shape")
    }

    fn table(shape: [usize; 3]) -> Vec<f64> {
        (0..*shape.iter().max().expect("a volume has three extents"))
            .map(|n| (n as f64).mul_add(-0.013, 1.0))
            .collect()
    }

    /// The lane pass computes the same expression, in the same order, as the
    /// per-element loop it replaced, so the fields agree to the bit.
    #[test]
    fn velocity_update_is_the_per_element_formula_to_the_bit() {
        let dt = 1.0e-7;
        for shape in SHAPES {
            let [nx, ny, nz] = shape;
            let pml = table(shape);
            let gradient = real_field(shape, 1.0);
            let rho = real_field(shape, 2.0).mapv(|v| v.mul_add(100.0, 1000.0));
            for axis in AXES {
                let mut velocity = real_field(shape, 3.0);
                let mut expected = velocity.clone();
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let p = pml[axis_index(axis, i, j, k)];
                            expected[[i, j, k]] = p
                                * (p * expected[[i, j, k]]
                                    - (dt / rho[[i, j, k]]) * gradient[[i, j, k]]);
                        }
                    }
                }
                update_velocity_fused(&mut velocity, &gradient, rho.view(), &pml, axis, dt);
                let same = velocity
                    .iter()
                    .zip(expected.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "velocity update diverges at {shape:?}");
            }
        }
    }

    #[test]
    fn shifted_kappa_is_the_per_element_formula_to_the_bit() {
        for shape in SHAPES {
            let [nx, ny, nz] = shape;
            let spectrum = real_field(shape, 1.0).mapv(|v| Complex64::new(v, -v * 0.5));
            let kappa = real_field(shape, 2.0);
            let shift_values: Vec<Complex64> = table(shape)
                .into_iter()
                .map(|v| Complex64::new(v, -v))
                .collect();
            let shift = Array1::from_vec([shift_values.len()], shift_values)
                .expect("values match the length");
            for axis in AXES {
                let mut gradient = real_field(shape, 5.0).mapv(|v| Complex64::new(v, v));
                let mut expected = gradient.clone();
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            expected[[i, j, k]] = (shift[axis_index(axis, i, j, k)]
                                * spectrum[[i, j, k]])
                                * kappa[[i, j, k]];
                        }
                    }
                }
                apply_shifted_kappa(&mut gradient, &spectrum, &kappa, &shift, axis);
                let same = gradient.iter().zip(expected.iter()).all(|(a, b)| {
                    a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                });
                assert!(same, "shifted kappa diverges at {shape:?}");
            }
        }
    }
}
