//! FDTD velocity field update — extracted from solver.rs for SRP compliance.
//!
//! Contains velocity-related update methods as an `impl FdtdSolver` block:
//! - `update_velocity` (dispatch to staggered or collocated)
//! - `update_velocity_staggered`

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

use moirai_parallel::{enumerate_mut_with, Adaptive};

use kwavers_math::numerics::operators::Axis;

use super::solver::FdtdSolver;

fn update_velocity_from_gradient(
    velocity: &mut Array3<f64>,
    gradient: &Array3<f64>,
    density: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        gradient.shape(),
        "invariant: FDTD pressure-gradient shape matches velocity field"
    );
    assert_eq!(
        velocity.shape(),
        density.shape(),
        "invariant: FDTD density shape matches velocity field"
    );

    if let (Some(velocity_values), Some(gradient_values), Some(density_values)) = (
        velocity.as_slice_mut(),
        gradient.as_slice(),
        density.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(velocity_values, |idx, velocity_value| {
            let rho = density_values[idx];
            if rho > 1e-9 {
                *velocity_value -= dt / rho * gradient_values[idx];
            }
        });
    } else {
        for ((velocity_value, &gradient_value), &rho) in
            velocity.iter_mut().zip(gradient.iter()).zip(density.iter())
        {
            if rho > 1e-9 {
                *velocity_value -= dt / rho * gradient_value;
            }
        }
    }
}

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
    /// - Returns [`crate::KwaversError::InternalError`] if `kspace_ops` is unexpectedly `None`
    ///   despite the caller having confirmed its presence.
    ///
    fn update_velocity_kspace(&mut self, dt: f64) -> KwaversResult<()> {
        {
            let kops = self.kspace_ops.as_mut().ok_or_else(|| {
                KwaversError::InternalError("kspace_ops unexpectedly None".into())
            })?;
            kops.compute_grad_pos(&self.fields.p);
        }

        let shape = self.fields.p.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        {
            let kops = self.kspace_ops.as_ref().ok_or_else(|| {
                KwaversError::InternalError("kspace_ops unexpectedly None".into())
            })?;

            update_velocity_from_gradient(
                &mut self.fields.ux,
                &kops.grad_x,
                &self.materials.rho0,
                dt,
            );
            {
                let ux = self.fields.ux.view_mut();
                let mut boundary = ux
                    .slice_mut(&[(nx - 1, nx, 1), (0, ny, 1), (0, nz, 1)])
                    .expect("invariant: last-row ux boundary within FDTD grid");
                boundary.fill(0.0);
            }

            update_velocity_from_gradient(
                &mut self.fields.uy,
                &kops.grad_y,
                &self.materials.rho0,
                dt,
            );
            {
                let uy = self.fields.uy.view_mut();
                let mut boundary = uy
                    .slice_mut(&[(0, nx, 1), (ny - 1, ny, 1), (0, nz, 1)])
                    .expect("invariant: last-col uy boundary within FDTD grid");
                boundary.fill(0.0);
            }

            update_velocity_from_gradient(
                &mut self.fields.uz,
                &kops.grad_z,
                &self.materials.rho0,
                dt,
            );
            {
                let uz = self.fields.uz.view_mut();
                let mut boundary = uz
                    .slice_mut(&[(0, nx, 1), (0, ny, 1), (nz - 1, nz, 1)])
                    .expect("invariant: last-layer uz boundary within FDTD grid");
                boundary.fill(0.0);
            }
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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        if self.kspace_ops.is_some() {
            return self.update_velocity_kspace(dt);
        }
        if self.config.staggered_grid {
            return self.update_velocity_staggered(dt);
        }

        // Summation by parts, not the general central difference: the leapfrog
        // conserves energy only when the boundary term vanishes, and a
        // one-sided row breaks that (KW-SOL-081). SBP conserves in the weighted
        // norm the operator carries, which additionally admits a rigid wall
        // (KW-SOL-086) -- the plain skew-symmetric closure it replaced could
        // only give a pressure-release one.
        self.conservative_operator
            .apply_into(Axis::X, self.fields.p.view(), &mut self.dvx_scratch);
        self.conservative_operator
            .apply_into(Axis::Y, self.fields.p.view(), &mut self.dvy_scratch);
        self.conservative_operator.apply_into(
            Axis::Z,
            self.fields.p.view(),
            &mut self.divergence_scratch,
        );

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_p_gradient_correction(&mut self.dvx_scratch, 0);
            cpml.update_and_apply_p_gradient_correction(&mut self.dvy_scratch, 1);
            cpml.update_and_apply_p_gradient_correction(&mut self.divergence_scratch, 2);
        }

        update_velocity_from_gradient(
            &mut self.fields.ux,
            &self.dvx_scratch,
            &self.materials.rho0,
            dt,
        );
        update_velocity_from_gradient(
            &mut self.fields.uy,
            &self.dvy_scratch,
            &self.materials.rho0,
            dt,
        );
        update_velocity_from_gradient(
            &mut self.fields.uz,
            &self.divergence_scratch,
            &self.materials.rho0,
            dt,
        );

        apply_rigid_wall(
            &mut self.fields.ux,
            &mut self.fields.uy,
            &mut self.fields.uz,
        );

        Ok(())
    }

    /// Staggered-grid velocity update.
    ///
    /// ## Algorithm (Yee leapfrog, Virieux 1986)
    ///
    /// ```text
    /// u^{n+½}[i+½] = u^{n-½}[i+½] − (Δt / ρ_avg) · (p^n[i+1] − p^n`i`) / Δx
    /// ```
    ///
    /// Density is linearly averaged at the half-cell interface: `ρ_avg = (ρ`i` + ρ[i+1]) / 2`.
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
        let shape = self.fields.p.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        // Gradient of pressure onto the faces, at the configured order. The
        // operator is full-shape and reflects taps at the walls, so there is no
        // half-shape scratch and no far-face zeroing: reflection makes the far
        // face vanish on its own, and the divergence the pressure update applies
        // is this operator's negative transpose by construction (ADR 106).
        self.leapfrog_operator
            .gradient_into(Axis::X, self.fields.p.view(), &mut self.dvx_scratch);
        self.leapfrog_operator
            .gradient_into(Axis::Y, self.fields.p.view(), &mut self.dvy_scratch);
        self.leapfrog_operator.gradient_into(
            Axis::Z,
            self.fields.p.view(),
            &mut self.divergence_scratch,
        );

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_p_gradient_correction(&mut self.dvx_scratch, 0);
            cpml.update_and_apply_p_gradient_correction(&mut self.dvy_scratch, 1);
            cpml.update_and_apply_p_gradient_correction(&mut self.divergence_scratch, 2);
        }

        // Density at face `i+½` is the average of the two cells it separates.
        // The last face has no cell beyond it, so the nearest cell's density is
        // used — density is a material property and is not zero-extended; doing
        // so would put a vacuum at the wall.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    let rho = 0.5
                        * (self.materials.rho0[index]
                            + self.materials.rho0[[(i + 1).min(nx - 1), j, k]]);
                    if rho > 1e-9 {
                        self.fields.ux[index] -= dt / rho * self.dvx_scratch[index];
                    }
                    let rho = 0.5
                        * (self.materials.rho0[index]
                            + self.materials.rho0[[i, (j + 1).min(ny - 1), k]]);
                    if rho > 1e-9 {
                        self.fields.uy[index] -= dt / rho * self.dvy_scratch[index];
                    }
                    let rho = 0.5
                        * (self.materials.rho0[index]
                            + self.materials.rho0[[i, j, (k + 1).min(nz - 1)]]);
                    if rho > 1e-9 {
                        self.fields.uz[index] -= dt / rho * self.divergence_scratch[index];
                    }
                }
            }
        }

        Ok(())
    }
}

/// Hold the wall-normal velocity at zero on every outer face.
///
/// This is the rigid wall, and on the collocated path it is what makes the
/// scheme conserve energy rather than merely approximately conserve it. The
/// summation-by-parts operator gives
///
/// ```text
///   d/dt ‖E‖_H = −( p_{N−1}u_{N−1} − p₀u₀ )
/// ```
///
/// per axis, so the energy is constant exactly when the wall-normal velocity
/// vanishes at the two end points. The condition is not decoration on top of a
/// conservative scheme — it is the half of it that lives at the boundary.
///
/// Only the normal component is held: a rigid wall is a slip wall, and the
/// tangential components carry no flux through it, so zeroing them would impose
/// a no-slip condition the inviscid equations do not have.
///
/// A PML, where configured, operates inside the domain and terminates against
/// this wall rather than replacing it.
fn apply_rigid_wall(ux: &mut Array3<f64>, uy: &mut Array3<f64>, uz: &mut Array3<f64>) {
    let [nx, ny, nz] = ux.shape();

    for j in 0..ny {
        for k in 0..nz {
            ux[[0, j, k]] = 0.0;
            ux[[nx - 1, j, k]] = 0.0;
        }
    }
    for i in 0..nx {
        for k in 0..nz {
            uy[[i, 0, k]] = 0.0;
            uy[[i, ny - 1, k]] = 0.0;
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            uz[[i, j, 0]] = 0.0;
            uz[[i, j, nz - 1]] = 0.0;
        }
    }
}
