//! FDTD velocity field update — extracted from solver.rs for SRP compliance.
//!
//! Contains velocity-related update methods as an `impl FdtdSolver` block:
//! - `update_velocity` (dispatch to staggered or collocated)
//! - `update_velocity_staggered`

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use leto::{ArrayView3, ArrayViewMut3};
use moirai_parallel::{enumerate_mut_with, Adaptive};

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

fn compute_forward_gradient(
    mut output: ArrayViewMut3<'_, f64>,
    high: ArrayView3<'_, f64>,
    low: ArrayView3<'_, f64>,
    spacing: f64,
) {
    assert_eq!(
        output.shape(),
        high.shape(),
        "invariant: FDTD forward-gradient upper slice shape matches output"
    );
    assert_eq!(
        output.shape(),
        low.shape(),
        "invariant: FDTD forward-gradient lower slice shape matches output"
    );

    if let (Some(output_values), Some(high_values), Some(low_values)) =
        (output.as_mut_slice(), high.as_slice(), low.as_slice())
    {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |idx, output_value| {
            *output_value = (high_values[idx] - low_values[idx]) / spacing;
        });
    } else {
        leto_ops::zip2_mut_with(
            &mut output,
            &high,
            &low,
            |output_value, high_value, low_value| {
                *output_value = (*high_value - *low_value) / spacing;
            },
        )
        .expect("invariant: FDTD forward-gradient shapes asserted equal above");
    }
}

fn update_staggered_velocity(
    mut velocity: ArrayViewMut3<'_, f64>,
    rho_lower: ArrayView3<'_, f64>,
    rho_upper: ArrayView3<'_, f64>,
    pressure_gradient: ArrayView3<'_, f64>,
    dt: f64,
) {
    assert_eq!(
        velocity.shape(),
        rho_lower.shape(),
        "invariant: FDTD staggered lower-density shape matches velocity field"
    );
    assert_eq!(
        velocity.shape(),
        rho_upper.shape(),
        "invariant: FDTD staggered upper-density shape matches velocity field"
    );
    assert_eq!(
        velocity.shape(),
        pressure_gradient.shape(),
        "invariant: FDTD staggered pressure-gradient shape matches velocity field"
    );

    if let (
        Some(velocity_values),
        Some(rho_lower_values),
        Some(rho_upper_values),
        Some(dp_values),
    ) = (
        velocity.as_mut_slice(),
        rho_lower.as_slice(),
        rho_upper.as_slice(),
        pressure_gradient.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(velocity_values, |idx, velocity_value| {
            let rho = 0.5 * (rho_lower_values[idx] + rho_upper_values[idx]);
            if rho > 1e-9 {
                *velocity_value -= dt / rho * dp_values[idx];
            }
        });
    } else {
        leto_ops::zip3_mut_with(
            &mut velocity,
            &rho_lower,
            &rho_upper,
            &pressure_gradient,
            |velocity_value, rho_lower_value, rho_upper_value, pressure_gradient_value| {
                let rho = 0.5 * (*rho_lower_value + *rho_upper_value);
                if rho > 1e-9 {
                    *velocity_value -= dt / rho * *pressure_gradient_value;
                }
            },
        )
        .expect("invariant: FDTD staggered velocity shapes asserted equal above");
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

        self.central_operator
            .apply_x_into(self.fields.p.view(), &mut self.dvx_scratch)?;
        self.central_operator
            .apply_y_into(self.fields.p.view(), &mut self.dvy_scratch)?;
        self.central_operator
            .apply_z_into(self.fields.p.view(), &mut self.divergence_scratch)?;

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
        let shape = self.fields.p.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        let dx = self.staggered_operator.dx;
        let dy = self.staggered_operator.dy;
        let dz = self.staggered_operator.dz;
        let pressure = self.fields.p.view();

        if nx > 1 {
            if let Some(ref mut dp_dx) = self.dp_dx_scratch {
                let hi = pressure
                    .slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)])
                    .expect("invariant: nx > 1 checked above; hi-x slice valid");
                let lo = pressure
                    .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
                    .expect("invariant: nx > 1 checked above; lo-x slice valid");
                compute_forward_gradient(dp_dx.view_mut(), hi, lo, dx);
            }
        }
        if ny > 1 {
            if let Some(ref mut dp_dy) = self.dp_dy_scratch {
                let hi = pressure
                    .slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)])
                    .expect("invariant: ny > 1 checked above; hi-y slice valid");
                let lo = pressure
                    .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
                    .expect("invariant: ny > 1 checked above; lo-y slice valid");
                compute_forward_gradient(dp_dy.view_mut(), hi, lo, dy);
            }
        }
        if nz > 1 {
            if let Some(ref mut dp_dz) = self.dp_dz_scratch {
                let hi = pressure
                    .slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)])
                    .expect("invariant: nz > 1 checked above; hi-z slice valid");
                let lo = pressure
                    .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
                    .expect("invariant: nz > 1 checked above; lo-z slice valid");
                compute_forward_gradient(dp_dz.view_mut(), hi, lo, dz);
            }
        }

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

        if let Some(ref dp_dx) = self.dp_dx_scratch {
            let rho_left = self
                .materials
                .rho0
                .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
                .expect("invariant: dp_dx_scratch implies nx > 1; rho left valid");
            let rho_right = self
                .materials
                .rho0
                .slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)])
                .expect("invariant: dp_dx_scratch implies nx > 1; rho right valid");
            update_staggered_velocity(
                self.fields
                    .ux
                    .view_mut()
                    .slice_mut(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
                    .expect("invariant: dp_dx_scratch implies nx > 1; ux slice valid"),
                rho_left,
                rho_right,
                dp_dx.view(),
                dt,
            );
            let mut boundary = self
                .fields
                .ux
                .view_mut()
                .slice_mut(&[(nx - 1, nx, 1), (0, ny, 1), (0, nz, 1)])
                .expect("invariant: dp_dx_scratch implies nx >= 1; ux boundary valid");
            boundary.fill(0.0);
        }

        if let Some(ref dp_dy) = self.dp_dy_scratch {
            let rho_front = self
                .materials
                .rho0
                .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
                .expect("invariant: dp_dy_scratch implies ny > 1; rho front valid");
            let rho_back = self
                .materials
                .rho0
                .slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)])
                .expect("invariant: dp_dy_scratch implies ny > 1; rho back valid");
            update_staggered_velocity(
                self.fields
                    .uy
                    .view_mut()
                    .slice_mut(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
                    .expect("invariant: dp_dy_scratch implies ny > 1; uy slice valid"),
                rho_front,
                rho_back,
                dp_dy.view(),
                dt,
            );
            let mut boundary = self
                .fields
                .uy
                .view_mut()
                .slice_mut(&[(0, nx, 1), (ny - 1, ny, 1), (0, nz, 1)])
                .expect("invariant: dp_dy_scratch implies ny >= 1; uy boundary valid");
            boundary.fill(0.0);
        }

        if let Some(ref dp_dz) = self.dp_dz_scratch {
            let rho_near = self
                .materials
                .rho0
                .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
                .expect("invariant: dp_dz_scratch implies nz > 1; rho near valid");
            let rho_far = self
                .materials
                .rho0
                .slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)])
                .expect("invariant: dp_dz_scratch implies nz > 1; rho far valid");
            update_staggered_velocity(
                self.fields
                    .uz
                    .view_mut()
                    .slice_mut(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
                    .expect("invariant: dp_dz_scratch implies nz > 1; uz slice valid"),
                rho_near,
                rho_far,
                dp_dz.view(),
                dt,
            );
            let mut boundary = self
                .fields
                .uz
                .view_mut()
                .slice_mut(&[(0, nx, 1), (0, ny, 1), (nz - 1, nz, 1)])
                .expect("invariant: dp_dz_scratch implies nz >= 1; uz boundary valid");
            boundary.fill(0.0);
        }

        Ok(())
    }
}
