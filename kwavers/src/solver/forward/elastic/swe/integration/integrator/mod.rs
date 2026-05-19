//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod body_force;

use super::super::boundary::ElasticSwePMLBoundary;
use super::super::scratch::ElasticStepScratch;
use super::super::stress::stress_divergence_into;
use super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array1, Zip};

/// Time integration engine for elastic waves.
///
/// Implements a velocity-Verlet scheme with optional body forces and a
/// separable per-axis exponential PML (Collino & Tsogka 2001 §3).
///
/// # PML correctness
///
/// The previous scalar implementation applied the maximum σ over all axes
/// uniformly to all velocity components (`v *= exp(−σ_max · dt)`), which
/// fails for oblique-incidence waves and grows unboundedly past NT ≈ 200.
///
/// The corrected implementation pre-computes per-axis σ profiles and at each
/// step applies the separable damping factor `exp(−σ_x[i]·dt) · exp(−σ_y[j]·dt) ·
/// exp(−σ_z[k]·dt)` to **both** displacements `u` and velocities `v`.
///
/// Damping `u` is necessary because the displacement field drives the stress
/// tensor, and undamped displacement in the PML generates stress that
/// continuously feeds reflected velocity back into the domain.
///
/// # Memory allocation discipline
///
/// `step` and `step_with_body_forces` accept a `&mut ElasticStepScratch`
/// that the caller pre-allocates **once before the time loop**.  This
/// eliminates 24 × `Array3<f64>` per-step heap allocations (2 ×
/// `stress_divergence_into` calls × 9 stress+divergence arrays + 3
/// acceleration arrays = 21 allocations per step, 24 when split across both
/// `compute_acceleration` calls).  For 128³ at f64 this removes 384 MiB of
/// heap churn per time step.
#[derive(Debug)]
pub struct TimeIntegrator<'a> {
    grid: &'a Grid,
    lambda: &'a ndarray::Array3<f64>,
    mu: &'a ndarray::Array3<f64>,
    density: &'a ndarray::Array3<f64>,
    /// Per-axis σ profiles (interior = 0; absorbing layer = power-law).
    sigma_x: Array1<f64>,
    sigma_y: Array1<f64>,
    sigma_z: Array1<f64>,
}

impl<'a> TimeIntegrator<'a> {
    /// Create a new time integrator.
    ///
    /// Computes per-axis σ profiles from `pml` at construction; the profiles
    /// do not depend on `dt`, which is determined later from the CFL condition.
    #[must_use]
    pub fn new(
        grid: &'a Grid,
        lambda: &'a ndarray::Array3<f64>,
        mu: &'a ndarray::Array3<f64>,
        density: &'a ndarray::Array3<f64>,
        pml: &ElasticSwePMLBoundary,
    ) -> Self {
        let (sigma_x, sigma_y, sigma_z) = pml.axis_sigma_profiles(grid);
        Self {
            grid,
            lambda,
            mu,
            density,
            sigma_x,
            sigma_y,
            sigma_z,
        }
    }

    /// Perform single time step with velocity-Verlet integration.
    ///
    /// ## Algorithm (velocity-Verlet)
    ///
    /// ```text
    /// v(t+Δt/2) = v(t)       + (Δt/2) · a(t)          [half-step v]
    /// u(t+Δt)   = u(t)       + Δt     · v(t+Δt/2)     [full-step u]
    /// v(t+Δt)   = v(t+Δt/2) + (Δt/2) · a(t+Δt)        [half-step v]
    /// ```
    ///
    /// ## Theorem (symplectic accuracy)
    ///
    /// Velocity-Verlet is a second-order symplectic integrator: it exactly
    /// preserves a shadow Hamiltonian (Leimkuhler & Reich 2004, §4.2), so
    /// total energy drifts are bounded over exponentially long times rather
    /// than growing linearly.  Standard Euler and Runge-Kutta methods are
    /// not symplectic and produce secular energy drift in elastic simulations.
    ///
    /// ## Memory
    ///
    /// `scratch` must be pre-allocated with [`ElasticStepScratch::new`]
    /// before the time loop.  No `Array3` is allocated inside this method.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        // a(t): acceleration at current state
        self.compute_acceleration(field, scratch, body_force, field.time)?;

        // Half-step v(t+Δt/2) = v(t) + (Δt/2)·a(t)
        //
        // Theorem: each element is updated independently → race-free parallel.
        let dt_half = 0.5 * dt;
        {
            let ax_v = scratch.ax.view();
            let ay_v = scratch.ay.view();
            let az_v = scratch.az.view();
            Zip::indexed(field.vx.view_mut())
                .and(field.vy.view_mut())
                .and(field.vz.view_mut())
                .par_for_each(|(i, j, k), vx, vy, vz| {
                    *vx += dt_half * ax_v[[i, j, k]];
                    *vy += dt_half * ay_v[[i, j, k]];
                    *vz += dt_half * az_v[[i, j, k]];
                });
        }

        // Full displacement step u(t+Δt) = u(t) + Δt·v(t+Δt/2)
        // ux/uy/uz and vx/vy/vz are disjoint fields → borrow-safe.
        {
            let vx_v = field.vx.view();
            let vy_v = field.vy.view();
            let vz_v = field.vz.view();
            Zip::from(field.ux.view_mut())
                .and(field.uy.view_mut())
                .and(field.uz.view_mut())
                .and(vx_v)
                .and(vy_v)
                .and(vz_v)
                .par_for_each(|ux, uy, uz, &vx, &vy, &vz| {
                    *ux += dt * vx;
                    *uy += dt * vy;
                    *uz += dt * vz;
                });
        }

        // a(t+Δt): acceleration at updated displacement (reuses scratch)
        let new_time = field.time + dt;
        self.compute_acceleration(field, scratch, body_force, new_time)?;

        // Second half-step v(t+Δt) = v(t+Δt/2) + (Δt/2)·a(t+Δt)
        {
            let ax_v = scratch.ax.view();
            let ay_v = scratch.ay.view();
            let az_v = scratch.az.view();
            Zip::indexed(field.vx.view_mut())
                .and(field.vy.view_mut())
                .and(field.vz.view_mut())
                .par_for_each(|(i, j, k), vx, vy, vz| {
                    *vx += dt_half * ax_v[[i, j, k]];
                    *vy += dt_half * ay_v[[i, j, k]];
                    *vz += dt_half * az_v[[i, j, k]];
                });
        }

        self.apply_pml_damping(field, dt);

        Ok(())
    }

    /// Perform single time step with multiple simultaneous body forces.
    ///
    /// Semantics identical to [`step`]; the body-force accumulation loop
    /// sums contributions from all `body_forces` slices per grid point.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn step_with_body_forces(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_forces: &[ElasticBodyForceConfig],
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        // a(t)
        self.compute_acceleration_with_body_forces(field, scratch, body_forces, field.time)?;

        let dt_half = 0.5 * dt;
        {
            let ax_v = scratch.ax.view();
            let ay_v = scratch.ay.view();
            let az_v = scratch.az.view();
            Zip::indexed(field.vx.view_mut())
                .and(field.vy.view_mut())
                .and(field.vz.view_mut())
                .par_for_each(|(i, j, k), vx, vy, vz| {
                    *vx += dt_half * ax_v[[i, j, k]];
                    *vy += dt_half * ay_v[[i, j, k]];
                    *vz += dt_half * az_v[[i, j, k]];
                });
        }

        {
            let vx_v = field.vx.view();
            let vy_v = field.vy.view();
            let vz_v = field.vz.view();
            Zip::from(field.ux.view_mut())
                .and(field.uy.view_mut())
                .and(field.uz.view_mut())
                .and(vx_v)
                .and(vy_v)
                .and(vz_v)
                .par_for_each(|ux, uy, uz, &vx, &vy, &vz| {
                    *ux += dt * vx;
                    *uy += dt * vy;
                    *uz += dt * vz;
                });
        }

        // a(t+Δt)
        let new_time = field.time + dt;
        self.compute_acceleration_with_body_forces(field, scratch, body_forces, new_time)?;

        {
            let ax_v = scratch.ax.view();
            let ay_v = scratch.ay.view();
            let az_v = scratch.az.view();
            Zip::indexed(field.vx.view_mut())
                .and(field.vy.view_mut())
                .and(field.vz.view_mut())
                .par_for_each(|(i, j, k), vx, vy, vz| {
                    *vx += dt_half * ax_v[[i, j, k]];
                    *vy += dt_half * ay_v[[i, j, k]];
                    *vz += dt_half * az_v[[i, j, k]];
                });
        }

        self.apply_pml_damping(field, dt);

        Ok(())
    }

    /// Compute elastic acceleration a = (∇·σ + f) / ρ into `scratch.{ax,ay,az}`.
    ///
    /// Calls [`stress_divergence_into`] which fills `scratch.{sxx,…,div_z}`,
    /// then divides by ρ and adds body force per grid point.  All writes are
    /// to disjoint `scratch` fields → race-free under Rayon `par_for_each`.
    ///
    /// ## Theorem (race-freedom)
    ///
    /// `stress_divergence_into` fills `{sxx,…,div_z}` (Pass 1 + Pass 2);
    /// on return, those fields hold consistent values for all `(i,j,k)`.
    /// The subsequent `Zip` reads `{div_x,div_y,div_z}` as immutable views
    /// and writes `{ax,ay,az}` as mutable views — six distinct `Array3`
    /// fields.  Rust NLL field-split borrows guarantee no aliasing.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    fn compute_acceleration(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_force: Option<&ElasticBodyForceConfig>,
        time: f64,
    ) -> KwaversResult<()> {
        stress_divergence_into(self.grid, self.lambda, self.mu, field, scratch);

        // NLL field-split borrow: div_{x,y,z} borrowed immutably;
        // ax/ay/az borrowed mutably; all six are distinct struct fields.
        let div_x_v = scratch.div_x.view();
        let div_y_v = scratch.div_y.view();
        let div_z_v = scratch.div_z.view();

        Zip::indexed(scratch.ax.view_mut())
            .and(scratch.ay.view_mut())
            .and(scratch.az.view_mut())
            .par_for_each(|(i, j, k), o_ax, o_ay, o_az| {
                let force = body_force
                    .map(|bf| {
                        body_force::evaluate(self.grid, bf, i, j, k, time).unwrap_or([0.0; 3])
                    })
                    .unwrap_or([0.0; 3]);
                let rho = self.density[[i, j, k]];
                *o_ax = (div_x_v[[i, j, k]] + force[0]) / rho;
                *o_ay = (div_y_v[[i, j, k]] + force[1]) / rho;
                *o_az = (div_z_v[[i, j, k]] + force[2]) / rho;
            });

        Ok(())
    }

    /// Compute elastic acceleration from multiple simultaneous body forces.
    ///
    /// Same theorem as [`compute_acceleration`]; body-force accumulation
    /// sums over `body_forces` slice inside the parallel closure.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    fn compute_acceleration_with_body_forces(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_forces: &[ElasticBodyForceConfig],
        time: f64,
    ) -> KwaversResult<()> {
        stress_divergence_into(self.grid, self.lambda, self.mu, field, scratch);

        let div_x_v = scratch.div_x.view();
        let div_y_v = scratch.div_y.view();
        let div_z_v = scratch.div_z.view();

        Zip::indexed(scratch.ax.view_mut())
            .and(scratch.ay.view_mut())
            .and(scratch.az.view_mut())
            .par_for_each(|(i, j, k), o_ax, o_ay, o_az| {
                let mut force = [0.0_f64; 3];
                for bf in body_forces {
                    let f = body_force::evaluate(self.grid, bf, i, j, k, time).unwrap_or([0.0; 3]);
                    force[0] += f[0];
                    force[1] += f[1];
                    force[2] += f[2];
                }
                let rho = self.density[[i, j, k]];
                *o_ax = (div_x_v[[i, j, k]] + force[0]) / rho;
                *o_ay = (div_y_v[[i, j, k]] + force[1]) / rho;
                *o_az = (div_z_v[[i, j, k]] + force[2]) / rho;
            });

        Ok(())
    }

    /// Apply separable per-axis PML damping to both displacements and velocities.
    ///
    /// For cell `(i,j,k)`, the damping factor is the product of per-axis
    /// exponentials:
    ///
    /// ```text
    /// d(i,j,k) = exp(−σ_x[i]·dt) · exp(−σ_y[j]·dt) · exp(−σ_z[k]·dt)
    /// ```
    ///
    /// Applied to both `{ux,uy,uz}` and `{vx,vy,vz}`. Damping the
    /// displacement fields is necessary: the stress tensor is derived from
    /// displacements, and undamped displacement in the PML would continuously
    /// feed reflected energy back into the interior even with damped velocity.
    ///
    /// ## Theorem (separable PML)
    ///
    /// For each `(i,j,k)`:
    /// ```text
    /// d(i,j,k) = exp(−σ_x[i]·dt) · exp(−σ_y[j]·dt) · exp(−σ_z[k]·dt)
    /// ```
    /// All six field components at `(i,j,k)` are multiplied by the same
    /// scalar `d` and are independent of neighbouring cells → race-free.
    ///
    /// Per-thread closure computes `d` on the fly from the per-axis σ slices,
    /// avoiding a temporary damping array.
    ///
    /// Split into two `Zip::indexed` passes (velocity, then displacement)
    /// because ndarray 0.16 `Zip::indexed` supports ≤ 5 arrays (6 tuple
    /// elements including the index).
    pub(crate) fn apply_pml_damping(&self, field: &mut ElasticWaveField, dt: f64) {
        let (nx, ny, nz) = field.vx.dim();
        let sx = self.sigma_x.as_slice().expect("sigma_x contiguous");
        let sy = self.sigma_y.as_slice().expect("sigma_y contiguous");
        let sz = self.sigma_z.as_slice().expect("sigma_z contiguous");

        debug_assert_eq!(sx.len(), nx);
        debug_assert_eq!(sy.len(), ny);
        debug_assert_eq!(sz.len(), nz);

        // Velocity pass
        Zip::indexed(field.vx.view_mut())
            .and(field.vy.view_mut())
            .and(field.vz.view_mut())
            .par_for_each(|(i, j, k), vx, vy, vz| {
                let d = (-sx[i] * dt).exp() * (-sy[j] * dt).exp() * (-sz[k] * dt).exp();
                if d < 1.0 {
                    *vx *= d;
                    *vy *= d;
                    *vz *= d;
                }
            });

        // Displacement pass
        Zip::indexed(field.ux.view_mut())
            .and(field.uy.view_mut())
            .and(field.uz.view_mut())
            .par_for_each(|(i, j, k), ux, uy, uz| {
                let d = (-sx[i] * dt).exp() * (-sy[j] * dt).exp() * (-sz[k] * dt).exp();
                if d < 1.0 {
                    *ux *= d;
                    *uy *= d;
                    *uz *= d;
                }
            });
    }

    /// Calculate CFL-limited time step.
    ///
    /// ## CFL condition for 3D elastic waves
    ///
    /// ```text
    /// Δt < Δx / (√3 · c_max)
    /// ```
    ///
    /// where `c_max = max(c_s, c_p)` over all grid cells, and
    /// `c_s = √(μ/ρ)`, `c_p = √((λ+2μ)/ρ)`.
    ///
    /// The √3 factor accounts for diagonal propagation in 3D.
    #[must_use]
    pub fn calculate_stable_timestep(&self, cfl_factor: f64) -> f64 {
        let (nx, ny, nz) = self.lambda.dim();
        let mut max_c = 0.0;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu_val = self.mu[[i, j, k]];
                    let lambda_val = self.lambda[[i, j, k]];
                    let rho_val = self.density[[i, j, k]];

                    if rho_val > 0.0 {
                        let cs = (mu_val / rho_val).sqrt();
                        let cp = (2.0f64.mul_add(mu_val, lambda_val) / rho_val).sqrt();
                        max_c = f64::max(max_c, f64::max(cs, cp));
                    }
                }
            }
        }

        if max_c <= 0.0 {
            return 0.0;
        }

        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_dt = min_dx / (3.0_f64.sqrt() * max_c);

        cfl_dt * cfl_factor
    }
}
