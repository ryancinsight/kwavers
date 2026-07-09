//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod body_force;

use super::super::boundary::ElasticSwePMLBoundary;
use super::super::scratch::ElasticStepScratch;
use super::super::stress::stress_divergence_into;
use super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::{
    Array1,
};

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
    lambda: &'a leto::Array3<f64>,
    mu: &'a leto::Array3<f64>,
    density: &'a leto::Array3<f64>,
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
        lambda: &'a leto::Array3<f64>,
        mu: &'a leto::Array3<f64>,
        density: &'a leto::Array3<f64>,
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

        let dt_half = 0.5 * dt;

        // --- Half-step v(t+Δt/2) = v(t) + (Δt/2)·a(t) ---
        //
        // Slice 8 site 1 (cluster A): migrated Zip::indexed 3-mut chain
        // to verbose is_standard_layout + flat-slice pattern. 6 verbose
        // asserts (3 mut on field.{vx,vy,vz} + 3 immut scratch.{ax,ay,az}).
        // The Zip chain is dropped entirely: each worker task writes to
        // disjoint vx_slice[idx] / vy_slice[idx] / vz_slice[idx] elements.
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let vx_slice = field.vx.as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch.ax.as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            vx_slice.iter_mut().enumerate(|idx, vx: &mut f64| {
                *vx += dt_half * ax_slice[idx];
                vy_slice[idx] += dt_half * ay_slice[idx];
                vz_slice[idx] += dt_half * az_slice[idx];
            });
        }

        // --- Full displacement step u(t+Δt) = u(t) + Δt·v(t+Δt/2) ---
        //
        // Slice 8 site 2 (cluster B): migrated Zip::from 6-element chain
        // (3 mut + 3 immut, at the ndarray producer limit) to verbose
        // is_standard_layout + flat-slice pattern. 6 verbose asserts (3 mut
        // on field.{ux,uy,uz} + 3 immut field.{vx,vy,vz}). The Zip chain is
        // dropped entirely: each worker task writes to disjoint
        // ux_slice[idx] / uy_slice[idx] / uz_slice[idx] elements.
        assert!(
            field.ux,
            "field.ux must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uy,
            "field.uy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uz,
            "field.uz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let ux_slice = field.ux.as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field.uy.as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field.uz.as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            let vx_slice = field.vx.as_slice()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            ux_slice.iter_mut().enumerate(|idx, ux: &mut f64| {
                *ux += dt * vx_slice[idx];
                uy_slice[idx] += dt * vy_slice[idx];
                uz_slice[idx] += dt * vz_slice[idx];
            });
        }

        // a(t+Δt): acceleration at updated displacement (reuses scratch)
        let new_time = field.time + dt;
        self.compute_acceleration(field, scratch, body_force, new_time)?;

        // --- Second half-step v(t+Δt) = v(t+Δt/2) + (Δt/2)·a(t+Δt) ---
        //
        // Slice 8 site 3 (cluster A): identical migration to site 1 (6
        // verbose asserts, Zip chain dropped).
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let vx_slice = field.vx.as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch.ax.as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            vx_slice.iter_mut().enumerate(|idx, vx: &mut f64| {
                *vx += dt_half * ax_slice[idx];
                vy_slice[idx] += dt_half * ay_slice[idx];
                vz_slice[idx] += dt_half * az_slice[idx];
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

        // --- Half-step v(t+Δt/2) = v(t) + (Δt/2)·a(t) ---
        //
        // Slice 8 site 4 (cluster A): identical migration to step() site 1.
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let vx_slice = field.vx.as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch.ax.as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            vx_slice.iter_mut().enumerate(|idx, vx: &mut f64| {
                *vx += dt_half * ax_slice[idx];
                vy_slice[idx] += dt_half * ay_slice[idx];
                vz_slice[idx] += dt_half * az_slice[idx];
            });
        }

        // --- Full displacement step u(t+Δt) = u(t) + Δt·v(t+Δt/2) ---
        //
        // Slice 8 site 5 (cluster B): identical migration to step() site 2.
        assert!(
            field.ux,
            "field.ux must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uy,
            "field.uy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uz,
            "field.uz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let ux_slice = field.ux.as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field.uy.as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field.uz.as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            let vx_slice = field.vx.as_slice()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            ux_slice.iter_mut().enumerate(|idx, ux: &mut f64| {
                *ux += dt * vx_slice[idx];
                uy_slice[idx] += dt * vy_slice[idx];
                uz_slice[idx] += dt * vz_slice[idx];
            });
        }

        // a(t+Δt)
        let new_time = field.time + dt;
        self.compute_acceleration_with_body_forces(field, scratch, body_forces, new_time)?;

        // --- Second half-step v(t+Δt) = v(t+Δt/2) + (Δt/2)·a(t+Δt) ---
        //
        // Slice 8 site 6 (cluster A): identical migration to step() site 3.
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let vx_slice = field.vx.as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch.ax.as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            vx_slice.iter_mut().enumerate(|idx, vx: &mut f64| {
                *vx += dt_half * ax_slice[idx];
                vy_slice[idx] += dt_half * ay_slice[idx];
                vz_slice[idx] += dt_half * az_slice[idx];
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
    /// The subsequent parallel pass reads `{div_x,div_y,div_z}` as
    /// immutable flat slices and writes `{ax,ay,az}` as mutable flat
    /// slices — six distinct `Array3` fields.  Rust NLL field-split
    /// borrows guarantee no aliasing.
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

        // Slice 8 site 7 (cluster C): 3-mut Zip::indexed chain with
        // body_force::evaluate requiring (i,j,k). Migrated to verbose
        // is_standard_layout + flat-slice pattern with idx-to-(i,j,k)
        // decomposition inline (i = idx/(ny*nz); j = (idx/nz)%ny;
        // k = idx%nz). 7 verbose asserts (3 mut on scratch.{ax,ay,az}
        // + 3 immut scratch.{div_x,div_y,div_z} + 1 immut self.density).
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_x,
            "scratch.div_x must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_y,
            "scratch.div_y must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_z,
            "scratch.div_z must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            self.density,
            "self.density must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let ax_slice = scratch.ax.as_slice_mut()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice_mut()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice_mut()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            let div_x_slice = scratch.div_x.as_slice()
                .expect("scratch.div_x: standard-layout asserted just above; layout matched");
            let div_y_slice = scratch.div_y.as_slice()
                .expect("scratch.div_y: standard-layout asserted just above; layout matched");
            let div_z_slice = scratch.div_z.as_slice()
                .expect("scratch.div_z: standard-layout asserted just above; layout matched");
            let rho_slice = self.density.as_slice()
                .expect("self.density: standard-layout asserted just above; layout matched");
            let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
            ax_slice.iter_mut().enumerate(|idx, o_ax: &mut f64| {
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let force = body_force
                    .map(|bf| {
                        body_force::evaluate(self.grid, bf, i, j, k, time).unwrap_or([0.0; 3])
                    })
                    .unwrap_or([0.0; 3]);
                ay_slice[idx] = (div_y_slice[idx] + force[1]) / rho_slice[idx];
                az_slice[idx] = (div_z_slice[idx] + force[2]) / rho_slice[idx];
                *o_ax = (div_x_slice[idx] + force[0]) / rho_slice[idx];
            });
        }

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

        // Slice 8 site 8 (cluster C): identical migration to
        // compute_acceleration (7 verbose asserts, flat-slice pattern,
        // idx-to-(i,j,k) inline). Body force accumulation loops over
        // `body_forces` slice — preserved.
        assert!(
            scratch.ax,
            "scratch.ax must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.ay,
            "scratch.ay must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.az,
            "scratch.az must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_x,
            "scratch.div_x must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_y,
            "scratch.div_y must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            scratch.div_z,
            "scratch.div_z must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            self.density,
            "self.density must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let ax_slice = scratch.ax.as_slice_mut()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch.ay.as_slice_mut()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch.az.as_slice_mut()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            let div_x_slice = scratch.div_x.as_slice()
                .expect("scratch.div_x: standard-layout asserted just above; layout matched");
            let div_y_slice = scratch.div_y.as_slice()
                .expect("scratch.div_y: standard-layout asserted just above; layout matched");
            let div_z_slice = scratch.div_z.as_slice()
                .expect("scratch.div_z: standard-layout asserted just above; layout matched");
            let rho_slice = self.density.as_slice()
                .expect("self.density: standard-layout asserted just above; layout matched");
            let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
            ax_slice.iter_mut().enumerate(|idx, o_ax: &mut f64| {
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let mut force = [0.0_f64; 3];
                for bf in body_forces {
                    let f = body_force::evaluate(self.grid, bf, i, j, k, time).unwrap_or([0.0; 3]);
                    force[0] += f[0];
                    force[1] += f[1];
                    force[2] += f[2];
                }
                ay_slice[idx] = (div_y_slice[idx] + force[1]) / rho_slice[idx];
                az_slice[idx] = (div_z_slice[idx] + force[2]) / rho_slice[idx];
                *o_ax = (div_x_slice[idx] + force[0]) / rho_slice[idx];
            });
        }

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
    /// because leto `Zip::indexed` supports ≤ 5 arrays (6 tuple
    /// elements including the index).
    pub(crate) fn apply_pml_damping(&self, field: &mut ElasticWaveField, dt: f64) {
        let (nx, ny, nz) = field.vx.shape();
        let sx = self.sigma_x.as_slice().expect("sigma_x contiguous");
        let sy = self.sigma_y.as_slice().expect("sigma_y contiguous");
        let sz = self.sigma_z.as_slice().expect("sigma_z contiguous");

        debug_assert_eq!((sx.shape()[0] * sx.shape()[1] * sx.shape()[2]), nx);
        debug_assert_eq!((sy.shape()[0] * sy.shape()[1] * sy.shape()[2]), ny);
        debug_assert_eq!((sz.shape()[0] * sz.shape()[1] * sz.shape()[2]), nz);

        // Slice 8 site 9 (cluster D, velocity pass): 3-mut Zip::indexed
        // chain with per-axis sigma_x[i]/sigma_y[j]/sigma_z[k] lookup
        // requiring (i,j,k). Migrated to verbose is_standard_layout +
        // flat-slice pattern with idx-to-(i,j,k) decomposition inline.
        // 3 verbose asserts on the 3 mut outs (sigma_* slices are
        // unconditionally C-contiguous Array1, asserted via debug_assert!).
        assert!(
            field.vx,
            "field.vx must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vy,
            "field.vy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.vz,
            "field.vz must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let vx_slice = field.vx.as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field.vy.as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field.vz.as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            vx_slice.iter_mut().enumerate(|idx, vx: &mut f64| {
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let d = (-sx[i] * dt).exp() * (-sy[j] * dt).exp() * (-sz[k] * dt).exp();
                if d < 1.0 {
                    *vx *= d;
                    vy_slice[idx] *= d;
                    vz_slice[idx] *= d;
                }
            });
        }

        // Slice 8 site 10 (cluster D, displacement pass): identical
        // migration pattern to the velocity pass, on field.{ux,uy,uz}.
        assert!(
            field.ux,
            "field.ux must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uy,
            "field.uy must be C-contiguous (default Array3 layout) for the migration"
        );
        assert!(
            field.uz,
            "field.uz must be C-contiguous (default Array3 layout) for the migration"
        );
        {
            let ux_slice = field.ux.as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field.uy.as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field.uz.as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            ux_slice.iter_mut().enumerate(|idx, ux: &mut f64| {
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let d = (-sx[i] * dt).exp() * (-sy[j] * dt).exp() * (-sz[k] * dt).exp();
                if d < 1.0 {
                    *ux *= d;
                    uy_slice[idx] *= d;
                    uz_slice[idx] *= d;
                }
            });
        }
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
        let (nx, ny, nz) = self.lambda.shape();
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
