//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod body_force;

use super::super::boundary::ElasticSwePMLBoundary;
use super::super::scratch::ElasticStepScratch;
use super::super::stress::stress_divergence_into;
use super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array1;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

const INTEGRATOR_CHUNK: usize = 4096;

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
/// step applies the separable damping factor `exp(−σ_x`i`·dt) · exp(−σ_y`J`·dt) ·
/// exp(−σ_z`K`·dt)` to **both** displacements `u` and velocities `v`.
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
    /// Reciprocal density when every cell has the same density.
    ///
    /// The homogeneous regime is selected once at construction so the dense
    /// no-body-force acceleration kernel avoids three divisions and one
    /// density load per cell on every evaluation.
    uniform_inverse_density: Option<f64>,
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
        let density_values = density
            .as_slice()
            .expect("invariant: elastic density uses standard layout");
        let uniform_inverse_density = density_values.split_first().and_then(|(&first, rest)| {
            rest.iter()
                .all(|&value| value == first)
                .then(|| first.recip())
        });
        Self {
            grid,
            lambda,
            mu,
            density,
            uniform_inverse_density,
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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        if let Some(body_force) = body_force {
            body_force::validate(body_force)?;
        }
        // a(t): acceleration at current state
        self.compute_acceleration(field, scratch, body_force, field.time)?;

        let dt_half = 0.5 * dt;

        // --- Half-step v(t+Δt/2) = v(t) + (Δt/2)·a(t) ---
        {
            let vx_slice = field
                .vx
                .as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch
                .ax
                .as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx_slice,
                vy_slice,
                vz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..vx_chunk.len() {
                        let idx = start + offset;
                        vx_chunk[offset] += dt_half * ax_slice[idx];
                        vy_chunk[offset] += dt_half * ay_slice[idx];
                        vz_chunk[offset] += dt_half * az_slice[idx];
                    }
                },
            );
        }

        // --- Full displacement step u(t+Δt) = u(t) + Δt·v(t+Δt/2) ---
        {
            let ux_slice = field
                .ux
                .as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field
                .uy
                .as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field
                .uz
                .as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            let vx_slice = field
                .vx
                .as_slice()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ux_slice,
                uy_slice,
                uz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, ux_chunk, uy_chunk, uz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..ux_chunk.len() {
                        let idx = start + offset;
                        ux_chunk[offset] += dt * vx_slice[idx];
                        uy_chunk[offset] += dt * vy_slice[idx];
                        uz_chunk[offset] += dt * vz_slice[idx];
                    }
                },
            );
        }

        // a(t+Δt): acceleration at updated displacement (reuses scratch)
        let new_time = field.time + dt;
        self.compute_acceleration(field, scratch, body_force, new_time)?;

        // --- Second half-step v(t+Δt) = v(t+Δt/2) + (Δt/2)·a(t+Δt) ---
        {
            let vx_slice = field
                .vx
                .as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch
                .ax
                .as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx_slice,
                vy_slice,
                vz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..vx_chunk.len() {
                        let idx = start + offset;
                        vx_chunk[offset] += dt_half * ax_slice[idx];
                        vy_chunk[offset] += dt_half * ay_slice[idx];
                        vz_chunk[offset] += dt_half * az_slice[idx];
                    }
                },
            );
        }

        self.apply_pml_damping(field, dt, scratch);

        Ok(())
    }

    /// Perform single time step with multiple simultaneous body forces.
    ///
    /// Semantics identical to [`Self::step`]; the body-force accumulation loop
    /// sums contributions from all `body_forces` slices per grid point.
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn step_with_body_forces(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_forces: &[ElasticBodyForceConfig],
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        for body_force in body_forces {
            body_force::validate(body_force)?;
        }
        // a(t)
        self.compute_acceleration_with_body_forces(field, scratch, body_forces, field.time)?;

        let dt_half = 0.5 * dt;

        // --- Half-step v(t+Δt/2) = v(t) + (Δt/2)·a(t) ---
        {
            let vx_slice = field
                .vx
                .as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch
                .ax
                .as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx_slice,
                vy_slice,
                vz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..vx_chunk.len() {
                        let idx = start + offset;
                        vx_chunk[offset] += dt_half * ax_slice[idx];
                        vy_chunk[offset] += dt_half * ay_slice[idx];
                        vz_chunk[offset] += dt_half * az_slice[idx];
                    }
                },
            );
        }

        // --- Full displacement step u(t+Δt) = u(t) + Δt·v(t+Δt/2) ---
        {
            let ux_slice = field
                .ux
                .as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field
                .uy
                .as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field
                .uz
                .as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            let vx_slice = field
                .vx
                .as_slice()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ux_slice,
                uy_slice,
                uz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, ux_chunk, uy_chunk, uz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..ux_chunk.len() {
                        let idx = start + offset;
                        ux_chunk[offset] += dt * vx_slice[idx];
                        uy_chunk[offset] += dt * vy_slice[idx];
                        uz_chunk[offset] += dt * vz_slice[idx];
                    }
                },
            );
        }

        // a(t+Δt)
        let new_time = field.time + dt;
        self.compute_acceleration_with_body_forces(field, scratch, body_forces, new_time)?;

        // --- Second half-step v(t+Δt) = v(t+Δt/2) + (Δt/2)·a(t+Δt) ---
        {
            let vx_slice = field
                .vx
                .as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            let ax_slice = scratch
                .ax
                .as_slice()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx_slice,
                vy_slice,
                vz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..vx_chunk.len() {
                        let idx = start + offset;
                        vx_chunk[offset] += dt_half * ax_slice[idx];
                        vy_chunk[offset] += dt_half * ay_slice[idx];
                        vz_chunk[offset] += dt_half * az_slice[idx];
                    }
                },
            );
        }

        self.apply_pml_damping(field, dt, scratch);

        Ok(())
    }

    /// Compute elastic acceleration a = (∇·σ + f) / ρ into `scratch.{ax,ay,az}`.
    ///
    /// Calls [`stress_divergence_into`] which fills `scratch.{sxx,…,div_z}`,
    /// then divides by ρ and adds body force per grid point.  All writes are
    /// to disjoint `scratch` fields → race-free under moirai parallel dispatch.
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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    fn compute_acceleration(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_force: Option<&ElasticBodyForceConfig>,
        time: f64,
    ) -> KwaversResult<()> {
        stress_divergence_into(self.grid, self.lambda, self.mu, field, scratch);

        {
            let ax_slice = scratch
                .ax
                .as_slice_mut()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice_mut()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice_mut()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            let div_x_slice = scratch
                .div_x
                .as_slice()
                .expect("scratch.div_x: standard-layout asserted just above; layout matched");
            let div_y_slice = scratch
                .div_y
                .as_slice()
                .expect("scratch.div_y: standard-layout asserted just above; layout matched");
            let div_z_slice = scratch
                .div_z
                .as_slice()
                .expect("scratch.div_z: standard-layout asserted just above; layout matched");
            let rho_slice = self
                .density
                .as_slice()
                .expect("self.density: standard-layout asserted just above; layout matched");

            if let Some(body_force) = body_force {
                let grid = self.grid;
                let (_nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
                for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                    ax_slice,
                    ay_slice,
                    az_slice,
                    INTEGRATOR_CHUNK,
                    |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                        let start = chunk_idx * INTEGRATOR_CHUNK;
                        for offset in 0..ax_chunk.len() {
                            let idx = start + offset;
                            let i = idx / (ny * nz);
                            let j = (idx / nz) % ny;
                            let k = idx % nz;
                            let force = body_force::evaluate(grid, body_force, i, j, k, time);
                            ax_chunk[offset] = (div_x_slice[idx] + force[0]) / rho_slice[idx];
                            ay_chunk[offset] = (div_y_slice[idx] + force[1]) / rho_slice[idx];
                            az_chunk[offset] = (div_z_slice[idx] + force[2]) / rho_slice[idx];
                        }
                    },
                );
                return Ok(());
            }

            // Point-force propagation injects velocity before each step and has
            // no distributed body-force field. Dispatch this regime once so its
            // dense kernel contains no coordinate division or optional branch.
            if let Some(inverse_density) = self.uniform_inverse_density {
                for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                    ax_slice,
                    ay_slice,
                    az_slice,
                    INTEGRATOR_CHUNK,
                    |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                        let start = chunk_idx * INTEGRATOR_CHUNK;
                        for offset in 0..ax_chunk.len() {
                            let idx = start + offset;
                            ax_chunk[offset] = div_x_slice[idx] * inverse_density;
                            ay_chunk[offset] = div_y_slice[idx] * inverse_density;
                            az_chunk[offset] = div_z_slice[idx] * inverse_density;
                        }
                    },
                );
                return Ok(());
            }

            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ax_slice,
                ay_slice,
                az_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..ax_chunk.len() {
                        let idx = start + offset;
                        ax_chunk[offset] = div_x_slice[idx] / rho_slice[idx];
                        ay_chunk[offset] = div_y_slice[idx] / rho_slice[idx];
                        az_chunk[offset] = div_z_slice[idx] / rho_slice[idx];
                    }
                },
            );
        }

        Ok(())
    }

    /// Compute elastic acceleration from multiple simultaneous body forces.
    ///
    /// Same theorem as [`compute_acceleration`]; body-force accumulation
    /// sums over `body_forces` slice inside the parallel closure.
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    fn compute_acceleration_with_body_forces(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_forces: &[ElasticBodyForceConfig],
        time: f64,
    ) -> KwaversResult<()> {
        stress_divergence_into(self.grid, self.lambda, self.mu, field, scratch);

        {
            let grid = self.grid;
            let ax_slice = scratch
                .ax
                .as_slice_mut()
                .expect("scratch.ax: standard-layout asserted just above; layout matched");
            let ay_slice = scratch
                .ay
                .as_slice_mut()
                .expect("scratch.ay: standard-layout asserted just above; layout matched");
            let az_slice = scratch
                .az
                .as_slice_mut()
                .expect("scratch.az: standard-layout asserted just above; layout matched");
            let div_x_slice = scratch
                .div_x
                .as_slice()
                .expect("scratch.div_x: standard-layout asserted just above; layout matched");
            let div_y_slice = scratch
                .div_y
                .as_slice()
                .expect("scratch.div_y: standard-layout asserted just above; layout matched");
            let div_z_slice = scratch
                .div_z
                .as_slice()
                .expect("scratch.div_z: standard-layout asserted just above; layout matched");
            let rho_slice = self
                .density
                .as_slice()
                .expect("self.density: standard-layout asserted just above; layout matched");
            let (_nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ax_slice,
                ay_slice,
                az_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..ax_chunk.len() {
                        let idx = start + offset;
                        let i = idx / (ny * nz);
                        let j = (idx / nz) % ny;
                        let k = idx % nz;
                        let mut force = [0.0_f64; 3];
                        for bf in body_forces {
                            let f = body_force::evaluate(grid, bf, i, j, k, time);
                            force[0] += f[0];
                            force[1] += f[1];
                            force[2] += f[2];
                        }
                        ay_chunk[offset] = (div_y_slice[idx] + force[1]) / rho_slice[idx];
                        az_chunk[offset] = (div_z_slice[idx] + force[2]) / rho_slice[idx];
                        ax_chunk[offset] = (div_x_slice[idx] + force[0]) / rho_slice[idx];
                    }
                },
            );
        }

        Ok(())
    }

    /// Apply separable per-axis PML damping to both displacements and velocities.
    ///
    /// For cell `(i,j,k)`, the damping factor is the product of per-axis
    /// exponentials:
    ///
    /// ```text
    /// d(i,j,k) = exp(−σ_x`i`·dt) · exp(−σ_y`J`·dt) · exp(−σ_z`K`·dt)
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
    /// d(i,j,k) = exp(−σ_x`i`·dt) · exp(−σ_y`J`·dt) · exp(−σ_z`K`·dt)
    /// ```
    /// All six field components at `(i,j,k)` are multiplied by the same
    /// scalar `d` and are independent of neighbouring cells → race-free.
    ///
    /// The reusable step workspace caches the three per-axis exponential
    /// factors for the active `dt`. The dense passes therefore perform only
    /// indexed multiplications and allocate no temporary damping array.
    ///
    /// Split into two passes (velocity, then displacement) to keep each
    /// parallel dispatch handling a 3-mut slice pattern.
    pub(crate) fn apply_pml_damping(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        scratch: &mut ElasticStepScratch,
    ) {
        let [nx, ny, nz] = field.vx.shape();
        let (dx, dy, dz) = scratch.pml_factors(&self.sigma_x, &self.sigma_y, &self.sigma_z, dt);

        debug_assert_eq!(dx.len(), nx);
        debug_assert_eq!(dy.len(), ny);
        debug_assert_eq!(dz.len(), nz);

        {
            let vx_slice = field
                .vx
                .as_slice_mut()
                .expect("field.vx: standard-layout asserted just above; layout matched");
            let vy_slice = field
                .vy
                .as_slice_mut()
                .expect("field.vy: standard-layout asserted just above; layout matched");
            let vz_slice = field
                .vz
                .as_slice_mut()
                .expect("field.vz: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx_slice,
                vy_slice,
                vz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..vx_chunk.len() {
                        let idx = start + offset;
                        let i = idx / (ny * nz);
                        let j = (idx / nz) % ny;
                        let k = idx % nz;
                        let d = dx[i] * dy[j] * dz[k];
                        if d < 1.0 {
                            vx_chunk[offset] *= d;
                            vy_chunk[offset] *= d;
                            vz_chunk[offset] *= d;
                        }
                    }
                },
            );
        }

        {
            let ux_slice = field
                .ux
                .as_slice_mut()
                .expect("field.ux: standard-layout asserted just above; layout matched");
            let uy_slice = field
                .uy
                .as_slice_mut()
                .expect("field.uy: standard-layout asserted just above; layout matched");
            let uz_slice = field
                .uz
                .as_slice_mut()
                .expect("field.uz: standard-layout asserted just above; layout matched");
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ux_slice,
                uy_slice,
                uz_slice,
                INTEGRATOR_CHUNK,
                |chunk_idx, ux_chunk, uy_chunk, uz_chunk| {
                    let start = chunk_idx * INTEGRATOR_CHUNK;
                    for offset in 0..ux_chunk.len() {
                        let idx = start + offset;
                        let i = idx / (ny * nz);
                        let j = (idx / nz) % ny;
                        let k = idx % nz;
                        let d = dx[i] * dy[j] * dz[k];
                        if d < 1.0 {
                            ux_chunk[offset] *= d;
                            uy_chunk[offset] *= d;
                            uz_chunk[offset] *= d;
                        }
                    }
                },
            );
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
        let [nx, ny, nz] = self.lambda.shape();
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
