//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod body_force;

use super::super::boundary::PMLBoundary;
use super::super::stress::stress_divergence;
use super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array1, Array3};

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
#[derive(Debug)]
pub struct TimeIntegrator<'a> {
    grid: &'a Grid,
    lambda: &'a Array3<f64>,
    mu: &'a Array3<f64>,
    density: &'a Array3<f64>,
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
        lambda: &'a Array3<f64>,
        mu: &'a Array3<f64>,
        density: &'a Array3<f64>,
        pml: &PMLBoundary,
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

    /// Perform single time step with velocity-Verlet integration
    ///
    /// ## Algorithm
    /// 1. Half-step velocity update: v(t+Δt/2) = v(t) + (Δt/2) * a(t)
    /// 2. Full-step displacement: u(t+Δt) = u(t) + Δt * v(t+Δt/2)
    /// 3. Half-step velocity update: v(t+Δt) = v(t+Δt/2) + (Δt/2) * a(t+Δt)
    /// 4. Apply PML damping
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.ux.dim();

        let mut ax = Array3::<f64>::zeros((nx, ny, nz));
        let mut ay = Array3::<f64>::zeros((nx, ny, nz));
        let mut az = Array3::<f64>::zeros((nx, ny, nz));

        self.compute_acceleration(field, &mut ax, &mut ay, &mut az, body_force, field.time)?;

        let dt_half = 0.5 * dt;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.vx[[i, j, k]] += dt_half * ax[[i, j, k]];
                    field.vy[[i, j, k]] += dt_half * ay[[i, j, k]];
                    field.vz[[i, j, k]] += dt_half * az[[i, j, k]];
                }
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }

        let new_time = field.time + dt;
        self.compute_acceleration(field, &mut ax, &mut ay, &mut az, body_force, new_time)?;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.vx[[i, j, k]] += dt_half * ax[[i, j, k]];
                    field.vy[[i, j, k]] += dt_half * ay[[i, j, k]];
                    field.vz[[i, j, k]] += dt_half * az[[i, j, k]];
                }
            }
        }

        self.apply_pml_damping(field, dt);

        Ok(())
    }

    /// Perform single time step with multiple simultaneous body forces.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn step_with_body_forces(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_forces: &[ElasticBodyForceConfig],
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.ux.dim();

        let mut ax = Array3::<f64>::zeros((nx, ny, nz));
        let mut ay = Array3::<f64>::zeros((nx, ny, nz));
        let mut az = Array3::<f64>::zeros((nx, ny, nz));

        self.compute_acceleration_with_body_forces(
            field,
            &mut ax,
            &mut ay,
            &mut az,
            body_forces,
            field.time,
        )?;

        let dt_half = 0.5 * dt;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.vx[[i, j, k]] += dt_half * ax[[i, j, k]];
                    field.vy[[i, j, k]] += dt_half * ay[[i, j, k]];
                    field.vz[[i, j, k]] += dt_half * az[[i, j, k]];
                }
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }

        let new_time = field.time + dt;
        self.compute_acceleration_with_body_forces(
            field,
            &mut ax,
            &mut ay,
            &mut az,
            body_forces,
            new_time,
        )?;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.vx[[i, j, k]] += dt_half * ax[[i, j, k]];
                    field.vy[[i, j, k]] += dt_half * ay[[i, j, k]];
                    field.vz[[i, j, k]] += dt_half * az[[i, j, k]];
                }
            }
        }

        self.apply_pml_damping(field, dt);

        Ok(())
    }

    /// Compute elastic acceleration a = (∇·σ + f) / ρ.
    ///
    /// Uses a two-pass stress divergence: first construct the full 6-component
    /// stress tensor from displacements and spatially-varying Lamé parameters,
    /// then differentiate to obtain ∇·σ.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn compute_acceleration(
        &self,
        field: &ElasticWaveField,
        ax: &mut Array3<f64>,
        ay: &mut Array3<f64>,
        az: &mut Array3<f64>,
        body_force: Option<&ElasticBodyForceConfig>,
        time: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.ux.dim();
        let (div_x, div_y, div_z) = stress_divergence(self.grid, self.lambda, self.mu, field);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let force = if let Some(bf) = body_force {
                        body_force::evaluate(self.grid, bf, i, j, k, time)?
                    } else {
                        [0.0, 0.0, 0.0]
                    };

                    let rho = self.density[[i, j, k]];
                    ax[[i, j, k]] = (div_x[[i, j, k]] + force[0]) / rho;
                    ay[[i, j, k]] = (div_y[[i, j, k]] + force[1]) / rho;
                    az[[i, j, k]] = (div_z[[i, j, k]] + force[2]) / rho;
                }
            }
        }

        Ok(())
    }

    fn compute_acceleration_with_body_forces(
        &self,
        field: &ElasticWaveField,
        ax: &mut Array3<f64>,
        ay: &mut Array3<f64>,
        az: &mut Array3<f64>,
        body_forces: &[ElasticBodyForceConfig],
        time: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.ux.dim();
        let (div_x, div_y, div_z) = stress_divergence(self.grid, self.lambda, self.mu, field);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut force = [0.0, 0.0, 0.0];
                    for bf in body_forces {
                        let f = body_force::evaluate(self.grid, bf, i, j, k, time)?;
                        force[0] += f[0];
                        force[1] += f[1];
                        force[2] += f[2];
                    }

                    let rho = self.density[[i, j, k]];
                    ax[[i, j, k]] = (div_x[[i, j, k]] + force[0]) / rho;
                    ay[[i, j, k]] = (div_y[[i, j, k]] + force[1]) / rho;
                    az[[i, j, k]] = (div_z[[i, j, k]] + force[2]) / rho;
                }
            }
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
    pub(crate) fn apply_pml_damping(&self, field: &mut ElasticWaveField, dt: f64) {
        let (nx, ny, nz) = field.vx.dim();
        let sx = self.sigma_x.as_slice().expect("sigma_x contiguous");
        let sy = self.sigma_y.as_slice().expect("sigma_y contiguous");
        let sz = self.sigma_z.as_slice().expect("sigma_z contiguous");

        debug_assert_eq!(sx.len(), nx);
        debug_assert_eq!(sy.len(), ny);
        debug_assert_eq!(sz.len(), nz);

        for (k, sigma_z) in sz.iter().copied().enumerate().take(nz) {
            let ez = (-sigma_z * dt).exp();
            for (j, sigma_y) in sy.iter().copied().enumerate().take(ny) {
                let eyz = ez * (-sigma_y * dt).exp();
                for (i, sigma_x) in sx.iter().copied().enumerate().take(nx) {
                    let d = eyz * (-sigma_x * dt).exp();
                    if d < 1.0 {
                        field.vx[[i, j, k]] *= d;
                        field.vy[[i, j, k]] *= d;
                        field.vz[[i, j, k]] *= d;
                        field.ux[[i, j, k]] *= d;
                        field.uy[[i, j, k]] *= d;
                        field.uz[[i, j, k]] *= d;
                    }
                }
            }
        }
    }

    /// Calculate CFL-limited time step.
    ///
    /// CFL condition for 3D elastic waves: `Δt < Δx / (√3 * c_max)`
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
