//! Time integration schemes for elastic wave propagation
//!
//! Implements velocity-Verlet (leapfrog) time integration for the elastic wave equation.
//!
//! ## Mathematical Background
//!
//! The elastic wave equation in first-order form:
//! ```text
//! ∂u/∂t = v
//! ρ ∂v/∂t = ∇·σ + f
//! ```
//!
//! Where:
//! - `u`: Displacement vector (m)
//! - `v`: Velocity vector (m/s)
//! - `σ`: Stress tensor (Pa)
//! - `f`: Body force (N/m³)
//! - `ρ`: Density (kg/m³)
//!
//! ## Velocity-Verlet Scheme
//!
//! Second-order accurate symplectic integrator:
//! ```text
//! v(t+Δt/2) = v(t) + (Δt/2) * a(t)
//! u(t+Δt)   = u(t) + Δt * v(t+Δt/2)
//! v(t+Δt)   = v(t+Δt/2) + (Δt/2) * a(t+Δt)
//! ```
//!
//! Where acceleration: `a = (∇·σ + f) / ρ`
//!
//! ## Energy Conservation
//!
//! The scheme conserves total energy (kinetic + elastic) in the absence of damping:
//! ```text
//! E = ∫(½ρv² + ½σ:ε) dV = const
//! ```
//!
//! ## Stability
//!
//! CFL condition for 3D elastic waves:
//! ```text
//! Δt < Δx / (√3 * c_max)
//! ```
//!
//! Where `c_max = max(c_p, c_s)` is the maximum wave speed.
//!
//! ## References
//!
//! - Verlet, L. (1967). "Computer experiments on classical fluids."
//!   *Physical Review*, 159(1), 98.
//! - Swope, W. C., et al. (1982). "A computer simulation method for the calculation
//!   of equilibrium constants." *J. Chem. Phys.*, 76(1), 637-649.

use super::stress::StressDerivatives;
use super::types::{ElasticBodyForceConfig, ElasticWaveField};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Time integration engine for elastic waves
///
/// Implements velocity-Verlet scheme with optional body forces and PML damping.
#[derive(Debug)]
pub struct TimeIntegrator<'a> {
    grid: &'a Grid,
    lambda: &'a Array3<f64>,
    mu: &'a Array3<f64>,
    density: &'a Array3<f64>,
    pml_sigma: &'a Array3<f64>,
    stress_calc: StressDerivatives<'a>,
}

impl<'a> TimeIntegrator<'a> {
    /// Create new time integrator
    #[must_use]
    pub fn new(
        grid: &'a Grid,
        lambda: &'a Array3<f64>,
        mu: &'a Array3<f64>,
        density: &'a Array3<f64>,
        pml_sigma: &'a Array3<f64>,
    ) -> Self {
        Self {
            grid,
            lambda,
            mu,
            density,
            pml_sigma,
            stress_calc: StressDerivatives::new(grid),
        }
    }

    /// Perform single time step with velocity-Verlet integration
    ///
    /// ## Arguments
    /// - `field`: Current wave field state (modified in-place)
    /// - `dt`: Time step size (seconds)
    /// - `body_force`: Optional body force configuration
    ///
    /// ## Algorithm
    /// 1. Half-step velocity update: v(t+Δt/2) = v(t) + (Δt/2) * a(t)
    /// 2. Full-step displacement: u(t+Δt) = u(t) + Δt * v(t+Δt/2)
    /// 3. Half-step velocity update: v(t+Δt) = v(t+Δt/2) + (Δt/2) * a(t+Δt)
    /// 4. Apply PML damping
    pub fn step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.ux.dim();

        // Compute initial acceleration
        let mut ax = Array3::<f64>::zeros((nx, ny, nz));
        let mut ay = Array3::<f64>::zeros((nx, ny, nz));
        let mut az = Array3::<f64>::zeros((nx, ny, nz));

        self.compute_acceleration(field, &mut ax, &mut ay, &mut az, body_force, field.time)?;

        // Half-step velocity update: v(t+Δt/2) = v(t) + (Δt/2) * a(t)
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

        // Full-step displacement: u(t+Δt) = u(t) + Δt * v(t+Δt/2)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }

        // Recompute acceleration at new time
        let new_time = field.time + dt;
        self.compute_acceleration(field, &mut ax, &mut ay, &mut az, body_force, new_time)?;

        // Second half-step velocity update: v(t+Δt) = v(t+Δt/2) + (Δt/2) * a(t+Δt)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.vx[[i, j, k]] += dt_half * ax[[i, j, k]];
                    field.vy[[i, j, k]] += dt_half * ay[[i, j, k]];
                    field.vz[[i, j, k]] += dt_half * az[[i, j, k]];
                }
            }
        }

        // Apply PML damping
        self.apply_pml_damping(field, dt);

        Ok(())
    }

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

    /// Compute acceleration from stress divergence and body forces
    ///
    /// ## Formula
    /// ```text
    /// a = (∇·σ + f) / ρ
    /// ```
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

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Compute stress divergence
                    let div = self.stress_calc.stress_divergence(i, j, k, field);

                    // Add body force if present
                    let force = if let Some(bf) = body_force {
                        self.evaluate_body_force(bf, i, j, k, time)?
                    } else {
                        [0.0, 0.0, 0.0]
                    };

                    // Compute acceleration: a = (∇·σ + f) / ρ
                    let rho = self.density[[i, j, k]];
                    ax[[i, j, k]] = (div[0] + force[0]) / rho;
                    ay[[i, j, k]] = (div[1] + force[1]) / rho;
                    az[[i, j, k]] = (div[2] + force[2]) / rho;
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

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let div = self.stress_calc.stress_divergence(i, j, k, field);

                    let mut force = [0.0, 0.0, 0.0];
                    for bf in body_forces {
                        let f = self.evaluate_body_force(bf, i, j, k, time)?;
                        force[0] += f[0];
                        force[1] += f[1];
                        force[2] += f[2];
                    }

                    let rho = self.density[[i, j, k]];
                    ax[[i, j, k]] = (div[0] + force[0]) / rho;
                    ay[[i, j, k]] = (div[1] + force[1]) / rho;
                    az[[i, j, k]] = (div[2] + force[2]) / rho;
                }
            }
        }

        Ok(())
    }

    /// Evaluate body force at a spatial location and time
    ///
    fn evaluate_body_force(
        &self,
        body_force: &ElasticBodyForceConfig,
        i: usize,
        j: usize,
        k: usize,
        time: f64,
    ) -> KwaversResult<[f64; 3]> {
        let x = i as f64 * self.grid.dx;
        let y = j as f64 * self.grid.dy;
        let z = k as f64 * self.grid.dz;

        match body_force {
            ElasticBodyForceConfig::GaussianImpulse {
                center_m,
                sigma_m,
                direction,
                t0_s,
                sigma_t_s,
                impulse_n_per_m3_s,
            } => {
                if !sigma_t_s.is_finite() || *sigma_t_s <= 0.0 {
                    return Ok([0.0, 0.0, 0.0]);
                }

                let dx = x - center_m[0];
                let dy = y - center_m[1];
                let dz = z - center_m[2];

                let sx = sigma_m[0];
                let sy = sigma_m[1];
                let sz = sigma_m[2];
                if !sx.is_finite()
                    || !sy.is_finite()
                    || !sz.is_finite()
                    || sx <= 0.0
                    || sy <= 0.0
                    || sz <= 0.0
                {
                    return Ok([0.0, 0.0, 0.0]);
                }

                let spatial_factor = (-0.5
                    * ((dx / sx) * (dx / sx) + (dy / sy) * (dy / sy) + (dz / sz) * (dz / sz)))
                    .exp();

                let dt = time - *t0_s;
                let temporal_factor = (-(dt * dt) / (2.0 * sigma_t_s * sigma_t_s)).exp()
                    / (sigma_t_s * (2.0 * std::f64::consts::PI).sqrt());

                let dir_norm = (direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2])
                    .sqrt();
                if !dir_norm.is_finite() || dir_norm < 1e-12 {
                    return Ok([0.0, 0.0, 0.0]);
                }

                let scale = impulse_n_per_m3_s * spatial_factor * temporal_factor / dir_norm;
                Ok([
                    scale * direction[0],
                    scale * direction[1],
                    scale * direction[2],
                ])
            }
        }
    }

    /// Apply PML damping to velocity field
    ///
    /// ## Mathematical Model
    /// PML acts as an exponential decay zone:
    /// ```text
    /// v(t+Δt) = v(t) * exp(-σ * Δt)
    /// ```
    ///
    /// Where σ is the PML attenuation coefficient (Np/m).
    fn apply_pml_damping(&self, field: &mut ElasticWaveField, dt: f64) {
        let (nx, ny, nz) = field.vx.dim();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let sigma = self.pml_sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping = (-sigma * dt).exp();
                        field.vx[[i, j, k]] *= damping;
                        field.vy[[i, j, k]] *= damping;
                        field.vz[[i, j, k]] *= damping;
                    }
                }
            }
        }
    }

    /// Calculate CFL-limited time step
    ///
    /// ## CFL Condition
    /// For elastic waves in 3D:
    /// ```text
    /// Δt < Δx / (√3 * c_max)
    /// ```
    ///
    /// Where:
    /// - `c_s = sqrt(μ/ρ)`: Shear wave speed
    /// - `c_p = sqrt((λ+2μ)/ρ)`: Compressional wave speed
    /// - `c_max = max(c_s, c_p)`: Maximum wave speed
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
                        let cp = ((lambda_val + 2.0 * mu_val) / rho_val).sqrt();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_time_integrator_creation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
        let pml_sigma = Array3::<f64>::zeros((10, 10, 10));

        let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);

        let dt = integrator.calculate_stable_timestep(0.5);
        assert!(dt > 0.0);
        assert!(dt < 1e-6); // Should be sub-microsecond for these parameters
    }

    #[test]
    fn test_velocity_verlet_step() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
        let pml_sigma = Array3::<f64>::zeros((10, 10, 10));

        let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);
        let mut field = ElasticWaveField::new(10, 10, 10);

        let dt = integrator.calculate_stable_timestep(0.5);
        let result = integrator.step(&mut field, dt, None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_pml_damping() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let lambda = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let mu = Array3::<f64>::from_elem((10, 10, 10), 1e9);
        let density = Array3::<f64>::from_elem((10, 10, 10), 1000.0);
        let mut pml_sigma = Array3::<f64>::zeros((10, 10, 10));

        // Add PML in boundary region
        pml_sigma[[0, 5, 5]] = 100.0;

        let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml_sigma);
        let mut field = ElasticWaveField::new(10, 10, 10);

        // Set initial velocity
        field.vx[[0, 5, 5]] = 1.0;
        let initial_velocity = field.vx[[0, 5, 5]];

        let dt = 1e-7;
        integrator.apply_pml_damping(&mut field, dt);

        // Velocity should be damped
        assert!(field.vx[[0, 5, 5]] < initial_velocity);
        assert!(field.vx[[0, 5, 5]] > 0.0);
    }
}
