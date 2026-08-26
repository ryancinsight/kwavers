//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod acceleration;
mod body_force;
mod damping;
mod step;

pub(crate) use body_force::PreparedBodyForces;

use super::super::boundary::ElasticSwePMLBoundary;
use kwavers_grid::Grid;
use leto::Array1;

/// Time integration engine for elastic waves.
///
/// Implements a velocity-Verlet scheme with optional body forces and a
/// separable per-axis exponential PML (Collino & Tsogka 2001 §3). Point-force
/// propagation can select a plane-strain stress mode once at its boundary;
/// both dimensional regimes share the same integration and damping logic.
#[derive(Debug)]
pub struct TimeIntegrator<'a> {
    pub(super) grid: &'a Grid,
    pub(super) lambda: &'a leto::Array3<f64>,
    pub(super) mu: &'a leto::Array3<f64>,
    pub(super) density: &'a leto::Array3<f64>,
    /// Reciprocal density when every cell has the same density.
    pub(super) uniform_inverse_density: Option<f64>,
    pub(super) sigma_x: Array1<f64>,
    pub(super) sigma_y: Array1<f64>,
    pub(super) sigma_z: Array1<f64>,
}

impl<'a> TimeIntegrator<'a> {
    /// Create a new time integrator.
    ///
    /// Computes per-axis σ profiles from `pml` at construction; the profiles
    /// do not depend on `dt`, which is determined later from the CFL condition.
    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
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

    /// Calculate CFL-limited time step.
    ///
    /// For 3-D elastic waves, `Δt < Δx / (√3 · c_max)`, where
    /// `c_s = √(μ/ρ)` and `c_p = √((λ+2μ)/ρ)`.
    #[must_use]
    pub fn calculate_stable_timestep(&self, cfl_factor: f64) -> f64 {
        let [nx, ny, nz] = self.lambda.shape();
        let mut max_c = 0.0_f64;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu = self.mu[[i, j, k]];
                    let lambda = self.lambda[[i, j, k]];
                    let density = self.density[[i, j, k]];
                    if density > 0.0 {
                        let shear_speed = (mu / density).sqrt();
                        let pressure_speed = (2.0f64.mul_add(mu, lambda) / density).sqrt();
                        max_c = max_c.max(shear_speed.max(pressure_speed));
                    }
                }
            }
        }

        if max_c <= 0.0 {
            return 0.0;
        }
        let min_spacing = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_dt = min_spacing / (3.0_f64.sqrt() * max_c);
        cfl_dt * cfl_factor
    }
}
