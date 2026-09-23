//! `TimeIntegrator` — velocity-Verlet time integration for elastic waves.

mod acceleration;
mod body_force;
mod damping;
mod step;

#[cfg(test)]
mod phase_split;

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
    /// `(λ, μ)` when the medium holds one pair everywhere.
    pub(super) uniform_lame: Option<(f64, f64)>,
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
        let uniform_inverse_density = uniform_value(density.view()).map(f64::recip);
        let uniform_lame = uniform_value(lambda.view()).zip(uniform_value(mu.view()));
        Self {
            grid,
            lambda,
            mu,
            density,
            uniform_inverse_density,
            uniform_lame,
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
        calculate_stable_timestep(self.grid, self.lambda, self.mu, self.density, cfl_factor)
    }
}

/// The value every element of `field` holds, or `None` when two differ or
/// the field is empty. Reads in logical order, so any layout serves.
fn uniform_value(field: leto::ArrayView3<'_, f64>) -> Option<f64> {
    let mut values = field.iter().copied();
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

/// Calculate the CFL-limited time step without constructing PML profiles.
///
/// Propagation preflight uses this scan before any simulation allocation. The
/// [`TimeIntegrator`] method delegates here so the CFL formula has one owner.
#[must_use]
pub(crate) fn calculate_stable_timestep(
    grid: &Grid,
    lambda: &leto::Array3<f64>,
    mu: &leto::Array3<f64>,
    density: &leto::Array3<f64>,
    cfl_factor: f64,
) -> f64 {
    let mut max_c = 0.0_f64;
    for ((&mu, &lambda), &density) in mu.iter().zip(lambda.iter()).zip(density.iter()) {
        if density > 0.0 {
            let shear_speed = (mu / density).sqrt();
            let pressure_speed = (2.0f64.mul_add(mu, lambda) / density).sqrt();
            max_c = max_c.max(shear_speed.max(pressure_speed));
        }
    }

    if max_c <= 0.0 {
        return 0.0;
    }
    let min_spacing = grid.dx.min(grid.dy).min(grid.dz);
    let cfl_dt = min_spacing / (3.0_f64.sqrt() * max_c);
    cfl_dt * cfl_factor
}

#[cfg(test)]
mod tests {
    use super::uniform_value;
    use leto::Array3;

    /// One value everywhere is found in any layout; a single differing
    /// element, however late, rules it out; an empty field has none.
    #[test]
    fn a_field_is_uniform_only_when_every_element_matches() {
        let constant = Array3::from_elem((4, 3, 5), 2.5e9);
        assert_eq!(uniform_value(constant.view()), Some(2.5e9));
        let transposed = constant.view().transpose([2, 1, 0]).expect("a permutation");
        assert!(
            transposed.as_slice().is_none(),
            "the case needs a strided view"
        );
        assert_eq!(uniform_value(transposed), Some(2.5e9));
        let mut late = constant.clone();
        late[[3, 2, 4]] = f64::from_bits(2.5e9_f64.to_bits() + 1);
        assert_eq!(uniform_value(late.view()), None);
        assert_eq!(
            uniform_value(Array3::from_elem((0, 3, 5), 1.0).view()),
            None
        );
    }
}
