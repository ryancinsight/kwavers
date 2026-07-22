use super::RadicalDiffusionSolver;
use crate::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::HashMap;

impl RadicalDiffusionSolver {
    /// Build the logarithmic radial grid `r`J` = R_bubble * exp(j * Δξ)`.
    #[must_use]
    pub fn radial_grid(&self) -> Vec<f64> {
        let n = self.n_points;
        let r_max = self.r_bubble_m * self.r_max_factor;
        let dxi = (r_max / self.r_bubble_m).ln() / (n - 1) as f64;

        (0..n)
            .map(|j| self.r_bubble_m * (j as f64 * dxi).exp())
            .collect()
    }

    /// Initialize species concentrations to zero on the full grid.
    ///
    /// Returns a `Vec<Vec<f64>>` of shape `[n_species][n_points]`.
    #[must_use]
    pub fn zero_concentrations(&self, n_species: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0_f64; self.n_points]; n_species]
    }

    /// Extract the bubble-wall concentrations as a species-indexed map.
    #[must_use]
    pub fn wall_concentrations(
        &self,
        concentrations: &[Vec<f64>],
        species_list: &[ROSSpecies],
    ) -> HashMap<ROSSpecies, f64> {
        species_list
            .iter()
            .zip(concentrations.iter())
            .map(|(&species, concentration)| {
                (species, concentration.first().copied().unwrap_or(0.0))
            })
            .collect()
    }

    /// Diffusion coefficient [m^2/s] for a radical species.
    #[must_use]
    pub fn diffusion_coefficient(species: ROSSpecies) -> f64 {
        species.diffusion_coefficient()
    }
}
