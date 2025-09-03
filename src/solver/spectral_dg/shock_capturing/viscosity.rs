//! Artificial viscosity methods for shock stabilization
//!
//! Implements various artificial viscosity approaches to stabilize shocks in spectral methods.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, Array4, Axis};

use crate::constants::numerical::{
    LINEAR_VISCOSITY_COEFF, MAX_VISCOSITY_LIMIT, QUADRATIC_VISCOSITY_COEFF,
    VON_NEUMANN_RICHTMYER_COEFF,
};

/// Artificial viscosity for shock stabilization
#[derive(Debug, Clone)]
pub struct ArtificialViscosity {
    /// Von Neumann-Richtmyer coefficient
    c_vnr: f64,
    /// Linear viscosity coefficient
    c_linear: f64,
    /// Quadratic viscosity coefficient
    c_quadratic: f64,
    /// Maximum viscosity limit
    max_viscosity: f64,
}

impl Default for ArtificialViscosity {
    fn default() -> Self {
        Self {
            c_vnr: VON_NEUMANN_RICHTMYER_COEFF,
            c_linear: LINEAR_VISCOSITY_COEFF,
            c_quadratic: QUADRATIC_VISCOSITY_COEFF,
            max_viscosity: MAX_VISCOSITY_LIMIT,
        }
    }
}

impl ArtificialViscosity {
    /// Create new artificial viscosity with custom coefficients
    #[must_use]
    pub fn new(c_vnr: f64, c_linear: f64, c_quadratic: f64, max_viscosity: f64) -> Self {
        Self {
            c_vnr,
            c_linear,
            c_quadratic,
            max_viscosity,
        }
    }

    /// Compute artificial viscosity coefficient
    pub fn compute_viscosity(
        &self,
        velocity: &Array4<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        shock_indicator: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (_, nx, ny, nz) = velocity.dim();
        let mut viscosity = Array3::zeros((nx, ny, nz));

        // Extract velocity components
        let vx = velocity.index_axis(Axis(0), 0);
        let vy = velocity.index_axis(Axis(0), 1);
        let vz = velocity.index_axis(Axis(0), 2);

        let dx = grid.dx.min(grid.dy).min(grid.dz);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Only apply where shocks are detected
                    if shock_indicator[[i, j, k]] > 0.1 {
                        // Compute velocity divergence
                        let div_v = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * grid.dx)
                            + (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * grid.dy)
                            + (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * grid.dz);

                        if div_v < 0.0 {
                            // Compression - shock forming
                            let c = sound_speed[[i, j, k]];
                            let rho = density[[i, j, k]];

                            // Von Neumann-Richtmyer viscosity
                            let q_vnr = self.c_vnr * rho * dx * dx * div_v.powi(2);

                            // Linear and quadratic terms
                            let q_linear = self.c_linear * rho * c * dx * div_v.abs();
                            let q_quadratic = self.c_quadratic * rho * dx * dx * div_v.powi(2) / c;

                            // Total viscosity with shock indicator weighting
                            let q_total =
                                (q_vnr + q_linear + q_quadratic) * shock_indicator[[i, j, k]];

                            // Limit maximum viscosity to prevent over-dissipation
                            viscosity[[i, j, k]] = q_total.min(self.max_viscosity * rho * c * c);
                        }
                    }
                }
            }
        }

        Ok(viscosity)
    }

    /// Apply viscous flux to momentum equation
    pub fn apply_viscous_flux(
        &self,
        momentum: &mut Array4<f64>,
        viscosity: &Array3<f64>,
        velocity: &Array4<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let (_, nx, ny, nz) = momentum.dim();

        // Apply viscous stress tensor
        for component in 0..3 {
            let mut momentum_component = momentum.index_axis_mut(Axis(0), component);
            let velocity_component = velocity.index_axis(Axis(0), component);

            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        // Compute viscous stress gradients
                        let visc_flux_x = (viscosity[[i + 1, j, k]]
                            * (velocity_component[[i + 1, j, k]] - velocity_component[[i, j, k]])
                            - viscosity[[i - 1, j, k]]
                                * (velocity_component[[i, j, k]]
                                    - velocity_component[[i - 1, j, k]]))
                            / (grid.dx * grid.dx);

                        let visc_flux_y = (viscosity[[i, j + 1, k]]
                            * (velocity_component[[i, j + 1, k]] - velocity_component[[i, j, k]])
                            - viscosity[[i, j - 1, k]]
                                * (velocity_component[[i, j, k]]
                                    - velocity_component[[i, j - 1, k]]))
                            / (grid.dy * grid.dy);

                        let visc_flux_z = (viscosity[[i, j, k + 1]]
                            * (velocity_component[[i, j, k + 1]] - velocity_component[[i, j, k]])
                            - viscosity[[i, j, k - 1]]
                                * (velocity_component[[i, j, k]]
                                    - velocity_component[[i, j, k - 1]]))
                            / (grid.dz * grid.dz);

                        // Update momentum with viscous flux
                        momentum_component[[i, j, k]] +=
                            dt * (visc_flux_x + visc_flux_y + visc_flux_z);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artificial_viscosity_compression() {
        let visc = ArtificialViscosity::default();
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);

        // Create test data with compression
        let mut velocity = Array4::<f64>::zeros((3, 10, 10, 10));
        let density = Array3::from_elem((10, 10, 10), 1.0);
        let sound_speed = Array3::from_elem((10, 10, 10), 340.0);
        let mut shock_indicator = Array3::zeros((10, 10, 10));

        // Add converging velocity field (compression)
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    velocity[[0, i, j, k]] = -(i as f64 - 5.0) * 10.0; // Converging in x
                    if i == 5 {
                        shock_indicator[[i, j, k]] = 1.0; // Shock at center
                    }
                }
            }
        }

        let viscosity = visc
            .compute_viscosity(&velocity, &density, &sound_speed, &shock_indicator, &grid)
            .unwrap();

        // Should have non-zero viscosity at compression region
        assert!(viscosity[[5, 5, 5]] > 0.0);
        // Should have zero viscosity away from shock
        assert_eq!(viscosity[[0, 0, 0]], 0.0);
    }
}
