// thermal/bioheat.rs - Pennes bioheat equation solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Zip};

/// Pennes bioheat equation
pub struct PennesEquation {
    blood_perfusion: f64,
    blood_temperature: f64,
    blood_specific_heat: f64,
}

impl PennesEquation {
    /// Create new Pennes equation solver
    pub fn new(perfusion: f64, blood_temp: f64) -> Self {
        Self {
            blood_perfusion: perfusion,
            blood_temperature: blood_temp,
            blood_specific_heat: 3617.0, // J/kg/K for blood
        }
    }

    /// Compute perfusion term: ωb * cb * (T - Ta)
    pub fn perfusion_term(&self, temperature: &Array3<f64>) -> Array3<f64> {
        let mut result = Array3::zeros(temperature.raw_dim());

        Zip::from(&mut result).and(temperature).for_each(|r, &t| {
            *r = self.blood_perfusion * self.blood_specific_heat * (t - self.blood_temperature);
        });

        result
    }
}

/// Bioheat solver for tissue heating
pub struct BioheatSolver {
    equation: PennesEquation,
    metabolic_heat: f64,
}

impl BioheatSolver {
    /// Create new bioheat solver
    pub fn new(perfusion: f64, blood_temp: f64, metabolic_heat: f64) -> Self {
        Self {
            equation: PennesEquation::new(perfusion, blood_temp),
            metabolic_heat,
        }
    }

    /// Solve bioheat equation for one time step
    pub fn solve(
        &self,
        temperature: &mut Array3<f64>,
        heat_source: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Get tissue properties
        let k = medium.thermal_conductivity(0.0, 0.0, 0.0, grid);
        let rho = medium.density(0.0, 0.0, 0.0, grid);
        let cp = medium.specific_heat(0.0, 0.0, 0.0, grid);

        // Compute diffusion term
        let alpha = k / (rho * cp);
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;

        let (nx, ny, nz) = temperature.dim();
        let mut update = Array3::zeros((nx, ny, nz));

        // Diffusion + perfusion + source
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Laplacian
                    let laplacian = (temperature[[i + 1, j, k]] - 2.0 * temperature[[i, j, k]]
                        + temperature[[i - 1, j, k]])
                        / dx2
                        + (temperature[[i, j + 1, k]] - 2.0 * temperature[[i, j, k]]
                            + temperature[[i, j - 1, k]])
                            / dy2
                        + (temperature[[i, j, k + 1]] - 2.0 * temperature[[i, j, k]]
                            + temperature[[i, j, k - 1]])
                            / dz2;

                    // Perfusion cooling
                    let perfusion = self.equation.blood_perfusion
                        * self.equation.blood_specific_heat
                        * (temperature[[i, j, k]] - self.equation.blood_temperature)
                        / (rho * cp);

                    // Total update
                    update[[i, j, k]] = alpha * laplacian - perfusion
                        + (heat_source[[i, j, k]] + self.metabolic_heat) / (rho * cp);
                }
            }
        }

        // Update temperature
        Zip::from(temperature).and(&update).for_each(|t, &u| {
            *t += u * dt;
        });

        Ok(())
    }
}
