//! Westervelt equation solver for nonlinear acoustics

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::Array3;

use super::config::AcousticSolverConfig;
use super::solver::AcousticSolver;

/// Westervelt equation solver
#[derive(Debug)]
pub struct WesterveltSolver {
    config: AcousticSolverConfig,
    #[allow(dead_code)]
    grid: Grid,
    prev_pressure: Option<Array3<f64>>,
    pressure_history: Option<Array3<f64>>,
}

impl WesterveltSolver {
    /// Create a new Westervelt solver
    pub fn new(config: AcousticSolverConfig, grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            config,
            grid: grid.clone(),
            prev_pressure: None,
            pressure_history: None,
        })
    }
}

impl AcousticSolver for WesterveltSolver {
    fn update(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Store pressure history for nonlinear term computation
        if let Some(ref mut prev) = self.prev_pressure {
            if let Some(ref mut history) = self.pressure_history {
                history.assign(prev);
            } else {
                self.pressure_history = Some(prev.clone());
            }
            prev.assign(pressure);
        } else {
            self.prev_pressure = Some(pressure.clone());
        }

        // Add source term
        *pressure = &*pressure + source_term;

        // Compute nonlinear term: β/(2ρ₀c₀⁴) * ∂²p²/∂t²
        // Reference: Hamilton & Blackstock, "Nonlinear Acoustics" (1998)
        if let (Some(prev), Some(history)) = (&self.prev_pressure, &self.pressure_history) {
            let beta = self.config.nonlinearity_scaling;

            // Estimate ∂²p²/∂t² using finite differences
            let p_squared_current = pressure.mapv(|x| x * x);
            let p_squared_prev = prev.mapv(|x| x * x);
            let p_squared_history = history.mapv(|x| x * x);

            let d2p2_dt2 =
                (&p_squared_current - &p_squared_prev * 2.0 + &p_squared_history) / (dt * dt);

            // Apply nonlinear correction
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;

                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        let c = crate::medium::sound_speed_at(medium, x, y, z, grid);

                        let nonlinear_factor = beta / (2.0 * rho * c.powi(4));
                        pressure[[i, j, k]] += dt * nonlinear_factor * d2p2_dt2[[i, j, k]];
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "Westervelt Nonlinear Acoustic Solver"
    }

    fn check_stability(&self, dt: f64, grid: &Grid, max_sound_speed: f64) -> KwaversResult<()> {
        // Westervelt equation requires stricter CFL for stability
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_sound_speed * dt / dx_min;
        let max_cfl = self.config.cfl_safety_factor * 0.8; // 80% of linear CFL

        if cfl > max_cfl {
            return Err(crate::error::ValidationError::OutOfRange {
                value: cfl,
                min: 0.0,
                max: max_cfl,
            } /* field: CFL (Westervelt) */
            .into());
        }

        Ok(())
    }
}
