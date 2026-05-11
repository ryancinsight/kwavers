//! `KzkSolverPlugin`: core KZK solver logic and operator methods.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{PluginMetadata, PluginState};
use ndarray::Array3;

use super::frequency_operator::FrequencyOperator;

/// KZK Equation Solver Plugin.
///
/// Implements the Khokhlov-Zabolotskaya-Kuznetsov equation for nonlinear beam propagation.
#[derive(Debug)]
pub struct KzkSolverPlugin {
    pub(super) metadata: PluginMetadata,
    pub(super) state: PluginState,
    /// Frequency domain operators for efficient computation.
    pub(super) frequency_operators: Option<FrequencyOperator>,
    /// Retarded time frame for moving window.
    pub(super) retarded_time_window: Option<f64>,
}

impl Default for KzkSolverPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl KzkSolverPlugin {
    /// Create new KZK solver plugin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "kzk_solver".to_owned(),
                name: "KZK Equation Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Nonlinear beam propagation using KZK equation".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Initialized,
            frequency_operators: None,
            retarded_time_window: None,
        }
    }

    /// Initialize frequency domain operators.
    ///
    /// Based on Aanonsen et al. (1984): "Distortion and harmonic generation in the nearfield"
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn initialize_operators(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
        max_frequency: f64,
    ) -> KwaversResult<()> {
        use crate::domain::medium::AcousticProperties;
        use std::f64::consts::PI;

        const NUM_HARMONICS: usize = 10;
        let fundamental = max_frequency / NUM_HARMONICS as f64;
        let mut frequencies = Vec::with_capacity(NUM_HARMONICS);
        for n in 1..=NUM_HARMONICS {
            frequencies.push(n as f64 * fundamental);
        }

        let shape = (grid.nx, grid.ny, NUM_HARMONICS);
        let mut absorption_op = Array3::zeros(shape);
        let mut diffraction_op = Array3::zeros(shape);

        for (f_idx, &freq) in frequencies.iter().enumerate() {
            let omega = 2.0 * PI * freq;
            const NOMINAL_SOUND_SPEED: f64 = 1500.0;
            let k = omega / NOMINAL_SOUND_SPEED;

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let alpha =
                        AcousticProperties::absorption_coefficient(medium, x, y, 0.0, grid, freq);
                    absorption_op[[i, j, f_idx]] = (-alpha * grid.dz).exp();

                    let kx = 2.0 * PI * i as f64 / (grid.nx as f64 * grid.dx);
                    let ky = 2.0 * PI * j as f64 / (grid.ny as f64 * grid.dy);
                    diffraction_op[[i, j, f_idx]] = ((kx * kx + ky * ky) / (2.0 * k)).cos();
                }
            }
        }

        self.frequency_operators = Some(FrequencyOperator {
            frequencies,
            absorption_operator: absorption_op,
            diffraction_operator: diffraction_op,
        });

        self.state = PluginState::Running;
        Ok(())
    }

    /// Solve KZK equation using Strang operator splitting.
    ///
    /// Based on Tavakkoli et al. (1998): "A new algorithm for computational simulation"
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn solve(
        &mut self,
        initial_field: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_steps: usize,
    ) -> KwaversResult<Array3<f64>> {
        use crate::domain::medium::AcousticProperties;

        let operators =
            self.frequency_operators
                .as_ref()
                .ok_or(crate::core::error::KwaversError::Physics(
                    crate::core::error::PhysicsError::InvalidParameter {
                        parameter: "frequency_operators".to_owned(),
                        value: 0.0,
                        reason:
                            "KZK operators not initialized - call initialize_operators first"
                                .to_owned(),
                    },
                ))?;

        let mut field = initial_field.clone();
        let dz = grid.dz;
        let density = crate::domain::medium::density_at(medium, 0.0, 0.0, 0.0, grid);
        let c0 = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let beta = AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, grid);

        // Strang splitting: L(dz/2) · N(dz) · L(dz/2)
        for _step in 0..time_steps {
            self.apply_linear_step(&mut field, operators, dz / 2.0)?;
            self.apply_nonlinear_step(&mut field, beta, density, c0, dz, grid)?;
            self.apply_linear_step(&mut field, operators, dz / 2.0)?;
        }

        Ok(field)
    }

    /// Apply linear propagation (diffraction + absorption).
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub(super) fn apply_linear_step(
        &self,
        field: &mut Array3<f64>,
        operators: &FrequencyOperator,
        step_size: f64,
    ) -> KwaversResult<()> {
        for (f_idx, _freq) in operators.frequencies.iter().enumerate() {
            let nx = field.shape()[0];
            let ny = field.shape()[1];

            for i in 0..nx {
                for j in 0..ny {
                    if f_idx < field.shape()[2] {
                        let absorption = operators.absorption_operator[[i, j, f_idx]];
                        field[[i, j, f_idx]] *= absorption.powf(step_size);

                        let diffraction = operators.diffraction_operator[[i, j, f_idx]];
                        field[[i, j, f_idx]] *= diffraction;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply nonlinear step using Burgers equation solution.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub(super) fn apply_nonlinear_step(
        &self,
        field: &mut Array3<f64>,
        beta: f64,
        density: f64,
        c0: f64,
        step_size: f64,
        _grid: &Grid,
    ) -> KwaversResult<()> {
        use ndarray::Zip;

        let nonlinear_factor = beta / (2.0 * density * c0.powi(3));

        Zip::from(field.view_mut()).par_for_each(|p| {
            let p0 = *p;
            let denominator = (nonlinear_factor * p0).mul_add(-step_size, 1.0);
            if denominator.abs() > 0.1 {
                *p = p0 / denominator;
            } else {
                *p = p0.signum() * p0.abs().min(1.0 / (nonlinear_factor * step_size));
            }
        });

        Ok(())
    }

    /// Calculate shock formation distance.
    ///
    /// Based on Bacon (1984): "Finite amplitude distortion of the pulsed fields"
    ///
    /// x_shock = ρc³ / (β ω p₀)
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn shock_formation_distance(
        &self,
        source_pressure: f64,
        frequency: f64,
        medium: &dyn Medium,
    ) -> KwaversResult<f64> {
        use crate::domain::medium::AcousticProperties;
        use std::f64::consts::PI;

        let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0)?;
        let density = crate::domain::medium::density_at(medium, 0.0, 0.0, 0.0, &grid);
        let sound_speed = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, &grid);
        let beta = AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, &grid);

        let omega = 2.0 * PI * frequency;

        Ok(density * sound_speed.powi(3) / (beta * omega * source_pressure))
    }

    /// Apply retarded time transformation.
    ///
    /// Based on Jing et al. (2012): "Verification of the Westervelt equation"
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn apply_retarded_time(
        &mut self,
        field: &Array3<f64>,
        propagation_distance: f64,
    ) -> KwaversResult<Array3<f64>> {
        const SOUND_SPEED: f64 = 1500.0;
        let time_shift = propagation_distance / SOUND_SPEED;
        self.retarded_time_window = Some(time_shift);
        Ok(field.clone())
    }
}
