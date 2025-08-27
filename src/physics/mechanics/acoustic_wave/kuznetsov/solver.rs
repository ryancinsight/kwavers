//! Core Kuznetsov equation solver implementation
//!
//! Implements the full Kuznetsov equation for nonlinear acoustic wave propagation:
//! ∇²p - (1/c₀²)∂²p/∂t² = -(β/ρ₀c₀⁴)∂²p²/∂t² - (δ/c₀⁴)∂³p/∂t³ + F

use super::config::KuznetsovConfig;
use super::diffusion::compute_diffusive_term_workspace;
use super::nonlinear::compute_nonlinear_term_workspace;
use super::workspace::KuznetsovWorkspace;
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use ndarray::{Array3, Array4, Zip};

/// Main Kuznetsov wave solver
#[derive(Debug)]
pub struct KuznetsovWave {
    config: KuznetsovConfig,
    grid: Grid,
    workspace: KuznetsovWorkspace,
    nonlinearity_scaling: f64,
    time_step_count: usize,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov wave solver
    pub fn new(config: KuznetsovConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;
        let workspace = KuznetsovWorkspace::new(grid)?;

        Ok(Self {
            config,
            grid: grid.clone(),
            workspace,
            nonlinearity_scaling: 1.0,
            time_step_count: 0,
        })
    }

    /// Compute the right-hand side of the Kuznetsov equation
    fn compute_rhs(
        &mut self,
        pressure: &Array3<f64>,
        source: &dyn Source,
        medium: &dyn Medium,
        t: f64,
        dt: f64,
    ) -> Array3<f64> {
        let mut rhs = Array3::zeros(pressure.dim());

        // Get medium properties at grid center (assuming homogeneous for now)
        use crate::constants::physics::GRID_CENTER_FACTOR;
        let center_x = self.grid.dx * (self.grid.nx as f64) / GRID_CENTER_FACTOR;
        let center_y = self.grid.dy * (self.grid.ny as f64) / GRID_CENTER_FACTOR;
        let center_z = self.grid.dz * (self.grid.nz as f64) / GRID_CENTER_FACTOR;
        let density = medium.density(center_x, center_y, center_z, &self.grid);
        let sound_speed = medium.sound_speed(center_x, center_y, center_z, &self.grid);
        let nonlinearity = self.config.nonlinearity_coefficient;
        let diffusivity = self.config.acoustic_diffusivity;

        // 1. Compute linear term: c₀²∇²p using spectral methods
        self.workspace.spectral_op.compute_laplacian_workspace(
            pressure,
            &mut self.workspace.laplacian,
            &self.grid,
        );

        // Add linear term to RHS
        let c0_squared = sound_speed * sound_speed;
        Zip::from(&mut rhs)
            .and(&self.workspace.laplacian)
            .for_each(|r, &lap| {
                *r += c0_squared * lap;
            });

        // 2. Compute nonlinear term if enabled (based on equation mode)
        let include_nonlinearity = matches!(
            self.config.equation_mode,
            super::config::AcousticEquationMode::FullKuznetsov
                | super::config::AcousticEquationMode::KZK
                | super::config::AcousticEquationMode::Westervelt
        );

        if include_nonlinearity {
            compute_nonlinear_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                dt,
                density,
                sound_speed,
                nonlinearity * self.nonlinearity_scaling,
                &mut self.workspace.nonlinear_term,
            );

            // Add nonlinear term to RHS
            Zip::from(&mut rhs)
                .and(&self.workspace.nonlinear_term)
                .for_each(|r, &nl| {
                    *r += nl;
                });
        }

        // 3. Compute diffusive term if enabled (based on equation mode)
        let include_diffusion = matches!(
            self.config.equation_mode,
            super::config::AcousticEquationMode::FullKuznetsov
                | super::config::AcousticEquationMode::KZK
        );

        if include_diffusion && diffusivity > 0.0 {
            compute_diffusive_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                &self.workspace.pressure_prev3,
                dt,
                sound_speed,
                diffusivity,
                &mut self.workspace.diffusive_term,
            );

            // Add diffusive term to RHS
            Zip::from(&mut rhs)
                .and(&self.workspace.diffusive_term)
                .for_each(|r, &diff| {
                    *r += diff;
                });
        }

        // 4. Add source term
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    let source_term = source.get_source_term(t, x, y, z, &self.grid);
                    rhs[[i, j, k]] += source_term;
                }
            }
        }

        rhs
    }
}

impl AcousticWaveModel for KuznetsovWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        // Extract pressure field (assuming it's at index 0)
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);

        // Update time history for finite difference schemes
        if self.time_step_count > 0 {
            self.workspace.update_time_history(&pressure.to_owned());
        }

        // Compute RHS of Kuznetsov equation
        let rhs = self.compute_rhs(&pressure.to_owned(), source, medium, t, dt);

        // Time integration using explicit Euler (can be upgraded to RK4)
        // ∂²p/∂t² = rhs  =>  p_new = p + dt * p_dot + 0.5 * dt² * rhs
        // Currently using forward Euler for second-order equation
        // This should be replaced with a proper time integration scheme

        Zip::from(&mut pressure).and(&rhs).for_each(|p, &r| {
            *p += dt * dt * r; // Simplified time stepping
        });

        self.time_step_count += 1;
    }

    fn report_performance(&self) {
        println!("KuznetsovWave solver performance:");
        println!("  Configuration: {:?}", self.config.equation_mode);
        println!(
            "  Grid size: {}x{}x{}",
            self.grid.nx, self.grid.ny, self.grid.nz
        );
        println!("  Time steps completed: {}", self.time_step_count);
        println!(
            "  Nonlinearity coefficient: {}",
            self.config.nonlinearity_coefficient
        );
        println!(
            "  Acoustic diffusivity: {:.2e}",
            self.config.acoustic_diffusivity
        );
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}
