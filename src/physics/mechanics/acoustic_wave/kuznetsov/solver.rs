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
#[derive(Debug, Debug))]
pub struct KuznetsovWave {
    config: KuznetsovConfig,
    grid: Grid,
    workspace: KuznetsovWorkspace,
    nonlinearity_scaling: f64,
    time_step_count: usize,
    /// Previous pressure field for leapfrog time integration
    pressure_current: Array3<f64>,
    /// Flag to track if this is the first time step
    first_step: bool,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov wave solver
    pub fn new(config: KuznetsovConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;
        let workspace = KuznetsovWorkspace::new(grid)?;
        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            config,
            grid: grid.clone(),
            workspace,
            nonlinearity_scaling: 1.0,
            time_step_count: 0,
            pressure_current: Array3::zeros(shape),
            first_step: true,
        })
    }

    /// Compute the right-hand side of the Kuznetsov equation
    ///
    /// This method now properly samples medium properties at each grid point
    /// for heterogeneous media support.
    fn compute_rhs(
        &mut self,
        pressure: &Array3<f64>, // Changed to reference to avoid cloning
        source: &dyn Source,
        medium: &dyn Medium,
        t: f64,
        dt: f64,
    ) -> Array3<f64> {
        let mut rhs = Array3::zeros(pressure.dim());

        // 1. Compute linear term: c₀²∇²p using spectral methods
        self.workspace.spectral_op.compute_laplacian_workspace(
            pressure,
            &mut self.workspace.laplacian,
            &self.grid,
        );

        // Process each grid point for heterogeneous media
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    // Get local medium properties at this grid point
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    let local_density = medium.density(x, y, z, &self.grid);
                    let local_sound_speed = medium.sound_speed(x, y, z, &self.grid);
                    let c0_squared = local_sound_speed * local_sound_speed;

                    // Add linear term with local sound speed
                    rhs[[i, j, k] += c0_squared * self.workspace.laplacian[[i, j, k];
                }
            }
        }

        // 2. Compute nonlinear term if enabled (based on equation mode)
        let include_nonlinearity = matches!(
            self.config.equation_mode,
            super::config::AcousticEquationMode::FullKuznetsov
                | super::config::AcousticEquationMode::KZK
                | super::config::AcousticEquationMode::Westervelt
        );

        if include_nonlinearity {
            // For nonlinear term, we still need representative values
            // In a fully heterogeneous implementation, this would also be per-point
            let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
            let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
            let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
            let avg_density = medium.density(center_x, center_y, center_z, &self.grid);
            let avg_sound_speed = medium.sound_speed(center_x, center_y, center_z, &self.grid);

            compute_nonlinear_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                dt,
                avg_density,
                avg_sound_speed,
                self.config.nonlinearity_coefficient * self.nonlinearity_scaling,
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

        if include_diffusion && self.config.acoustic_diffusivity > 0.0 {
            // Similar to nonlinear, use average for now
            let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
            let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
            let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
            let avg_sound_speed = medium.sound_speed(center_x, center_y, center_z, &self.grid);

            compute_diffusive_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                &self.workspace.pressure_prev3,
                dt,
                avg_sound_speed,
                self.config.acoustic_diffusivity,
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
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    let source_term = source.get_source_term(t, x, y, z, &self.grid);
                    rhs[[i, j, k] += source_term;
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
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        // Extract pressure field (assuming it's at index 0)
        let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0);

        // Store current pressure for next iteration
        if self.first_step {
            // First step: use forward Euler to bootstrap the leapfrog scheme
            self.pressure_current.assign(&pressure_field);

            // Compute RHS without cloning
            let rhs = self.compute_rhs(&pressure_field.to_owned(), source, medium, t, dt);

            // Forward Euler for first step: p_next = p_current + dt * velocity + 0.5 * dt^2 * acceleration
            // Since we're starting from rest, velocity = 0
            Zip::from(&mut pressure_field)
                .and(&self.pressure_current)
                .and(&rhs)
                .for_each(|p_next, &p_curr, &accel| {
                    *p_next = p_curr + 0.5 * dt * dt * accel;
                });

            // Update time history
            self.workspace.update_time_history(&self.pressure_current);
            self.first_step = false;
        } else {
            // Leapfrog time integration for second-order wave equation
            // p_next = 2 * p_current - p_previous + dt^2 * rhs

            // Store current pressure before updating
            let p_current = pressure_field.to_owned();

            // Use prev_pressure parameter (which is p_previous in the leapfrog scheme)
            let p_previous = if self.time_step_count > 0 {
                prev_pressure.to_owned()
            } else {
                self.workspace.pressure_prev.clone()
            };

            // Compute RHS using reference to avoid clone
            let rhs = self.compute_rhs(&p_current, source, medium, t, dt);

            // Apply leapfrog scheme
            Zip::from(&mut pressure_field)
                .and(&p_current)
                .and(&p_previous)
                .and(&rhs)
                .for_each(|p_next, &p_curr, &p_prev, &accel| {
                    *p_next = 2.0 * p_curr - p_prev + dt * dt * accel;
                });

            // Update time history for next iteration
            self.workspace.update_time_history(&p_current);
            self.pressure_current.assign(&p_current);
        }

        self.time_step_count += 1;
        Ok(())
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
