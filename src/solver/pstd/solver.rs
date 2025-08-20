//! Core PSTD solver implementation

use ndarray::{Array3, Zip};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use crate::boundary::Boundary;
use crate::error::KwaversResult;
use super::config::PstdConfig;
use super::spectral_ops::SpectralOperations;

/// PSTD solver state
pub struct PstdSolver {
    config: PstdConfig,
    spectral: SpectralOperations,
    pressure: Array3<f64>,
    velocity_x: Array3<f64>,
    velocity_y: Array3<f64>,
    velocity_z: Array3<f64>,
}

impl PstdSolver {
    /// Create new PSTD solver
    pub fn new(config: PstdConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate()?;
        
        let spectral = SpectralOperations::new(grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        
        Ok(Self {
            config,
            spectral,
            pressure: Array3::zeros(shape),
            velocity_x: Array3::zeros(shape),
            velocity_y: Array3::zeros(shape),
            velocity_z: Array3::zeros(shape),
        })
    }
    
    /// Advance one time step
    pub fn step(
        &mut self,
        medium: &dyn Medium,
        source: &dyn Source,
        boundary: &dyn Boundary,
        grid: &Grid,
        time: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Update pressure: ∂p/∂t = -ρc²∇·v
        self.update_pressure(medium, source, grid, time, dt)?;
        
        // Apply boundary conditions to pressure
        boundary.apply(&mut self.pressure, grid, time)?;
        
        // Update velocity: ∂v/∂t = -∇p/ρ
        self.update_velocity(medium, grid, dt)?;
        
        // Apply boundary conditions to velocity
        boundary.apply(&mut self.velocity_x, grid, time)?;
        boundary.apply(&mut self.velocity_y, grid, time)?;
        boundary.apply(&mut self.velocity_z, grid, time)?;
        
        Ok(())
    }
    
    /// Update pressure field
    pub fn update_pressure(
        &mut self,
        medium: &dyn Medium,
        source: &dyn Source,
        grid: &Grid,
        time: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute divergence of velocity in spectral space
        let (dvx_dx, _, _) = self.spectral.compute_gradient(&self.velocity_x, grid)?;
        let (_, dvy_dy, _) = self.spectral.compute_gradient(&self.velocity_y, grid)?;
        let (_, _, dvz_dz) = self.spectral.compute_gradient(&self.velocity_z, grid)?;
        
        let mut divergence = Array3::zeros(self.pressure.dim());
        Zip::from(&mut divergence)
            .and(&dvx_dx)
            .and(&dvy_dy)
            .and(&dvz_dz)
            .for_each(|d, &dx, &dy, &dz| {
                *d = dx + dy + dz;
            });
        
        // Update pressure with source term
        let source_term = source.evaluate(grid, time);
        
        Zip::indexed(&mut self.pressure)
            .and(&divergence)
            .and(&source_term)
            .for_each(|(i, j, k), p, &div, &s| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                
                *p -= dt * (rho * c * c * div - s);
            });
        
        Ok(())
    }
    
    /// Update velocity field
    pub fn update_velocity(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute pressure gradient
        let (grad_x, grad_y, grad_z) = self.spectral.compute_gradient(&self.pressure, grid)?;
        
        // Update velocity components
        Zip::indexed(&mut self.velocity_x)
            .and(&grad_x)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
        
        Zip::indexed(&mut self.velocity_y)
            .and(&grad_y)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
        
        Zip::indexed(&mut self.velocity_z)
            .and(&grad_z)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
        
        Ok(())
    }
    
    /// Get pressure field
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }
    
    /// Get velocity fields
    pub fn velocity(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.velocity_x, &self.velocity_y, &self.velocity_z)
    }
    
    /// Compute divergence of velocity field
    pub fn compute_divergence(&self, vx: &Array3<f64>, vy: &Array3<f64>, vz: &Array3<f64>) -> Array3<f64> {
        self.spectral.compute_divergence(vx, vy, vz)
    }
}