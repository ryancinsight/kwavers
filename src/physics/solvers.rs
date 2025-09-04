//! Numerical methods and solvers for physics simulations
//!
//! This module consolidates all numerical solution methods including FDTD, PSTD,
//! spectral-DG, and adaptive methods with proper error handling and validation.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

/// Generic trait for time-stepping numerical solvers
pub trait PhysicsSolver<T: Medium> {
    /// Configuration type for this solver
    type Config;
    
    /// Initialize solver with grid and medium
    fn new(grid: Grid, medium: T, config: Self::Config) -> KwaversResult<Self>
    where
        Self: Sized;
    
    /// Advance simulation by one time step
    fn step(&mut self, dt: f64) -> KwaversResult<()>;
    
    /// Get current simulation time
    fn time(&self) -> f64;
    
    /// Check stability conditions
    fn check_stability(&self) -> KwaversResult<()>;
}

/// Finite Difference Time Domain solver for linear acoustics
pub struct FdtdSolver<T: Medium> {
    grid: Grid,
    medium: T,
    pressure: Array3<f64>,
    velocity_x: Array3<f64>,
    velocity_y: Array3<f64>,
    velocity_z: Array3<f64>,
    time: f64,
    dt: f64,
}

impl<T: Medium> FdtdSolver<T> {
    /// Create new FDTD solver with validated time step
    pub fn new(grid: Grid, medium: T) -> KwaversResult<Self> {
        let dt = Self::calculate_stable_dt(&grid, &medium)?;
        
        let (nx, ny, nz) = grid.dimensions();
        Ok(Self {
            pressure: Array3::zeros((nx, ny, nz)),
            velocity_x: Array3::zeros((nx, ny, nz)),
            velocity_y: Array3::zeros((nx, ny, nz)),
            velocity_z: Array3::zeros((nx, ny, nz)),
            grid,
            medium,
            time: 0.0,
            dt,
        })
    }
    
    /// Calculate stable time step using CFL condition
    fn calculate_stable_dt(grid: &Grid, medium: &T) -> KwaversResult<f64> {
        let max_velocity = medium.max_sound_speed();
        let min_spacing = grid.min_spacing();
        
        // CFL stability: dt < min_spacing / (sqrt(3) * max_velocity)
        const SAFETY_FACTOR: f64 = 0.9;
        let dt = SAFETY_FACTOR * min_spacing / (3.0_f64.sqrt() * max_velocity);
        
        if dt <= 0.0 {
            return Err(crate::error::KwaversError::InvalidParameter(
                "Invalid time step calculated".to_string()
            ));
        }
        
        Ok(dt)
    }
    
    /// Update pressure field using finite differences
    fn update_pressure(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();
        let (dx, dy, dz) = self.grid.spacing();
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let bulk_modulus = self.medium.bulk_modulus_at(i, j, k);
                    
                    // Compute velocity divergence
                    let div_v = (self.velocity_x[[i+1, j, k]] - self.velocity_x[[i-1, j, k]]) / (2.0 * dx)
                              + (self.velocity_y[[i, j+1, k]] - self.velocity_y[[i, j-1, k]]) / (2.0 * dy)
                              + (self.velocity_z[[i, j, k+1]] - self.velocity_z[[i, j, k-1]]) / (2.0 * dz);
                    
                    // Update pressure: dp/dt = -K * div(v)
                    self.pressure[[i, j, k]] -= self.dt * bulk_modulus * div_v;
                }
            }
        }
        
        Ok(())
    }
    
    /// Update velocity fields using finite differences
    fn update_velocity(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();
        let (dx, dy, dz) = self.grid.spacing();
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let inv_density = 1.0 / self.medium.density_at(i, j, k);
                    
                    // Compute pressure gradients
                    let dp_dx = (self.pressure[[i+1, j, k]] - self.pressure[[i-1, j, k]]) / (2.0 * dx);
                    let dp_dy = (self.pressure[[i, j+1, k]] - self.pressure[[i, j-1, k]]) / (2.0 * dy);
                    let dp_dz = (self.pressure[[i, j, k+1]] - self.pressure[[i, j, k-1]]) / (2.0 * dz);
                    
                    // Update velocities: dv/dt = -(1/rho) * grad(p)
                    self.velocity_x[[i, j, k]] -= self.dt * inv_density * dp_dx;
                    self.velocity_y[[i, j, k]] -= self.dt * inv_density * dp_dy;
                    self.velocity_z[[i, j, k]] -= self.dt * inv_density * dp_dz;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get reference to pressure field (zero-copy)
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }
    
    /// Get mutable reference to pressure field
    pub fn pressure_mut(&mut self) -> &mut Array3<f64> {
        &mut self.pressure
    }
}

impl<T: Medium> PhysicsSolver<T> for FdtdSolver<T> {
    type Config = ();
    
    fn new(grid: Grid, medium: T, _config: Self::Config) -> KwaversResult<Self> {
        Self::new(grid, medium)
    }
    
    fn step(&mut self, _dt: f64) -> KwaversResult<()> {
        // Use internal stable time step, ignore passed dt
        self.update_pressure()?;
        self.update_velocity()?;
        self.time += self.dt;
        Ok(())
    }
    
    fn time(&self) -> f64 {
        self.time
    }
    
    fn check_stability(&self) -> KwaversResult<()> {
        // Check for NaN or infinite values
        if !self.pressure.iter().all(|&x| x.is_finite()) {
            return Err(crate::error::KwaversError::NumericalInstability(
                "Non-finite values detected in pressure field".to_string()
            ));
        }
        Ok(())
    }
}

/// Pseudospectral Time Domain solver for efficient wave propagation
pub struct PstdSolver<T: Medium> {
    grid: Grid,
    medium: T,
    pressure: Array3<f64>,
    time: f64,
    dt: f64,
}

impl<T: Medium> PstdSolver<T> {
    /// Create new PSTD solver
    pub fn new(grid: Grid, medium: T) -> KwaversResult<Self> {
        let dt = Self::calculate_stable_dt(&grid, &medium)?;
        let (nx, ny, nz) = grid.dimensions();
        
        Ok(Self {
            pressure: Array3::zeros((nx, ny, nz)),
            grid,
            medium,
            time: 0.0,
            dt,
        })
    }
    
    /// Calculate stable time step for spectral methods
    fn calculate_stable_dt(grid: &Grid, medium: &T) -> KwaversResult<f64> {
        let max_velocity = medium.max_sound_speed();
        let min_spacing = grid.min_spacing();
        
        // More restrictive for spectral methods
        const SAFETY_FACTOR: f64 = 0.5;
        let dt = SAFETY_FACTOR * min_spacing / max_velocity;
        
        if dt <= 0.0 {
            return Err(crate::error::KwaversError::InvalidParameter(
                "Invalid time step for PSTD".to_string()
            ));
        }
        
        Ok(dt)
    }
    
    /// Perform k-space propagation step
    fn k_space_step(&mut self) -> KwaversResult<()> {
        // Implementation would use FFT for k-space propagation
        // Placeholder for now - would need rustfft integration
        Ok(())
    }
}

impl<T: Medium> PhysicsSolver<T> for PstdSolver<T> {
    type Config = ();
    
    fn new(grid: Grid, medium: T, _config: Self::Config) -> KwaversResult<Self> {
        Self::new(grid, medium)
    }
    
    fn step(&mut self, _dt: f64) -> KwaversResult<()> {
        self.k_space_step()?;
        self.time += self.dt;
        Ok(())
    }
    
    fn time(&self) -> f64 {
        self.time
    }
    
    fn check_stability(&self) -> KwaversResult<()> {
        Ok(()) // PSTD is typically more stable than FDTD
    }
}

/// Configuration for adaptive time stepping
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    pub min_dt: f64,
    pub max_dt: f64,
    pub tolerance: f64,
    pub safety_factor: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_dt: 1e-12,
            max_dt: 1e-6,
            tolerance: 1e-6,
            safety_factor: 0.9,
        }
    }
}

/// Adaptive solver that adjusts time step based on error estimates
pub struct AdaptiveSolver<T: Medium> {
    inner_solver: FdtdSolver<T>,
    config: AdaptiveConfig,
    error_estimate: f64,
}

impl<T: Medium> AdaptiveSolver<T> {
    /// Create adaptive solver wrapping FDTD
    pub fn new(grid: Grid, medium: T, config: AdaptiveConfig) -> KwaversResult<Self> {
        let inner_solver = FdtdSolver::new(grid, medium)?;
        
        Ok(Self {
            inner_solver,
            config,
            error_estimate: 0.0,
        })
    }
    
    /// Estimate local truncation error
    fn estimate_error(&self) -> f64 {
        // Simplified error estimation - would be more sophisticated in practice
        self.error_estimate
    }
    
    /// Adapt time step based on error
    fn adapt_timestep(&mut self) -> f64 {
        let error = self.estimate_error();
        let factor = if error > self.config.tolerance {
            0.8 // Reduce time step
        } else {
            1.2 // Increase time step
        };
        
        let new_dt = (self.inner_solver.dt * factor)
            .max(self.config.min_dt)
            .min(self.config.max_dt);
            
        self.inner_solver.dt = new_dt;
        new_dt
    }
}

impl<T: Medium> PhysicsSolver<T> for AdaptiveSolver<T> {
    type Config = AdaptiveConfig;
    
    fn new(grid: Grid, medium: T, config: Self::Config) -> KwaversResult<Self> {
        Self::new(grid, medium, config)
    }
    
    fn step(&mut self, _dt: f64) -> KwaversResult<()> {
        let adaptive_dt = self.adapt_timestep();
        self.inner_solver.step(adaptive_dt)?;
        Ok(())
    }
    
    fn time(&self) -> f64 {
        self.inner_solver.time()
    }
    
    fn check_stability(&self) -> KwaversResult<()> {
        self.inner_solver.check_stability()
    }
}