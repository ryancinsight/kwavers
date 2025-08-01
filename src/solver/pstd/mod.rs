//! Pseudo-Spectral Time Domain (PSTD) solver
//! 
//! This module implements the PSTD method for solving acoustic wave equations
//! with high accuracy and minimal numerical dispersion.
//! 
//! # Design Principles
//! - SOLID: Single responsibility for spectral wave propagation
//! - CUPID: Composable with other solvers via plugin architecture
//! - KISS: Simple interface hiding complex spectral operations
//! - DRY: Reuses spectral utilities from utils module

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::utils::{fft_3d, ifft_3d};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginContext, PluginState};
use crate::physics::composable::FieldType;
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex;
use std::f64::consts::PI;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};

/// PSTD solver configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PstdConfig {
    /// Enable k-space correction for improved accuracy
    pub k_space_correction: bool,
    /// Order of k-space correction (typically 2-8)
    pub k_space_order: usize,
    /// Apply anti-aliasing filter (2/3 rule)
    pub anti_aliasing: bool,
    /// Stencil size for PML regions (if using finite differences there)
    pub pml_stencil_size: usize,
    /// CFL safety factor (typically 0.3-0.5 for PSTD)
    pub cfl_factor: f64,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: true,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
        }
    }
}

/// PSTD solver for acoustic wave propagation
pub struct PstdSolver {
    /// Configuration
    config: PstdConfig,
    /// Grid reference
    grid: Grid,
    /// Wavenumber arrays
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    /// K-squared array
    k_squared: Array3<f64>,
    /// Anti-aliasing filter
    anti_alias_filter: Option<Array3<f64>>,
    /// K-space correction
    kappa: Option<Array3<f64>>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Workspace arrays to avoid allocations
    workspace_real: Array3<f64>,
    workspace_complex: Array3<Complex<f64>>,
}

impl PstdSolver {
    /// Create a new PSTD solver
    pub fn new(config: PstdConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing PSTD solver with config: {:?}", config);
        
        // Validate configuration
        if config.k_space_order % 2 != 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "k_space_order".to_string(),
                value: config.k_space_order.to_string(),
                constraint: "must be even (2, 4, 6, or 8)".to_string(),
            }));
        }
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Initialize wavenumber arrays
        let (kx, ky, kz) = Self::compute_wavenumbers(grid);
        
        // Compute k-squared
        let k_squared = &kx * &kx + &ky * &ky + &kz * &kz;
        
        // Create anti-aliasing filter if enabled
        let filter = if config.anti_aliasing {
            Some(Self::create_anti_aliasing_filter(&kx, &ky, &kz, grid))
        } else {
            None
        };
        
        // Create k-space correction if enabled
        let kappa = if config.k_space_correction {
            Some(Self::compute_k_space_correction(&k_squared, grid, config.k_space_order))
        } else {
            None
        };
        
        Ok(Self {
            config,
            grid: grid.clone(),
            kx,
            ky,
            kz,
            k_squared,
            anti_alias_filter: filter,
            kappa,
            metrics: HashMap::new(),
            workspace_real: Array3::zeros((nx, ny, nz)),
            workspace_complex: Array3::zeros((nx, ny, nz)),
        })
    }
    
    /// Compute wavenumber arrays for FFT
    fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut kx = Array3::zeros((nx, ny, nz));
        let mut ky = Array3::zeros((nx, ny, nz));
        let mut kz = Array3::zeros((nx, ny, nz));
        
        // Compute 1D wavenumbers
        let kx_1d: Vec<f64> = (0..nx).map(|i| {
            if i <= nx / 2 {
                2.0 * PI * i as f64 / (nx as f64 * grid.dx)
            } else {
                2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * grid.dx)
            }
        }).collect();
        
        let ky_1d: Vec<f64> = (0..ny).map(|j| {
            if j <= ny / 2 {
                2.0 * PI * j as f64 / (ny as f64 * grid.dy)
            } else {
                2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * grid.dy)
            }
        }).collect();
        
        let kz_1d: Vec<f64> = (0..nz).map(|k| {
            if k <= nz / 2 {
                2.0 * PI * k as f64 / (nz as f64 * grid.dz)
            } else {
                2.0 * PI * (k as f64 - nz as f64) / (nz as f64 * grid.dz)
            }
        }).collect();
        
        // Fill 3D arrays
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    kx[[i, j, k]] = kx_1d[i];
                    ky[[i, j, k]] = ky_1d[j];
                    kz[[i, j, k]] = kz_1d[k];
                }
            }
        }
        
        (kx, ky, kz)
    }
    
    /// Create anti-aliasing filter (2/3 rule)
    fn create_anti_aliasing_filter(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        grid: &Grid,
    ) -> Array3<f64> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut filter = Array3::ones((nx, ny, nz));
        
        // Maximum wavenumbers (2/3 of Nyquist)
        let kx_max = 2.0 * PI / (3.0 * grid.dx);
        let ky_max = 2.0 * PI / (3.0 * grid.dy);
        let kz_max = 2.0 * PI / (3.0 * grid.dz);
        
        Zip::from(&mut filter)
            .and(kx)
            .and(ky)
            .and(kz)
            .for_each(|f, &kx_val, &ky_val, &kz_val| {
                if kx_val.abs() > kx_max || ky_val.abs() > ky_max || kz_val.abs() > kz_max {
                    *f = 0.0;
                }
            });
        
        filter
    }
    
    /// Compute k-space correction factors
    fn compute_k_space_correction(
        k_squared: &Array3<f64>,
        grid: &Grid,
        order: usize,
    ) -> Array3<f64> {
        let mut kappa = Array3::ones(k_squared.raw_dim());
        
        // Fix: Properly compute wavenumber components for k-space correction
        // The current implementation incorrectly uses average spacing
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Compute individual wavenumber components correctly
                    let kx = if i <= nx / 2 {
                        2.0 * PI * i as f64 / (nx as f64 * grid.dx)
                    } else {
                        2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * grid.dx)
                    };
                    
                    let ky = if j <= ny / 2 {
                        2.0 * PI * j as f64 / (ny as f64 * grid.dy)
                    } else {
                        2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * grid.dy)
                    };
                    
                    let kz = if k <= nz / 2 {
                        2.0 * PI * k as f64 / (nz as f64 * grid.dz)
                    } else {
                        2.0 * PI * (k as f64 - nz as f64) / (nz as f64 * grid.dz)
                    };
                    
                    // Apply proper k-space correction for each direction
                    let arg_x = kx * grid.dx / 2.0;
                    let arg_y = ky * grid.dy / 2.0;
                    let arg_z = kz * grid.dz / 2.0;
                    
                    let sinc_x = if arg_x.abs() < 1e-12 { 1.0 } else { arg_x.sin() / arg_x };
                    let sinc_y = if arg_y.abs() < 1e-12 { 1.0 } else { arg_y.sin() / arg_y };
                    let sinc_z = if arg_z.abs() < 1e-12 { 1.0 } else { arg_z.sin() / arg_z };
                    
                    // Combine corrections with proper order handling
                    kappa[[i, j, k]] = match order {
                        2 => sinc_x * sinc_y * sinc_z,
                        4 => (sinc_x * sinc_y * sinc_z).powi(2),
                        6 => (sinc_x * sinc_y * sinc_z).powi(3),
                        8 => (sinc_x * sinc_y * sinc_z).powi(4),
                        _ => sinc_x * sinc_y * sinc_z,
                    };
                }
            }
        }
        
        kappa
    }
    
    /// Update pressure field using PSTD
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity_divergence: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD pressure update, dt={}", dt);
        let start = std::time::Instant::now();
        
        // Transform velocity divergence to k-space
        let mut fields_4d = Array4::zeros((1, velocity_divergence.shape()[0], velocity_divergence.shape()[1], velocity_divergence.shape()[2]));
        fields_4d.index_axis_mut(Axis(0), 0).assign(velocity_divergence);
        
        let mut div_v_hat = fft_3d(&fields_4d, 0, &self.grid);
        
        // Apply anti-aliasing filter if enabled
        if let Some(ref filter) = self.anti_alias_filter {
            Zip::from(&mut div_v_hat)
                .and(filter)
                .for_each(|d, &f| *d *= f);
        }
        
        // Apply k-space correction if enabled
        if let Some(ref kappa) = self.kappa {
            Zip::from(&mut div_v_hat)
                .and(kappa)
                .for_each(|d, &k| *d *= k);
        }
        
        // Update pressure in k-space: ∂p/∂t = -ρc²∇·v
        // Apply proper FFT scaling factor
        let scale_factor = -dt / (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let pressure_update_hat = div_v_hat.mapv(|d| d * Complex::new(scale_factor, 0.0));
        
        // Transform back to physical space
        let pressure_update = ifft_3d(&pressure_update_hat, &self.grid);
        
        // Apply the update with spatially varying ρc² - Fix: Handle this properly
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;
                    
                    // Get local medium properties
                    let rho = medium.density(x, y, z, &self.grid);
                    let c = medium.sound_speed(x, y, z, &self.grid);
                    let rho_c2 = rho * c * c;
                    
                    pressure[[i, j, k]] += pressure_update[[i, j, k]] * rho_c2;
                }
            }
        }
        
        // Update metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("pressure_update_time".to_string(), elapsed);
        
        Ok(())
    }
    
    /// Update velocity field using PSTD
    pub fn update_velocity(
        &mut self,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD velocity update, dt={}", dt);
        let start = std::time::Instant::now();
        
        // Get density array
        let rho_array = medium.density_array();
        
        // Transform pressure to k-space
        let mut fields_4d = Array4::zeros((1, pressure.shape()[0], pressure.shape()[1], pressure.shape()[2]));
        fields_4d.index_axis_mut(Axis(0), 0).assign(pressure);
        
        let mut pressure_hat = fft_3d(&fields_4d, 0, &self.grid);
        
        // Apply anti-aliasing filter if enabled
        if let Some(ref filter) = self.anti_alias_filter {
            Zip::from(&mut pressure_hat)
                .and(filter)
                .for_each(|p, &f| *p *= f);
        }
        
        // Compute gradients in k-space
        let grad_x_hat = &pressure_hat * &self.kx.mapv(|k| Complex::new(0.0, k));
        let grad_y_hat = &pressure_hat * &self.ky.mapv(|k| Complex::new(0.0, k));
        let grad_z_hat = &pressure_hat * &self.kz.mapv(|k| Complex::new(0.0, k));
        
        // Apply k-space correction if enabled
        let (grad_x_hat, grad_y_hat, grad_z_hat) = if let Some(ref kappa) = self.kappa {
            let kappa_complex = kappa.mapv(|k| Complex::new(k, 0.0));
            (grad_x_hat * &kappa_complex,
             grad_y_hat * &kappa_complex,
             grad_z_hat * &kappa_complex)
        } else {
            (grad_x_hat, grad_y_hat, grad_z_hat)
        };
        
        // Transform back to physical space
        let grad_x = ifft_3d(&grad_x_hat, &self.grid);
        let grad_y = ifft_3d(&grad_y_hat, &self.grid);
        let grad_z = ifft_3d(&grad_z_hat, &self.grid);
        
        // Update velocities: ∂v/∂t = -∇p/ρ
        Zip::from(velocity_x)
            .and(&grad_x)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        Zip::from(velocity_y)
            .and(&grad_y)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        Zip::from(velocity_z)
            .and(&grad_z)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        // Update metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("velocity_update_time".to_string(), elapsed);
        
        Ok(())
    }
    
    /// Compute velocity divergence using spectral derivatives
    pub fn compute_divergence(
        &self,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Transform velocities to k-space
        let mut fields_x = Array4::zeros((1, velocity_x.shape()[0], velocity_x.shape()[1], velocity_x.shape()[2]));
        let mut fields_y = Array4::zeros((1, velocity_y.shape()[0], velocity_y.shape()[1], velocity_y.shape()[2]));
        let mut fields_z = Array4::zeros((1, velocity_z.shape()[0], velocity_z.shape()[1], velocity_z.shape()[2]));
        
        fields_x.index_axis_mut(Axis(0), 0).assign(velocity_x);
        fields_y.index_axis_mut(Axis(0), 0).assign(velocity_y);
        fields_z.index_axis_mut(Axis(0), 0).assign(velocity_z);
        
        let vx_hat = fft_3d(&fields_x, 0, &self.grid);
        let vy_hat = fft_3d(&fields_y, 0, &self.grid);
        let vz_hat = fft_3d(&fields_z, 0, &self.grid);
        
        // Compute derivatives in k-space
        let dvx_dx_hat = &vx_hat * &self.kx.mapv(|k| Complex::new(0.0, k));
        let dvy_dy_hat = &vy_hat * &self.ky.mapv(|k| Complex::new(0.0, k));
        let dvz_dz_hat = &vz_hat * &self.kz.mapv(|k| Complex::new(0.0, k));
        
        // Sum to get divergence
        let div_hat = dvx_dx_hat + dvy_dy_hat + dvz_dz_hat;
        
        // Apply k-space correction if enabled
        let div_hat = if let Some(ref kappa) = self.kappa {
            div_hat * &kappa.mapv(|k| Complex::new(k, 0.0))
        } else {
            div_hat
        };
        
        // Transform back to physical space
        let divergence = ifft_3d(&div_hat, &self.grid);
        Ok(divergence)
    }
    
    /// Get maximum stable time step
    pub fn max_stable_dt(&self, c_max: f64) -> f64 {
        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        self.config.cfl_factor * dx_min / c_max
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}

/// PSTD solver as a physics plugin
pub struct PstdPlugin {
    solver: PstdSolver,
    metadata: PluginMetadata,
}

impl PstdPlugin {
    pub fn new(config: PstdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = PstdSolver::new(config, grid)?;
        let metadata = PluginMetadata {
            id: "pstd_solver".to_string(),
            name: "PSTD Solver".to_string(),
            version: "1.0.0".to_string(),
            description: "Pseudo-Spectral Time Domain solver for high-accuracy wave propagation".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        Ok(Self {
            solver,
            metadata,
        })
    }
}

impl std::fmt::Debug for PstdPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PstdPlugin")
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl PhysicsPlugin for PstdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        PluginState::Created
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Velocity,  // Needs velocity fields for divergence
        ]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Pressure,  // Updates pressure field
            FieldType::Velocity,  // Updates velocity fields
        ]
    }
    
    fn initialize(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Validate grid compatibility
        if grid.nx < 16 || grid.ny < 16 || grid.nz < 16 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "grid_size".to_string(),
                value: format!("{}x{}x{}", grid.nx, grid.ny, grid.nz),
                constraint: "minimum 16 points in each dimension".to_string(),
            }));
        }
        
        // Check if grid sizes are suitable for FFT (power of 2 is optimal)
        let is_power_of_2 = |n: usize| (n & (n - 1)) == 0;
        if !is_power_of_2(grid.nx) || !is_power_of_2(grid.ny) || !is_power_of_2(grid.nz) {
            warn!("Grid dimensions are not powers of 2, FFT performance may be suboptimal");
        }
        
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Extract velocity fields as owned arrays for divergence computation
        let velocity_x = fields.index_axis(Axis(0), 4).to_owned();
        let velocity_y = fields.index_axis(Axis(0), 5).to_owned();
        let velocity_z = fields.index_axis(Axis(0), 6).to_owned();
        let divergence = self.solver.compute_divergence(&velocity_x, &velocity_y, &velocity_z)?;
        
        // Update pressure
        let mut pressure = fields.index_axis(Axis(0), 0).to_owned();
        self.solver.update_pressure(&mut pressure, &divergence, medium, dt)?;
        fields.index_axis_mut(Axis(0), 0).assign(&pressure);
        
        // Update velocities
        let mut velocity_x = fields.index_axis(Axis(0), 4).to_owned();
        let mut velocity_y = fields.index_axis(Axis(0), 5).to_owned();
        let mut velocity_z = fields.index_axis(Axis(0), 6).to_owned();
        
        self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, medium, dt)?;
        
        // Copy back to fields
        fields.index_axis_mut(Axis(0), 4).assign(&velocity_x);
        fields.index_axis_mut(Axis(0), 5).assign(&velocity_y);
        fields.index_axis_mut(Axis(0), 6).assign(&velocity_z);
        
        Ok(())
    }
    
    fn performance_metrics(&self) -> HashMap<String, f64> {
        self.solver.get_metrics().clone()
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(Self {
            metadata: self.metadata.clone(),
            solver: PstdSolver::new(self.solver.config, &self.solver.grid).unwrap(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    
    #[test]
    fn test_pstd_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = PstdConfig::default();
        let solver = PstdSolver::new(config, &grid);
        assert!(solver.is_ok());
    }
    
    #[test]
    fn test_wavenumber_computation() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let (kx, ky, kz) = PstdSolver::compute_wavenumbers(&grid);
        
        // Check DC component
        assert_eq!(kx[[0, 0, 0]], 0.0);
        assert_eq!(ky[[0, 0, 0]], 0.0);
        assert_eq!(kz[[0, 0, 0]], 0.0);
        
        // Check positive frequencies
        assert!(kx[[1, 0, 0]] > 0.0);
        assert!(ky[[0, 1, 0]] > 0.0);
        assert!(kz[[0, 0, 1]] > 0.0);
        
        // Check negative frequencies
        assert!(kx[[7, 0, 0]] < 0.0);
        assert!(ky[[0, 7, 0]] < 0.0);
        assert!(kz[[0, 0, 7]] < 0.0);
    }
    
    #[test]
    fn test_anti_aliasing_filter() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let (kx, ky, kz) = PstdSolver::compute_wavenumbers(&grid);
        let filter = PstdSolver::create_anti_aliasing_filter(&kx, &ky, &kz, &grid);
        
        // Check that DC component is not filtered
        assert_eq!(filter[[0, 0, 0]], 1.0);
        
        // Check that high frequencies are filtered
        assert_eq!(filter[[16, 16, 16]], 0.0);
    }
    
    #[test]
    fn test_k_space_correction() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let (kx, ky, kz) = PstdSolver::compute_wavenumbers(&grid);
        let k_squared = &kx * &kx + &ky * &ky + &kz * &kz;
        let kappa = PstdSolver::compute_k_space_correction(&k_squared, &grid, 4);
        
        // Check that DC component has no correction
        assert_eq!(kappa[[0, 0, 0]], 1.0);
        
        // Check that correction decreases for higher frequencies
        assert!(kappa[[1, 0, 0]] < 1.0);
        assert!(kappa[[2, 0, 0]] < kappa[[1, 0, 0]]);
    }
    
    #[test]
    fn test_divergence_computation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = PstdConfig::default();
        let solver = PstdSolver::new(config, &grid).unwrap();
        
        // Create uniform velocity field (should have zero divergence)
        let vx = Array3::ones((32, 32, 32));
        let vy = Array3::zeros((32, 32, 32));
        let vz = Array3::zeros((32, 32, 32));
        
        let divergence = solver.compute_divergence(&vx, &vy, &vz).unwrap();
        
        // Check that divergence is approximately zero
        let max_div = divergence.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_div < 1e-10, "Uniform field should have zero divergence");
    }
}

#[cfg(test)]
mod validation_tests;