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
use crate::boundary::Boundary;
use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::utils::{fft_3d, ifft_3d};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginConfig, PluginContext};
use crate::physics::composable::{FieldType, ValidationResult};
use ndarray::{Array3, Array4, Axis, Zip, s};
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
    /// k-squared array for Laplacian
    k_squared: Array3<f64>,
    /// Anti-aliasing filter
    filter: Option<Array3<f64>>,
    /// k-space correction factors
    kappa: Option<Array3<f64>>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
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
            filter,
            kappa,
            metrics: HashMap::new(),
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
        
        // k-space correction based on exact dispersion relation
        // κ = sinc(k·Δx/2)^order
        Zip::from(&mut kappa)
            .and(k_squared)
            .for_each(|kap, &k2| {
                if k2 > 0.0 {
                    let k = k2.sqrt();
                    let dx_eff = (grid.dx + grid.dy + grid.dz) / 3.0; // Average spacing
                    let arg = k * dx_eff / 2.0;
                    let sinc = if arg.abs() < 1e-10 {
                        1.0
                    } else {
                        arg.sin() / arg
                    };
                    *kap = sinc.powi(order as i32);
                }
            });
        
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
        
        // Get medium properties
        let rho_array = medium.density_array();
        let c_array = medium.sound_speed_array();
        
        // Compute ρc² array
        let rho_c2 = Zip::from(&rho_array)
            .and(&c_array)
            .map_collect(|&rho, &c| rho * c * c);
        
        // Transform velocity divergence to k-space
        let mut fields_4d = Array4::zeros((1, velocity_divergence.shape()[0], velocity_divergence.shape()[1], velocity_divergence.shape()[2]));
        fields_4d.index_axis_mut(Axis(0), 0).assign(velocity_divergence);
        
        let mut div_v_hat = fft_3d(&fields_4d, 0, &self.grid);
        
        // Apply anti-aliasing filter if enabled
        if let Some(ref filter) = self.filter {
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
        // In k-space, this becomes a simple multiplication
        let pressure_update_hat = div_v_hat.mapv(|d| d * Complex::new(-dt, 0.0));
        
        // Transform back to physical space
        let pressure_update = ifft_3d(&pressure_update_hat, &self.grid);
        
        // Apply the update with spatially varying ρc²
        Zip::from(pressure)
            .and(&pressure_update)
            .and(&rho_c2)
            .for_each(|p, &update, &rho_c2_val| {
                *p += update * rho_c2_val;
            });
        
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
        if let Some(ref filter) = self.filter {
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
        _config: Option<Box<dyn PluginConfig>>,
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
        // Extract velocity and pressure fields
        let mut pressure = fields.index_axis(Axis(0), 0).to_owned();
        let mut velocity_x = fields.index_axis(Axis(0), 4).to_owned();
        let mut velocity_y = fields.index_axis(Axis(0), 5).to_owned();
        let mut velocity_z = fields.index_axis(Axis(0), 6).to_owned();
        
        // Compute velocity divergence
        let divergence = self.solver.compute_divergence(&velocity_x, &velocity_y, &velocity_z)?;
        
        // Update pressure
        self.solver.update_pressure(&mut pressure, &divergence, medium, dt)?;
        
        // Update velocities
        self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, medium, dt)?;
        
        // Write back to fields array
        fields.index_axis_mut(Axis(0), 0).assign(&pressure);
        fields.index_axis_mut(Axis(0), 4).assign(&velocity_x);
        fields.index_axis_mut(Axis(0), 5).assign(&velocity_y);
        fields.index_axis_mut(Axis(0), 6).assign(&velocity_z);
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.solver.get_metrics().clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.solver.metrics.clear();
        Ok(())
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