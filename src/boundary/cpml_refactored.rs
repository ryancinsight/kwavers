//! Refactored Convolutional Perfectly Matched Layer (C-PML) implementation
//! 
//! This module provides an optimized C-PML boundary condition implementation with:
//! - Plane-by-plane iteration for optimal cache performance
//! - Type-safe PmlAxis enum instead of magic numbers
//! - Separate psi arrays for cleaner code structure

use crate::grid::Grid;
use crate::error::{KwaversResult, KwaversError, ConfigError, ValidationError};
use ndarray::{Array3, Axis, Zip, s};

use std::f64::consts::PI;
use log::debug;

/// Type-safe axis specification for PML operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PmlAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl PmlAxis {
    /// Get all axes in order
    pub fn all() -> [PmlAxis; 3] {
        [PmlAxis::X, PmlAxis::Y, PmlAxis::Z]
    }
    
    /// Get the axis name as a string
    pub fn name(&self) -> &'static str {
        match self {
            PmlAxis::X => "X",
            PmlAxis::Y => "Y",
            PmlAxis::Z => "Z",
        }
    }
}

/// Minimum cosine theta value to prevent division by zero in reflection estimation
const MIN_COS_THETA_FOR_REFLECTION: f64 = 0.1;

/// Configuration for Convolutional PML
#[derive(Debug, Clone)]
pub struct CPMLConfig {
    /// Number of PML cells in each direction
    pub thickness: usize,
    
    /// Polynomial order for profile grading (typically 3-4)
    pub polynomial_order: f64,
    
    /// Maximum conductivity scaling factor
    pub sigma_factor: f64,
    
    /// Maximum κ (coordinate stretching) value
    pub kappa_max: f64,
    
    /// Maximum α (frequency shifting) value
    pub alpha_max: f64,
    
    /// Target reflection coefficient (e.g., 1e-6)
    pub target_reflection: f64,
    
    /// Enable grazing angle absorption
    pub grazing_angle_absorption: bool,
}

impl Default for CPMLConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            polynomial_order: 3.0,
            sigma_factor: 0.8,
            kappa_max: 15.0,
            alpha_max: 0.24,
            target_reflection: 1e-6,
            grazing_angle_absorption: true,
        }
    }
}

impl CPMLConfig {
    /// Create config for grazing angles
    pub fn for_grazing_angles() -> Self {
        Self {
            thickness: 20,
            polynomial_order: 4.0,
            sigma_factor: 1.0,
            kappa_max: 25.0,
            alpha_max: 0.3,
            target_reflection: 1e-8,
            grazing_angle_absorption: true,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "thickness".to_string(),
                value: self.thickness.to_string(),
                constraint: "C-PML thickness must be > 0".to_string(),
            }));
        }
        
        if self.polynomial_order < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "polynomial_order".to_string(),
                value: self.polynomial_order.to_string(),
                constraint: "Polynomial order must be >= 0".to_string(),
            }));
        }
        
        if self.kappa_max < 1.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kappa_max".to_string(),
                value: self.kappa_max.to_string(),
                constraint: "kappa_max must be >= 1.0".to_string(),
            }));
        }
        
        Ok(())
    }
}

/// Optimized C-PML boundary condition with separate memory arrays
pub struct CPMLBoundary {
    config: CPMLConfig,
    
    // Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    
    // 1D profile arrays for each axis
    sigma_x: Vec<f64>,
    sigma_y: Vec<f64>,
    sigma_z: Vec<f64>,
    
    kappa_x: Vec<f64>,
    kappa_y: Vec<f64>,
    kappa_z: Vec<f64>,
    
    // Pre-computed reciprocals for efficiency
    inv_kappa_x: Vec<f64>,
    inv_kappa_y: Vec<f64>,
    inv_kappa_z: Vec<f64>,
    
    alpha_x: Vec<f64>,
    alpha_y: Vec<f64>,
    alpha_z: Vec<f64>,
    
    // Recursive convolution coefficients
    b_x: Vec<f64>,
    b_y: Vec<f64>,
    b_z: Vec<f64>,
    
    c_x: Vec<f64>,
    c_y: Vec<f64>,
    c_z: Vec<f64>,
    
    // Separate memory arrays for each component (no more Array4)
    psi_x: Array3<f64>,
    psi_y: Array3<f64>,
    psi_z: Array3<f64>,
    
    // Optional dispersive media support
    psi_dispersive_x: Option<Array3<f64>>,
    psi_dispersive_y: Option<Array3<f64>>,
    psi_dispersive_z: Option<Array3<f64>>,
    
    // Time step
    dt: f64,
}

impl CPMLBoundary {
    /// Create a new C-PML boundary condition
    pub fn new(config: CPMLConfig, grid: &Grid, dt: f64, sound_speed: f64) -> KwaversResult<Self> {
        config.validate()?;
        
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        
        let mut cpml = Self {
            config,
            nx,
            ny,
            nz,
            sigma_x: vec![0.0; nx],
            sigma_y: vec![0.0; ny],
            sigma_z: vec![0.0; nz],
            kappa_x: vec![1.0; nx],
            kappa_y: vec![1.0; ny],
            kappa_z: vec![1.0; nz],
            inv_kappa_x: vec![1.0; nx],
            inv_kappa_y: vec![1.0; ny],
            inv_kappa_z: vec![1.0; nz],
            alpha_x: vec![0.0; nx],
            alpha_y: vec![0.0; ny],
            alpha_z: vec![0.0; nz],
            b_x: vec![0.0; nx],
            b_y: vec![0.0; ny],
            b_z: vec![0.0; nz],
            c_x: vec![0.0; nx],
            c_y: vec![0.0; ny],
            c_z: vec![0.0; nz],
            psi_x: Array3::zeros((nx, ny, nz)),
            psi_y: Array3::zeros((nx, ny, nz)),
            psi_z: Array3::zeros((nx, ny, nz)),
            psi_dispersive_x: None,
            psi_dispersive_y: None,
            psi_dispersive_z: None,
            dt,
        };
        
        cpml.compute_profiles(grid, sound_speed)?;
        Ok(cpml)
    }
    
    /// Compute PML profile coefficients
    fn compute_profiles(&mut self, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        // X-direction profiles
        self.compute_profile_1d(
            self.nx, grid.dx, sound_speed,
            &mut self.sigma_x, &mut self.kappa_x, &mut self.inv_kappa_x,
            &mut self.alpha_x, &mut self.b_x, &mut self.c_x
        );
        
        // Y-direction profiles
        self.compute_profile_1d(
            self.ny, grid.dy, sound_speed,
            &mut self.sigma_y, &mut self.kappa_y, &mut self.inv_kappa_y,
            &mut self.alpha_y, &mut self.b_y, &mut self.c_y
        );
        
        // Z-direction profiles
        self.compute_profile_1d(
            self.nz, grid.dz, sound_speed,
            &mut self.sigma_z, &mut self.kappa_z, &mut self.inv_kappa_z,
            &mut self.alpha_z, &mut self.b_z, &mut self.c_z
        );
        
        Ok(())
    }
    
    /// Compute 1D profile for a given direction
    fn compute_profile_1d(
        &self,
        n: usize,
        d: f64,
        sound_speed: f64,
        sigma: &mut [f64],
        kappa: &mut [f64],
        inv_kappa: &mut [f64],
        alpha: &mut [f64],
        b: &mut [f64],
        c: &mut [f64],
    ) {
        let thickness = self.config.thickness.min(n / 4);
        
        if thickness == 0 {
            return;
        }
        
        let sigma_max = self.theoretical_sigma_max(d, sound_speed) * self.config.sigma_factor;
        
        for i in 0..thickness {
            let xi = (i as f64 + 0.5) / thickness as f64;
            let poly = xi.powf(self.config.polynomial_order);
            
            sigma[i] = sigma_max * poly;
            kappa[i] = 1.0 + (self.config.kappa_max - 1.0) * poly;
            inv_kappa[i] = 1.0 / kappa[i];
            alpha[i] = self.config.alpha_max * (1.0 - xi);
            
            let sum = (sigma[i] / kappa[i]) + alpha[i];
            b[i] = (-sum * self.dt).exp();
            c[i] = if sum > 0.0 {
                sigma[i] / (kappa[i] * sum) * (b[i] - 1.0)
            } else {
                0.0
            };
            
            // Mirror for the other end
            let j = n - 1 - i;
            sigma[j] = sigma[i];
            kappa[j] = kappa[i];
            inv_kappa[j] = inv_kappa[i];
            alpha[j] = alpha[i];
            b[j] = b[i];
            c[j] = c[i];
        }
    }
    
    /// Calculate theoretical maximum conductivity
    fn theoretical_sigma_max(&self, d: f64, sound_speed: f64) -> f64 {
        let n = self.config.polynomial_order;
        let r = self.config.target_reflection;
        let l = self.config.thickness as f64 * d;
        
        let cos_theta = if self.config.grazing_angle_absorption {
            MIN_COS_THETA_FOR_REFLECTION
        } else {
            1.0
        };
        
        -(n + 1.0) * sound_speed * r.ln() * cos_theta / (2.0 * l)
    }
    
    /// Optimized update for X-component acoustic memory using plane-by-plane iteration
    pub fn update_acoustic_memory_x(&mut self, pressure_grad: &Array3<f64>) {
        // Process each X-plane
        for i in 0..self.nx {
            let b_i = self.b_x[i];
            let c_i = self.c_x[i];
            
            // Skip planes in the interior where PML is inactive
            if b_i == 0.0 && c_i == 0.0 {
                // Could also check if psi plane is all zeros for early exit
                let psi_plane = self.psi_x.slice(s![i, .., ..]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            // Get mutable view of the psi plane and immutable view of gradient plane
            let mut psi_plane = self.psi_x.slice_mut(s![i, .., ..]);
            let grad_plane = pressure_grad.slice(s![i, .., ..]);
            
            // Apply the update to the entire 2D plane at once
            Zip::from(&mut psi_plane)
                .and(&grad_plane)
                .for_each(|psi_val, &grad| {
                    *psi_val = b_i * *psi_val + c_i * grad;
                });
        }
    }
    
    /// Optimized update for Y-component acoustic memory using plane-by-plane iteration
    pub fn update_acoustic_memory_y(&mut self, pressure_grad: &Array3<f64>) {
        // Process each Y-plane
        for j in 0..self.ny {
            let b_j = self.b_y[j];
            let c_j = self.c_y[j];
            
            // Skip inactive planes
            if b_j == 0.0 && c_j == 0.0 {
                let psi_plane = self.psi_y.slice(s![.., j, ..]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            let mut psi_plane = self.psi_y.slice_mut(s![.., j, ..]);
            let grad_plane = pressure_grad.slice(s![.., j, ..]);
            
            Zip::from(&mut psi_plane)
                .and(&grad_plane)
                .for_each(|psi_val, &grad| {
                    *psi_val = b_j * *psi_val + c_j * grad;
                });
        }
    }
    
    /// Optimized update for Z-component acoustic memory using plane-by-plane iteration
    pub fn update_acoustic_memory_z(&mut self, pressure_grad: &Array3<f64>) {
        // Process each Z-plane
        for k in 0..self.nz {
            let b_k = self.b_z[k];
            let c_k = self.c_z[k];
            
            // Skip inactive planes
            if b_k == 0.0 && c_k == 0.0 {
                let psi_plane = self.psi_z.slice(s![.., .., k]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            let mut psi_plane = self.psi_z.slice_mut(s![.., .., k]);
            let grad_plane = pressure_grad.slice(s![.., .., k]);
            
            Zip::from(&mut psi_plane)
                .and(&grad_plane)
                .for_each(|psi_val, &grad| {
                    *psi_val = b_k * *psi_val + c_k * grad;
                });
        }
    }
    
    /// Type-safe update for acoustic memory with axis parameter
    pub fn update_acoustic_memory(&mut self, pressure_grad: &Array3<f64>, axis: PmlAxis) {
        match axis {
            PmlAxis::X => self.update_acoustic_memory_x(pressure_grad),
            PmlAxis::Y => self.update_acoustic_memory_y(pressure_grad),
            PmlAxis::Z => self.update_acoustic_memory_z(pressure_grad),
        }
    }
    
    /// Optimized gradient application for X-component using plane-by-plane iteration
    pub fn apply_cpml_gradient_x(&self, gradient: &mut Array3<f64>) {
        // Process each X-plane
        for i in 0..self.nx {
            let inv_kappa_i = self.inv_kappa_x[i];
            
            // Skip planes where kappa is 1.0 (no stretching) and psi is zero
            if inv_kappa_i == 1.0 {
                let psi_plane = self.psi_x.slice(s![i, .., ..]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            let mut grad_plane = gradient.slice_mut(s![i, .., ..]);
            let psi_plane = self.psi_x.slice(s![i, .., ..]);
            
            // Apply the scalar coefficient to the entire 2D plane
            Zip::from(&mut grad_plane)
                .and(&psi_plane)
                .for_each(|grad, &psi_val| {
                    *grad = grad.mul_add(inv_kappa_i, psi_val);
                });
        }
    }
    
    /// Optimized gradient application for Y-component using plane-by-plane iteration
    pub fn apply_cpml_gradient_y(&self, gradient: &mut Array3<f64>) {
        // Process each Y-plane
        for j in 0..self.ny {
            let inv_kappa_j = self.inv_kappa_y[j];
            
            // Skip inactive planes
            if inv_kappa_j == 1.0 {
                let psi_plane = self.psi_y.slice(s![.., j, ..]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            let mut grad_plane = gradient.slice_mut(s![.., j, ..]);
            let psi_plane = self.psi_y.slice(s![.., j, ..]);
            
            Zip::from(&mut grad_plane)
                .and(&psi_plane)
                .for_each(|grad, &psi_val| {
                    *grad = grad.mul_add(inv_kappa_j, psi_val);
                });
        }
    }
    
    /// Optimized gradient application for Z-component using plane-by-plane iteration
    pub fn apply_cpml_gradient_z(&self, gradient: &mut Array3<f64>) {
        // Process each Z-plane
        for k in 0..self.nz {
            let inv_kappa_k = self.inv_kappa_z[k];
            
            // Skip inactive planes
            if inv_kappa_k == 1.0 {
                let psi_plane = self.psi_z.slice(s![.., .., k]);
                if psi_plane.iter().all(|&v| v.abs() < 1e-15) {
                    continue;
                }
            }
            
            let mut grad_plane = gradient.slice_mut(s![.., .., k]);
            let psi_plane = self.psi_z.slice(s![.., .., k]);
            
            Zip::from(&mut grad_plane)
                .and(&psi_plane)
                .for_each(|grad, &psi_val| {
                    *grad = grad.mul_add(inv_kappa_k, psi_val);
                });
        }
    }
    
    /// Type-safe gradient application with axis parameter
    pub fn apply_cpml_gradient(&self, gradient: &mut Array3<f64>, axis: PmlAxis) {
        match axis {
            PmlAxis::X => self.apply_cpml_gradient_x(gradient),
            PmlAxis::Y => self.apply_cpml_gradient_y(gradient),
            PmlAxis::Z => self.apply_cpml_gradient_z(gradient),
        }
    }
    
    /// Update the time step and recompute coefficients if needed
    pub fn update_dt(&mut self, new_dt: f64, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        const DT_TOLERANCE: f64 = 1e-12;
        
        if (self.dt - new_dt).abs() > DT_TOLERANCE {
            debug!("Updating CPML coefficients for new dt: {} -> {}", self.dt, new_dt);
            self.dt = new_dt;
            self.compute_profiles(grid, sound_speed)?;
        }
        Ok(())
    }
    
    /// Reset all memory variables to zero
    pub fn reset(&mut self) {
        self.psi_x.fill(0.0);
        self.psi_y.fill(0.0);
        self.psi_z.fill(0.0);
        
        if let Some(ref mut psi) = self.psi_dispersive_x {
            psi.fill(0.0);
        }
        if let Some(ref mut psi) = self.psi_dispersive_y {
            psi.fill(0.0);
        }
        if let Some(ref mut psi) = self.psi_dispersive_z {
            psi.fill(0.0);
        }
    }
    
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let array_memory = 3 * self.nx * self.ny * self.nz * std::mem::size_of::<f64>();
        let profile_memory = 6 * (self.nx + self.ny + self.nz) * std::mem::size_of::<f64>();
        let dispersive_memory = if self.psi_dispersive_x.is_some() {
            3 * self.nx * self.ny * self.nz * std::mem::size_of::<f64>()
        } else {
            0
        };
        
        array_memory + profile_memory + dispersive_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pml_axis_enum() {
        assert_eq!(PmlAxis::X as usize, 0);
        assert_eq!(PmlAxis::Y as usize, 1);
        assert_eq!(PmlAxis::Z as usize, 2);
        assert_eq!(PmlAxis::X.name(), "X");
        assert_eq!(PmlAxis::all().len(), 3);
    }
    
    #[test]
    fn test_plane_iteration_efficiency() {
        let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let cpml = CPMLBoundary::new(config, &grid, 1e-6, 1500.0).unwrap();
        
        // Memory usage should be 3 separate arrays, not a 4D array
        let expected_array_size = 3 * 100 * 100 * 100 * 8; // 3 arrays × size × f64
        let actual_usage = cpml.memory_usage();
        
        // Should be approximately the same (plus profile arrays)
        assert!(actual_usage >= expected_array_size);
    }
}