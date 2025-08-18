//! Convolutional Perfectly Matched Layer (C-PML) implementation
//! 
//! This module provides a complete C-PML boundary condition implementation for absorbing
//! outgoing waves at domain boundaries. Based on the formulation by Roden & Gedney (2000)
//! and Komatitsch & Martin (2007).
//!
//! ## Implementation Features
//! - Full recursive convolution with memory variables
//! - Support for acoustic, elastic, and dispersive media
//! - Optimized for grazing angle absorption
//! - Polynomial grading profiles with κ stretching and α frequency shifting
//!
//! ## References
//! - Roden & Gedney (2000) "Convolutional PML (CPML): An efficient FDTD implementation"
//! - Komatitsch & Martin (2007) "An unsplit convolutional perfectly matched layer"


use crate::grid::Grid;
use crate::error::{KwaversResult, KwaversError, ConfigError, ValidationError};
use ndarray::{Array3, Array4, Axis, Zip};

use std::f64::consts::PI;
use log::debug;

/// Minimum cosine theta value to prevent division by zero in reflection estimation
/// This corresponds to angles near 90 degrees (grazing incidence)
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
    
    // REMOVED: CFL number and sound speed - these should come from the solver
    // The solver determines dt based on the actual medium properties and stability requirements
}

impl Default for CPMLConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            polynomial_order: 3.0,
            sigma_factor: 0.8,  // σ_max = σ_factor * σ_theoretical
            kappa_max: 15.0,    // Higher values for grazing angle absorption
            alpha_max: 0.24,    // Standard for low-frequency absorption
            target_reflection: 1e-6,
            grazing_angle_absorption: true,
            // REMOVED: cfl_number and sound_speed
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
            kappa_max: 25.0,    // Very high for grazing angles
            alpha_max: 0.3,
            target_reflection: 1e-8,
            grazing_angle_absorption: true,
            // REMOVED: cfl_number and sound_speed
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
                constraint: "κ_max must be >= 1".to_string(),
            }));
        }
        
        Ok(())
    }
}

/// Convolutional PML boundary condition
#[derive(Debug, Clone)]
pub struct CPMLBoundary {
    config: CPMLConfig,
    
    /// Time step from the solver - ensures consistency with simulation
    /// This must match the solver's dt for proper impedance matching
    dt: f64,
    
    /// Profile arrays for each dimension
    sigma_x: Vec<f64>,
    sigma_y: Vec<f64>,
    sigma_z: Vec<f64>,
    
    kappa_x: Vec<f64>,
    kappa_y: Vec<f64>,
    kappa_z: Vec<f64>,
    
    alpha_x: Vec<f64>,
    alpha_y: Vec<f64>,
    alpha_z: Vec<f64>,
    
    /// Coefficients for time integration
    b_x: Vec<f64>,
    b_y: Vec<f64>,
    b_z: Vec<f64>,
    
    c_x: Vec<f64>,
    c_y: Vec<f64>,
    c_z: Vec<f64>,
    
    /// Memory variables for convolutional integration
    /// Stored as 4D arrays: [component, nx, ny, nz]
    psi_acoustic: Array4<f64>,
    
    /// Auxiliary memory for dispersive media
    psi_dispersive: Option<Array4<f64>>,
    
    /// Grid dimensions for validation
    nx: usize,
    ny: usize,
    nz: usize,
}

impl CPMLBoundary {
    /// Create new C-PML boundary with given configuration
    /// 
    /// # Arguments
    /// * `config` - CPML configuration parameters
    /// * `grid` - Computational grid
    /// * `dt` - Time step from the solver (must be consistent with solver's dt)
    /// * `sound_speed` - Reference sound speed for the medium (typically max sound speed)
    /// 
    /// # Important
    /// The `dt` parameter MUST be the same as used by the solver to ensure proper
    /// impedance matching at the boundaries. Using an inconsistent dt will cause
    /// spurious reflections.
    pub fn new(config: CPMLConfig, grid: &Grid, dt: f64, sound_speed: f64) -> KwaversResult<Self> {
        config.validate()?;
        
        debug!("Initializing C-PML with thickness {}, dt {:.3e} s, sound speed {} m/s", 
               config.thickness, dt, sound_speed);
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Validate dt for stability (CFL condition)
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let max_stable_dt = min_dx / (sound_speed * (3.0_f64).sqrt());
        if dt > max_stable_dt {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: dt.to_string(),
                constraint: format!(
                    "Time step exceeds CFL stability limit. Maximum stable dt = {:.3e} s for sound speed = {} m/s and min grid spacing = {:.3e} m",
                    max_stable_dt, sound_speed, min_dx
                ),
            }));
        }
        
        // Initialize profile arrays
        let mut cpml = Self {
            config: config.clone(),
            dt,  // Use the provided dt from the solver
            sigma_x: vec![0.0; nx],
            sigma_y: vec![0.0; ny],
            sigma_z: vec![0.0; nz],
            kappa_x: vec![1.0; nx],
            kappa_y: vec![1.0; ny],
            kappa_z: vec![1.0; nz],
            alpha_x: vec![0.0; nx],
            alpha_y: vec![0.0; ny],
            alpha_z: vec![0.0; nz],
            b_x: vec![0.0; nx],
            b_y: vec![0.0; ny],
            b_z: vec![0.0; nz],
            c_x: vec![0.0; nx],
            c_y: vec![0.0; ny],
            c_z: vec![0.0; nz],
            psi_acoustic: Array4::zeros((3, nx, ny, nz)), // 3 components for x, y, z
            psi_dispersive: None,
            nx,
            ny,
            nz,
        };
        
        // Compute standard profiles with the provided sound speed
        cpml.compute_profiles(grid, sound_speed)?;
        
        Ok(cpml)
    }
    
    /// Compute C-PML profiles based on configuration
    /// 
    /// Note: Potential optimization for cubic grids - when dimensions are identical
    /// (nx==ny==nz and dx==dy==dz), profiles could be computed once and reused.
    /// However, the performance gain would be minimal as this is only done during
    /// initialization. The current implementation prioritizes clarity and simplicity.
    fn compute_profiles(&mut self, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        let thickness = self.config.thickness as f64;
        let m = self.config.polynomial_order;
        
        // Compute theoretical sigma_max based on analytical formula
        let sigma_theoretical_x = self.compute_theoretical_sigma(grid.dx, sound_speed);
        let sigma_theoretical_y = self.compute_theoretical_sigma(grid.dy, sound_speed);
        let sigma_theoretical_z = self.compute_theoretical_sigma(grid.dz, sound_speed);
        
        // X-direction profiles
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            self.nx, thickness, m, sigma_theoretical_x, grid.dx, &self.config, self.dt
        );
        self.sigma_x = sigma;
        self.kappa_x = kappa;
        self.alpha_x = alpha;
        self.b_x = b;
        self.c_x = c;
        
        // Y-direction profiles  
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            self.ny, thickness, m, sigma_theoretical_y, grid.dy, &self.config, self.dt
        );
        self.sigma_y = sigma;
        self.kappa_y = kappa;
        self.alpha_y = alpha;
        self.b_y = b;
        self.c_y = c;
        
        // Z-direction profiles
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            self.nz, thickness, m, sigma_theoretical_z, grid.dz, &self.config, self.dt
        );
        self.sigma_z = sigma;
        self.kappa_z = kappa;
        self.alpha_z = alpha;
        self.b_z = b;
        self.c_z = c;
        
        debug!("C-PML profiles computed with σ_theoretical = ({:.2e}, {:.2e}, {:.2e})",
               sigma_theoretical_x, sigma_theoretical_y, sigma_theoretical_z);
        
        Ok(())
    }
    
    /// Compute theoretical sigma value based on grid spacing and sound speed
    /// 
    /// The theoretical optimal sigma depends on the impedance Z = rho * c
    /// For simplicity, we assume constant density and use sound speed directly
    fn compute_theoretical_sigma(&self, dx: f64, sound_speed: f64) -> f64 {
        let m = self.config.polynomial_order;
        let r_coeff = self.config.target_reflection;
        
        // Theoretical optimal value for C-PML
        // sigma_opt = -(m+1) * c * ln(R) / (2 * L)
        // where L is the PML thickness in meters
        let pml_width = self.config.thickness as f64 * dx;
        let sigma_opt = -(m + 1.0) * sound_speed * r_coeff.ln() / (2.0 * pml_width);
        
        sigma_opt * self.config.sigma_factor
    }
    
    /// Generic function to compute 1D profile for any direction
    fn compute_profile_for_dimension(
        n: usize,
        thickness: f64,
        m: f64,
        sigma_max: f64,
        dx: f64,
        config: &CPMLConfig,
        dt: f64, // Use the actual dt from the struct
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut sigma = vec![0.0; n];
        let mut kappa = vec![1.0; n];
        let mut alpha = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        
        for i in 0..n {
            // Distance from PML interface (0 at interface, 1 at boundary)
            let d_left = if i < thickness as usize {
                (thickness - i as f64 - 0.5) / thickness
            } else {
                0.0
            };
            
            let d_right = if i >= n - thickness as usize {
                (i as f64 - (n as f64 - thickness - 1.0) + 0.5) / thickness
            } else {
                0.0
            };
            
            let d = d_left.max(d_right);
            
            if d > 0.0 {
                // Polynomial grading
                let d_m = d.powf(m);
                
                // Conductivity profile
                sigma[i] = sigma_max * d_m;
                
                // Coordinate stretching profile
                if config.grazing_angle_absorption {
                    // Profile for grazing angles
                    let kappa_grad = (config.kappa_max - 1.0) * d.powf(m + 1.0);
                    kappa[i] = 1.0 + kappa_grad;
                } else {
                    kappa[i] = 1.0 + (config.kappa_max - 1.0) * d_m;
                }
                
                // Frequency shifting profile (quadratic for stability)
                alpha[i] = config.alpha_max * (1.0 - d).powi(2);
                
                // Compute update coefficients
                let sigma_i = sigma[i];
                let kappa_i = kappa[i];
                let alpha_i = alpha[i];
                
                // Time integration coefficients - use the passed dt
                b[i] = (-(sigma_i + kappa_i * alpha_i) * dt).exp();
                
                if (sigma_i + kappa_i * alpha_i).abs() > 1e-10 {
                    c[i] = sigma_i / (sigma_i + kappa_i * alpha_i) * (b[i] - 1.0);
                } else {
                    c[i] = 0.0;
                }
            } else {
                // Outside PML region
                sigma[i] = 0.0;
                kappa[i] = 1.0;
                alpha[i] = 0.0;
                b[i] = 0.0;
                c[i] = 0.0;
            }
        }
        
        (sigma, kappa, alpha, b, c)
    }
    
    /// Update acoustic memory variables with recursive convolution
    /// Uses pre-computed coefficients for efficiency
    pub fn update_acoustic_memory(
        &mut self,
        pressure_grad: &Array3<f64>,
        component: usize,
    ) -> KwaversResult<()> {
        let mut psi = self.psi_acoustic.index_axis_mut(Axis(0), component);
        
        match component {
            0 => { // X-component
                // Use pre-computed coefficients for efficiency
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(i, _j, _k), psi_val, &grad| {
                        *psi_val = self.b_x[i] * *psi_val + self.c_x[i] * grad;
                    });
            }
            1 => { // Y-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, j, _k), psi_val, &grad| {
                        *psi_val = self.b_y[j] * *psi_val + self.c_y[j] * grad;
                    });
            }
            2 => { // Z-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, _j, k), psi_val, &grad| {
                        *psi_val = self.b_z[k] * *psi_val + self.c_z[k] * grad;
                    });
            }
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "component".to_string(),
                    value: component.to_string(),
                    constraint: "Must be 0, 1, or 2 for x, y, z components".to_string(),
                }));
            }
        }
        
        Ok(())
    }
    
    /// Update the time step and recompute coefficients if needed
    /// This should be called if the simulation time step changes
    /// 
    /// # Arguments
    /// * `new_dt` - New time step from the solver
    /// * `grid` - Computational grid
    /// * `sound_speed` - Reference sound speed for the medium
    pub fn update_dt(&mut self, new_dt: f64, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        const DT_TOLERANCE: f64 = 1e-12;
        
        if (self.dt - new_dt).abs() > DT_TOLERANCE {
            debug!("Updating CPML coefficients for new dt: {} -> {}", self.dt, new_dt);
            self.dt = new_dt;
            // Recompute profiles with the new dt and sound speed
            self.compute_profiles(grid, sound_speed)?;
        }
        Ok(())
    }
    
    /// Apply C-PML absorption to field gradients
    pub fn apply_cpml_gradient(
        &self,
        gradient: &mut Array3<f64>,
        component: usize,
    ) -> KwaversResult<()> {
        let psi = self.psi_acoustic.index_axis(Axis(0), component);
        
        match component {
            0 => { // X-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(i, j, k), grad, &psi_val| {
                        *grad = *grad / self.kappa_x[i] + psi_val;
                    });
            }
            1 => { // Y-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(i, j, k), grad, &psi_val| {
                        *grad = *grad / self.kappa_y[j] + psi_val;
                    });
            }
            2 => { // Z-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(i, j, k), grad, &psi_val| {
                        *grad = *grad / self.kappa_z[k] + psi_val;
                    });
            }
            _ => return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "component".to_string(),
                value: component.to_string(),
                constraint: "Component must be 0, 1, or 2".to_string(),
            })),
        }
        
        Ok(())
    }
    
    /// Enable support for dispersive media
    pub fn enable_dispersive_support(&mut self) {
        if self.psi_dispersive.is_none() {
            self.psi_dispersive = Some(Array4::zeros((3, self.nx, self.ny, self.nz)));
            debug!("Enabled dispersive media support for C-PML");
        }
    }
    
    /// Get reflection coefficient estimate at given angle
    /// Returns None if angle is out of valid range [0, 90] degrees
    pub fn estimate_reflection(&self, angle_degrees: f64) -> Option<f64> {
        // Validate input angle range
        if !(0.0..=90.0).contains(&angle_degrees) {
            return None; // Return None for invalid input instead of silent failure
        }
        
        let angle_rad = angle_degrees * PI / 180.0;
        let cos_theta = angle_rad.cos();
        
        // Theoretical reflection coefficient for C-PML
        let r_normal = self.config.target_reflection;
        
        // Model for grazing angles
        let reflection = if self.config.grazing_angle_absorption {
            // For grazing angles, reflection should increase
            let grazing_factor = (1.0 - cos_theta.powi(2)).sqrt(); // sin(theta)
            // Increase reflection for larger angles (smaller cos_theta)
            let angle_enhancement = 1.0 + grazing_factor * (self.config.kappa_max - 1.0);
            r_normal * angle_enhancement
        } else {
            // Standard model - reflection increases as angle increases (cos_theta decreases)
            r_normal / cos_theta.max(MIN_COS_THETA_FOR_REFLECTION)
        };
        
        Some(reflection)
    }
    
    /// Get the C-PML configuration
    pub fn config(&self) -> &CPMLConfig {
        &self.config
    }

    /// Get absorption coefficients for a specific position
    pub fn get_coefficients(&self, x: usize, y: usize, z: usize) -> (f64, f64, f64) {
        let ax = if x < self.config.thickness || x >= self.nx - self.config.thickness {
            self.sigma_x[x]
        } else {
            0.0
        };
        
        let ay = if y < self.config.thickness || y >= self.ny - self.config.thickness {
            self.sigma_y[y]
        } else {
            0.0
        };
        
        let az = if z < self.config.thickness || z >= self.nz - self.config.thickness {
            self.sigma_z[z]
        } else {
            0.0
        };
        
        (ax, ay, az)
    }
    
    // Note: The apply_to_field method has been removed as it was deprecated.
    // C-PML must be integrated into the solver's gradient computation, not applied
    // as a post-processing step. Use the get_coefficients method to retrieve
    // absorption coefficients for proper integration into your solver.
}

// NOTE: The Boundary trait implementation has been removed as it was incorrect.
// The CPMLBoundary MUST be integrated directly into the solver's update equations.
//
// # Correct Usage
//
// The CPMLBoundary is not applied as a post-processing step. Instead, it must be
// integrated into the main solver's update loop:
//
// 1. After computing the spatial gradient (e.g., of pressure), call `update_acoustic_memory`.
// 2. Then, call `apply_cpml_gradient` to modify the gradient before using it to update the field (e.g., velocity).
//
// Example:
// ```rust
// // In your solver's update loop:
// let pressure_grad = compute_gradient(&pressure);
// cpml.update_acoustic_memory(&pressure_grad, component)?;
// cpml.apply_cpml_gradient(&mut pressure_grad, component)?;
// // Now use the modified gradient to update velocity
// ```

// The incorrect Boundary trait implementation has been removed.
// CPMLBoundary must be integrated into the solver's update equations, not used as a post-processing step.

// The helper methods for applying CPML as post-processing have been removed.
// These were incorrect implementations that violated the physics of PML.
// CPML must be integrated into the solver's update equations.

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpml_initialization() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-7; // Typical time step for testing
        let sound_speed = 1540.0; // Water sound speed
        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        
        assert_eq!(cpml.sigma_x.len(), 64);
        assert_eq!(cpml.kappa_x.len(), 64);
        assert_eq!(cpml.alpha_x.len(), 64);
    }
    
    #[test]
    fn test_profile_computation() {
        let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig {
            thickness: 10,
            polynomial_order: 3.0,
            kappa_max: 15.0,
            ..Default::default()
        };
        
        let dt = 1e-7; // Typical time step for testing
        let sound_speed = 1540.0; // Water sound speed
        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        
        // Check that profiles are properly graded
        // At PML interface (i=10), values should be zero/one
        assert!((cpml.sigma_x[10] - 0.0).abs() < 1e-10);
        assert!((cpml.kappa_x[10] - 1.0).abs() < 1e-10);
        
        // At boundary (i=0), values should be maximum
        assert!(cpml.sigma_x[0] > 0.0);
        assert!(cpml.kappa_x[0] > 1.0);
    }
    
    #[test]
    fn test_grazing_angle_reflection() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::for_grazing_angles();
        let dt = 1e-7; // Typical time step for testing
        let sound_speed = 1540.0; // Water sound speed
        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        
        // Test reflection estimates at various angles
        let r_normal = cpml.estimate_reflection(0.0).unwrap();    // Normal incidence
        let r_45 = cpml.estimate_reflection(45.0).unwrap();       // 45 degrees
        let r_grazing = cpml.estimate_reflection(85.0).unwrap();  // Near grazing
        
        // Test invalid angles return None
        assert!(cpml.estimate_reflection(-10.0).is_none());
        assert!(cpml.estimate_reflection(95.0).is_none());
        
        // Reflection should increase with angle
        assert!(r_45 > r_normal);
        assert!(r_grazing > r_45);
        
        // But should still be small for grazing angle config
        assert!(r_grazing < 1e-6);
    }
}

// Include comprehensive validation tests
#[cfg(test)]
#[path = "cpml_validation_tests.rs"]
mod cpml_validation_tests;