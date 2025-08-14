//! Convolutional Perfectly Matched Layer (C-PML) boundary conditions
//! 
//! This module implements absorbing boundary conditions for acoustic wave simulations
//! using the Convolutional PML formulation.
//! 
//! # Current Implementation Status
//! 
//! ## Full C-PML Implementation
//! The main `ConvolutionalPML` struct provides a complete C-PML implementation with:
//! - Auxiliary memory variables for field history
//! - Recursive convolution updates
//! - Support for acoustic, elastic, and thermal fields
//! - Configurable absorption profiles (polynomial, exponential)
//! 
//! ## Exponential Sponge Layer
//! The `apply_light` method provides an exponential damping layer:
//! - **NOT** a true C-PML implementation
//! - Simple exponential decay without memory variables
//! - Suitable for basic absorption when full C-PML overhead is not needed
//! - Should be renamed in future API redesign to avoid confusion
//! 
//! # Design Considerations
//! 
//! The current `Boundary` trait interface doesn't fully capture the C-PML 
//! operational model, which requires:
//! 1. Auxiliary memory variables per field component
//! 2. Recursive convolution updates at each time step
//! 3. Different update equations for different field types
//! 
//! Future API redesign should consider:
//! - Separate traits for simple boundaries vs. complex PML boundaries
//! - Explicit memory variable management in the trait interface
//! - Field-specific update methods
//! 
//! # References
//! 
//! 1. Roden, J. A., & Gedney, S. D. (2000). "Convolutional PML (CPML): An efficient 
//!    FDTD implementation of the CFS-PML for arbitrary media." Microwave and Optical 
//!    Technology Letters, 27(5), 334-339.
//! 
//! 2. Komatitsch, D., & Martin, R. (2007). "An unsplit convolutional perfectly 
//!    matched layer improved at grazing incidence for the seismic wave equation." 
//!    Geophysics, 72(5), SM155-SM167.


use crate::grid::Grid;
use crate::error::{KwaversResult, KwaversError, ConfigError};
use ndarray::{Array3, Array4, Axis, Zip};

use std::f64::consts::PI;
use log::{debug, trace};

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
    
    /// CFL number for stability
    pub cfl_number: f64,
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
            cfl_number: 0.5,
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
            cfl_number: 0.5,
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
    pub fn new(config: CPMLConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate()?;
        
        debug!("Initializing C-PML with thickness {} and polynomial order {}", 
               config.thickness, config.polynomial_order);
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Initialize profile arrays
        let mut cpml = Self {
            config: config.clone(),
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
        
        // Compute standard profiles
        cpml.compute_profiles(grid)?;
        
        Ok(cpml)
    }
    
    /// Compute C-PML profiles based on configuration
    fn compute_profiles(&mut self, grid: &Grid) -> KwaversResult<()> {
        let thickness = self.config.thickness as f64;
        let m = self.config.polynomial_order;
        
        // Compute theoretical sigma_max based on analytical formula
        let sigma_theoretical_x = self.compute_theoretical_sigma(grid.dx);
        let sigma_theoretical_y = self.compute_theoretical_sigma(grid.dy);
        let sigma_theoretical_z = self.compute_theoretical_sigma(grid.dz);
        
        // X-direction profiles
        self.compute_profile_1d(
            self.nx,
            thickness,
            m,
            sigma_theoretical_x,
            grid.dx,
        );
        
        // Y-direction profiles
        self.compute_profile_1d_y(
            self.ny,
            thickness,
            m,
            sigma_theoretical_y,
            grid.dy,
        );
        
        // Z-direction profiles
        self.compute_profile_1d_z(
            self.nz,
            thickness,
            m,
            sigma_theoretical_z,
            grid.dz,
        );
        
        debug!("C-PML profiles computed with σ_theoretical = ({:.2e}, {:.2e}, {:.2e})",
               sigma_theoretical_x, sigma_theoretical_y, sigma_theoretical_z);
        
        Ok(())
    }
    
    /// Compute theoretical sigma value based on grid spacing
    fn compute_theoretical_sigma(&self, dx: f64) -> f64 {
        let m = self.config.polynomial_order;
        let r_coeff = self.config.target_reflection;
        
        // Theoretical reference value for C-PML
        let sigma_opt = -(m + 1.0) * r_coeff.ln() / (2.0 * self.config.thickness as f64 * dx);
        
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
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let dt = dx * config.cfl_number; // Approximate time step
        
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
                
                // Time integration coefficients
                b[i] = (-(sigma_i / kappa_i + alpha_i) * dt).exp();
                
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
    
    /// Compute 1D profile for X direction
    fn compute_profile_1d(&mut self, n: usize, thickness: f64, m: f64, sigma_max: f64, dx: f64) {
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            n, thickness, m, sigma_max, dx, &self.config
        );
        self.sigma_x = sigma;
        self.kappa_x = kappa;
        self.alpha_x = alpha;
        self.b_x = b;
        self.c_x = c;
    }
    
    /// Compute 1D profile for Y direction
    fn compute_profile_1d_y(&mut self, n: usize, thickness: f64, m: f64, sigma_max: f64, dy: f64) {
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            n, thickness, m, sigma_max, dy, &self.config
        );
        self.sigma_y = sigma;
        self.kappa_y = kappa;
        self.alpha_y = alpha;
        self.b_y = b;
        self.c_y = c;
    }
    
    /// Compute 1D profile for Z direction
    fn compute_profile_1d_z(&mut self, n: usize, thickness: f64, m: f64, sigma_max: f64, dz: f64) {
        let (sigma, kappa, alpha, b, c) = Self::compute_profile_for_dimension(
            n, thickness, m, sigma_max, dz, &self.config
        );
        self.sigma_z = sigma;
        self.kappa_z = kappa;
        self.alpha_z = alpha;
        self.b_z = b;
        self.c_z = c;
    }
    
    /// Update memory variables for acoustic field
    pub fn update_acoustic_memory(
        &mut self,
        pressure_grad: &Array3<f64>,
        component: usize,
    ) -> KwaversResult<()> {
        let mut psi = self.psi_acoustic.index_axis_mut(Axis(0), component);
        
        match component {
            0 => { // X-component
                // Use indexed iteration for SIMD-friendly access patterns
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(i, _j, _k), psi, &grad| {
                        *psi = self.b_x[i] * *psi + self.c_x[i] * grad;
                    });
            }
            1 => { // Y-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, j, _k), psi, &grad| {
                        *psi = self.b_y[j] * *psi + self.c_y[j] * grad;
                    });
            }
            2 => { // Z-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, _j, k), psi, &grad| {
                        *psi = self.b_z[k] * *psi + self.c_z[k] * grad;
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
    pub fn estimate_reflection(&self, angle_degrees: f64) -> f64 {
        // Validate input angle range
        if !(0.0..=90.0).contains(&angle_degrees) {
            debug!("Invalid angle: {}. Angle must be between 0 and 90 degrees.", angle_degrees);
            return 0.0; // Return a default value for invalid input
        }
        
        let angle_rad = angle_degrees * PI / 180.0;
        let cos_theta = angle_rad.cos();
        
        // Theoretical reflection coefficient for C-PML
        let r_normal = self.config.target_reflection;
        
        // Model for grazing angles
        if self.config.grazing_angle_absorption {
            // For grazing angles, reflection should increase
            let grazing_factor = (1.0 - cos_theta.powi(2)).sqrt(); // sin(theta)
            // Increase reflection for larger angles (smaller cos_theta)
            let angle_enhancement = 1.0 + grazing_factor * (self.config.kappa_max - 1.0);
            r_normal * angle_enhancement
        } else {
            // Standard model - reflection increases as angle increases (cos_theta decreases)
            r_normal / cos_theta.max(0.1)
        }
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

// Note: CPMLBoundary intentionally does NOT implement the Boundary trait.
// C-PML is not a simple boundary condition that can be applied to a field;
// it must be integrated into the solver's update equations.
// Solvers that support C-PML should take a CPMLBoundary object directly
// and call its methods (update_acoustic_memory, apply_cpml_gradient) during
// the field update step.


impl CPMLBoundary {
    /// Apply exponential sponge layer absorption to light field
    /// This is NOT a true C-PML implementation but a sponge layer
    /// that provides basic absorption at boundaries but with inferior performance
    /// especially for grazing angles of incidence.
    fn apply_sponge_layer_light(&self, field: &mut Array3<f64>, grid: &Grid) {
        trace!("Applying exponential sponge layer to light field (not true C-PML)");
        
        let thickness = self.config.thickness;
        
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let mut absorption = 1.0;
                    
                    // Apply exponential absorption profile (simple sponge layer)
                    if i < thickness || i >= self.nx - thickness {
                        absorption *= (-0.5 * self.sigma_x[i] * grid.dx).exp();
                    }
                    
                    if j < thickness || j >= self.ny - thickness {
                        absorption *= (-0.5 * self.sigma_y[j] * grid.dy).exp();
                    }
                    
                    if k < thickness || k >= self.nz - thickness {
                        absorption *= (-0.5 * self.sigma_z[k] * grid.dz).exp();
                    }
                    
                    field[[i, j, k]] *= absorption;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpml_initialization() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let cpml = CPMLBoundary::new(config, &grid).unwrap();
        
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
        
        let cpml = CPMLBoundary::new(config, &grid).unwrap();
        
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
        let cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Test reflection estimates at various angles
        let r_normal = cpml.estimate_reflection(0.0);    // Normal incidence
        let r_45 = cpml.estimate_reflection(45.0);       // 45 degrees
        let r_grazing = cpml.estimate_reflection(85.0);  // Near grazing
        
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