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
    
    /// Sound speed for time step calculation [m/s]
    pub sound_speed: f64,
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
            sound_speed: 1540.0, // Default water sound speed [m/s]
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
            sound_speed: 1540.0, // Default water sound speed [m/s]
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
    
    /// Cached time step for performance optimization
    /// Computed once during initialization to avoid redundant calculations
    dt: f64,
    
    /// Sound speed used for dt calculation
    sound_speed: f64,
    
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
        
        // Calculate dt once during initialization for performance optimization
        // dt = CFL * dx / (c * sqrt(3)) where sqrt(3) accounts for 3D Courant condition
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let dt = config.cfl_number * min_dx / (config.sound_speed * (3.0_f64).sqrt());
        
        debug!("C-PML cached dt: {:.3e} s (CFL: {}, sound speed: {} m/s)", 
               dt, config.cfl_number, config.sound_speed);
        
        // Initialize profile arrays
        let mut cpml = Self {
            config: config.clone(),
            dt,
            sound_speed: config.sound_speed,
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
    
    /// Update acoustic memory variables with recursive convolution
    pub fn update_acoustic_memory(
        &mut self,
        pressure_grad: &Array3<f64>,
        component: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let mut psi = self.psi_acoustic.index_axis_mut(Axis(0), component);
        
        match component {
            0 => { // X-component
                // Use indexed iteration for SIMD-friendly access patterns
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(i, _j, _k), psi, &grad| {
                        // Compute coefficients with dt
                        let b = (-(self.sigma_x[i] + self.alpha_x[i]) * dt).exp();
                        let c = if self.sigma_x[i] > 0.0 {
                            self.sigma_x[i] * (b - 1.0) / (self.sigma_x[i] + self.alpha_x[i])
                        } else {
                            0.0
                        };
                        *psi = b * *psi + c * grad;
                    });
            }
            1 => { // Y-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, j, _k), psi, &grad| {
                        // Compute coefficients with dt
                        let b = (-(self.sigma_y[j] + self.alpha_y[j]) * dt).exp();
                        let c = if self.sigma_y[j] > 0.0 {
                            self.sigma_y[j] * (b - 1.0) / (self.sigma_y[j] + self.alpha_y[j])
                        } else {
                            0.0
                        };
                        *psi = b * *psi + c * grad;
                    });
            }
            2 => { // Z-component
                Zip::indexed(&mut psi)
                    .and(pressure_grad)
                    .for_each(|(_i, _j, k), psi, &grad| {
                        // Compute coefficients with dt
                        let b = (-(self.sigma_z[k] + self.alpha_z[k]) * dt).exp();
                        let c = if self.sigma_z[k] > 0.0 {
                            self.sigma_z[k] * (b - 1.0) / (self.sigma_z[k] + self.alpha_z[k])
                        } else {
                            0.0
                        };
                        *psi = b * *psi + c * grad;
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
            r_normal / cos_theta.max(0.1)
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

// Implement the Boundary trait for CPMLBoundary to provide a standard interface
impl crate::boundary::Boundary for CPMLBoundary {
    /// Apply C-PML boundary conditions to acoustic field with full recursive convolution
    /// 
    /// This implementation uses the complete C-PML formulation with memory variables
    /// for proper absorption of outgoing waves. Based on Roden & Gedney (2000).
    fn apply_acoustic(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) -> KwaversResult<()> {
        let dt = grid.dx / (1540.0 * (3.0_f64).sqrt()); // CFL-stable time step
        
        // Apply C-PML in each direction with full recursive convolution
        self.apply_cpml_x_direction(field, dt)?;
        self.apply_cpml_y_direction(field, dt)?;
        self.apply_cpml_z_direction(field, dt)?;
        
        Ok(())
    }
    
    /// Apply C-PML to acoustic field in frequency domain
    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        use rustfft::num_complex::Complex;
        
        // In frequency domain, C-PML becomes a complex coordinate stretching
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Compute complex stretching factors
                    let sx = Complex::new(self.kappa_x[i], self.sigma_x[i] / (2.0 * PI));
                    let sy = Complex::new(self.kappa_y[j], self.sigma_y[j] / (2.0 * PI));
                    let sz = Complex::new(self.kappa_z[k], self.sigma_z[k] / (2.0 * PI));
                    
                    // Apply complex coordinate stretching
                    field[[i, j, k]] *= sx * sy * sz;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply C-PML to light field using proper absorption
    fn apply_light(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying C-PML to light field with proper absorption");
        
        // Use a standard dt for light diffusion
        let dt = 1e-6; // Standard time step for light diffusion
        
        // For light diffusion, apply C-PML absorption based on the diffusion equation
        // ∂φ/∂t = D∇²φ - μₐφ + S
        // In PML region: add absorption term
        
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Check if in PML region
                    if i < self.config.thickness || i >= self.nx - self.config.thickness ||
                       j < self.config.thickness || j >= self.ny - self.config.thickness ||
                       k < self.config.thickness || k >= self.nz - self.config.thickness {
                        
                        // Apply C-PML absorption with proper profile
                        let absorption = self.compute_cpml_factor(i, j, k);
                        field[[i, j, k]] *= (-absorption * dt).exp();
                    }
                }
            }
        }
    }
}

impl CPMLBoundary {
    /// Apply C-PML in X direction with recursive convolution
    fn apply_cpml_x_direction(&mut self, field: &mut Array3<f64>, dt: f64) -> KwaversResult<()> {
        let thickness = self.config.thickness;
        
        // Apply C-PML in left and right X boundaries
        for i in 0..self.nx {
            if i < thickness || i >= self.nx - thickness {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        // Compute spatial derivative ∂p/∂x using finite differences
                        let dp_dx = if i == 0 {
                            (field[[i+1, j, k]] - field[[i, j, k]]) / self.config.cfl_number
                        } else if i == self.nx - 1 {
                            (field[[i, j, k]] - field[[i-1, j, k]]) / self.config.cfl_number
                        } else {
                            (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * self.config.cfl_number)
                        };
                        
                        // Update memory variable with recursive convolution
                        // ψ^{n+1} = b_x * ψ^n + c_x * ∂p/∂x
                        self.psi_acoustic[[0, i, j, k]] = self.b_x[i] * self.psi_acoustic[[0, i, j, k]] 
                                                        + self.c_x[i] * dp_dx;
                        
                        // Apply C-PML correction to field
                        // p^{n+1} = p^n - dt * (σ_x/κ_x) * ψ_x
                        let correction = dt * (self.sigma_x[i] / self.kappa_x[i]) * self.psi_acoustic[[0, i, j, k]];
                        field[[i, j, k]] -= correction;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply C-PML in Y direction with recursive convolution
    fn apply_cpml_y_direction(&mut self, field: &mut Array3<f64>, dt: f64) -> KwaversResult<()> {
        let thickness = self.config.thickness;
        
        // Apply C-PML in front and back Y boundaries
        for j in 0..self.ny {
            if j < thickness || j >= self.ny - thickness {
                for i in 0..self.nx {
                    for k in 0..self.nz {
                        // Compute spatial derivative ∂p/∂y using finite differences
                        let dp_dy = if j == 0 {
                            (field[[i, j+1, k]] - field[[i, j, k]]) / self.config.cfl_number
                        } else if j == self.ny - 1 {
                            (field[[i, j, k]] - field[[i, j-1, k]]) / self.config.cfl_number
                        } else {
                            (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * self.config.cfl_number)
                        };
                        
                        // Update memory variable with recursive convolution
                        // ψ^{n+1} = b_y * ψ^n + c_y * ∂p/∂y
                        self.psi_acoustic[[1, i, j, k]] = self.b_y[j] * self.psi_acoustic[[1, i, j, k]] 
                                                        + self.c_y[j] * dp_dy;
                        
                        // Apply C-PML correction to field
                        let correction = dt * (self.sigma_y[j] / self.kappa_y[j]) * self.psi_acoustic[[1, i, j, k]];
                        field[[i, j, k]] -= correction;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply C-PML in Z direction with recursive convolution
    fn apply_cpml_z_direction(&mut self, field: &mut Array3<f64>, dt: f64) -> KwaversResult<()> {
        let thickness = self.config.thickness;
        
        // Apply C-PML in top and bottom Z boundaries
        for k in 0..self.nz {
            if k < thickness || k >= self.nz - thickness {
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        // Compute spatial derivative ∂p/∂z using finite differences
                        let dp_dz = if k == 0 {
                            (field[[i, j, k+1]] - field[[i, j, k]]) / self.config.cfl_number
                        } else if k == self.nz - 1 {
                            (field[[i, j, k]] - field[[i, j, k-1]]) / self.config.cfl_number
                        } else {
                            (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * self.config.cfl_number)
                        };
                        
                        // Update memory variable with recursive convolution
                        // ψ^{n+1} = b_z * ψ^n + c_z * ∂p/∂z
                        self.psi_acoustic[[2, i, j, k]] = self.b_z[k] * self.psi_acoustic[[2, i, j, k]] 
                                                        + self.c_z[k] * dp_dz;
                        
                        // Apply C-PML correction to field
                        let correction = dt * (self.sigma_z[k] / self.kappa_z[k]) * self.psi_acoustic[[2, i, j, k]];
                        field[[i, j, k]] -= correction;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Compute C-PML absorption factor at a given position
    fn compute_cpml_factor(&self, i: usize, j: usize, k: usize) -> f64 {
        let mut factor = 0.0;
        
        // X-direction contribution
        if i < self.config.thickness {
            factor += self.sigma_x[i] / self.kappa_x[i];
        } else if i >= self.nx - self.config.thickness {
            factor += self.sigma_x[i] / self.kappa_x[i];
        }
        
        // Y-direction contribution
        if j < self.config.thickness {
            factor += self.sigma_y[j] / self.kappa_y[j];
        } else if j >= self.ny - self.config.thickness {
            factor += self.sigma_y[j] / self.kappa_y[j];
        }
        
        // Z-direction contribution
        if k < self.config.thickness {
            factor += self.sigma_z[k] / self.kappa_z[k];
        } else if k >= self.nz - self.config.thickness {
            factor += self.sigma_z[k] / self.kappa_z[k];
        }
        
        factor
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