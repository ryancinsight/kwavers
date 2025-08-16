//! Pseudo-Spectral Time Domain (PSTD) solver
//! 
//! This module implements the PSTD method for solving acoustic wave equations
//! with high accuracy and minimal numerical dispersion.
//! 
//! # Theory
//! 
//! The PSTD method computes spatial derivatives in the frequency domain using
//! the Fast Fourier Transform (FFT), which provides spectral accuracy for
//! smooth fields. The key advantages are:
//! 
//! - **Spectral accuracy**: Exponential convergence for smooth solutions
//! - **No numerical dispersion**: Exact representation of wave propagation
//! - **Large time steps**: Limited only by physical CFL condition
//! - **Efficient for large domains**: O(N log N) complexity via FFT
//! 
//! # Algorithm
//! 
//! The acoustic wave equation:
//! ```text
//! ∂p/∂t = -ρc²∇·v
//! ∂v/∂t = -∇p/ρ
//! ```
//! 
//! In PSTD, spatial derivatives are computed as:
//! ```text
//! ∂f/∂x = F⁻¹{ikₓ F{f}}
//! ```
//! where F denotes the FFT and kₓ is the wavenumber.
//! 
//! # Literature References
//! 
//! 1. **Liu, Q. H. (1997)**. "The PSTD algorithm: A time-domain method requiring 
//!    only two cells per wavelength." *Microwave and Optical Technology Letters*, 
//!    15(3), 158-165. DOI: 10.1002/(SICI)1098-2760(19970620)15:3<158::AID-MOP11>3.0.CO;2-3
//!    - Original PSTD formulation for electromagnetic waves
//! 
//! 2. **Tabei, M., Mast, T. D., & Waag, R. C. (2002)**. "A k-space method for 
//!    coupled first-order acoustic propagation equations." *The Journal of the 
//!    Acoustical Society of America*, 111(1), 53-63. DOI: 10.1121/1.1421344
//!    - Extension to acoustic wave propagation
//!    - k-space correction for improved accuracy
//! 
//! 3. **Mast, T. D., Souriau, L. P., Liu, D. L., Tabei, M., Nachman, A. I., & 
//!    Waag, R. C. (2001)**. "A k-space method for large-scale models of wave 
//!    propagation in tissue." *IEEE Transactions on Ultrasonics, Ferroelectrics, 
//!    and Frequency Control*, 48(2), 341-354. DOI: 10.1109/58.911717
//!    - Application to medical ultrasound
//!    - Handling of heterogeneous media
//! 
//! 4. **Treeby, B. E., & Cox, B. T. (2010)**. "k-Wave: MATLAB toolbox for the 
//!    simulation and reconstruction of photoacoustic wave fields." *Journal of 
//!    Biomedical Optics*, 15(2), 021314. DOI: 10.1117/1.3360308
//!    - Comprehensive k-Wave implementation
//!    - Validation and benchmarking
//! 
//! # Implementation Details
//! 
//! ## Anti-aliasing (2/3 Rule)
//! 
//! To prevent aliasing from nonlinear operations, we apply the 2/3 rule:
//! - Zero out wavenumbers above 2/3 of the Nyquist frequency
//! - Based on: Orszag, S. A. (1971). "On the elimination of aliasing in 
//!   finite-difference schemes by filtering high-wavenumber components." 
//!   *Journal of the Atmospheric Sciences*, 28(6), 1074-1074.
//! 
//! ## k-space Correction
//! 
//! For improved accuracy at high frequencies, we apply k-space corrections:
//! ```text
//! κ = sinc(kΔx/2) × sinc(kΔt·c/2)
//! ```
//! This corrects for the finite difference approximation in time.
//! 
//! # Design Principles
//! - SOLID: Single responsibility for spectral wave propagation
//! - CUPID: Composable with other solvers via plugin architecture
//! - KISS: Simple interface hiding complex spectral operations
//! - DRY: Reuses spectral utilities from utils module

use crate::grid::Grid;
use crate::medium::Medium;
use crate::medium::absorption::{PowerLawAbsorption, apply_power_law_absorption};
use crate::error::{KwaversResult, KwaversError, ValidationError};
use crate::utils::{fft_3d, ifft_3d, spectral};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginContext, PluginState, PluginConfig};
use crate::validation::ValidationResult;
use crate::constants::cfl;
use crate::solver::kspace_correction::{compute_kspace_correction, KSpaceCorrectionConfig, CorrectionMethod};
use crate::boundary::cpml::{CPMLBoundary, CPMLConfig};
use ndarray::{Array3, Array4, Axis, Zip, s};
use num_complex::Complex;
use std::f64::consts::PI;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, info};

/// PSTD solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PstdConfig {
    /// Enable k-space correction
    pub k_space_correction: bool,
    /// Order of k-space correction (typically 2-8)
    pub k_space_order: usize,
    /// Apply anti-aliasing filter (2/3 rule)
    pub anti_aliasing: bool,
    /// Stencil size for PML regions (if using finite differences there)
    pub pml_stencil_size: usize,
    /// CFL safety factor (typically 0.3-0.5 for PSTD)
    pub cfl_factor: f64,
    /// Use leapfrog time integration (second-order accurate)
    pub use_leapfrog: bool,
    /// Enable power-law absorption
    pub enable_absorption: bool,
    /// Absorption model configuration
    pub absorption_model: Option<PowerLawAbsorption>,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: true,
            pml_stencil_size: 4,
            cfl_factor: cfl::PSTD_DEFAULT,
            use_leapfrog: true,  // Default to second-order time integration
            enable_absorption: false,  // Disabled by default for backward compatibility
            absorption_model: None,
        }
    }
}

impl PluginConfig for PstdConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        
        // Validate k-space order
        if self.k_space_order < 2 || self.k_space_order > 8 {
            errors.push(ValidationError::FieldValidation {
                field: "k_space_order".to_string(),
                value: self.k_space_order.to_string(),
                constraint: "Must be between 2 and 8".to_string(),
            });
        }
        
        // Validate CFL factor
        if self.cfl_factor <= 0.0 || self.cfl_factor > 1.0 {
            errors.push(ValidationError::FieldValidation {
                field: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "Must be in (0, 1]".to_string(),
            });
        }
        // Note: CFL factor > 0.5 warning removed for simplicity
        
        // Validate PML stencil size
        if self.pml_stencil_size < 2 || self.pml_stencil_size > 10 {
            errors.push(ValidationError::FieldValidation {
                field: "pml_stencil_size".to_string(),
                value: self.pml_stencil_size.to_string(),
                constraint: "Must be between 2 and 10".to_string(),
            });
        }
        
        if errors.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failure(errors)
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn std::any::Any + Send + Sync> {
        Box::new(self.clone())
    }
}

/// PSTD solver for acoustic wave propagation
#[derive(Clone, Debug)]
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
    /// CPML boundary condition (optional for backward compatibility)
    boundary: Option<CPMLBoundary>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Workspace arrays to avoid allocations
    workspace_real: Array3<f64>,
    workspace_complex: Array3<Complex<f64>>,
    /// 4D workspace arrays for FFT operations (Issue #5)
    workspace_real_4d: Array4<f64>,
    workspace_complex_4d: Array4<Complex<f64>>,
    /// Previous pressure field for leapfrog integration
    pressure_prev: Option<Array3<f64>>,
    /// Previous velocity field for leapfrog integration  
    velocity_prev: Option<Array4<f64>>,
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
        
        // Estimate dt and reference sound speed
        let c_ref = 1500.0; // Default reference sound speed
        let dt = config.cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c_ref;
        
        // Create k-space correction using unified approach
        let kappa = if config.k_space_correction {
            let kspace_config = KSpaceCorrectionConfig {
                enabled: true,
                method: CorrectionMethod::ExactDispersion,  // Use most accurate method
                cfl_number: config.cfl_factor,
                max_correction: 2.0,
            };
            Some(compute_kspace_correction(grid, &kspace_config, dt, c_ref))
        } else {
            None
        };
        
        // Initialize CPML boundary if PML is enabled in config
        // Using default CPML config for now - could be extended to take custom config
        let boundary = if config.pml_stencil_size > 0 {
            match CPMLBoundary::new(CPMLConfig::default(), grid, dt, c_ref) {
                Ok(cpml) => {
                    info!("CPML boundary initialized for PSTD solver");
                    Some(cpml)
                },
                Err(e) => {
                    log::warn!("Failed to initialize CPML boundary: {}. \
                              Continuing without boundary conditions.", e);
                    None
                }
            }
        } else {
            debug!("CPML boundary not enabled (pml_stencil_size = 0)");
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
            boundary,
            metrics: HashMap::new(),
            workspace_real: Array3::zeros((nx, ny, nz)),
            workspace_complex: Array3::zeros((nx, ny, nz)),
            workspace_real_4d: Array4::zeros((1, nx, ny, nz)),
            workspace_complex_4d: Array4::zeros((1, nx, ny, nz)),
            pressure_prev: None,
            velocity_prev: None,
        })
    }
    
    /// Enable CPML boundary conditions
    /// 
    /// # Arguments
    /// * `config` - CPML configuration
    /// * `sound_speed` - Reference sound speed for the medium
    pub fn enable_cpml(&mut self, config: CPMLConfig, sound_speed: f64) -> KwaversResult<()> {
        let dt = self.config.cfl_factor * self.grid.dx.min(self.grid.dy).min(self.grid.dz) / sound_speed;
        
        match CPMLBoundary::new(config, &self.grid, dt, sound_speed) {
            Ok(cpml) => {
                info!("CPML boundary enabled for PSTD solver");
                self.boundary = Some(cpml);
                Ok(())
            },
            Err(e) => {
                log::error!("Failed to enable CPML boundary: {}", e);
                Err(e)
            }
        }
    }
    
    /// Disable CPML boundary conditions
    pub fn disable_cpml(&mut self) {
        info!("CPML boundary disabled for PSTD solver");
        self.boundary = None;
    }
    
    /// Check if CPML is enabled
    pub fn has_cpml(&self) -> bool {
        self.boundary.is_some()
    }
    
    /// Compute wavenumber arrays for FFT
    fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        // Use centralized spectral utilities
        spectral::compute_wavenumbers(grid)
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
    // Note: k-space correction is handled by compute_kspace_correction from solver::kspace_correction module
    
    /// Update pressure field using k-space divergence (more efficient)
    pub fn update_pressure_kspace(
        &mut self,
        pressure: &mut ndarray::ArrayViewMut3<f64>,
        div_v_hat: &Array3<Complex<f64>>,  // Already in k-space
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD pressure update (k-space), dt={}, leapfrog={}", dt, self.config.use_leapfrog);
        let start = std::time::Instant::now();
        
        // Clone the input to avoid modifying it
        let mut div_v_hat = div_v_hat.clone();
        
        // Apply anti-aliasing filter if enabled
        if let Some(ref filter) = self.anti_alias_filter {
            Zip::from(&mut div_v_hat)
                .and(filter)
                .for_each(|d, &f| *d *= f);
        }
        
        // Apply k-space correction if enabled (already applied in compute_divergence_kspace)
        // Skip this step since kappa was already applied
        
        // Apply power-law absorption if enabled
        if self.config.enable_absorption {
            if let Some(ref absorption) = self.config.absorption_model {
                // Get reference sound speed (use average for simplicity)
                let c_ref = medium.sound_speed(
                    self.grid.dx * self.grid.nx as f64 / 2.0,
                    self.grid.dy * self.grid.ny as f64 / 2.0,
                    self.grid.dz * self.grid.nz as f64 / 2.0,
                    &self.grid
                );
                apply_power_law_absorption(&mut div_v_hat, &self.k_squared, absorption, c_ref, dt);
            }
        }
        
        // Pre-compute rho_c2 array for efficiency (Issue #2)
        let rho_c2_array = medium.density_array() * &medium.sound_speed_array().mapv(|c| c.powi(2));
        
        // Save current pressure before update for leapfrog scheme
        let pressure_current_copy = if self.config.use_leapfrog {
            Some(pressure.to_owned())
        } else {
            None
        };
        
        // Choose time integration scheme
        if self.config.use_leapfrog && self.pressure_prev.is_some() {
            // Leapfrog scheme: p^{n+1} = p^{n-1} + 2*dt*(-ρc²∇·v^n)
            let scale_factor = -2.0 * dt;
            let pressure_update_hat = div_v_hat.mapv(|d| d * Complex::new(scale_factor, 0.0));
            
            // Transform back to physical space
            let pressure_update = ifft_3d(&pressure_update_hat, &self.grid);
            
            // Apply leapfrog update with pre-computed ρc²
            let pressure_prev = self.pressure_prev.as_ref().unwrap();
            Zip::from(pressure)
                .and(pressure_prev)
                .and(&pressure_update)
                .and(&rho_c2_array)
                .for_each(|p, &p_prev, &update, &rho_c2| {
                    *p = p_prev + update * rho_c2;
                });
        } else if self.config.use_leapfrog && self.pressure_prev.is_none() {
            // Second-order accurate initialization for leapfrog using RK2
            // Issue #3: Eliminate redundant inverse FFT
            
            // Perform iFFT once to get the pressure update kernel
            let pressure_update_kernel_hat = div_v_hat.mapv(|d| d * Complex::new(-1.0, 0.0));
            let pressure_update_kernel = ifft_3d(&pressure_update_kernel_hat, &self.grid);
            
            // Step 1: Compute half-step pressure using scaled kernel
            let scale_factor_half = dt / 2.0;
            let mut pressure_half = pressure.to_owned();
            Zip::from(&mut pressure_half)
                .and(&pressure_update_kernel)
                .and(&rho_c2_array)
                .for_each(|p, &kernel, &rho_c2| {
                    *p += kernel * scale_factor_half * rho_c2;
                });
            
            // Step 2: Apply full step update using scaled kernel
            let scale_factor_full = dt;
            Zip::from(pressure)
                .and(&pressure_update_kernel)
                .and(&rho_c2_array)
                .for_each(|p, &kernel, &rho_c2| {
                    *p += kernel * scale_factor_full * rho_c2;
                });
        } else {
            // Standard first-order Euler scheme
            let scale_factor = -dt;
            let pressure_update_hat = div_v_hat.mapv(|d| d * Complex::new(scale_factor, 0.0));
            
            let pressure_update = ifft_3d(&pressure_update_hat, &self.grid);
            
            // Apply update with pre-computed ρc²
            Zip::from(pressure)
                .and(&pressure_update)
                .and(&rho_c2_array)
                .for_each(|p, &update, &rho_c2| {
                    *p += update * rho_c2;
                });
        }
        
        // Store current pressure for next leapfrog step
        if self.config.use_leapfrog {
            self.pressure_prev = pressure_current_copy;
        }
        
        // Update performance metrics
        self.metrics.insert("pressure_update_time".to_string(), start.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    /// Update pressure field (wrapper for backward compatibility)
    pub fn update_pressure(
        &mut self,
        pressure: &mut ndarray::ArrayViewMut3<f64>,
        velocity_divergence: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Transform velocity divergence to k-space using workspace
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(velocity_divergence);
        
        let div_v_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
        // Delegate to the k-space version
        self.update_pressure_kspace(pressure, &div_v_hat, medium, dt)
    }
    
    /// Update velocity field using PSTD
    pub fn update_velocity(
        &mut self,
        velocity_x: &mut ndarray::ArrayViewMut3<f64>,
        velocity_y: &mut ndarray::ArrayViewMut3<f64>,
        velocity_z: &mut ndarray::ArrayViewMut3<f64>,
        pressure: &ndarray::ArrayView3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Delegate to CPML version if boundary is configured
        if self.boundary.is_some() {
            return self.update_velocity_with_cpml(
                velocity_x, velocity_y, velocity_z, pressure, medium, dt
            );
        }
        
        debug!("PSTD velocity update (no CPML), dt={}", dt);
        let start = std::time::Instant::now();
        
        // Get density array
        let rho_array = medium.density_array();
        
        // Transform pressure to k-space using workspace
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(pressure);
        
        let mut pressure_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
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
        // Note: ifft_3d applies proper 1/(nx*ny*nz) normalization
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
    
    /// Update velocity field using pre-computed pressure gradients (more efficient when gradients are already available)
    pub fn update_velocity_with_gradient(
        &mut self,
        velocity_x: &mut ndarray::ArrayViewMut3<f64>,
        velocity_y: &mut ndarray::ArrayViewMut3<f64>,
        velocity_z: &mut ndarray::ArrayViewMut3<f64>,
        grad_x: &Array3<f64>,
        grad_y: &Array3<f64>,
        grad_z: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD velocity update with gradients, dt={}", dt);
        let start = std::time::Instant::now();
        
        // Get density array
        let rho_array = medium.density_array();
        
        // Update velocity components using pre-computed gradients
        Zip::from(velocity_x)
            .and(grad_x)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
            
        Zip::from(velocity_y)
            .and(grad_y)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
            
        Zip::from(velocity_z)
            .and(grad_z)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        // Update metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("velocity_update_time".to_string(), elapsed);
        
        Ok(())
    }
    
    /// Compute velocity divergence in k-space (avoids redundant FFT/iFFT)
    /// Returns the k-space representation of the divergence
    /// Uses workspace arrays to avoid repeated allocations (Issue #5)
    pub fn compute_divergence_kspace(
        &mut self,
        velocity_x: &ndarray::ArrayView3<f64>,
        velocity_y: &ndarray::ArrayView3<f64>,
        velocity_z: &ndarray::ArrayView3<f64>,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        // Transform velocities to k-space using workspace
        // Note: we reuse workspace_real_4d for each component sequentially
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(velocity_x);
        let vx_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(velocity_y);
        let vy_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(velocity_z);
        let vz_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
        // Compute derivatives in k-space
        let dvx_dx_hat = &vx_hat * &self.kx.mapv(|k| Complex::new(0.0, k));
        let dvy_dy_hat = &vy_hat * &self.ky.mapv(|k| Complex::new(0.0, k));
        let dvz_dz_hat = &vz_hat * &self.kz.mapv(|k| Complex::new(0.0, k));
        
        // Sum to get divergence
        let mut div_hat = dvx_dx_hat + dvy_dy_hat + dvz_dz_hat;
        
        // Apply k-space correction if enabled
        if let Some(ref kappa) = self.kappa {
            div_hat = div_hat * &kappa.mapv(|k| Complex::new(k, 0.0));
        }
        
        Ok(div_hat)
    }
    
    /// Compute velocity divergence in real space (wrapper for backward compatibility)
    pub fn compute_divergence(
        &mut self,
        velocity_x: &ndarray::ArrayView3<f64>,
        velocity_y: &ndarray::ArrayView3<f64>,
        velocity_z: &ndarray::ArrayView3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Get k-space divergence
        let div_hat = self.compute_divergence_kspace(velocity_x, velocity_y, velocity_z)?;
        
        // Transform back to physical space
        // Note: ifft_3d applies proper 1/(nx*ny*nz) normalization
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
    
    /// Merge metrics from another solver instance
    pub fn merge_metrics(&mut self, other_metrics: &HashMap<String, f64>) {
        for (key, value) in other_metrics {
            // For most metrics, we'll take the maximum value
            // This can be customized based on the metric type
            if key.contains("time") || key.contains("elapsed") {
                // For time-based metrics, accumulate
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current + value);
            } else if key.contains("count") || key.contains("calls") {
                // For counters, accumulate
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current + value);
            } else {
                // For other metrics (like errors, norms), take the maximum
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current.max(*value));
            }
        }
    }

    /// Compute spatial gradient of a field using spectral differentiation
    /// 
    /// # Arguments
    /// * `field` - Input field
    /// * `direction` - 0 for x, 1 for y, 2 for z
    pub fn compute_gradient(&mut self, field: &Array3<f64>, direction: usize) -> KwaversResult<Array3<f64>> {
        // Transform to k-space
        self.workspace_real_4d.index_axis_mut(Axis(0), 0).assign(field);
        let field_hat = fft_3d(&self.workspace_real_4d, 0, &self.grid);
        
        // Select wavenumber array based on direction
        let k_array = match direction {
            0 => &self.kx,
            1 => &self.ky,
            2 => &self.kz,
            _ => return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "direction".to_string(),
                value: direction.to_string(),
                constraint: "Must be 0, 1, or 2".to_string(),
            })),
        };
        
        // Compute gradient in k-space: ∂f/∂x = F⁻¹{ikₓ F{f}}
        let grad_hat = &field_hat * &k_array.mapv(|k| Complex::new(0.0, k));
        
        // Transform back to real space
        self.workspace_complex_4d.index_axis_mut(Axis(0), 0).assign(&grad_hat);
        let grad_complex = self.workspace_complex_4d.index_axis(Axis(0), 0);
        let grad_real = ifft_3d(&grad_complex.to_owned(), &self.grid);
        
        Ok(grad_real)
    }
    
    /// Update velocity field with CPML boundary conditions
    pub fn update_velocity_with_cpml(
        &mut self,
        velocity_x: &mut ndarray::ArrayViewMut3<f64>,
        velocity_y: &mut ndarray::ArrayViewMut3<f64>,
        velocity_z: &mut ndarray::ArrayViewMut3<f64>,
        pressure: &ndarray::ArrayView3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD velocity update with CPML, dt={}", dt);
        
        // Compute pressure gradients
        let mut grad_x = self.compute_gradient(&pressure.to_owned(), 0)?;
        let mut grad_y = self.compute_gradient(&pressure.to_owned(), 1)?;
        let mut grad_z = self.compute_gradient(&pressure.to_owned(), 2)?;
        
        // Apply CPML if boundary is configured
        if let Some(ref mut boundary) = self.boundary {
            // Update CPML memory variables
            boundary.update_acoustic_memory(&grad_x, 0)?;
            boundary.update_acoustic_memory(&grad_y, 1)?;
            boundary.update_acoustic_memory(&grad_z, 2)?;
            
            // Apply CPML correction to gradients
            boundary.apply_cpml_gradient(&mut grad_x, 0)?;
            boundary.apply_cpml_gradient(&mut grad_y, 1)?;
            boundary.apply_cpml_gradient(&mut grad_z, 2)?;
        }
        
        // Update velocity using corrected gradients
        let rho_array = medium.density_array();
        
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
        
        Ok(())
    }
    
    /// Update pressure field with CPML boundary conditions
    pub fn update_pressure_with_cpml(
        &mut self,
        pressure: &mut ndarray::ArrayViewMut3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("PSTD pressure update with CPML, dt={}", dt);
        
        // Compute velocity gradients
        let mut grad_vx = self.compute_gradient(velocity_x, 0)?;
        let mut grad_vy = self.compute_gradient(velocity_y, 1)?;
        let mut grad_vz = self.compute_gradient(velocity_z, 2)?;
        
        // Apply CPML if boundary is configured
        if let Some(ref mut boundary) = self.boundary {
            // Update CPML memory variables for velocity gradients
            boundary.update_acoustic_memory(&grad_vx, 0)?;
            boundary.update_acoustic_memory(&grad_vy, 1)?;
            boundary.update_acoustic_memory(&grad_vz, 2)?;
            
            // Apply CPML correction
            boundary.apply_cpml_gradient(&mut grad_vx, 0)?;
            boundary.apply_cpml_gradient(&mut grad_vy, 1)?;
            boundary.apply_cpml_gradient(&mut grad_vz, 2)?;
        }
        
        // Compute divergence from corrected gradients
        let divergence = &grad_vx + &grad_vy + &grad_vz;
        
        // Update pressure
        // Compute bulk modulus from density and sound speed: K = ρc²
        let density_array = medium.density_array();
        let sound_speed_array = medium.sound_speed_array();
        let bulk_modulus_array = &density_array * &sound_speed_array.mapv(|c| c * c);
        
        Zip::from(pressure)
            .and(&divergence)
            .and(&bulk_modulus_array)
            .for_each(|p, &div, &bulk| {
                *p -= dt * bulk * div;
            });
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
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
        // K-space correction is now handled by compute_kspace_correction from solver module
        use crate::solver::kspace_correction::{compute_kspace_correction, KSpaceCorrectionConfig, CorrectionMethod};
        let config = KSpaceCorrectionConfig {
            enabled: true,
            method: CorrectionMethod::ExactDispersion,
            cfl_number: 0.5,
            max_correction: 2.0,
        };
        let kappa = compute_kspace_correction(&grid, &config, 1e-6, 1500.0);
        
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
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Create uniform velocity field (should have zero divergence)
        let vx = Array3::ones((32, 32, 32));
        let vy = Array3::zeros((32, 32, 32));
        let vz = Array3::zeros((32, 32, 32));
        
        let divergence = solver.compute_divergence(
            &vx.view(),
            &vy.view(),
            &vz.view()
        ).unwrap();
        
        // Check that divergence is approximately zero
        let max_div = divergence.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_div < 1e-10, "Uniform field should have zero divergence");
    }
}

#[cfg(test)]
mod validation_tests;
mod fft_scaling_test;

// Plugin implementation for PSTD solver

/// PSTD solver plugin for integration with the physics pipeline
#[derive(Debug)]
pub struct PstdPlugin {
    solver: PstdSolver,
    metadata: PluginMetadata,
}

impl PstdPlugin {
    /// Create a new PSTD plugin
    pub fn new(config: PstdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = PstdSolver::new(config, &grid)?;
        let metadata = PluginMetadata {
            id: "pstd_solver".to_string(),
            name: "PSTD Solver".to_string(),
            version: "1.0.0".to_string(),
            author: "Kwavers Team".to_string(),
            description: "Pseudo-Spectral Time Domain solver with k-space methods".to_string(),
            license: "MIT".to_string(),
        };
        Ok(Self { solver, metadata })
    }
    
    /// Helper method to compute gradient using spectral derivatives
    fn compute_gradient(&self, field: &ndarray::ArrayView3<f64>, direction: usize) -> KwaversResult<Array3<f64>> {
        use crate::utils::spectral;
        
        // Convert ArrayView to Array3 for spectral gradient functions
        let field_owned = field.to_owned();
        
        // Use centralized spectral gradient functions
        match direction {
            0 => spectral::gradient_x(&field_owned, &self.solver.grid),
            1 => spectral::gradient_y(&field_owned, &self.solver.grid),
            2 => spectral::gradient_z(&field_owned, &self.solver.grid),
            _ => Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "gradient_direction".to_string(),
                value: direction.to_string(),
                constraint: "must be 0 (x), 1 (y), or 2 (z)".to_string(),
            })),
        }
    }
}

impl PhysicsPlugin for PstdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        PluginState::Initialized
    }
    
    fn initialize(&mut self, _grid: &Grid, _medium: &dyn crate::medium::Medium) -> KwaversResult<()> {
        // Solver is already initialized in new()
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn crate::medium::Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        use ndarray::s;
        use crate::physics::field_mapping::UnifiedFieldType;
        
        // Get field indices using the unified system
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();
        
        // Work directly with mutable views using correct indices
        let mut fields_view = fields.view_mut();
        let (mut pressure, mut velocity_x, mut velocity_y, mut velocity_z) = 
            fields_view.multi_slice_mut((
                s![pressure_idx, .., .., ..],
                s![vx_idx, .., .., ..],
                s![vy_idx, .., .., ..],
                s![vz_idx, .., .., ..]
            ));
        
        // Initialize velocities if they are all zero (first step)
        let v_max = velocity_x.iter().chain(velocity_y.iter()).chain(velocity_z.iter())
            .fold(0.0f64, |acc, &v| acc.max(v.abs()));
        
        if v_max < 1e-15 && context.step == 0 {
            // Initialize velocities from pressure gradient for first step
            // This ensures the wave starts propagating
            let grad_x = self.compute_gradient(&pressure.view(), 0)?;
            let grad_y = self.compute_gradient(&pressure.view(), 1)?;
            let grad_z = self.compute_gradient(&pressure.view(), 2)?;
            
            let rho_array = medium.density_array();
            Zip::from(&mut velocity_x)
                .and(&grad_x)
                .and(&rho_array)
                .for_each(|v, &grad, &rho| {
                    *v = -dt * grad / rho;
                });
            Zip::from(&mut velocity_y)
                .and(&grad_y)
                .and(&rho_array)
                .for_each(|v, &grad, &rho| {
                    *v = -dt * grad / rho;
                });
            Zip::from(&mut velocity_z)
                .and(&grad_z)
                .and(&rho_array)
                .for_each(|v, &grad, &rho| {
                    *v = -dt * grad / rho;
                });
        }
        
        // Update pressure and velocity with CPML if configured
        if self.solver.boundary.is_some() {
            // Use CPML-aware updates
            let vx = velocity_x.to_owned();
            let vy = velocity_y.to_owned();
            let vz = velocity_z.to_owned();
            self.solver.update_pressure_with_cpml(&mut pressure, &vx, &vy, &vz, medium, dt)?;
            self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure.view(), medium, dt)?;
        } else {
            // Use standard k-space updates
            let divergence_kspace = self.solver.compute_divergence_kspace(&velocity_x.view(), &velocity_y.view(), &velocity_z.view())?;
            self.solver.update_pressure_kspace(&mut pressure, &divergence_kspace, medium, dt)?;
            self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure.view(), medium, dt)?;
        }
        
        // No copy back needed - we modified the views directly!
        
        Ok(())
    }
    
    fn required_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        use crate::physics::field_mapping::UnifiedFieldType;
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
            UnifiedFieldType::Density,
            UnifiedFieldType::SoundSpeed,
        ]
    }
    
    fn provided_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        use crate::physics::field_mapping::UnifiedFieldType;
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(PstdPlugin {
            solver: PstdSolver::new(self.solver.config.clone(), &self.solver.grid).unwrap(),
            metadata: self.metadata.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}