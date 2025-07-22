use crate::boundary::Boundary;
use crate::grid::Grid;
use crate::error::{KwaversResult, ConfigError};
use log::trace;
use ndarray::{Array3, Zip}; // Removed parallel prelude
use rustfft::num_complex::Complex;

/// Perfectly Matched Layer (PML) boundary condition for absorbing outgoing waves.
///
/// This implementation uses a polynomial grading of the absorption profile
/// with optional backing by a theoretical model for automatic parameter selection.
#[derive(Debug, Clone)]
pub struct PMLBoundary {
    // thickness: usize, // Removed
    // sigma_max_acoustic: f64, // Removed
    // sigma_max_light: f64, // Removed
    /// Pre-computed damping profiles for each dimension
    acoustic_damping_x: Vec<f64>,
    acoustic_damping_y: Vec<f64>,
    acoustic_damping_z: Vec<f64>,
    light_damping_x: Vec<f64>,
    light_damping_y: Vec<f64>,
    light_damping_z: Vec<f64>,
    // polynomial_order: usize, // Removed
    // target_reflection: f64, // Removed
    /// Pre-computed combined damping factors for optimization
    acoustic_damping_3d: Option<Array3<f64>>,
    light_damping_3d: Option<Array3<f64>>,
}

/// Configuration for PML boundary layer
/// Follows SOLID principles by grouping related parameters together
#[derive(Debug, Clone)]
pub struct PMLConfig {
    pub thickness: usize,
    pub sigma_max_acoustic: f64,
    pub sigma_max_light: f64,
    pub alpha_max_acoustic: f64,
    pub alpha_max_light: f64,
    pub kappa_max_acoustic: f64,
    pub kappa_max_light: f64,
    pub target_reflection: Option<f64>,
}

impl Default for PMLConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            sigma_max_acoustic: 2.0,
            sigma_max_light: 1.0,
            alpha_max_acoustic: 0.0,
            alpha_max_light: 0.0,
            kappa_max_acoustic: 1.0,
            kappa_max_light: 1.0,
            target_reflection: Some(1e-4),
        }
    }
}

impl PMLConfig {
    /// Set PML thickness
    pub fn with_thickness(mut self, thickness: usize) -> Self {
        self.thickness = thickness;
        self
    }

    /// Set reflection coefficient
    pub fn with_reflection_coefficient(mut self, reflection: f64) -> Self {
        self.target_reflection = Some(reflection);
        self
    }

    /// Validate PML configuration parameters
    /// Follows SOLID Single Responsibility Principle
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "thickness".to_string(),
                value: self.thickness.to_string(),
                reason: "PML thickness must be > 0".to_string(),
            }.into());
        }
        
        if self.sigma_max_acoustic < 0.0 || self.sigma_max_light < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_max".to_string(),
                value: format!("acoustic: {}, light: {}", self.sigma_max_acoustic, self.sigma_max_light),
                reason: "Sigma values must be >= 0".to_string(),
            }.into());
        }
        
        Ok(())
    }
}

impl PMLBoundary {
    /// Create new PML boundary with configuration struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn new(config: PMLConfig) -> KwaversResult<Self> {
        config.validate()?;
        
        // Create damping profiles based on configuration
        let acoustic_profile = Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_acoustic, 2);
        let light_profile = Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_light, 2);
        
        Ok(Self {
            acoustic_damping_x: acoustic_profile.clone(),
            acoustic_damping_y: acoustic_profile.clone(),
            acoustic_damping_z: acoustic_profile.clone(),
            light_damping_x: light_profile.clone(),
            light_damping_y: light_profile.clone(),
            light_damping_z: light_profile,
            acoustic_damping_3d: None,
            light_damping_3d: None,
        })
    }
    
    /// Create with default configuration
    pub fn with_defaults() -> KwaversResult<Self> {
        Self::new(PMLConfig::default())
    }

    /// Creates a new PML boundary with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points (0 for auto-selection)
    /// * `sigma_max_acoustic` - Maximum acoustic absorption coefficient (0 for auto-selection)
    /// * `sigma_max_light` - Maximum light absorption coefficient (0 for auto-selection)
    /// * `medium` - The simulation medium
    /// * `grid` - The simulation grid
    /// * `acoustic_freq` - Characteristic acoustic frequency
    /// * `polynomial_order` - Order of polynomial grading (default: 3)
    /// * `target_reflection` - Target theoretical reflection coefficient (default: 1e-6)
    ///
    /// Creates a damping profile for a PML layer.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points
    /// * `length` - Total domain length in grid points
    /// * `dx` - Grid spacing
    /// * `sigma_max` - Maximum absorption coefficient
    /// * `order` - Polynomial order for profile grading
    fn damping_profile(thickness: usize, length: usize, _dx: f64, sigma_max: f64, order: usize) -> Vec<f64> {
        let mut profile = vec![0.0; length];
        
        // Apply PML at both domain boundaries (left/right or top/bottom)
        // Left/bottom boundary
        for (i, profile_val) in profile.iter_mut().enumerate().take(thickness) {
            let normalized_distance = (thickness - i) as f64 / thickness as f64;
            *profile_val = sigma_max * normalized_distance.powi(order as i32);
        }
        
        // Right/top boundary
        for i in 0..thickness {
            let idx = length - i - 1;
            let normalized_distance = i as f64 / thickness as f64;
            profile[idx] = sigma_max * normalized_distance.powi(order as i32);
        }
        
        profile
    }
    
    /// Applies a pre-computed damping factor to a field value
    #[inline]
    fn apply_damping(val: &mut f64, damping: f64, dx: f64) {
        if damping > 0.0 {
            *val *= (-damping * dx).exp();
        }
    }
    
    /// Applies a pre-computed damping factor to a complex field value
    #[inline]
    fn apply_complex_damping(val: &mut Complex<f64>, damping: f64, dx: f64) {
        if damping > 0.0 {
            let decay = (-damping * dx).exp();
            val.re *= decay;
            val.im *= decay;
        }
    }

    /// Precomputes the 3D damping factors for acoustic fields to avoid repeated calculations
    fn precompute_acoustic_damping_3d(&mut self, grid: &Grid) {
        if self.acoustic_damping_3d.is_none() {
            trace!("Precomputing 3D acoustic damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            Zip::indexed(&mut damping_3d).par_for_each(|(i, j, k), val| {
                *val = self.acoustic_damping_x[i] + self.acoustic_damping_y[j] + self.acoustic_damping_z[k];
            });
            
            self.acoustic_damping_3d = Some(damping_3d);
        }
    }
    
    /// Precomputes the 3D damping factors for light fields to avoid repeated calculations
    fn precompute_light_damping_3d(&mut self, grid: &Grid) {
        if self.light_damping_3d.is_none() {
            trace!("Precomputing 3D light damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            Zip::indexed(&mut damping_3d).par_for_each(|(i, j, k), val| {
                *val = self.light_damping_x[i] + self.light_damping_y[j] + self.light_damping_z[k];
            });
            
            self.light_damping_3d = Some(damping_3d);
        }
    }
}

impl Boundary for PMLBoundary {
    fn apply_acoustic(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying spatial acoustic PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
    }

    fn apply_acoustic_freq(&mut self, field: &mut Array3<Complex<f64>>, grid: &Grid, time_step: usize) {
        trace!("Applying frequency domain acoustic PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_complex_damping(val, damping, dx);
            });
    }

    fn apply_light(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying light PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_light_damping_3d(grid);
        let damping_3d = self.light_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
    }
}
