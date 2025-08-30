use crate::boundary::Boundary;
use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;
use log::trace;
use ndarray::{Array3, ArrayViewMut3, Zip};

use rustfft::num_complex::Complex;

// Physical constants for PML boundary parameters
/// Exponential scaling factor for PML absorption profile
/// This factor adds a small exponential component to the polynomial PML profile
/// to modify absorption characteristics at grazing angles. The value 0.1 provides
/// a 10% scaling without destabilizing the absorption profile.
/// Based on: Berenger, "A perfectly matched layer for absorption of electromagnetic waves"
const PML_EXPONENTIAL_SCALING_FACTOR: f64 = 0.1;

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
    /// Configuration for this PML boundary
    config: PMLConfig,
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
                constraint: "PML thickness must be > 0".to_string(),
            }
            .into());
        }

        if self.sigma_max_acoustic < 0.0 || self.sigma_max_light < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_max".to_string(),
                value: format!(
                    "acoustic: {}, light: {}",
                    self.sigma_max_acoustic, self.sigma_max_light
                ),
                constraint: "Sigma values must be >= 0".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl PMLBoundary {
    /// Create new PML boundary with configuration struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn new(config: PMLConfig) -> KwaversResult<Self> {
        config.validate()?;

        // Create empty damping profiles - will be initialized on first use
        Ok(Self {
            acoustic_damping_x: Vec::new(),
            acoustic_damping_y: Vec::new(),
            acoustic_damping_z: Vec::new(),
            light_damping_x: Vec::new(),
            light_damping_y: Vec::new(),
            light_damping_z: Vec::new(),
            acoustic_damping_3d: None,
            light_damping_3d: None,
            config,
        })
    }

    /// Initialize damping profiles for the given grid
    fn initialize_for_grid(&mut self, grid: &Grid) {
        // Only initialize if not already done or grid size changed
        if self.acoustic_damping_x.len() != grid.nx {
            self.acoustic_damping_x = Self::damping_profile(
                self.config.thickness,
                grid.nx,
                grid.dx,
                self.config.sigma_max_acoustic,
                2,
            );
        }
        if self.acoustic_damping_y.len() != grid.ny {
            self.acoustic_damping_y = Self::damping_profile(
                self.config.thickness,
                grid.ny,
                grid.dy,
                self.config.sigma_max_acoustic,
                2,
            );
        }
        if self.acoustic_damping_z.len() != grid.nz {
            self.acoustic_damping_z = Self::damping_profile(
                self.config.thickness,
                grid.nz,
                grid.dz,
                self.config.sigma_max_acoustic,
                2,
            );
        }
        if self.light_damping_x.len() != grid.nx {
            self.light_damping_x = Self::damping_profile(
                self.config.thickness,
                grid.nx,
                grid.dx,
                self.config.sigma_max_light,
                2,
            );
        }
        if self.light_damping_y.len() != grid.ny {
            self.light_damping_y = Self::damping_profile(
                self.config.thickness,
                grid.ny,
                grid.dy,
                self.config.sigma_max_light,
                2,
            );
        }
        if self.light_damping_z.len() != grid.nz {
            self.light_damping_z = Self::damping_profile(
                self.config.thickness,
                grid.nz,
                grid.dz,
                self.config.sigma_max_light,
                2,
            );
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> KwaversResult<Self> {
        Self::new(PMLConfig::default())
    }

    /// Creates a damping profile for a PML layer with frequency-dependent absorption.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points
    /// * `length` - Total domain length in grid points
    /// * `dx` - Grid spacing
    /// * `sigma_max` - Maximum absorption coefficient
    /// * `order` - Polynomial order for profile grading
    fn damping_profile(
        thickness: usize,
        length: usize,
        dx: f64,
        sigma_max: f64,
        order: usize,
    ) -> Vec<f64> {
        let mut profile = vec![0.0; length];

        // PML profile with exponential absorption characteristics
        // Theoretical reference sigma for reflection coefficient R
        let target_reflection: f64 = 1e-6; // -120 dB reflection
        let reference_sigma =
            -((order + 1) as f64) * target_reflection.ln() / (2.0 * thickness as f64 * dx);
        let sigma_eff = sigma_max.min(reference_sigma * 2.0); // Don't exceed theoretical reference

        // Apply PML at both domain boundaries (left/right or top/bottom)
        // Left/bottom boundary - polynomial grading
        for (i, profile_val) in profile.iter_mut().enumerate().take(thickness) {
            let normalized_distance = (thickness - i) as f64 / thickness as f64;
            let polynomial_factor = normalized_distance.powi(order as i32);

            // Add exponential component for grazing angle absorption
            let exponential_factor = (-2.0 * normalized_distance).exp();

            *profile_val = sigma_eff
                * polynomial_factor
                * (1.0 + PML_EXPONENTIAL_SCALING_FACTOR * exponential_factor);
        }

        // Right/top boundary
        (0..thickness).for_each(|i| {
            let idx = length - i - 1;
            let normalized_distance = i as f64 / thickness as f64;
            let polynomial_factor = normalized_distance.powi(order as i32);

            // Add exponential component for grazing angle absorption
            let exponential_factor = (-2.0 * normalized_distance).exp();

            profile[idx] = sigma_eff
                * polynomial_factor
                * (1.0 + PML_EXPONENTIAL_SCALING_FACTOR * exponential_factor);
        });

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
        // Ensure damping profiles are initialized for this grid
        self.initialize_for_grid(grid);

        if self.acoustic_damping_3d.is_none() {
            trace!("Precomputing 3D acoustic damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));

            Zip::indexed(&mut damping_3d).for_each(|(i, j, k), val| {
                *val = self.acoustic_damping_x[i]
                    + self.acoustic_damping_y[j]
                    + self.acoustic_damping_z[k];
            });

            self.acoustic_damping_3d = Some(damping_3d);
        }
    }

    /// Precomputes the 3D damping factors for light fields to avoid repeated calculations
    fn precompute_light_damping_3d(&mut self, grid: &Grid) {
        // Ensure damping profiles are initialized for this grid
        self.initialize_for_grid(grid);

        if self.light_damping_3d.is_none() {
            trace!("Precomputing 3D light damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));

            Zip::indexed(&mut damping_3d).for_each(|(i, j, k), val| {
                *val = self.light_damping_x[i] + self.light_damping_y[j] + self.light_damping_z[k];
            });

            self.light_damping_3d = Some(damping_3d);
        }
    }
}

impl Boundary for PMLBoundary {
    fn apply_acoustic(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
    ) -> crate::KwaversResult<()> {
        trace!("Applying spatial acoustic PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();

        // Apply damping using precomputed factors
        Zip::from(&mut field)
            .and(damping_3d)
            .for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    ) -> crate::KwaversResult<()> {
        trace!(
            "Applying frequency domain acoustic PML at step {}",
            time_step
        );
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();

        // Apply damping using precomputed factors
        Zip::from(field).and(damping_3d).for_each(|val, &damping| {
            Self::apply_complex_damping(val, damping, dx);
        });
        Ok(())
    }

    fn apply_light(&mut self, mut field: ArrayViewMut3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying light PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_light_damping_3d(grid);
        let damping_3d = self.light_damping_3d.as_ref().unwrap();

        // Apply damping using precomputed factors
        Zip::from(&mut field)
            .and(damping_3d)
            .for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
    }
}

impl PMLBoundary {
    /// Apply acoustic PML with custom damping factor
    /// Follows Open/Closed Principle: Extends functionality without modifying existing code
    pub fn apply_acoustic_with_factor(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
        damping_factor: f64,
    ) {
        trace!(
            "Applying acoustic PML with factor {} at step {}",
            damping_factor,
            time_step
        );
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();

        // Apply damping using precomputed factors with custom scaling
        Zip::from(&mut field)
            .and(damping_3d)
            .for_each(|val, &damping| {
                let scaled_damping = damping * damping_factor;
                Self::apply_damping(val, scaled_damping, dx);
            });
    }
}
