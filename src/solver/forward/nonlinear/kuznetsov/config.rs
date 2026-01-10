//! Configuration structures and constants for Kuznetsov equation solver
//!
//! This module provides configuration options for the Kuznetsov equation solver,
//! including numerical parameters, physical constants, and equation mode selection.

use crate::core::error::{KwaversError, ValidationError};
use crate::domain::grid::Grid;

// Physical constants and numerical parameters
/// Default CFL factor for Kuznetsov equation stability
pub const DEFAULT_CFL_FACTOR: f64 = 0.3;

/// Maximum allowed CFL factor
pub const MAX_CFL_FACTOR: f64 = 1.0;

/// Default k-space correction order
pub const DEFAULT_K_SPACE_CORRECTION_ORDER: usize = 4;

/// Maximum k-space correction order
pub const MAX_K_SPACE_CORRECTION_ORDER: usize = 4;

/// Default spatial accuracy order
pub const DEFAULT_SPATIAL_ORDER: usize = 4;

/// Default nonlinearity scaling factor
pub const DEFAULT_NONLINEARITY_SCALING: f64 = 1.0;

/// Default diffusivity coefficient
pub const DEFAULT_DIFFUSIVITY: f64 = 1.0;

/// Maximum k-space correction factor
pub const MAX_K_SPACE_CORRECTION: f64 = 2.0;

/// Central difference coefficient for gradient
pub const CENTRAL_DIFF_COEFF: f64 = 2.0;

/// Second derivative finite difference coefficient
pub const SECOND_DERIV_COEFF: f64 = 2.0;

/// Minimum history levels for time integration schemes
pub const MIN_HISTORY_LEVELS: usize = 2;

/// Default history levels for RK4 time stepping
pub const DEFAULT_HISTORY_LEVELS: usize = 4;

/// Maximum pressure threshold for shock detection \[Pa\]
pub const MAX_PRESSURE_THRESHOLD: f64 = 1e8;

/// Minimum grid spacing for stability \[m\]
pub const MIN_GRID_SPACING: f64 = 1e-6;

/// Maximum nonlinearity coefficient B/A
pub const MAX_NONLINEARITY_COEFFICIENT: f64 = 20.0;

/// k-space correction factors for different orders
pub const KUZNETSOV_K_SPACE_CORRECTION_SECOND_ORDER: f64 = 0.05;
pub const KUZNETSOV_K_SPACE_CORRECTION_FOURTH_ORDER: f64 = 0.01;

/// Acoustic equation mode selector
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AcousticEquationMode {
    /// Full Kuznetsov equation with all nonlinear and diffusive terms
    #[default]
    FullKuznetsov,
    /// KZK parabolic approximation for focused beams
    KZK,
    /// Westervelt equation for nonlinear acoustic wave propagation
    Westervelt,
    /// Linear wave equation (no nonlinearity)
    Linear,
}

/// Configuration for Kuznetsov equation solver
#[derive(Debug, Clone)]
pub struct KuznetsovConfig {
    /// Equation mode selector
    pub equation_mode: AcousticEquationMode,

    /// CFL factor for time step calculation (0 < `cfl_factor` <= 1)
    pub cfl_factor: f64,

    /// Nonlinearity coefficient B/A (typical values: 3.5-12 for biological tissues)
    pub nonlinearity_coefficient: f64,

    /// Acoustic diffusivity δ [m²/s] for thermoviscous losses
    pub acoustic_diffusivity: f64,

    /// Enable k-space correction for accurate time stepping
    pub use_k_space_correction: bool,

    /// Order of k-space correction (2 or 4)
    pub k_space_correction_order: usize,

    /// Spatial accuracy order (2, 4, or 6)
    pub spatial_order: usize,

    /// Enable adaptive time stepping
    pub adaptive_time_stepping: bool,

    /// Maximum pressure before limiting \[Pa\]
    pub max_pressure: f64,

    /// Enable shock capturing for strong nonlinearity
    pub shock_capturing: bool,

    /// Number of history levels for time integration
    pub history_levels: usize,

    /// Nonlinearity scaling factor (for testing/calibration)
    pub nonlinearity_scaling: f64,

    /// Diffusivity scaling factor (for testing/calibration)
    pub diffusivity: f64,
}

impl Default for KuznetsovConfig {
    fn default() -> Self {
        Self {
            equation_mode: AcousticEquationMode::FullKuznetsov,
            cfl_factor: DEFAULT_CFL_FACTOR,
            nonlinearity_coefficient: 5.0, // Typical for soft tissue
            acoustic_diffusivity: 4.5e-6,  // Typical for water at 20°C
            use_k_space_correction: true,
            k_space_correction_order: DEFAULT_K_SPACE_CORRECTION_ORDER,
            spatial_order: DEFAULT_SPATIAL_ORDER,
            adaptive_time_stepping: false,
            max_pressure: MAX_PRESSURE_THRESHOLD,
            shock_capturing: true,
            history_levels: DEFAULT_HISTORY_LEVELS,
            nonlinearity_scaling: DEFAULT_NONLINEARITY_SCALING,
            diffusivity: DEFAULT_DIFFUSIVITY,
        }
    }
}

impl KuznetsovConfig {
    /// Create configuration for KZK parabolic approximation
    #[must_use]
    pub fn kzk() -> Self {
        Self {
            equation_mode: AcousticEquationMode::KZK,
            cfl_factor: 0.5, // More conservative for paraxial approximation
            ..Default::default()
        }
    }

    /// Create configuration for Westervelt equation
    #[must_use]
    pub fn westervelt() -> Self {
        Self {
            equation_mode: AcousticEquationMode::Westervelt,
            acoustic_diffusivity: 0.0, // Westervelt doesn't include diffusivity term
            ..Default::default()
        }
    }

    /// Create configuration for linear wave equation
    #[must_use]
    pub fn linear() -> Self {
        Self {
            equation_mode: AcousticEquationMode::Linear,
            nonlinearity_coefficient: 0.0,
            acoustic_diffusivity: 0.0,
            shock_capturing: false,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self, grid: &Grid) -> Result<(), KwaversError> {
        // Check CFL factor
        if self.cfl_factor <= 0.0 || self.cfl_factor > MAX_CFL_FACTOR {
            return Err(KwaversError::Validation(ValidationError::OutOfRange {
                value: self.cfl_factor,
                min: 0.0,
                max: MAX_CFL_FACTOR,
            }));
        }

        // Check grid spacing
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        if min_dx <= MIN_GRID_SPACING {
            return Err(KwaversError::Validation(ValidationError::OutOfRange {
                value: min_dx,
                min: MIN_GRID_SPACING,
                max: f64::INFINITY,
            }));
        }

        // Check nonlinearity coefficient
        if self.nonlinearity_coefficient < 0.0
            || self.nonlinearity_coefficient > MAX_NONLINEARITY_COEFFICIENT
        {
            return Err(KwaversError::Validation(ValidationError::OutOfRange {
                value: self.nonlinearity_coefficient,
                min: 0.0,
                max: MAX_NONLINEARITY_COEFFICIENT,
            }));
        }

        // Check k-space correction order
        if self.use_k_space_correction
            && (self.k_space_correction_order != 2 && self.k_space_correction_order != 4)
        {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "k_space_correction_order".to_string(),
                value: self.k_space_correction_order.to_string(),
                constraint: "must be 2 or 4".to_string(),
            }));
        }

        // Check spatial order
        if self.spatial_order != 2 && self.spatial_order != 4 && self.spatial_order != 6 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "spatial_order".to_string(),
                value: self.spatial_order.to_string(),
                constraint: "must be 2, 4, or 6".to_string(),
            }));
        }

        Ok(())
    }
}
