//! Seismic Imaging Parameters and Configuration
//!
//! Configuration structures for Full Waveform Inversion and Reverse Time Migration
//! Following GRASP principles: Information Expert pattern for parameter management

/// Full Waveform Inversion Parameters
#[derive(Debug, Clone)]
pub struct FwiParameters {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Regularization weight
    pub regularization: RegularizationParameters,
}

/// Regularization parameters for inversion
#[derive(Debug, Clone)]
pub struct RegularizationParameters {
    /// Tikhonov regularization weight
    pub tikhonov_weight: f64,
    /// Total variation weight
    pub tv_weight: f64,
    /// Smoothness constraint weight
    pub smoothness_weight: f64,
}

/// Convergence criteria for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Absolute tolerance
    pub absolute_tolerance: f64,
}

/// Reverse Time Migration Settings
#[derive(Debug, Clone)]
pub struct RtmSettings {
    /// Imaging condition type
    pub imaging_condition: ImagingCondition,
    /// Source wavefield storage strategy
    pub storage_strategy: StorageStrategy,
    /// Boundary handling method
    pub boundary_type: BoundaryType,
    /// Apply Laplacian filtering
    pub apply_laplacian: bool,
}

/// Available imaging conditions for RTM
#[derive(Debug, Clone)]
pub enum ImagingCondition {
    /// Zero-lag cross-correlation
    ZeroLag,
    /// Normalized cross-correlation
    Normalized,
}

/// Wavefield storage strategies for memory management
#[derive(Debug, Clone)]
pub enum StorageStrategy {
    /// Store all timeframes (high memory)
    Full,
    /// Checkpointing strategy (balanced)
    Checkpoints(usize),
}

/// Boundary condition types for imaging
#[derive(Debug, Clone)]
pub enum BoundaryType {
    /// Absorbing boundary layers
    Absorbing,
    /// Free surface boundary
    FreeSurface,
    /// Periodic boundaries
    Periodic,
}

/// Migration aperture for focusing
#[derive(Debug, Clone)]
pub struct MigrationAperture {
    /// Half-aperture angle in radians
    pub half_angle: f64,
    /// Maximum migration distance
    pub max_distance: f64,
    /// Aperture taper function
    pub taper: TaperFunction,
}

/// Taper functions for aperture control
#[derive(Debug, Clone)]
pub enum TaperFunction {
    /// No tapering
    None,
    /// Cosine taper
    Cosine,
    /// Gaussian taper
    Gaussian { sigma: f64 },
}

impl Default for FwiParameters {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            step_size: 0.01,
            regularization: RegularizationParameters::default(),
        }
    }
}

impl Default for RegularizationParameters {
    fn default() -> Self {
        Self {
            tikhonov_weight: 1e-3,
            tv_weight: 1e-4,
            smoothness_weight: 1e-2,
        }
    }
}

impl Default for RtmSettings {
    fn default() -> Self {
        Self {
            imaging_condition: ImagingCondition::ZeroLag,
            storage_strategy: StorageStrategy::Checkpoints(10),
            boundary_type: BoundaryType::Absorbing,
            apply_laplacian: true,
        }
    }
}

impl Default for MigrationAperture {
    fn default() -> Self {
        Self {
            half_angle: std::f64::consts::PI / 6.0, // 30 degrees
            max_distance: 1000.0,                   // 1km
            taper: TaperFunction::Cosine,
        }
    }
}
