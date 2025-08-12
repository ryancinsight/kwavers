//! Common constants used throughout the Kwavers simulation framework
//!
//! This module defines named constants to avoid magic numbers and ensure
//! consistency across the codebase, following SSOT and DRY principles.

use std::f64::consts::PI;

// Re-export standard mathematical constants for convenience
pub use std::f64::consts::{E, PI as PI_CONST, TAU};

/// Numerical tolerance constants
pub mod tolerance {
    /// Default tolerance for floating-point comparisons
    pub const DEFAULT: f64 = 1e-10;
    
    /// Tolerance for conservation law enforcement
    pub const CONSERVATION: f64 = 1e-6;
    
    /// Tolerance for iterative solver convergence
    pub const CONVERGENCE: f64 = 1e-8;
    
    /// Tolerance for stability checks
    pub const STABILITY: f64 = 1e-4;
    
    /// Tolerance for error estimation
    pub const ERROR_ESTIMATION: f64 = 1e-3;
}

/// CFL (Courant-Friedrichs-Lewy) condition constants
pub mod cfl {
    /// Default CFL safety factor for FDTD
    pub const FDTD_DEFAULT: f64 = 0.95;
    
    /// Default CFL safety factor for PSTD
    pub const PSTD_DEFAULT: f64 = 0.3;
    
    /// Conservative CFL safety factor
    pub const CONSERVATIVE: f64 = 0.5;
    
    /// Aggressive CFL safety factor (use with caution)
    pub const AGGRESSIVE: f64 = 0.8;
}

/// Physical constants
pub mod physics {
    /// Speed of sound in water at 20°C (m/s)
    pub const SOUND_SPEED_WATER: f64 = 1480.0;
    
    /// Speed of sound in soft tissue (m/s)
    pub const SOUND_SPEED_TISSUE: f64 = 1540.0;
    
    /// Density of water at 20°C (kg/m³)
    pub const DENSITY_WATER: f64 = 998.0;
    
    /// Density of soft tissue (kg/m³)
    pub const DENSITY_TISSUE: f64 = 1050.0;
    
    /// Nonlinearity parameter B/A for water
    pub const NONLINEARITY_WATER: f64 = 5.0;
    
    /// Nonlinearity parameter B/A for soft tissue
    pub const NONLINEARITY_TISSUE: f64 = 6.5;
}

/// Grid and discretization constants
pub mod grid {
    /// Minimum grid points per wavelength for accurate simulation
    pub const MIN_POINTS_PER_WAVELENGTH: usize = 6;
    
    /// Optimal grid points per wavelength
    pub const OPTIMAL_POINTS_PER_WAVELENGTH: usize = 10;
    
    /// Default PML thickness in grid points
    pub const DEFAULT_PML_THICKNESS: usize = 10;
    
    /// Default buffer zone width for domain coupling
    pub const DEFAULT_BUFFER_WIDTH: usize = 4;
}

/// Stability and threshold constants
pub mod stability {
    /// Default stability threshold for interface coupling
    pub const INTERFACE_THRESHOLD: f64 = 0.8;
    
    /// Maximum allowed pressure (Pa) to prevent numerical overflow
    pub const MAX_PRESSURE: f64 = 1e8;
    
    /// Maximum allowed gradient for stability
    pub const MAX_GRADIENT: f64 = 1e6;
    
    /// Smoothing factor for interface transitions
    pub const SMOOTHING_FACTOR: f64 = 0.1;
}

/// Performance and optimization constants
pub mod performance {
    /// Chunk size for small grids
    pub const CHUNK_SIZE_SMALL: usize = 4 * 1024;
    
    /// Chunk size for medium grids
    pub const CHUNK_SIZE_MEDIUM: usize = 16 * 1024;
    
    /// Chunk size for large grids
    pub const CHUNK_SIZE_LARGE: usize = 64 * 1024;
    
    /// Threshold for considering a grid "large"
    pub const LARGE_GRID_THRESHOLD: usize = 1_000_000;
    
    /// Threshold for considering a grid "medium"
    pub const MEDIUM_GRID_THRESHOLD: usize = 100_000;
    
    /// Threshold for enabling chunked processing
    pub const CHUNKED_PROCESSING_THRESHOLD: usize = 10_000;
}

/// Interpolation and reconstruction constants
pub mod interpolation {
    /// Order for cubic spline interpolation
    pub const CUBIC_SPLINE_ORDER: usize = 4;
    
    /// Order for spectral interpolation
    pub const SPECTRAL_ORDER: usize = 8;
    
    /// Default WENO epsilon for avoiding division by zero
    pub const WENO_EPSILON: f64 = 1e-6;
}