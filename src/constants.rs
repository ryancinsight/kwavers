//! Common constants used throughout the Kwavers simulation framework
//!
//! This module defines named constants to avoid magic numbers and ensure
//! consistency across the codebase, following SSOT and DRY principles.


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

/// Bubble dynamics constants
pub mod bubble_dynamics {
    /// Peclet number scaling factor for effective polytropic index
    /// Based on heat transfer considerations in bubble dynamics
    /// Reference: Prosperetti & Lezzi (1986), "Bubble dynamics in a compressible liquid"
    pub const PECLET_SCALING_FACTOR: f64 = 10.0;
    
    /// Minimum Peclet number to avoid division by zero
    pub const MIN_PECLET_NUMBER: f64 = 0.1;
    
    /// Default accommodation coefficient for mass transfer
    pub const DEFAULT_ACCOMMODATION_COEFF: f64 = 0.04;
    
    /// Nusselt number base value for spherical heat transfer
    pub const NUSSELT_BASE: f64 = 2.0;
    
    /// Nusselt number Peclet coefficient
    pub const NUSSELT_PECLET_COEFF: f64 = 0.6;
    
    /// Nusselt number Peclet exponent
    pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;
    
    /// Minimum bubble radius ratio for Rayleigh regime
    pub const RAYLEIGH_REGIME_RATIO: f64 = 0.1;
    
    /// Default initial bubble radius (microns converted to meters)
    pub const DEFAULT_R0: f64 = 5e-6;
    
    /// Typical cavitation threshold pressure (Pa)
    pub const CAVITATION_THRESHOLD: f64 = -1e5;
    
    /// Minimum bubble radius (1 nm)
    pub const MIN_RADIUS: f64 = 1e-9;
    
    /// Maximum bubble radius (1 cm)
    pub const MAX_RADIUS: f64 = 1e-2;
    
    /// Air composition - Nitrogen fraction
    pub const N2_FRACTION: f64 = 0.79;
    
    /// Air composition - Oxygen fraction
    pub const O2_FRACTION: f64 = 0.21;
    
    /// Van der Waals constant a for N2 (bar·L²/mol²)
    pub const VDW_A_N2: f64 = 1.370;
    
    /// Van der Waals constant b for N2 (L/mol)
    pub const VDW_B_N2: f64 = 0.0387;
    
    /// Van der Waals constant a for O2 (bar·L²/mol²)
    pub const VDW_A_O2: f64 = 1.382;
    
    /// Van der Waals constant b for O2 (L/mol)
    pub const VDW_B_O2: f64 = 0.0319;
    
    /// Unit conversion: bar·L²/mol² to Pa·m⁶/mol²
    pub const BAR_L2_TO_PA_M6: f64 = 0.1;
    
    /// Unit conversion: L/mol to m³/mol
    pub const L_TO_M3: f64 = 1e-3;
}

/// Thermodynamics constants
pub mod thermodynamics {
    /// Universal gas constant (J/(mol·K))
    pub const R_GAS: f64 = 8.314462618;
    
    /// Avogadro's number (molecules/mol)
    pub const AVOGADRO: f64 = 6.02214076e23;
    
    /// Molecular weight of water (kg/mol)
    pub const M_WATER: f64 = 0.018015;
    
    /// Ambient temperature (K)
    pub const T_AMBIENT: f64 = 293.15;
    
    /// Standard atmospheric pressure (Pa)
    pub const P_ATMOSPHERIC: f64 = 101325.0;
    
    /// Surface tension of water at 20°C (N/m)
    pub const SIGMA_WATER: f64 = 0.0728;
    
    /// Dynamic viscosity of water at 20°C (Pa·s)
    pub const MU_WATER: f64 = 1.002e-3;
    
    /// Thermal conductivity of water at 20°C (W/(m·K))
    pub const K_THERMAL_WATER: f64 = 0.598;
    
    /// Specific heat capacity of water at constant pressure (J/(kg·K))
    pub const CP_WATER: f64 = 4182.0;
    
    /// Specific heat capacity of water at constant volume (J/(kg·K))
    pub const CV_WATER: f64 = 4150.0;
}

/// Optical and visualization constants
pub mod optics {
    /// Wavelength ranges for visible spectrum (nm)
    pub const VIOLET_WAVELENGTH: f64 = 440.0;
    pub const CYAN_WAVELENGTH: f64 = 490.0;
    pub const GREEN_WAVELENGTH: f64 = 510.0;
    pub const YELLOW_WAVELENGTH: f64 = 580.0;
    pub const RED_WAVELENGTH: f64 = 645.0;
    
    /// Default pulse width factor for Gaussian pulses
    pub const GAUSSIAN_PULSE_WIDTH_FACTOR: f64 = 10.0;
    
    /// Default pulse center factor
    pub const GAUSSIAN_PULSE_CENTER_FACTOR: f64 = 20.0;
    
    /// Window width factor for edge effect reduction
    pub const WINDOW_WIDTH_FACTOR: f64 = 10.0;
    
    /// Typical tissue diffusion coefficient (mm²/ns)
    pub const TISSUE_DIFFUSION_COEFFICIENT: f64 = 1e-3;
    
    /// Typical tissue absorption coefficient (mm⁻¹)
    pub const TISSUE_ABSORPTION_COEFFICIENT: f64 = 0.1;
    
    /// Default polarization factor
    pub const DEFAULT_POLARIZATION_FACTOR: f64 = 0.5;
    
    /// Laplacian stencil center coefficient
    pub const LAPLACIAN_CENTER_COEFF: f64 = -2.0;
}

/// Shock capturing and numerical methods constants
pub mod numerical {
    /// WENO scheme optimal weights
    pub const WENO_WEIGHT_0: f64 = 0.1;
    pub const WENO_WEIGHT_1: f64 = 0.6;
    pub const WENO_WEIGHT_2: f64 = 0.3;
    
    /// Artificial viscosity coefficients
    pub const VON_NEUMANN_RICHTMYER_COEFF: f64 = 2.0;
    pub const LINEAR_VISCOSITY_COEFF: f64 = 0.1;
    pub const QUADRATIC_VISCOSITY_COEFF: f64 = 1.5;
    pub const MAX_VISCOSITY_LIMIT: f64 = 0.1;
    
    /// WENO smoothness indicator epsilon
    pub const WENO_EPSILON: f64 = 1e-6;
    
    /// Stencil coefficients for numerical differentiation
    pub const STENCIL_COEFF_3_4: f64 = 0.75;  // 3/4
    pub const STENCIL_COEFF_1_4: f64 = 0.25;  // 1/4
    pub const STENCIL_COEFF_1_2: f64 = 0.5;   // 1/2
}

/// Chemistry and ROS constants
pub mod chemistry {
    /// Relative damage weights for ROS species
    pub const HYDROXYL_RADICAL_WEIGHT: f64 = 10.0;  // Most damaging
    pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 1.0;
    pub const SUPEROXIDE_WEIGHT: f64 = 2.0;
    pub const SINGLET_OXYGEN_WEIGHT: f64 = 5.0;
    pub const PEROXYNITRITE_WEIGHT: f64 = 8.0;
    pub const NITRIC_OXIDE_WEIGHT: f64 = 0.5;
}

/// Cavitation damage constants
pub mod cavitation {
    /// Default damage threshold pressure (Pa)
    pub const DEFAULT_THRESHOLD_PRESSURE: f64 = 100e6;  // 100 MPa
    
    /// Default pit formation efficiency (fraction of impacts that cause pits)
    pub const DEFAULT_PIT_EFFICIENCY: f64 = 0.01;  // 1%
    
    /// Default fatigue damage rate per cycle
    pub const DEFAULT_FATIGUE_RATE: f64 = 1e-6;
    
    /// Default stress concentration factor
    pub const DEFAULT_CONCENTRATION_FACTOR: f64 = 2.0;
    
    /// Material removal efficiency coefficient (empirical)
    pub const MATERIAL_REMOVAL_EFFICIENCY: f64 = 1e-9;
    
    /// Impact energy coefficient for erosion model
    pub const IMPACT_ENERGY_COEFFICIENT: f64 = 0.5;
    
    /// Compression factor exponent for cavitation intensity
    pub const COMPRESSION_FACTOR_EXPONENT: f64 = 1.5;
}

/// Test and validation constants
pub mod validation {
    /// Default temperature rise for thermal tests (K)
    pub const DEFAULT_TEMP_RISE: f64 = 10.0;
    
    /// Measurement radius factors for validation
    pub const MEASUREMENT_RADIUS_FACTOR_1: f64 = 5.0;
    pub const MEASUREMENT_RADIUS_FACTOR_2: f64 = 10.0;
    
    /// Minimum time scale separation ratio for multi-rate methods
    pub const MIN_TIMESCALE_SEPARATION: f64 = 10.0;
    
    /// Smooth background sigma factor
    pub const SMOOTH_BACKGROUND_SIGMA_FACTOR: f64 = 10.0;
    
    /// Wavelet coefficient ratio threshold for sharp feature detection
    pub const WAVELET_SHARP_FEATURE_RATIO: f64 = 10.0;
    
    /// Default compression ratio for bubble tests
    pub const DEFAULT_COMPRESSION_RATIO: f64 = 10.0;
    
    /// Default bubble wall velocities for interaction tests (m/s)
    pub const DEFAULT_EXPANSION_VELOCITY_1: f64 = 10.0;
    pub const DEFAULT_EXPANSION_VELOCITY_2: f64 = 5.0;
}

/// Adaptive integration stability constants
pub mod adaptive_integration {
    /// Minimum radius safety factor (fraction of MIN_RADIUS)
    pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.1;
    
    /// Maximum radius safety factor (multiple of MAX_RADIUS)
    pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 10.0;
    
    /// Maximum velocity fraction of sound speed
    pub const MAX_VELOCITY_FRACTION: f64 = 0.9;
    
    /// Minimum temperature for stability (K)
    pub const MIN_TEMPERATURE: f64 = 100.0;
    
    /// Maximum temperature for stability (K)
    pub const MAX_TEMPERATURE: f64 = 100000.0;
}