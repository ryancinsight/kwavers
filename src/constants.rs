//! Common constants used throughout the Kwavers simulation framework
//!
//! This module defines named constants to avoid magic numbers and ensure
//! consistency across the codebase, following SSOT and DRY principles.


// Re-export standard mathematical constants for convenience
pub use std::f64::consts::{E, PI as PI_CONST, TAU};

/// Numerical constants for finite differences and FFT
pub mod numerical {
    /// Coefficient for second-order central difference
    pub const SECOND_ORDER_DIFF_COEFF: f64 = 2.0;
    
    /// Coefficient for third-order finite difference
    pub const THIRD_ORDER_DIFF_COEFF: f64 = 3.0;
    
    /// FFT wavenumber scaling factor
    pub const FFT_K_SCALING: f64 = 2.0;
    
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
    
    /// Default numerical tolerance
    pub const DEFAULT_TOLERANCE: f64 = 1e-10;
    
    /// Machine epsilon for f64
    pub const EPSILON: f64 = f64::EPSILON;
    
    /// Default maximum iterations
    pub const MAX_ITERATIONS: usize = 1000;
    
    /// Convergence tolerance for iterative methods
    pub const CONVERGENCE_TOLERANCE: f64 = 1e-12;
    
    /// Default relaxation parameter
    pub const RELAXATION_PARAMETER: f64 = 0.5;
}

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
    
    /// Speed of sound in water at 25°C (m/s) - commonly used reference
    pub const SOUND_SPEED_WATER_25C: f64 = 1500.0;
    
    /// Speed of sound in air at 20°C (m/s)
    pub const SOUND_SPEED_AIR: f64 = 343.0;
    
    /// Speed of sound in soft tissue (m/s)
    pub const SOUND_SPEED_TISSUE: f64 = 1540.0;
    
    /// Density of water at 20°C (kg/m³)
    pub const DENSITY_WATER: f64 = 998.0;
    
    /// Reference density for HU calculations (kg/m³)
    pub const DENSITY_WATER_HU_REF: f64 = 1000.0;
    
    /// Default ultrasound frequency (Hz)
    pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;
    
    /// High pressure threshold for cavitation (Pa)
    pub const HIGH_PRESSURE_THRESHOLD: f64 = 1e8;
    
    /// Density of water at 20°C (kg/m³)
    pub const DENSITY_WATER: f64 = 998.0;
    
    /// Density of soft tissue (kg/m³)
    pub const DENSITY_TISSUE: f64 = 1050.0;
    
    /// Nonlinearity parameter B/A for water
    pub const NONLINEARITY_WATER: f64 = 5.0;
    
    /// Nonlinearity parameter B/A for soft tissue
    pub const NONLINEARITY_TISSUE: f64 = 6.5;
    
    /// Standard test pressure amplitude (Pa)
    pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e5;
    
    /// Standard spatial resolution for tests (m)
    pub const STANDARD_SPATIAL_RESOLUTION: f64 = 1e-4;
    
    /// Standard Gaussian beam width (m)
    pub const STANDARD_BEAM_WIDTH: f64 = 2e-3;
    
    /// Minimal nonlinearity coefficient for testing linear approximation
    pub const NEAR_LINEAR_NONLINEARITY: f64 = 1e-10;
    
    /// Offset for nonlinearity coefficient β = 1 + B/2A
    pub const NONLINEARITY_COEFFICIENT_OFFSET: f64 = 1.0;
    
    /// Divisor for B/A term in nonlinearity coefficient
    pub const B_OVER_A_DIVISOR: f64 = 2.0;
    
    /// Reference frequency for absorption coefficient (Hz)
    pub const REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ: f64 = 1_000_000.0;
    
    /// Grid center factor for coordinate calculations
    pub const GRID_CENTER_FACTOR: f64 = 2.0;
}

/// Grid and discretization constants
pub mod grid {
    /// Minimum grid points per wavelength for numerical stability
    pub const MIN_POINTS_PER_WAVELENGTH: usize = 6;
    
    /// Recommended grid points per wavelength
    pub const RECOMMENDED_POINTS_PER_WAVELENGTH: usize = 10;
    
    /// Default PML thickness in grid points
    pub const DEFAULT_PML_THICKNESS: usize = 10;
    
    /// Default buffer zone width for domain coupling
    pub const DEFAULT_BUFFER_WIDTH: usize = 4;
    
    /// Minimum grid points for stability
    pub const MIN_GRID_POINTS: usize = 16;
    
    /// Default grid spacing [m]
    pub const DEFAULT_GRID_SPACING: f64 = 1e-3;
    
    /// Minimum grid spacing for numerical stability [m]
    pub const MIN_GRID_SPACING: f64 = 1e-6;
    
    /// Maximum grid spacing for accuracy [m]
    pub const MAX_GRID_SPACING: f64 = 1e-2;
}

/// Stability and threshold constants
pub mod stability {
    /// Default stability threshold for interface coupling
    pub const INTERFACE_THRESHOLD: f64 = 0.8;
    
    /// Pressure limit (Pa) to prevent numerical overflow
    pub const PRESSURE_LIMIT: f64 = 1e12;
    
    /// Gradient limit for stability
    pub const GRADIENT_LIMIT: f64 = 1e8;
    
    /// Smoothing factor for interface transitions
    pub const SMOOTHING_FACTOR: f64 = 0.1;
    
    /// Default PSTD weight in hybrid solver blending (favors accuracy)
    pub const DEFAULT_PSTD_WEIGHT: f64 = 0.6;
    
    /// Default FDTD weight in hybrid solver blending (complements PSTD)
    pub const DEFAULT_FDTD_WEIGHT: f64 = 0.4;
}

/// Boundary condition constants
pub mod boundary {
    /// Default PML exponential enhancement factor
    pub const PML_EXPONENTIAL_ENHANCEMENT_FACTOR: f64 = 0.1;
    
    /// Standard damping factor for boundary tests
    pub const BOUNDARY_TEST_DAMPING: f64 = 0.98;
    
    /// C-PML damping factor for validation
    pub const CPML_TEST_DAMPING: f64 = 0.99;
    
    /// Grazing angle boundary effectiveness threshold
    pub const GRAZING_EFFECTIVENESS_THRESHOLD: f64 = 0.95;
    
    /// Standard reflection coefficient target
    pub const DEFAULT_REFLECTION_TARGET: f64 = 1e-6;
}

/// Solver and integration constants
pub mod solver {
    /// Standard Runge-Kutta timestep coefficient
    pub const RK_TIMESTEP_COEFFICIENT: f64 = 0.5;
    
    /// Default time integration safety factor  
    pub const TIME_INTEGRATION_SAFETY: f64 = 0.9;
    
    /// Field damping factor for tests
    pub const TEST_FIELD_DAMPING: f64 = 0.999;
    
    /// Energy conservation threshold for validation
    pub const ENERGY_CONSERVATION_THRESHOLD: f64 = 0.1;
    
    /// Default smoothing factor for interfaces
    pub const INTERFACE_SMOOTHING_FACTOR: f64 = 0.1;
    
    /// Stress PML damping factor
    pub const STRESS_PML_DAMPING_FACTOR: f64 = 1.2;
    
    /// Adaptive mesh refinement threshold
    pub const AMR_REFINEMENT_THRESHOLD: f64 = 0.1;
    
    /// Grid scaling factor
    pub const GRID_SCALING_FACTOR: f64 = 1.0;
    
    /// Boundary damping strength
    pub const BOUNDARY_DAMPING_STRENGTH: f64 = 1.2;
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
    
    /// Surface tension coefficient (factor of 2) from Young-Laplace equation
    pub const SURFACE_TENSION_COEFF: f64 = 2.0;
    
    /// Viscous stress coefficient (factor of 4) from Navier-Stokes
    pub const VISCOUS_STRESS_COEFF: f64 = 4.0;
    
    /// Kinetic energy coefficient (factor of 1.5) in Rayleigh-Plesset equation  
    pub const KINETIC_ENERGY_COEFF: f64 = 1.5;
    
    /// Latent heat of vaporization for water at standard conditions [J/kg]
    pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.26e6;
    
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
    /// Universal gas constant (J/mol·K)
    pub const R_GAS: f64 = 8.314462618;
    
    /// Avogadro's number
    pub const AVOGADRO: f64 = 6.02214076e23;
    
    /// Molecular weight of water (kg/mol)
    pub const M_WATER: f64 = 0.018015;
    
    /// Ambient temperature (K)
    pub const T_AMBIENT: f64 = 293.15;
    
    /// Van der Waals constant a for N2 (bar·L²/mol²)
    pub const VDW_A_N2: f64 = 1.370;
    
    /// Van der Waals constant b for N2 (L/mol)
    pub const VDW_B_N2: f64 = 0.0387;
    
    /// Van der Waals constant a for O2 (bar·L²/mol²)
    pub const VDW_A_O2: f64 = 1.382;
    
    /// Van der Waals constant b for O2 (L/mol)
    pub const VDW_B_O2: f64 = 0.0319;
    
    /// Van der Waals constant a for H2O (bar·L²/mol²)
    pub const VDW_A_H2O: f64 = 5.537;
    
    /// Van der Waals constant b for H2O (L/mol)
    pub const VDW_B_H2O: f64 = 0.0305;
    
    /// Conversion factor from bar·L² to Pa·m⁶
    pub const BAR_L2_TO_PA_M6: f64 = 0.1;
    
    /// Conversion factor from L to m³
    pub const L_TO_M3: f64 = 1e-3;
    
    /// Fraction of N2 in air
    pub const N2_FRACTION: f64 = 0.79;
    
    /// Fraction of O2 in air
    pub const O2_FRACTION: f64 = 0.21;
    
    /// Vapor diffusion coefficient in air at standard conditions (m²/s)
    pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.5e-5;
    
    /// Nusselt number constant term
    pub const NUSSELT_CONSTANT: f64 = 2.0;
    
    /// Nusselt number Peclet coefficient
    pub const NUSSELT_PECLET_COEFF: f64 = 0.6;
    
    /// Nusselt number Peclet exponent for heat transfer
    pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;
    
    /// Reference temperature for reaction rates (K)
    pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;
    
    /// Base reaction rate constant for sonochemistry (1/s)
    pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-4;
    
    /// Secondary reaction rate constant (1/s)
    pub const SECONDARY_REACTION_RATE: f64 = 5e-5;
    
    /// Sherwood number Peclet exponent for mass transfer
    pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;
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




/// Chemistry and ROS constants
pub mod chemistry {
    /// Relative damage weights for ROS species
    pub const HYDROXYL_RADICAL_WEIGHT: f64 = 10.0;  // Most damaging
    pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 1.0;
    pub const SUPEROXIDE_WEIGHT: f64 = 2.0;
    pub const SINGLET_OXYGEN_WEIGHT: f64 = 5.0;
    pub const PEROXYNITRITE_WEIGHT: f64 = 8.0;
    pub const NITRIC_OXIDE_WEIGHT: f64 = 0.5;
    
    /// Base photochemical initiation rate coefficient (1/s per unit light intensity)
    pub const BASE_PHOTOCHEMICAL_RATE: f64 = 1e-7;
    
    /// Default thermal diffusion coefficient (mm²/ns)
    pub const DEFAULT_THERMAL_DIFFUSION: f64 = 2.0;
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
    
    /// Field growth tolerance for stability checks
    pub const FIELD_GROWTH_TOLERANCE: f64 = 1.1;
    
    /// Amplitude conservation tolerance
    pub const AMPLITUDE_TOLERANCE: f64 = 0.85;
    
    /// Energy conservation threshold
    pub const ENERGY_CONSERVATION_THRESHOLD: f64 = 0.01;
    
    /// Phase error tolerance (radians)
    pub const PHASE_ERROR_TOLERANCE: f64 = 0.1;
}

/// Domain decomposition constants
pub mod domain_decomposition {
    /// Smoothness threshold for domain analysis
    pub const SMOOTHNESS_THRESHOLD: f64 = 0.1;
    
    /// Heterogeneity threshold for medium analysis
    pub const HETEROGENEITY_THRESHOLD: f64 = 0.2;
    
    /// Frequency cutoff as fraction of Nyquist frequency
    pub const FREQUENCY_CUTOFF_FRACTION: f64 = 0.3;
}

/// Adaptive integration constants
pub mod adaptive_integration {
    /// Maximum time step for bubble dynamics (100 ns - limited by acoustic frequency)
    pub const MAX_TIME_STEP: f64 = 1e-7;
    
    /// Minimum time step for bubble dynamics (1 ps - for extreme collapse)
    pub const MIN_TIME_STEP: f64 = 1e-12;
    
    /// Default relative tolerance for adaptive stepping
    pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-6;
    
    /// Default absolute tolerance for adaptive stepping
    pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-9;
    
    /// Safety factor for time step adjustment (standard value)
    pub const SAFETY_FACTOR: f64 = 0.9;
    
    /// Maximum time step increase factor
    pub const MAX_TIME_STEP_INCREASE: f64 = 1.5;
    
    /// Maximum time step decrease factor
    pub const MAX_TIME_STEP_DECREASE: f64 = 0.1;
    
    /// Maximum number of substeps
    pub const MAX_SUBSTEPS: usize = 1000;
    
    /// Initial time step fraction (start conservatively)
    pub const INITIAL_TIME_STEP_FRACTION: f64 = 0.1;
    
    /// Error control exponent for 4th order method (1/5 = 0.2)
    pub const ERROR_CONTROL_EXPONENT: f64 = 0.2;
    
    /// Half step factor for Richardson extrapolation
    pub const HALF_STEP_FACTOR: f64 = 0.5;
    
    /// Minimum temperature for bubble dynamics (K)
    pub const MIN_TEMPERATURE: f64 = 100.0;
    
    /// Maximum temperature for bubble dynamics (K)
    pub const MAX_TEMPERATURE: f64 = 10000.0;
    
    /// Minimum radius safety factor (fraction of MIN_RADIUS)
    pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.1;
    
    /// Maximum radius safety factor (multiple of MAX_RADIUS)
    pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 10.0;
    
    /// Maximum velocity fraction of sound speed
    pub const MAX_VELOCITY_FRACTION: f64 = 0.9;
}

// Acoustic properties of common media
pub mod acoustic {
    /// Speed of sound in water at 20°C [m/s]
    pub const SOUND_SPEED_WATER: f64 = 1482.0;
    
    /// Speed of sound in water at body temperature (37°C) [m/s]
    pub const SOUND_SPEED_WATER_BODY_TEMP: f64 = 1540.0;
    
    /// Reference speed of sound for general calculations [m/s]
    pub const SOUND_SPEED_REFERENCE: f64 = 1500.0;
    
    /// Density of water at 20°C [kg/m³]
    pub const DENSITY_WATER: f64 = 998.0;
    
    /// Density of water at body temperature [kg/m³]
    pub const DENSITY_WATER_BODY_TEMP: f64 = 993.0;
    
    /// Typical density of soft tissue [kg/m³]
    pub const DENSITY_SOFT_TISSUE: f64 = 1050.0;
    
    /// Typical absorption coefficient for water [dB/cm/MHz]
    pub const ABSORPTION_WATER: f64 = 0.002;
    
    /// Typical absorption coefficient for soft tissue [dB/cm/MHz]
    pub const ABSORPTION_SOFT_TISSUE: f64 = 0.5;
}

// Numerical simulation parameters
pub mod simulation {
    /// Default CFL factor for stability
    pub const CFL_FACTOR_DEFAULT: f64 = 0.3;
    
    /// Conservative CFL factor for nonlinear simulations
    pub const CFL_FACTOR_CONSERVATIVE: f64 = 0.2;
    
    /// Maximum safe CFL factor
    pub const CFL_FACTOR_MAX: f64 = 1.0;
    
    /// Default time step for acoustic simulations [s]
    pub const TIME_STEP_DEFAULT: f64 = 1e-7;
    
    /// Minimum points per wavelength for accurate simulation
    pub const POINTS_PER_WAVELENGTH_MIN: usize = 5;
    
    /// Recommended points per wavelength
    pub const POINTS_PER_WAVELENGTH_DEFAULT: usize = 10;
    
    /// Maximum pressure for biological tissues [Pa]
    pub const PRESSURE_MAX_TISSUE: f64 = 100e6;
    
    /// Typical ambient pressure [Pa]
    pub const PRESSURE_AMBIENT: f64 = 101325.0;
}

// Nonlinear acoustics parameters
pub mod nonlinear {
    /// Nonlinearity parameter B/A for water
    pub const B_OVER_A_WATER: f64 = 5.0;
    
    /// Nonlinearity parameter B/A for soft tissue (average)
    pub const B_OVER_A_SOFT_TISSUE: f64 = 7.0;
    
    /// Acoustic diffusivity for water at 20°C [m²/s]
    pub const DIFFUSIVITY_WATER: f64 = 4.5e-6;
}

// Test and validation parameters
pub mod test {
    /// Default test grid size
    pub const TEST_GRID_SIZE: usize = 32;
    
    /// Default test frequency [Hz]
    pub const TEST_FREQUENCY: f64 = 1e6;
    
    /// Default test amplitude [Pa]
    pub const TEST_AMPLITUDE: f64 = 1e6;
    
    /// Test tolerance for floating point comparisons
    pub const TEST_TOLERANCE: f64 = 1e-6;
}