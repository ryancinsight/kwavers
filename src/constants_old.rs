//! Physical and numerical constants used throughout the codebase

/// Numerical constants
pub mod numerical {
    pub const EPSILON: f64 = 1e-10;
    pub const MAX_ITERATIONS: usize = 1000;
    pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;
    pub const SECOND_ORDER_DIFF_COEFF: f64 = 0.5;  // Coefficient for second-order differentiation
    pub const THIRD_ORDER_DIFF_COEFF: f64 = 1.0 / 6.0;  // Coefficient for third-order differentiation
    pub const FFT_K_SCALING: f64 = 2.0 * std::f64::consts::PI;  // Scaling factor for FFT wavenumbers
    pub const WENO_WEIGHT_0: f64 = 0.1;  // WENO scheme weight 0
    pub const WENO_WEIGHT_1: f64 = 0.6;  // WENO scheme weight 1
    pub const WENO_WEIGHT_2: f64 = 0.3;  // WENO scheme weight 2
    pub const VON_NEUMANN_RICHTMYER_COEFF: f64 = 2.0;  // Von Neumann-Richtmyer viscosity coefficient
    pub const LINEAR_VISCOSITY_COEFF: f64 = 0.06;  // Linear artificial viscosity coefficient
    pub const QUADRATIC_VISCOSITY_COEFF: f64 = 1.5;  // Quadratic artificial viscosity coefficient
    pub const MAX_VISCOSITY_LIMIT: f64 = 0.1;  // Maximum artificial viscosity limit
    pub const WENO_EPSILON: f64 = 1e-6;  // WENO scheme epsilon for avoiding division by zero
    pub const STENCIL_COEFF_1_4: f64 = 0.25;  // Stencil coefficient 1/4
}

/// Stability constants
pub mod stability {
    pub const MIN_STABLE_TIMESTEP: f64 = 1e-12;
    pub const MAX_STABLE_TIMESTEP: f64 = 1e-3;
    pub const STABILITY_FACTOR: f64 = 0.9;
    pub const PRESSURE_LIMIT: f64 = 1e10;  // Pa - Maximum pressure limit for stability
}

/// Performance constants
pub mod performance {
    pub const CACHE_SIZE: usize = 1024;
    pub const CHUNK_SIZE: usize = 64;
    pub const PARALLEL_THRESHOLD: usize = 1000;
    pub const LARGE_GRID_THRESHOLD: usize = 1000000;  // Threshold for large grid processing
    pub const MEDIUM_GRID_THRESHOLD: usize = 100000;  // Threshold for medium grid processing
    pub const CHUNK_SIZE_LARGE: usize = 256;  // Chunk size for large grids
    pub const CHUNK_SIZE_MEDIUM: usize = 128;  // Chunk size for medium grids
    pub const CHUNK_SIZE_SMALL: usize = 32;  // Chunk size for small grids
    pub const CHUNKED_PROCESSING_THRESHOLD: usize = 10000;  // Threshold for chunked processing
}

/// Chemistry constants
pub mod chemistry {
    pub const ACTIVATION_ENERGY: f64 = 50000.0;  // J/mol
    pub const PRE_EXPONENTIAL_FACTOR: f64 = 1e10;  // 1/s
    pub const REACTION_RATE: f64 = 1e-3;  // mol/(L·s)
    pub const HYDROXYL_RADICAL_WEIGHT: f64 = 17.008;  // g/mol - OH radical molecular weight
    pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 34.014;  // g/mol - H2O2 molecular weight
    pub const SUPEROXIDE_WEIGHT: f64 = 32.00;  // g/mol - O2- molecular weight
    pub const SINGLET_OXYGEN_WEIGHT: f64 = 32.00;  // g/mol - 1O2 molecular weight
    pub const PEROXYNITRITE_WEIGHT: f64 = 62.005;  // g/mol - ONOO- molecular weight
    pub const NITRIC_OXIDE_WEIGHT: f64 = 30.006;  // g/mol - NO molecular weight
    pub const BASE_PHOTOCHEMICAL_RATE: f64 = 1e-5;  // mol/(L·s) - Base photochemical reaction rate
}

/// Acoustic constants
pub mod optics {
    /// Gaussian pulse width factor for optical simulations
    pub const GAUSSIAN_PULSE_WIDTH_FACTOR: f64 = 2.0;
    /// Gaussian pulse center factor
    pub const GAUSSIAN_PULSE_CENTER_FACTOR: f64 = 0.5;
    /// Window width factor for spectral analysis
    pub const WINDOW_WIDTH_FACTOR: f64 = 4.0;
}

pub mod physics {
    /// Standard spatial resolution in meters
    pub const STANDARD_SPATIAL_RESOLUTION: f64 = 1e-3;
}

pub mod validation {
    /// Tolerance for floating point comparisons
    pub const TOLERANCE: f64 = 1e-10;
    /// Maximum iterations for convergence
    pub const MAX_ITERATIONS: usize = 1000;
}

pub mod acoustic {
    pub const REFERENCE_PRESSURE: f64 = 20e-6;  // Pa (20 μPa)
    pub const REFERENCE_INTENSITY: f64 = 1e-12;  // W/m²
    pub const IMPEDANCE_AIR: f64 = 413.0;  // Pa·s/m
}

/// Optics constants
pub mod optics {
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;  // m/s
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;  // J·s
    pub const PHOTON_ENERGY_CONVERSION: f64 = 1.602176634e-19;  // J/eV
    pub const TISSUE_DIFFUSION_COEFFICIENT: f64 = 0.03;  // cm - Tissue optical diffusion coefficient
    pub const TISSUE_ABSORPTION_COEFFICIENT: f64 = 0.1;  // 1/cm - Tissue optical absorption coefficient
    pub const DEFAULT_POLARIZATION_FACTOR: f64 = 1.0;  // Default polarization factor
    pub const LAPLACIAN_CENTER_COEFF: f64 = -4.0;  // Laplacian center coefficient for 2D
}

/// Cavitation constants
pub mod cavitation {
    pub const BLAKE_THRESHOLD: f64 = 0.85;  // Blake threshold ratio
    pub const INERTIAL_THRESHOLD: f64 = 2.0;  // Inertial cavitation threshold
    pub const STABLE_THRESHOLD: f64 = 0.5;  // Stable cavitation threshold
    pub const DEFAULT_THRESHOLD_PRESSURE: f64 = 1e5;  // Pa - Default cavitation threshold pressure
    pub const DEFAULT_PIT_EFFICIENCY: f64 = 0.1;  // Default pit formation efficiency
    pub const DEFAULT_FATIGUE_RATE: f64 = 1e-6;  // Default material fatigue rate
    pub const DEFAULT_CONCENTRATION_FACTOR: f64 = 2.0;  // Default stress concentration factor
    pub const MATERIAL_REMOVAL_EFFICIENCY: f64 = 0.01;  // Material removal efficiency
    pub const IMPACT_ENERGY_COEFFICIENT: f64 = 0.5;  // Impact energy coefficient
    pub const COMPRESSION_FACTOR_EXPONENT: f64 = 1.4;  // Compression factor exponent (adiabatic)
}

/// Tolerance constants
pub mod tolerance {
    pub const RELATIVE: f64 = 1e-6;
    pub const ABSOLUTE: f64 = 1e-9;
    pub const MACHINE_EPSILON: f64 = f64::EPSILON;
    pub const CONVERGENCE: f64 = 1e-6;  // Convergence tolerance for iterative methods
}

/// CFL constants
pub mod cfl {
    pub const CFL_SAFETY_FACTOR: f64 = 0.3;
    pub const MAX_CFL_NUMBER: f64 = 1.0;
    pub const MIN_CFL_NUMBER: f64 = 0.01;
    pub const CONSERVATIVE: f64 = 0.1;
    pub const AGGRESSIVE: f64 = 0.9;
    pub const PSTD_DEFAULT: f64 = 0.5;
    pub const FDTD_DEFAULT: f64 = 0.3;
}

/// Thermodynamics constants
pub mod thermodynamics {
    pub const R_GAS: f64 = 8.314462618;  // J/(mol·K) - Universal gas constant
    pub const AVOGADRO: f64 = 6.02214076e23;  // 1/mol - Avogadro's number
    pub const M_WATER: f64 = 0.018015;  // kg/mol - Molar mass of water
    pub const BOLTZMANN: f64 = 1.380649e-23;  // J/K - Boltzmann constant
    pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;  // W/(m²·K⁴)
    pub const T_AMBIENT: f64 = 293.15;  // K - Ambient temperature (20°C)
    pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.42e-5;  // m²/s at 20°C
    pub const NUSSELT_CONSTANT: f64 = 2.0;  // Nusselt number constant
    pub const NUSSELT_PECLET_COEFF: f64 = 0.6;  // Nusselt-Peclet coefficient
    pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;  // Nusselt-Peclet exponent
    pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;  // Sherwood-Peclet exponent
    pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;  // K - Reference temperature for reactions (25°C)
    pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-6;  // mol/(L·s) - Base sonochemical reaction rate
    pub const SECONDARY_REACTION_RATE: f64 = 1e-7;  // mol/(L·s) - Secondary reaction rate
}

/// Bubble dynamics constants
pub mod bubble_dynamics {
    pub const SURFACE_TENSION_WATER: f64 = 0.0728;  // N/m at 20°C
    pub const VAPOR_PRESSURE_WATER: f64 = 2338.0;  // Pa at 20°C
    pub const POLYTROPIC_INDEX_AIR: f64 = 1.4;  // Adiabatic index for air
    pub const WATER_VISCOSITY: f64 = 1.0e-3;  // Pa·s at 20°C
    pub const MIN_BUBBLE_RADIUS: f64 = 1e-9;  // m - Minimum physical bubble radius
    pub const MAX_BUBBLE_RADIUS: f64 = 1e-2;  // m - Maximum physical bubble radius
    pub const MIN_RADIUS: f64 = MIN_BUBBLE_RADIUS;  // Alias
    pub const MAX_RADIUS: f64 = MAX_BUBBLE_RADIUS;  // Alias
    pub const VISCOUS_STRESS_COEFF: f64 = 4.0;  // Viscous stress coefficient
    pub const SURFACE_TENSION_COEFF: f64 = 2.0;  // Surface tension coefficient
    pub const KINETIC_ENERGY_COEFF: f64 = 1.5;  // Kinetic energy coefficient
    pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.453e6;  // J/kg at 20°C
    pub const BAR_L2_TO_PA_M6: f64 = 1e11;  // Conversion: bar·L² to Pa·m⁶
    pub const L_TO_M3: f64 = 1e-3;  // Conversion: L to m³
    pub const PECLET_SCALING_FACTOR: f64 = 0.1;  // Peclet number scaling
    pub const MIN_PECLET_NUMBER: f64 = 0.01;  // Minimum Peclet number
}

/// Adaptive integration constants
pub mod adaptive_integration {
    pub const MIN_TIME_STEP: f64 = 1e-12;  // s - Minimum time step
    pub const MAX_TIME_STEP: f64 = 1e-6;  // s - Maximum time step
    pub const RELATIVE_TOLERANCE: f64 = 1e-6;  // Relative error tolerance
    pub const ABSOLUTE_TOLERANCE: f64 = 1e-9;  // Absolute error tolerance
    pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-6;  // Default relative error tolerance
    pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-9;  // Default absolute error tolerance
    pub const SAFETY_FACTOR: f64 = 0.9;  // Safety factor for step size
    pub const MAX_ITERATIONS: usize = 1000;  // Maximum iterations
    pub const MAX_TIME_STEP_INCREASE: f64 = 2.0;  // Maximum factor for time step increase
    pub const MAX_TIME_STEP_DECREASE: f64 = 0.1;  // Minimum factor for time step decrease
    pub const MAX_SUBSTEPS: usize = 100;  // Maximum substeps in adaptive integration
    pub const INITIAL_TIME_STEP_FRACTION: f64 = 0.01;  // Initial time step as fraction of max
    pub const ERROR_CONTROL_EXPONENT: f64 = 0.25;  // Exponent for error control (1/4 for RK4)
    pub const HALF_STEP_FACTOR: f64 = 0.5;  // Factor for half-step in error estimation
    pub const MIN_TEMPERATURE: f64 = 273.15;  // Minimum temperature in Kelvin (0°C)
    pub const MAX_TEMPERATURE: f64 = 10000.0;  // Maximum temperature in Kelvin
    pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.1;  // Safety factor for minimum radius
    pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 10.0;  // Safety factor for maximum radius
    pub const MAX_VELOCITY_FRACTION: f64 = 0.5;  // Maximum velocity as fraction of sound speed
}

/// Physics constants submodule
pub mod physics {
    // Water properties at standard conditions (20°C, 1 atm)
    pub const WATER_DENSITY: f64 = 1000.0;  // kg/m³
    pub const WATER_SOUND_SPEED: f64 = 1500.0;  // m/s
    pub const WATER_ATTENUATION: f64 = 0.0022;  // Np/m/MHz
    pub const WATER_NONLINEARITY: f64 = 3.5;  // B/A parameter
    pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.6;  // W/(m·K)
    pub const WATER_SPECIFIC_HEAT: f64 = 4180.0;  // J/(kg·K)
    pub const WATER_REFRACTIVE_INDEX: f64 = 1.33;
    pub const WATER_GRUNEISEN: f64 = 0.12;
    
    // Aliases for compatibility
    pub const DENSITY_WATER: f64 = WATER_DENSITY;
    pub const SOUND_SPEED_WATER: f64 = WATER_SOUND_SPEED;
    
    // Tissue properties (soft tissue average)
    pub const DENSITY_TISSUE: f64 = 1050.0;  // kg/m³
    pub const SOUND_SPEED_TISSUE: f64 = 1540.0;  // m/s
    pub const TISSUE_ATTENUATION: f64 = 0.5;  // dB/cm/MHz
    pub const TISSUE_NONLINEARITY: f64 = 6.0;  // B/A parameter
    
    // Ultrasound parameters
    pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;  // 1 MHz
    pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e6;  // 1 MPa
    pub const STANDARD_BEAM_WIDTH: f64 = 0.01;  // 10 mm
    
    // Default mode conversion efficiency
    pub const DEFAULT_MODE_CONVERSION_EFFICIENCY: f64 = 0.3;
    
    // Power law absorption
    pub const DEFAULT_POWER_LAW_EXPONENT: f64 = 1.05;  // Typical for biological tissues
    
    // Additional physics constants
    pub const HIGH_PRESSURE_THRESHOLD: f64 = 10e6;  // Pa - High pressure threshold (10 MPa)
    pub const GRID_CENTER_FACTOR: f64 = 0.5;  // Factor for grid centering
    pub const NONLINEARITY_COEFFICIENT_OFFSET: f64 = 1.0;  // Offset for nonlinearity coefficient
    pub const B_OVER_A_DIVISOR: f64 = 2.0;  // Divisor for B/A parameter
    pub const REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ: f64 = 1e6;  // Hz - Reference frequency for absorption (1 MHz)
}

// Numerical tolerances
pub const FLOAT_EQUALITY_TOLERANCE: f64 = 1e-10;
pub const NUMERICAL_EPSILON: f64 = 1e-6;
pub const SYMMETRY_TOLERANCE: f64 = 1e-10;
pub const SINC_ARGUMENT_THRESHOLD: f64 = 1e-10;

// CFL and stability
pub const DEFAULT_CFL_SAFETY_FACTOR: f64 = 0.3;
pub const MAX_CFL_NUMBER: f64 = 1.0;

// Water properties at standard conditions (20°C, 1 atm)
pub const WATER_DENSITY: f64 = 1000.0;  // kg/m³
pub const WATER_SOUND_SPEED: f64 = 1500.0;  // m/s
pub const WATER_ATTENUATION: f64 = 0.0022;  // Np/m/MHz
pub const WATER_NONLINEARITY: f64 = 3.5;  // B/A parameter
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.6;  // W/(m·K)
pub const WATER_SPECIFIC_HEAT: f64 = 4180.0;  // J/(kg·K)
pub const WATER_REFRACTIVE_INDEX: f64 = 1.33;
pub const WATER_GRUNEISEN: f64 = 0.12;

// Conversion factors
pub const MS_TO_S: f64 = 1e-3;
pub const MHZ_TO_HZ: f64 = 1e6;

// Default mode conversion efficiency
pub const DEFAULT_MODE_CONVERSION_EFFICIENCY: f64 = 0.3;

// Power law absorption
pub const DEFAULT_POWER_LAW_EXPONENT: f64 = 1.05;  // Typical for biological tissues

/// Grid constants
pub mod grid {
    pub const MIN_GRID_SPACING: f64 = 1e-6;  // m - Minimum grid spacing
    pub const MAX_GRID_SPACING: f64 = 1.0;  // m - Maximum grid spacing
    pub const DEFAULT_GRID_SPACING: f64 = 1e-3;  // m - Default grid spacing
    pub const MIN_GRID_POINTS: usize = 10;  // Minimum number of grid points per dimension
}