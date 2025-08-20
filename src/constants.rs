//! Physical and numerical constants used throughout the codebase

/// Numerical constants
pub mod numerical {
    pub const EPSILON: f64 = 1e-10;
    pub const MAX_ITERATIONS: usize = 1000;
    pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;
}

/// Stability constants
pub mod stability {
    pub const MIN_STABLE_TIMESTEP: f64 = 1e-12;
    pub const MAX_STABLE_TIMESTEP: f64 = 1e-3;
    pub const STABILITY_FACTOR: f64 = 0.9;
}

/// Performance constants
pub mod performance {
    pub const CACHE_SIZE: usize = 1024;
    pub const CHUNK_SIZE: usize = 64;
    pub const PARALLEL_THRESHOLD: usize = 1000;
}

/// Chemistry constants
pub mod chemistry {
    pub const ACTIVATION_ENERGY: f64 = 50000.0;  // J/mol
    pub const PRE_EXPONENTIAL_FACTOR: f64 = 1e10;  // 1/s
    pub const REACTION_RATE: f64 = 1e-3;  // mol/(L·s)
}

/// Acoustic constants
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
}

/// Cavitation constants
pub mod cavitation {
    pub const BLAKE_THRESHOLD: f64 = 0.85;  // Blake threshold ratio
    pub const INERTIAL_THRESHOLD: f64 = 2.0;  // Inertial cavitation threshold
    pub const STABLE_THRESHOLD: f64 = 0.5;  // Stable cavitation threshold
}

/// Tolerance constants
pub mod tolerance {
    pub const RELATIVE: f64 = 1e-6;
    pub const ABSOLUTE: f64 = 1e-9;
    pub const MACHINE_EPSILON: f64 = f64::EPSILON;
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
    pub const SAFETY_FACTOR: f64 = 0.9;  // Safety factor for step size
    pub const MAX_ITERATIONS: usize = 1000;  // Maximum iterations
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