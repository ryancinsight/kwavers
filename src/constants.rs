//! Physical and numerical constants used throughout the library

use std::f64::consts::PI;

/// Chemistry constants for sonodynamic therapy
pub mod chemistry {
    pub const SINGLET_OXYGEN_LIFETIME: f64 = 3.5e-6;  // seconds - Singlet oxygen lifetime in water
    pub const SINGLET_OXYGEN_DIFFUSION: f64 = 2e-9;  // m²/s - Singlet oxygen diffusion coefficient
    pub const SINGLET_OXYGEN_WEIGHT: f64 = 32.0;  // g/mol - O2 molecular weight
    pub const HYDROXYL_RADICAL_LIFETIME: f64 = 1e-9;  // seconds - Hydroxyl radical lifetime
    pub const HYDROXYL_RADICAL_DIFFUSION: f64 = 2.3e-9;  // m²/s - Hydroxyl radical diffusion coefficient
    pub const HYDROXYL_RADICAL_WEIGHT: f64 = 17.0;  // g/mol - OH molecular weight
    pub const SUPEROXIDE_WEIGHT: f64 = 32.0;  // g/mol - O2- molecular weight
    pub const PEROXYNITRITE_WEIGHT: f64 = 62.0;  // g/mol - ONOO- molecular weight
    pub const HYDROGEN_PEROXIDE_DIFFUSION: f64 = 1.4e-9;  // m²/s - H2O2 diffusion coefficient
    pub const SUPEROXIDE_DIFFUSION: f64 = 2e-9;  // m²/s - Superoxide diffusion coefficient
    pub const NITRIC_OXIDE_DIFFUSION: f64 = 3.3e-9;  // m²/s - NO diffusion coefficient
    pub const PEROXYNITRITE_DIFFUSION: f64 = 1.8e-9;  // m²/s - ONOO- diffusion coefficient
    pub const OXYGEN_HENRY_CONSTANT: f64 = 1.3e-3;  // mol/(L·atm) - Henry's law constant for O2 at 25°C
    pub const NITROGEN_HENRY_CONSTANT: f64 = 6.1e-4;  // mol/(L·atm) - Henry's law constant for N2 at 25°C
    pub const CO2_HENRY_CONSTANT: f64 = 3.4e-2;  // mol/(L·atm) - Henry's law constant for CO2 at 25°C
    pub const ARGON_HENRY_CONSTANT: f64 = 1.4e-3;  // mol/(L·atm) - Henry's law constant for Ar at 25°C
    pub const TISSUE_OXYGEN_CONSUMPTION: f64 = 2.5e-4;  // mol/(m³·s) - Typical tissue O2 consumption rate
    pub const CELL_MEMBRANE_PERMEABILITY: f64 = 1e-5;  // m/s - Cell membrane permeability to ROS
    pub const MITOCHONDRIAL_ROS_PRODUCTION: f64 = 1e-8;  // mol/(m³·s) - Basal mitochondrial ROS production
    pub const ANTIOXIDANT_CAPACITY: f64 = 1e-3;  // mol/m³ - Typical cellular antioxidant capacity
    pub const DNA_DAMAGE_THRESHOLD: f64 = 1e-6;  // mol/m³ - ROS concentration for DNA damage
    pub const LIPID_PEROXIDATION_RATE: f64 = 1e-7;  // 1/s - Rate constant for lipid peroxidation
    pub const PROTEIN_OXIDATION_RATE: f64 = 5e-8;  // 1/s - Rate constant for protein oxidation
    pub const GLUTATHIONE_CONCENTRATION: f64 = 5e-3;  // mol/m³ - Typical cellular glutathione concentration
    pub const CATALASE_ACTIVITY: f64 = 1e4;  // 1/(mol·s) - Catalase rate constant
    pub const SOD_ACTIVITY: f64 = 2e9;  // 1/(mol·s) - Superoxide dismutase rate constant
    pub const PEROXIDASE_ACTIVITY: f64 = 1e7;  // 1/(mol·s) - Peroxidase rate constant
    pub const SONOLUMINESCENCE_THRESHOLD: f64 = 1e5;  // Pa - Pressure threshold for sonoluminescence
    pub const CAVITATION_ROS_YIELD: f64 = 1e-12;  // mol/J - ROS yield per unit cavitation energy
    pub const SONOSENSITIZER_QUANTUM_YIELD: f64 = 0.5;  // Typical sonosensitizer quantum yield
    pub const TISSUE_PH: f64 = 7.4;  // Physiological pH
    pub const TUMOR_PH: f64 = 6.8;  // Typical tumor microenvironment pH
    pub const HYPOXIA_OXYGEN_LEVEL: f64 = 1e-5;  // mol/m³ - O2 level defining hypoxia
    pub const NORMOXIA_OXYGEN_LEVEL: f64 = 2e-4;  // mol/m³ - Normal tissue O2 level
    pub const ROS_SCAVENGING_RATE: f64 = 1e6;  // 1/(mol·s) - General ROS scavenging rate
    pub const MEMBRANE_LIPID_CONCENTRATION: f64 = 0.5;  // mol/m³ - Membrane lipid concentration
    pub const CYTOPLASM_VOLUME_FRACTION: f64 = 0.7;  // Fraction of cell volume that is cytoplasm
    pub const NUCLEUS_VOLUME_FRACTION: f64 = 0.1;  // Fraction of cell volume that is nucleus
    pub const MITOCHONDRIA_VOLUME_FRACTION: f64 = 0.2;  // Fraction of cell volume that is mitochondria
    pub const EXTRACELLULAR_VOLUME_FRACTION: f64 = 0.2;  // Fraction of tissue that is extracellular
    pub const VASCULAR_VOLUME_FRACTION: f64 = 0.05;  // Fraction of tissue that is vascular
    pub const CELL_DIAMETER: f64 = 10e-6;  // m - Typical cell diameter
    pub const NUCLEUS_DIAMETER: f64 = 5e-6;  // m - Typical nucleus diameter
    pub const MITOCHONDRIA_DIAMETER: f64 = 1e-6;  // m - Typical mitochondria diameter
    pub const CAPILLARY_DIAMETER: f64 = 8e-6;  // m - Typical capillary diameter
    pub const INTERCAPILLARY_DISTANCE: f64 = 100e-6;  // m - Typical distance between capillaries
    pub const OXYGEN_WEIGHT: f64 = 32.0;  // g/mol - O2 molecular weight
    pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 34.014;  // g/mol - H2O2 molecular weight
    pub const NITRIC_OXIDE_WEIGHT: f64 = 30.006;  // g/mol - NO molecular weight
    pub const BASE_PHOTOCHEMICAL_RATE: f64 = 1e-5;  // mol/(L·s) - Base photochemical reaction rate
}

/// Optical constants (merged)
pub mod optics {
    // Basic optical constants
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;  // m/s
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;  // J·s
    pub const PHOTON_ENERGY_CONVERSION: f64 = 1.602176634e-19;  // J/eV
    
    // Tissue optical properties
    pub const TISSUE_DIFFUSION_COEFFICIENT: f64 = 0.03;  // cm - Tissue optical diffusion coefficient
    pub const TISSUE_ABSORPTION_COEFFICIENT: f64 = 0.1;  // 1/cm - Tissue optical absorption coefficient
    pub const DEFAULT_POLARIZATION_FACTOR: f64 = 1.0;  // Default polarization factor
    pub const LAPLACIAN_CENTER_COEFF: f64 = -4.0;  // Laplacian center coefficient for 2D
    
    // Pulse and window parameters
    pub const GAUSSIAN_PULSE_WIDTH_FACTOR: f64 = 2.0;
    pub const GAUSSIAN_PULSE_CENTER_FACTOR: f64 = 0.5;
    pub const WINDOW_WIDTH_FACTOR: f64 = 4.0;
}

/// Physics constants (merged)
pub mod physics {
    // Spatial resolution
    pub const STANDARD_SPATIAL_RESOLUTION: f64 = 1e-3;  // meters
    
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

/// Validation constants
pub mod validation {
    /// Tolerance for floating point comparisons
    pub const TOLERANCE: f64 = 1e-10;
    /// Maximum iterations for convergence
    pub const MAX_ITERATIONS: usize = 1000;
}

/// Acoustic constants
pub mod acoustic {
    pub const REFERENCE_PRESSURE: f64 = 20e-6;  // Pa (20 μPa)
    pub const REFERENCE_INTENSITY: f64 = 1e-12;  // W/m²
    pub const IMPEDANCE_AIR: f64 = 413.0;  // Pa·s/m
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
    pub const CFL_MAX: f64 = 0.5;
    pub const CFL_MIN: f64 = 0.1;
    pub const FDTD_DEFAULT: f64 = 0.3;  // Default CFL for FDTD
    pub const CONSERVATIVE: f64 = 0.2;  // Conservative CFL value
    pub const AGGRESSIVE: f64 = 0.4;  // Aggressive CFL value
}

/// Numerical constants
pub mod numerical {
    pub const EPSILON: f64 = 1e-10;
    pub const MAX_ITERATIONS: usize = 1000;
    pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;
    
    // Finite difference coefficients for second-order accurate schemes
    pub const FD2_CENTRAL_COEFF: f64 = 0.5;  // Central difference coefficient
    pub const FD2_FORWARD_COEFF: [f64; 3] = [-1.5, 2.0, -0.5];  // Forward difference
    pub const FD2_BACKWARD_COEFF: [f64; 3] = [0.5, -2.0, 1.5];  // Backward difference
    pub const SECOND_ORDER_DIFF_COEFF: f64 = 1.0;  // Second order difference coefficient
    pub const THIRD_ORDER_DIFF_COEFF: f64 = 1.0;  // Third order difference coefficient
    
    // Fourth-order accurate finite difference coefficients
    pub const FD4_CENTRAL_COEFF: [f64; 5] = [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0];
    pub const FD4_LAPLACIAN_COEFF: [f64; 5] = [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0];
    
    // FFT scaling factors
    pub const FFT_FORWARD_SCALE: f64 = 1.0;
    pub const FFT_INVERSE_SCALE_2D: f64 = 1.0;  // Will be divided by N*M at runtime
    pub const FFT_INVERSE_SCALE_3D: f64 = 1.0;  // Will be divided by N*M*L at runtime
    pub const FFT_K_SCALING: f64 = 2.0 * std::f64::consts::PI;  // k-space scaling factor
    
    // WENO scheme weights and coefficients
    pub const WENO_WEIGHT_0: f64 = 0.1;
    pub const WENO_WEIGHT_1: f64 = 0.6;
    pub const WENO_WEIGHT_2: f64 = 0.3;
    pub const WENO_EPSILON: f64 = 1e-6;
    pub const STENCIL_COEFF_1_4: f64 = 0.25;
    
    // Artificial viscosity coefficients
    pub const VON_NEUMANN_RICHTMYER_COEFF: f64 = 2.0;
    pub const LINEAR_VISCOSITY_COEFF: f64 = 0.06;
    pub const QUADRATIC_VISCOSITY_COEFF: f64 = 1.5;
    pub const MAX_VISCOSITY_LIMIT: f64 = 2.0;
    
    // PML parameters
    pub const PML_ALPHA_MAX: f64 = 0.0;  // Maximum PML alpha value
    pub const PML_POLYNOMIAL_ORDER: f64 = 2.0;  // PML polynomial order
    pub const PML_SIGMA_OPTIMAL: f64 = 0.8;  // Optimal PML sigma factor
}

/// Material constants
pub mod material {
    // Steel properties
    pub const STEEL_DENSITY: f64 = 7850.0;  // kg/m³
    pub const STEEL_SOUND_SPEED: f64 = 5960.0;  // m/s
    pub const STEEL_IMPEDANCE: f64 = 46.7e6;  // Pa·s/m
    
    // Aluminum properties
    pub const ALUMINUM_DENSITY: f64 = 2700.0;  // kg/m³
    pub const ALUMINUM_SOUND_SPEED: f64 = 6420.0;  // m/s
    pub const ALUMINUM_IMPEDANCE: f64 = 17.3e6;  // Pa·s/m
    
    // Glass properties
    pub const GLASS_DENSITY: f64 = 2500.0;  // kg/m³
    pub const GLASS_SOUND_SPEED: f64 = 5640.0;  // m/s
    pub const GLASS_IMPEDANCE: f64 = 14.1e6;  // Pa·s/m
    
    // Bone properties
    pub const BONE_DENSITY: f64 = 1900.0;  // kg/m³
    pub const BONE_SOUND_SPEED: f64 = 4080.0;  // m/s
    pub const BONE_IMPEDANCE: f64 = 7.75e6;  // Pa·s/m
    
    // Blood properties
    pub const BLOOD_DENSITY: f64 = 1060.0;  // kg/m³
    pub const BLOOD_SOUND_SPEED: f64 = 1584.0;  // m/s
    pub const BLOOD_IMPEDANCE: f64 = 1.68e6;  // Pa·s/m
    
    // Fat properties
    pub const FAT_DENSITY: f64 = 925.0;  // kg/m³
    pub const FAT_SOUND_SPEED: f64 = 1450.0;  // m/s
    pub const FAT_IMPEDANCE: f64 = 1.34e6;  // Pa·s/m
    
    // Muscle properties
    pub const MUSCLE_DENSITY: f64 = 1090.0;  // kg/m³
    pub const MUSCLE_SOUND_SPEED: f64 = 1589.0;  // m/s
    pub const MUSCLE_IMPEDANCE: f64 = 1.73e6;  // Pa·s/m
    
    // Liver properties
    pub const LIVER_DENSITY: f64 = 1060.0;  // kg/m³
    pub const LIVER_SOUND_SPEED: f64 = 1595.0;  // m/s
    pub const LIVER_IMPEDANCE: f64 = 1.69e6;  // Pa·s/m
    
    // Brain properties
    pub const BRAIN_DENSITY: f64 = 1040.0;  // kg/m³
    pub const BRAIN_SOUND_SPEED: f64 = 1541.0;  // m/s
    pub const BRAIN_IMPEDANCE: f64 = 1.60e6;  // Pa·s/m
    
    // Skull properties
    pub const SKULL_DENSITY: f64 = 1900.0;  // kg/m³
    pub const SKULL_SOUND_SPEED: f64 = 2800.0;  // m/s
    pub const SKULL_IMPEDANCE: f64 = 5.32e6;  // Pa·s/m
}

/// Thermal constants
pub mod thermal {
    pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;  // W/(m²·K⁴)
    pub const TISSUE_THERMAL_CONDUCTIVITY: f64 = 0.5;  // W/(m·K)
    pub const TISSUE_SPECIFIC_HEAT: f64 = 3600.0;  // J/(kg·K)
    pub const TISSUE_THERMAL_DIFFUSIVITY: f64 = 1.4e-7;  // m²/s
    pub const BLOOD_PERFUSION_RATE: f64 = 0.5;  // kg/(m³·s)
    pub const METABOLIC_HEAT_GENERATION: f64 = 400.0;  // W/m³
    pub const BODY_TEMPERATURE: f64 = 310.15;  // K (37°C)
    pub const ROOM_TEMPERATURE: f64 = 293.15;  // K (20°C)
}

/// Absorption model constants
pub mod absorption {
    pub const POWER_LAW_PREFACTOR: f64 = 0.1;  // dB/(MHz^y·cm)
    pub const POWER_LAW_EXPONENT: f64 = 1.0;  // Frequency power law exponent
    pub const STOKES_VISCOSITY_FACTOR: f64 = 4.0/3.0;  // Stokes' hypothesis
    pub const RELAXATION_TIME: f64 = 1e-12;  // s - Molecular relaxation time
}

/// Dispersion model constants
pub mod dispersion {
    pub const KRAMERS_KRONIG_FACTOR: f64 = 2.0 / super::PI;
    pub const CAUSALITY_EPSILON: f64 = 1e-10;
    pub const PHASE_VELOCITY_REFERENCE: f64 = 1500.0;  // m/s
}

/// Boundary condition constants
pub mod boundary {
    pub const PML_LAYERS: usize = 10;
    pub const PML_ALPHA: f64 = 0.0;
    pub const PML_POLYNOMIAL_ORDER: f64 = 2.0;
    pub const ABC_COEFFICIENT: f64 = 0.5;
}

/// Grid constants
pub mod grid {
    pub const MIN_POINTS_PER_WAVELENGTH: usize = 6;
    pub const MAX_ASPECT_RATIO: f64 = 10.0;
    pub const DEFAULT_GRID_SPACING: f64 = 1e-4;  // 0.1 mm
    pub const MIN_GRID_SPACING: f64 = 1e-6;  // 1 μm minimum grid spacing
    pub const MIN_GRID_POINTS: usize = 16;  // Minimum grid points per dimension
}

/// Source constants
pub mod source {
    pub const DEFAULT_FREQUENCY: f64 = 1e6;  // 1 MHz
    pub const DEFAULT_AMPLITUDE: f64 = 1e6;  // 1 MPa
    pub const DEFAULT_PHASE: f64 = 0.0;  // radians
    pub const DEFAULT_CYCLES: f64 = 3.0;  // tone burst cycles
}

/// Sensor constants
pub mod sensor {
    pub const DEFAULT_SAMPLING_FREQUENCY: f64 = 100e6;  // 100 MHz
    pub const DEFAULT_RECORDING_TIME: f64 = 100e-6;  // 100 μs
    pub const DEFAULT_TRIGGER_LEVEL: f64 = 0.1;  // 10% of max
}

/// Thermodynamics constants for bubble dynamics
pub mod thermodynamics {
    pub const WATER_VAPOR_PRESSURE: f64 = 2339.0;  // Pa at 20°C
    pub const SURFACE_TENSION_WATER: f64 = 0.0728;  // N/m
    pub const POLYTROPIC_EXPONENT: f64 = 1.4;  // For air
    pub const GAS_CONSTANT: f64 = 8.314;  // J/(mol·K)
    pub const R_GAS: f64 = 8.314;  // J/(mol·K) - Universal gas constant (alias)
    pub const AMBIENT_TEMPERATURE: f64 = 293.15;  // K (20°C)
    pub const T_AMBIENT: f64 = 293.15;  // K (20°C) - alias
    pub const WATER_VISCOSITY: f64 = 1.002e-3;  // Pa·s at 20°C
    pub const AIR_THERMAL_CONDUCTIVITY: f64 = 0.026;  // W/(m·K)
    pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.598;  // W/(m·K)
    pub const ACCOMMODATION_COEFFICIENT: f64 = 0.04;  // For air-water interface
    pub const AVOGADRO: f64 = 6.022e23;  // 1/mol - Avogadro's number
    pub const M_WATER: f64 = 0.018;  // kg/mol - Molecular weight of water
    pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.42e-5;  // m²/s - Water vapor in air
    pub const NUSSELT_CONSTANT: f64 = 2.0;  // Nusselt number for sphere
    pub const NUSSELT_PECLET_COEFF: f64 = 0.6;  // Coefficient for Peclet correction
    pub const NUSSELT_PECLET_EXPONENT: f64 = 0.33;  // Exponent for Peclet correction
    pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;  // Sherwood number Peclet exponent
    pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;  // K - Reference temperature for reactions
    pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-6;  // mol/(L·s) - Base sonochemical reaction rate
    pub const SECONDARY_REACTION_RATE: f64 = 1e-7;  // 1/s - Secondary reaction rate constant
}

/// Bubble dynamics constants
pub mod bubble_dynamics {
    pub const INITIAL_RADIUS: f64 = 5e-6;  // m - Initial bubble radius
    pub const AMBIENT_PRESSURE: f64 = 101325.0;  // Pa - Atmospheric pressure
    pub const BLAKE_RADIUS_RATIO: f64 = 0.915;  // Blake threshold radius ratio
    pub const MINIMUM_RADIUS_RATIO: f64 = 0.1;  // Minimum radius as fraction of R0
    pub const MAXIMUM_RADIUS_RATIO: f64 = 10.0;  // Maximum radius as fraction of R0
    pub const MIN_RADIUS: f64 = 1e-9;  // m - Absolute minimum bubble radius
    pub const MAX_RADIUS: f64 = 1e-3;  // m - Absolute maximum bubble radius
    pub const RAYLEIGH_COLLAPSE_TIME_FACTOR: f64 = 0.915;  // Rayleigh collapse time factor
    pub const VISCOUS_DAMPING_FACTOR: f64 = 4.0;  // Viscous damping coefficient
    pub const THERMAL_DAMPING_FACTOR: f64 = 3.0;  // Thermal damping coefficient
    pub const BAR_L2_TO_PA_M6: f64 = 1e11;  // Conversion from bar·L² to Pa·m⁶
    pub const L_TO_M3: f64 = 1e-3;  // Conversion from L to m³
    pub const PECLET_SCALING_FACTOR: f64 = 1.0;  // Peclet number scaling
    pub const MIN_PECLET_NUMBER: f64 = 0.1;  // Minimum Peclet number for heat transfer
    pub const VISCOUS_STRESS_COEFF: f64 = 4.0;  // Viscous stress coefficient
    pub const SURFACE_TENSION_COEFF: f64 = 2.0;  // Surface tension coefficient
    pub const KINETIC_ENERGY_COEFF: f64 = 0.5;  // Kinetic energy coefficient
    pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.257e6;  // J/kg - Latent heat of vaporization for water
}

/// Stability constants
pub mod stability {
    pub const COURANT_NUMBER: f64 = 0.3;  // CFL condition
    pub const DIFFUSION_NUMBER: f64 = 0.25;  // Diffusion stability
    pub const PECLET_NUMBER_MAX: f64 = 2.0;  // Maximum Peclet number
    pub const REYNOLDS_NUMBER_CRITICAL: f64 = 2300.0;  // Critical Reynolds number
    pub const DAMPING_COEFFICIENT: f64 = 0.01;  // Numerical damping
    pub const PRESSURE_LIMIT: f64 = 1e9;  // Pa - Maximum pressure limit for stability
}

/// Performance constants
pub mod performance {
    pub const CACHE_LINE_SIZE: usize = 64;  // Bytes
    pub const SIMD_WIDTH: usize = 8;  // AVX2 double precision
    pub const THREAD_POOL_SIZE: usize = 4;  // Default thread count
    pub const CHUNK_SIZE: usize = 1024;  // Default chunk size for parallel processing
    pub const CHUNK_SIZE_LARGE: usize = 4096;  // Large chunk size for parallel processing
    pub const CHUNK_SIZE_MEDIUM: usize = 1024;  // Medium chunk size for parallel processing
    pub const CHUNK_SIZE_SMALL: usize = 256;  // Small chunk size for parallel processing
    pub const PREFETCH_DISTANCE: usize = 8;  // Cache prefetch distance
    pub const LARGE_GRID_THRESHOLD: usize = 1000000;  // Threshold for large grid optimizations
    pub const MEDIUM_GRID_THRESHOLD: usize = 100000;  // Threshold for medium grid optimizations
    pub const SMALL_GRID_THRESHOLD: usize = 10000;  // Threshold for small grid optimizations
    pub const CHUNKED_PROCESSING_THRESHOLD: usize = 10000;  // Threshold for chunked processing
}

/// Adaptive integration constants
pub mod adaptive_integration {
    pub const ABSOLUTE_TOLERANCE: f64 = 1e-10;
    pub const RELATIVE_TOLERANCE: f64 = 1e-8;
    pub const MINIMUM_TIMESTEP: f64 = 1e-15;  // seconds
    pub const MAXIMUM_TIMESTEP: f64 = 1e-6;  // seconds
    pub const MIN_TIME_STEP: f64 = 1e-15;  // Alias for MINIMUM_TIMESTEP
    pub const MAX_TIME_STEP: f64 = 1e-6;  // Alias for MAXIMUM_TIMESTEP
    pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-10;
    pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-8;
    pub const SAFETY_FACTOR: f64 = 0.9;
    pub const ERROR_EXPONENT: f64 = 0.2;  // For step size control
    pub const ERROR_CONTROL_EXPONENT: f64 = 0.2;  // Alias
    pub const GROWTH_LIMIT: f64 = 5.0;  // Maximum step size growth factor
    pub const SHRINK_LIMIT: f64 = 0.1;  // Minimum step size shrink factor
    pub const MAX_TIME_STEP_INCREASE: f64 = 5.0;  // Maximum increase factor
    pub const MAX_TIME_STEP_DECREASE: f64 = 0.1;  // Maximum decrease factor
    pub const MAX_SUBSTEPS: usize = 1000;  // Maximum substeps in integration
    pub const INITIAL_TIME_STEP_FRACTION: f64 = 0.01;  // Initial timestep as fraction of total
    pub const HALF_STEP_FACTOR: f64 = 0.5;  // Factor for half-step method
    pub const MIN_TEMPERATURE: f64 = 273.15;  // Minimum temperature (K)
    pub const MAX_TEMPERATURE: f64 = 373.15;  // Maximum temperature (K)
    pub const MIN_RADIUS_SAFETY_FACTOR: f64 = 0.01;  // Minimum radius safety
    pub const MAX_RADIUS_SAFETY_FACTOR: f64 = 100.0;  // Maximum radius safety
    pub const MAX_VELOCITY_FRACTION: f64 = 0.1;  // Maximum velocity as fraction of sound speed
}