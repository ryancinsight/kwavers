//! Fundamental physical constants

/// Speed of sound in water at 20°C (m/s) — k-Wave simulation default
///
/// Value: 1500.0 m/s — the standard nominal reference used in k-Wave and most
/// ultrasound simulation/beamforming literature. This is a round-number approximation
/// suitable for simulation defaults and signal processing defaults.
/// Note: the physically precise value at 20°C is 1482 m/s (see `SOUND_SPEED_WATER_PRECISE`).
/// Reference: Treeby & Cox (2010), k-Wave Toolbox default; Duck (1990)
pub const SOUND_SPEED_WATER_SIM: f64 = 1500.0;

/// Speed of sound in water at 20°C (m/s) — physically precise value
///
/// Value: 1482.0 m/s at 20°C, 1 atm.
///
/// For temperature-dependent speed of sound use
/// [`crate::core::constants::water::WaterProperties::sound_speed`], which
/// implements the Del Grosso & Mader (1972) 5th-order polynomial accurate to
/// ±0.1 m/s over 0–100 °C.
///
/// For temperature-dependent viscosity use
/// [`crate::core::constants::state_dependent::StateDependent::dynamic_viscosity_water`]
/// (Dortmund Data Bank VFT formula, < 2% vs NIST over 0–100 °C) or
/// [`crate::core::constants::state_dependent::StateDependent::viscosity_arrhenius`]
/// for generic Arrhenius fluids.
///
/// Reference: National Physical Laboratory acoustic properties database.
pub const SOUND_SPEED_WATER: f64 = 1482.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s)
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of water at 20°C (kg/m³)
/// Value: 998.2 kg/m³ (precise value)
/// Reference: NIST Chemistry WebBook
pub const DENSITY_WATER: f64 = 998.2;

/// Nominal density of water (kg/m³) — round-number simulation default
///
/// Value: 1000.0 kg/m³ — the standard round-number approximation used throughout
/// ultrasound simulation, HIFU planning, and acoustic modelling literature when
/// sub-percent density accuracy is not required.
///
/// Use `DENSITY_WATER = 998.2 kg/m³` when physical precision is needed.
///
/// Reference: k-Wave toolbox default; Duck, F. A. (1990). Physical Properties of
/// Tissue. Academic Press, London.
pub const DENSITY_WATER_NOMINAL: f64 = 1000.0;

/// Sound speed in water at 20°C (m/s) - Alias for compatibility
pub const C_WATER: f64 = SOUND_SPEED_WATER;

/// Bulk modulus of water at 20°C (Pa)
/// K = ρ * c², where ρ = 998.2 kg/m³, c = 1482 m/s
pub const BULK_MODULUS_WATER: f64 = 2.19e9;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Density of whole blood at 37°C (kg/m³)
///
/// Value: 1060.0 kg/m³ — measured value for normal adult whole blood.
///
/// Reference: ICRP Publication 23 (1975), *Report of the Task Group on Reference Man*,
/// Pergamon Press, Table 22 (p. 346). Also confirmed in:
/// Duck, F. A. (1990). *Physical Properties of Tissue*. Academic Press, London, p. 119.
pub const DENSITY_BLOOD: f64 = 1060.0;

/// Density of air at 20°C (kg/m³)
/// Value: 1.204 kg/m³ (at 20°C, 1 atm)
/// Reference: NIST Standard Reference Database
pub const DENSITY_AIR: f64 = 1.204;

/// Standard atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER_20C: f64 = 2339.0;

/// Gravitational acceleration (m/s²)
pub const GRAVITY: f64 = 9.80665;

/// Universal gas constant (J/(mol·K))
pub const GAS_CONSTANT: f64 = 8.314462618;

/// Avogadro's number (1/mol)
pub const AVOGADRO: f64 = 6.02214076e23;

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Planck constant (J·s)
pub const PLANCK: f64 = 6.62607015e-34;

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Elementary charge (C)
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

/// Vacuum permittivity (F/m)
/// Value: 8.8541878128e-12 F/m
/// Reference: 2018 CODATA recommended values
pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

/// Electron invariant mass (kg)
/// Value: 9.1093837015e-31 kg
/// Reference: 2018 CODATA recommended values
pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

// Pi is already available through std::f64::consts::PI

// ============================================================================
// Elastic Constants
// ============================================================================

/// Bond transformation factor for anisotropic media
pub const BOND_TRANSFORM_FACTOR: f64 = 1.0;

/// Lamé to stiffness conversion factor
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 1.0;

/// Symmetry tolerance for elastic tensors
pub const SYMMETRY_TOLERANCE: f64 = 1e-6;
