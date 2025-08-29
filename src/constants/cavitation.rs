//! Cavitation-related constants

/// Blake threshold pressure for cavitation inception [Pa]
pub const BLAKE_THRESHOLD: f64 = 1.5e5;

/// Cavitation index threshold
pub const CAVITATION_INDEX_THRESHOLD: f64 = 1.0;

/// Minimum bubble radius [m]
pub const MIN_BUBBLE_RADIUS: f64 = 1e-9;

/// Maximum bubble radius [m]
pub const MAX_BUBBLE_RADIUS: f64 = 1e-3;

/// Initial bubble radius [m]
pub const INITIAL_BUBBLE_RADIUS: f64 = 1e-6;

/// Vapor pressure of water at 20°C [Pa]
pub const VAPOR_PRESSURE_WATER: f64 = 2339.0;

/// Surface tension of water [N/m]
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Polytrophic gas constant
pub const POLYTROPIC_GAS_CONSTANT: f64 = 1.4;

/// Compression factor exponent
pub const COMPRESSION_FACTOR_EXPONENT: f64 = 3.0;

/// Default concentration factor
pub const DEFAULT_CONCENTRATION_FACTOR: f64 = 1e-6;

/// Default fatigue rate
pub const DEFAULT_FATIGUE_RATE: f64 = 1e-3;

/// Default pit efficiency
pub const DEFAULT_PIT_EFFICIENCY: f64 = 0.1;

/// Default threshold pressure
pub const DEFAULT_THRESHOLD_PRESSURE: f64 = 1e5;

/// Impact energy coefficient
pub const IMPACT_ENERGY_COEFFICIENT: f64 = 0.5;

/// Material removal efficiency
pub const MATERIAL_REMOVAL_EFFICIENCY: f64 = 0.01;
