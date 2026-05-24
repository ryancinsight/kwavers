//! Fundamental physical constants: simulation defaults, water properties,
//! universal physics, and elastic/optical invariants.
//!
//! Tissue-specific acoustic properties live in [`super::tissue_acoustics`].
//! CT Hounsfield-unit thresholds live in [`super::ct_acoustics`].
//!
//! # Compatibility shims
//!
//! The `pub use` lines below preserve backward compatibility for the ~549 call
//! sites that import directly from this module path.  A cleanup task
//! ([backlog.md §CLEAN-001]) will migrate all callers to the canonical submodule
//! paths and remove these shims.
pub use crate::core::constants::ct_acoustics::*;
pub use crate::core::constants::tissue_acoustics::*;

// ── Simulation defaults ───────────────────────────────────────────────────────

/// Speed of sound in water at 20°C (m/s) — k-Wave simulation default.
///
/// Value: 1500.0 m/s — the standard nominal reference used in k-Wave and most
/// ultrasound simulation/beamforming literature.
/// Note: the physically precise value at 20°C is 1482 m/s (see `SOUND_SPEED_WATER`).
/// Reference: Treeby & Cox (2010), k-Wave Toolbox default; Duck (1990).
pub const SOUND_SPEED_WATER_SIM: f64 = 1500.0;

/// Speed of sound in water at 20°C (m/s) — physically precise value.
///
/// Value: 1482.0 m/s at 20°C, 1 atm.
/// Reference: National Physical Laboratory acoustic properties database.
pub const SOUND_SPEED_WATER: f64 = 1482.0;

/// Speed of sound in soft tissue (m/s) — generic simulation default.
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s).
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of soft tissue (kg/m³) — generic simulation default.
pub const DENSITY_TISSUE: f64 = 1050.0;

// ── Water properties ──────────────────────────────────────────────────────────

/// Density of water at 20°C (kg/m³).
///
/// Value: 998.2 kg/m³ (precise value). Reference: NIST Chemistry WebBook.
pub const DENSITY_WATER: f64 = 998.2;

/// Nominal density of water (kg/m³) — round-number simulation default.
///
/// Value: 1000.0 kg/m³. Use `DENSITY_WATER = 998.2 kg/m³` when physical
/// precision is needed. Reference: k-Wave toolbox default; Duck (1990).
pub const DENSITY_WATER_NOMINAL: f64 = 1000.0;

/// Speed of sound in water at 37°C / body temperature (m/s).
///
/// Value: 1524.0 m/s. Reference: Del Grosso VA, Mader CW (1972).
/// J. Acoust. Soc. Am. **52**(5):1442–1446; Duck FA (1990) Table 2.1.
pub const SOUND_SPEED_WATER_37C: f64 = 1524.0;

/// Density of water at 37°C (kg/m³).
///
/// Value: 993.3 kg/m³. Reference: NIST Chemistry WebBook SRD 69.
pub const DENSITY_WATER_37C: f64 = 993.3;

/// Speed of sound in water at 20°C (m/s) — alias for `SOUND_SPEED_WATER`.
pub const C_WATER: f64 = SOUND_SPEED_WATER;

/// Bulk modulus of water at 20°C (Pa).
///
/// K = ρ·c² ≈ 998.2 × 1482² ≈ 2.19 GPa.
pub const BULK_MODULUS_WATER: f64 = 2.19e9;

/// Nominal acoustic impedance of water / water-like tissue (Pa·s/m = Rayl).
///
/// Z = ρ·c = DENSITY_WATER_NOMINAL × SOUND_SPEED_WATER_SIM = 1.5 MRayl.
///
/// Reference: Duck FA (1990); k-Wave defaults.
pub const ACOUSTIC_IMPEDANCE_WATER_NOMINAL: f64 = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM;

/// Default acoustic absorption coefficient of soft tissue [dB/(cm·MHz)].
///
/// Value: 0.5 dB/(cm·MHz) — mid-range for generic soft tissue.
/// Reference: Duck, F.A. (1990) Table 4.1.
pub const ACOUSTIC_ABSORPTION_TISSUE: f64 = 0.5; // dB/(cm·MHz)

// ── Universal physical constants ──────────────────────────────────────────────

/// Standard atmospheric pressure (Pa).
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Vapor pressure of water at 20°C (Pa).
pub const VAPOR_PRESSURE_WATER_20C: f64 = 2339.0;

/// Gravitational acceleration (m/s²).
pub const GRAVITY: f64 = 9.80665;

/// Universal gas constant (J/(mol·K)).
pub const GAS_CONSTANT: f64 = 8.314462618;

/// Avogadro's number (1/mol).
pub const AVOGADRO: f64 = 6.02214076e23;

/// Boltzmann constant (J/K).
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Planck constant (J·s). Exact defined value since the 2019 SI redefinition.
pub const PLANCK: f64 = 6.62607015e-34;

/// Reduced Planck constant ℏ = h / (2π) (J·s). CODATA 2018.
pub const REDUCED_PLANCK: f64 = 1.054_571_817e-34;

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Stefan-Boltzmann constant (W/(m²·K⁴)).
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Elementary charge (C).
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

/// Vacuum permittivity (F/m). CODATA 2018.
pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

/// Vacuum permeability (H/m). CODATA 2018.
///
/// Satisfies c² · ε₀ · μ₀ = 1 with the other defined constants.
pub const VACUUM_PERMEABILITY: f64 = 1.25663706212e-6;

/// Vacuum impedance Z₀ = √(μ₀ / ε₀) (Ω). CODATA 2018.
pub const VACUUM_IMPEDANCE: f64 = 376.730_313_668;

/// Electron invariant mass (kg). CODATA 2018.
pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

// ── Elastic / structural constants ────────────────────────────────────────────

/// Bond transformation factor for anisotropic elastic media.
pub const BOND_TRANSFORM_FACTOR: f64 = 2.0;

/// Lamé-to-stiffness conversion factor.
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 2.0;

/// Symmetry tolerance for elastic tensor checks.
pub const SYMMETRY_TOLERANCE: f64 = 1e-6;

// ── Optical tissue constants (near-infrared, 750–900 nm) ─────────────────────

/// Optical absorption coefficient of soft tissue in the NIR (750–900 nm) [m⁻¹].
///
/// Reference: Jacques, S.L. (2013). Phys. Med. Biol. 58(11), R37–R61.
pub const OPTICAL_ABSORPTION_TISSUE_NIR: f64 = 10.0; // m⁻¹

/// Reduced scattering coefficient of soft tissue in the NIR (750–900 nm) [m⁻¹].
///
/// Reference: Jacques, S.L. (2013). Phys. Med. Biol. 58(11), R37–R61.
pub const REDUCED_SCATTERING_TISSUE_NIR: f64 = 1000.0; // m⁻¹
