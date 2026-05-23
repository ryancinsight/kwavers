//! Cavitation-related constants

/// Blake threshold pressure ratio for inertial cavitation
pub const BLAKE_THRESHOLD: f64 = 0.541;

/// Surface tension of water at 20°C (N/m)
pub const SURFACE_TENSION_WATER: f64 = 0.0728;

/// Effective surface tension at the soft-tissue–gas interface (N/m).
///
/// Value: 0.060 N/m — lower than pure water due to proteins, lipids, and
/// surfactant-like molecules in biological tissue. This value governs the
/// Laplace pressure in tissue-embedded bubble models.
///
/// Reference: Cheeke JDN (2002). *Fundamentals and Applications of
/// Ultrasonic Waves*. CRC Press, p. 199.
pub const SURFACE_TENSION_TISSUE: f64 = 0.060;

/// Dynamic viscosity of air at 20°C (Pa·s).
///
/// Value: 1.81×10⁻⁵ Pa·s — standard reference at 293.15 K, 1 atm.
///
/// Reference: NIST Chemistry WebBook, thermophysical properties of air.
/// Sutherland's law at T = 293.15 K yields μ = 1.813×10⁻⁵ Pa·s.
pub const VISCOSITY_AIR: f64 = 1.81e-5;

/// Dissolved-gas diffusion coefficient in soft tissue (m²/s).
///
/// O₂ in soft tissue at 37°C; directly measured by Krogh-cylinder methods
/// and confirmed by fluorescence quenching.
/// Value: 1.8×10⁻⁹ m²/s.
///
/// Reference: Pittman RN (2011). *Regulation of Tissue Oxygenation*.
/// Morgan & Claypool Life Sciences, Chapter 4.
pub const GAS_DIFFUSION_COEFFICIENT_TISSUE: f64 = 1.8e-9;

/// Binary gas diffusion coefficient for O₂/N₂ in air at 20°C, 1 atm (m²/s).
///
/// Chapman–Enskog kinetic theory; confirmed by experimental measurements.
/// Value: 2.0×10⁻⁵ m²/s.
///
/// Reference: Bird RB, Stewart WE, Lightfoot EN (2002).
/// *Transport Phenomena*, 2nd ed. Wiley, Appendix A.
pub const GAS_DIFFUSION_COEFFICIENT_AIR: f64 = 2.0e-5;

/// Viscosity of water at 20°C (Pa·s)
pub const VISCOSITY_WATER: f64 = 1.002e-3;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER: f64 = 2339.0;

/// Vapor pressure of water at 25°C (Pa)
///
/// Value: 3169.0 Pa — computed from the Antoine equation at 25 °C (298.15 K).
/// Used as the ambient-lab-temperature reference in bubble-dynamics models
/// that operate near room temperature (25 °C) rather than at 20 °C.
///
/// Reference: CRC Handbook of Chemistry and Physics, 97th ed., Table 6-5.
pub const VAPOR_PRESSURE_WATER_25C: f64 = 3169.0;

/// Polytropic exponent for air [-]
///
/// γ = 1.4 for diatomic ideal gas. SSOT: delegates to `HEAT_CAPACITY_RATIO_DIATOMIC`.
pub const POLYTROPIC_EXPONENT_AIR: f64 =
    crate::core::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;

/// Van der Waals hard core radius for bubble (m)
pub const VAN_DER_WAALS_RADIUS: f64 = 8.86e-10;

/// Mechanical index threshold for bioeffects
pub const MECHANICAL_INDEX_THRESHOLD: f64 = 0.7;

/// Flynn collapse coefficient for transient cavitation threshold.
///
/// The threshold acoustic pressure for violent inertial collapse is:
/// $$ P_{\text{Flynn}} = \alpha \left( P_0 + \frac{2\sigma}{R_n} \right) - P_v $$
///
/// where $\alpha = 0.83$ is the empirical collapse fraction derived from
/// numerical integration of the Rayleigh–Plesset equation showing that
/// $R_{\max}/R_0 \ge 2$ (violent collapse) occurs at approximately 83%
/// of the combined hydrostatic and Laplace pressure.
///
/// # References
/// - Flynn, H. G. (1964). "Physics of Acoustic Cavitation in Liquids".
///   In *Methods of Experimental Physics*, Vol. 1, Academic Press, pp. 57–172.
pub const FLYNN_COLLAPSE_COEFFICIENT: f64 = 0.83;

/// Cavitation inception threshold (MPa)
pub const CAVITATION_THRESHOLD_WATER: f64 = -30.0;

/// Typical bubble damping constant
pub const BUBBLE_DAMPING_CONSTANT: f64 = 1.5e-9;

// ============================================================================
// Cavitation Damage Parameters
// ============================================================================

/// Compression factor exponent for damage calculation
pub const COMPRESSION_FACTOR_EXPONENT: f64 = 2.0;

/// Default bubble concentration factor
pub const DEFAULT_CONCENTRATION_FACTOR: f64 = 1e5;

/// Default fatigue rate for material
pub const DEFAULT_FATIGUE_RATE: f64 = 0.01;

/// Default pit formation efficiency
pub const DEFAULT_PIT_EFFICIENCY: f64 = 0.1;

/// Default cavitation threshold pressure (Pa)
pub const DEFAULT_THRESHOLD_PRESSURE: f64 = -1e5;

/// Impact energy coefficient
pub const IMPACT_ENERGY_COEFFICIENT: f64 = 0.5;

/// Material removal efficiency factor
pub const MATERIAL_REMOVAL_EFFICIENCY: f64 = 0.05;

// ============================================================================
// Bubble Dynamics Limits
// ============================================================================

/// Maximum bubble radius (m)
pub const MAX_RADIUS: f64 = 1e-3;

/// Minimum bubble radius (m)
pub const MIN_RADIUS: f64 = 1e-9;

/// Initial (equilibrium) bubble radius for a 5 μm air bubble in water at 20°C (m).
///
/// Representative value for ultrasound contrast agents and dissolved-gas nuclei.
///
/// Reference: Leighton T. G. (1994). The Acoustic Bubble. Academic Press. §2.1.
pub const INITIAL_BUBBLE_RADIUS: f64 = 5e-6;

/// Conversion factor from `bar·L²` to `Pa·m⁶`.
///
/// Derivation: `1 bar = 10⁵ Pa`, `1 L² = (10⁻³ m³)² = 10⁻⁶ m⁶`, so
/// `1 bar·L² = 10⁵ × 10⁻⁶ Pa·m⁶ = 10⁻¹ Pa·m⁶`. Used to convert tabulated
/// Van der Waals `a` constants (bar·L²/mol²) into SI (Pa·m⁶/mol²).
///
/// Was previously `1e-7` — six orders of magnitude too small, which made
/// the VdW attraction term in `RayleighPlessetSolver::calculate_internal_pressure`
/// effectively zero and silently reduced thermal-mode VdW to ideal-gas
/// behaviour.
pub const BAR_L2_TO_PA_M6: f64 = 1e-1;

/// Conversion factor from liters to cubic meters
pub const L_TO_M3: f64 = 1e-3;

/// Minimum Peclet number for thermal effects
pub const MIN_PECLET_NUMBER: f64 = 10.0;

/// Peclet number scaling factor
pub const PECLET_SCALING_FACTOR: f64 = 0.1;

/// Surface tension coefficient multiplier
pub const SURFACE_TENSION_COEFF: f64 = 2.0;

// ============================================================================
// Power Modulation Limits
// ============================================================================

/// Minimum duty cycle for power modulation
pub const MIN_DUTY_CYCLE: f64 = 0.01;

/// Maximum duty cycle for power modulation
pub const MAX_DUTY_CYCLE: f64 = 1.0;
