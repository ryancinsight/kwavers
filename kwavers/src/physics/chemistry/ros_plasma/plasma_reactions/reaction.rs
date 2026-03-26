//! Plasma reaction type definitions
//!
//! High-temperature reactions occurring in the bubble interior during collapse.
//!
//! # Mathematical Specification
//!
//! ## Theorem: Modified Arrhenius Rate Law
//! The temperature-dependent rate constant for elementary reactions follows
//! the modified Arrhenius form:
//!
//! $$ k(T) = A \cdot T^n \cdot \exp\left(-\frac{E_a}{R \cdot T}\right) $$
//!
//! where $A$ is the pre-exponential factor, $n$ the temperature exponent,
//! $E_a$ the activation energy [J/mol], and $R$ the universal gas constant.
//!
//! **Invariant:** $k(T) > 0$ for all $T > 0$ (rate constants are strictly positive).
//!
//! ## Theorem: Zeldovich Thermal NO Mechanism
//! Thermal NO production proceeds via the extended Zeldovich mechanism:
//! 1. $N_2 + O \rightleftharpoons NO + N$ (rate-limiting)
//! 2. $N + O_2 \rightleftharpoons NO + O$
//!
//! The onset temperature threshold ($T \ge 1800\,\text{K}$) derives from
//! the high activation energy of N₂ triple-bond dissociation ($E_a \approx 315\,\text{kJ/mol}$).
//!
//! # References
//! - Zeldovich, Y. B. (1946). "The oxidation of nitrogen in combustion and explosions"
//! - Eller & Flynn (1965). "Rectified diffusion during nonlinear pulsations"

use crate::core::constants::GAS_CONSTANT;

/// Plasma reaction types
#[derive(Debug, Clone)]
pub struct PlasmaReaction {
    /// Reaction name
    pub name: String,
    /// Reactants and their stoichiometric coefficients
    pub reactants: Vec<(String, f64)>,
    /// Products and their stoichiometric coefficients
    pub products: Vec<(String, f64)>,
    /// Activation energy (J/mol)
    pub activation_energy: f64,
    /// Pre-exponential factor (units depend on reaction order)
    pub pre_exponential: f64,
    /// Temperature exponent for modified Arrhenius
    pub temperature_exponent: f64,
}

impl PlasmaReaction {
    /// Calculate rate constant at given temperature
    ///
    /// Modified Arrhenius: k(T) = A·T^n·exp(-Ea/(R·T))
    #[must_use]
    pub fn rate_constant(&self, temperature: f64) -> f64 {
        self.pre_exponential
            * temperature.powf(self.temperature_exponent)
            * (-self.activation_energy / (GAS_CONSTANT * temperature)).exp()
    }
}

/// Calculate NO production rate (Zeldovich mechanism)
///
/// Reference: Zeldovich (1946) - thermal NO formation
/// N₂ + O ⇌ NO + N (rate-limiting step)
#[must_use]
pub fn zeldovich_no_rate(temperature: f64, o2_conc: f64, n2_conc: f64) -> f64 {
    if temperature < 1800.0 {
        return 0.0; // Too cold for thermal NO
    }

    // Equilibrium O atom concentration
    let k_o2_diss =
        3.6e18 * temperature.powf(-1.0) * (-495e3 / (GAS_CONSTANT * temperature)).exp();
    let o_eq = (k_o2_diss * o2_conc).sqrt();

    // NO formation rate
    let k_no = 1.8e14 * (-315e3 / (GAS_CONSTANT * temperature)).exp();
    k_no * o_eq * n2_conc
}

