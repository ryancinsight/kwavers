//! Tissue ablation kinetics for thermal therapy
//!
//! Implements thermal ablation kinetics based on classical thermodynamic models
//! for protein denaturation and subsequent tissue necrosis.
//!
//! # Mathematical Specifications
//!
//! ## Theorem 1: Arrhenius Damage Accumulation
//! The irreversible thermal damage $\Omega$ accumulated in tissue over time $t$ obeys
//! the Arrhenius rate equation, positing that cell death behaves as a first-order
//! chemical reaction.
//!
//! $$ \Omega(t) = \int_0^t A \exp\left(-\frac{E_a}{R T(\tau)}\right) d\tau $$
//!
//! **Proof / Invariants:** 
//! The survival fraction of cells $S(t) / S(0)$ is modeled exponentially:
//! $S(t) = S(0) \exp(-\Omega(t))$. Ergo, $\Omega = 1$ mathematically guarantees
//! $1 - e^{-1} \approx 63.2\%$ cell death, establishing the threshold for irreversible
//! protein denaturation.
//!
//! ## Theorem 2: Cumulative Equivalent Minutes at 43°C (CEM43)
//! Thermal dose is normalized to equivalent minutes at 43°C (Sapareto & Dewey):
//!
//! $$ \text{CEM}_{43} = \int_{0}^{t} R^{43 - T(\tau)} d\tau $$
//!
//! Where $R$ represents the empirical compensation constant. For $T > 43^\circ\text{C}$, 
//! $R \approx 0.5$, indicating damage rate doubles for every 1°C increase.
//!
//! # References
//! - Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in cancer therapy". Int. J. Radiation Oncology Biol. Phys., 10(6), 787-800.
//! - Henriques, F. C. (1947). "Studies of thermal injury". Archives of Pathology, 43(5), 489-502.
//! - Lepock, J. R., et al. (1993). "Thermal denaturation of proteins". Int. J. Hyperthermia, 9(2), 263-270.

/// Ablation kinetics parameters (Arrhenius model)
///
/// The Arrhenius equation for thermal damage: Ω(t) = ∫A·exp(-E_a/RT)dt
/// where:
/// - A = frequency factor (1/s)
/// - E_a = activation energy (J/mol)
/// - R = universal gas constant (8.314 J/mol/K)
/// - T = absolute temperature (K)
#[derive(Debug, Clone, Copy)]
pub struct AblationKinetics {
    /// Frequency factor [1/s] - typical: 1.0e69 to 1.0e75
    pub frequency_factor: f64,
    /// Activation energy [J/mol] - typical: 250,000 to 650,000 J/mol
    pub activation_energy: f64,
    /// Damage threshold (Ω) - typical: 1.0 for 63.2% protein denaturation
    pub damage_threshold: f64,
    /// Temperature at which tissue begins ablation [°C] - typical: 45-60°C
    pub ablation_threshold: f64,
}

impl AblationKinetics {
    /// Create custom ablation kinetics
    pub fn new(
        frequency_factor: f64,
        activation_energy: f64,
        damage_threshold: f64,
        ablation_threshold: f64,
    ) -> Self {
        Self {
            frequency_factor,
            activation_energy,
            damage_threshold,
            ablation_threshold,
        }
    }

    /// Protein denaturation kinetics (Henriques model)
    /// Reference: Henriques (1947) - thermal injury to skin
    /// A = 1.0e44 [1/s], E_a = 284 kJ/mol
    pub fn protein_denaturation() -> Self {
        Self {
            frequency_factor: 1.0e44,     // [1/s]
            activation_energy: 284_000.0, // [J/mol] (67.8 kcal/mol)
            damage_threshold: 1.0,
            ablation_threshold: 45.0,
        }
    }

    /// Collagen denaturation kinetics
    /// Reference: Lepock et al. (1993) - collagen triple helix dissociation
    /// Lower activation energy than protein, faster denaturation
    pub fn collagen_denaturation() -> Self {
        Self {
            frequency_factor: 1.0e44,     // [1/s]
            activation_energy: 250_000.0, // [J/mol] (59.8 kcal/mol) - lower than protein
            damage_threshold: 1.0,
            ablation_threshold: 55.0,
        }
    }

    /// HIFU ablation kinetics (tissue necrosis)
    /// Reference: Sapareto & Dewey (1984) - thermal dose model
    /// Higher frequency factor for faster ablation kinetics
    pub fn hifu_ablation() -> Self {
        Self {
            frequency_factor: 1.0e47,     // [1/s] - higher for faster tissue necrosis
            activation_energy: 284_000.0, // [J/mol] (67.8 kcal/mol)
            damage_threshold: 1.0,
            ablation_threshold: 50.0,
        }
    }

    /// Temperature-dependent damage rate
    ///
    /// dΩ/dt = A·exp(-E_a/RT)
    ///
    /// # Arguments
    /// * `temperature` - Absolute temperature [K]
    ///
    /// # Returns
    /// Damage accumulation rate [1/s]
    pub fn damage_rate(&self, temperature: f64) -> f64 {
        let r = 8.314; // Universal gas constant [J/mol/K]
        self.frequency_factor * (-self.activation_energy / (r * temperature)).exp()
    }

    /// Accumulated thermal damage (Ω)
    ///
    /// Ω(t) = ∫₀ᵗ A·exp(-E_a/RT(τ))dτ
    ///
    /// # Arguments
    /// * `damage_rate` - Damage accumulation rate at current time [1/s]
    /// * `dt` - Time step [s]
    ///
    /// # Returns
    /// Accumulated damage (unitless, 0-∞)
    /// - Ω < 1: No significant damage
    /// - Ω = 1: 63.2% protein denaturation
    /// - Ω > 1: Irreversible damage
    pub fn accumulated_damage(current_damage: f64, damage_rate: f64, dt: f64) -> f64 {
        current_damage + damage_rate * dt
    }

    /// Tissue viability (thermal damage extent)
    ///
    /// Returns fraction of viable tissue: exp(-Ω)
    /// - 1.0 = fully viable
    /// - 0.37 = 63% damage (Ω=1)
    /// - 0.0 = completely ablated
    pub fn viability(&self, damage: f64) -> f64 {
        (-damage).exp()
    }

    /// Is tissue ablated?
    pub fn is_ablated(&self, damage: f64) -> bool {
        damage >= self.damage_threshold
    }
}

impl Default for AblationKinetics {
    fn default() -> Self {
        Self::hifu_ablation()
    }
}
