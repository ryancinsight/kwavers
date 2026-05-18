//! Chemical reaction enthalpy effects

use uom::si::f64::Power;
use uom::si::power::watt;

use crate::core::constants::fundamental::GAS_CONSTANT as R_GAS;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;

impl EnergyBalanceCalculator {
    /// Calculate thermal dissociation energy rate for water vapor.
    ///
    /// # Reaction
    ///
    /// During extreme compression (T > 2000 K), water vapor dissociates endothermically:
    ///
    /// ```text
    /// H₂O + M → H• + OH•  + M     ΔH°₂₉₈ = +498.4 kJ/mol
    /// ```
    ///
    /// (Net reaction with recombination channels accounts for reduced effective ΔH.)
    ///
    /// # Arrhenius Rate Law
    ///
    /// The first-order thermal dissociation rate constant (Yasui 1997, Eq. 16;
    /// Baulch et al. 2005, Reaction R5):
    ///
    /// ```text
    /// k(T) = A · exp(−Eₐ / (R T))   [s⁻¹]
    ///
    /// A   = 1.912 × 10¹⁶  s⁻¹   (high-pressure limit, third-body enhanced)
    /// Eₐ  = 495.4 kJ/mol          (O–H bond energy; NIST WebBook)
    /// ```
    ///
    /// Molar dissociation rate: `ṅ = k(T) · n_vapor`  [mol/s]
    ///
    /// Volumetric heat absorption rate (W):
    /// ```text
    /// Q̇_chem = −ΔH_diss · ṅ
    /// ```
    ///
    /// Negative sign: endothermic reaction absorbs energy from the bubble.
    ///
    /// # References
    ///
    /// - Yasui K (1997). "Alternative model of single-bubble sonoluminescence."
    ///   *Phys Rev E* 56(6):6750–6760.
    /// - Baulch DL et al. (2005). "Evaluated kinetic data for combustion modelling."
    ///   *J Phys Chem Ref Data* 34(3):757–1397. Reaction R5.
    /// - Storey BD & Szeri AJ (2000). "Water vapour, sonoluminescence and sonochemistry."
    ///   *J Fluid Mech* 396:203–229.
    #[must_use]
    pub fn calculate_chemical_reaction_rate(&self, state: &BubbleState) -> Power {
        if !self.enable_chemical_reactions || state.temperature < 2000.0 {
            return Power::new::<watt>(0.0);
        }

        // Water dissociation enthalpy (Baulch et al. 2005): ΔH = 498.4 kJ/mol
        const H_DISSOCIATION: f64 = 498_400.0; // J/mol

        // Arrhenius parameters (Yasui 1997, Table I; Baulch 2005 Reaction R5)
        const ACTIVATION_ENERGY: f64 = 495_400.0; // J/mol (O−H bond dissociation energy)
        const PRE_EXPONENTIAL: f64 = 1.912e16; // s⁻¹ (high-pressure limit)

        // k(T) = A · exp(−Eₐ / RT)  [s⁻¹]
        let rate_constant =
            PRE_EXPONENTIAL * (-ACTIVATION_ENERGY / (R_GAS * state.temperature)).exp();

        // Number of moles of water vapor that can dissociate
        let n_vapor_moles = state.n_vapor / crate::core::constants::AVOGADRO;

        // Molar dissociation rate: ṅ = k × n_vapor  [mol/s]
        let dissociation_rate_mol_per_s = rate_constant * n_vapor_moles;

        // Energy absorption rate: Q̇ = −ΔH · ṅ  [W] (negative = endothermic)
        Power::new::<watt>(-dissociation_rate_mol_per_s * H_DISSOCIATION)
    }
}
