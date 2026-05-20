//! Plasma ionization energy rate during cavitation collapses

use uom::si::f64::Power;
use uom::si::power::watt;

use crate::core::constants::fundamental::{
    AVOGADRO, BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE, PLANCK,
};
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleState, GasSpecies};
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;

impl EnergyBalanceCalculator {
    /// Calculate plasma ionization energy rate using the full Saha equation.
    ///
    /// During extreme compression (T > 10 000 K), gas ionization occurs, e.g.:
    /// ```text
    /// Ar → Ar⁺ + e⁻     E_ion = 15.76 eV
    /// ```
    /// Ionization absorbs energy endothermically, reducing peak temperature and
    /// light emission.
    ///
    /// # Saha Equation (Theorem)
    ///
    /// At thermal equilibrium, the degree of ionization `α = n_i/(n_i + n_0)`
    /// satisfies (Saha 1921; Griem 1964, *Plasma Spectroscopy*, §3.2):
    ///
    /// ```text
    /// α² / (1 − α) = Φ(T) / n_total
    ///
    /// Φ(T) = (2 g_i/g_0) · (2π m_e k_B T / h²)^(3/2) · exp(−E_ion / k_B T)
    /// ```
    ///
    /// where:
    /// - `g_i/g_0` = ratio of ionic to neutral statistical weights (≈ 1 for first ionization)
    /// - `m_e = 9.109×10⁻³¹ kg` (electron mass)
    /// - `k_B = 1.381×10⁻²³ J/K` (Boltzmann constant)
    /// - `h = 6.626×10⁻³⁴ J·s` (Planck constant)
    /// - `n_total` = total particle number density [m⁻³]
    ///
    /// Solving the quadratic for `α`:
    /// ```text
    /// α = (−Φ + √(Φ² + 4 Φ n_total)) / (2 n_total)   ∈ [0, 1]
    /// ```
    ///
    /// The energy absorption rate (W) = `α · n_total · E_ion / τ_eq`
    /// where `τ_eq = 1e-12 s` is the collisional equilibration time
    /// (Moss et al. 1997, Table 1).
    ///
    /// # References
    ///
    /// - Saha MN (1921). "On a physical theory of stellar spectra." *Proc R Soc Lond A* 99, 135–153.
    /// - Moss WC et al. (1997). "Sonoluminescence and the prospects for table-top micro-thermonuclear
    ///   fusion." *Phys Fluids* 9(6):1535–1538.
    /// - Hilgenfeldt S et al. (1999). "A simple explanation of light emission in sonoluminescence."
    ///   *J Fluid Mech* 365:171–204.
    #[must_use]
    pub fn calculate_plasma_ionization_rate(&self, state: &BubbleState) -> Power {
        if !self.enable_plasma_effects || state.temperature < 10_000.0 {
            return Power::new::<watt>(0.0);
        }

        // Ionization energy [eV → J] for each species (NIST Atomic Spectra Database).
        // `ELEMENTARY_CHARGE` is numerically equal to the eV-to-Joule conversion.
        let ev_to_joules = ELEMENTARY_CHARGE;
        let e_ion_j = match state.gas_species {
            GasSpecies::Argon => 15.759_610 * ev_to_joules,
            GasSpecies::Xenon => 12.129_843 * ev_to_joules,
            GasSpecies::Nitrogen => 14.534_135 * ev_to_joules,
            GasSpecies::Oxygen => 13.618_055 * ev_to_joules,
            GasSpecies::Air => 14.53 * ev_to_joules, // N₂-weighted average
            GasSpecies::Custom { .. } => 15.0 * ev_to_joules,
        };

        let t = state.temperature;
        let kt = BOLTZMANN * t;

        // Saha factor Φ(T) = 2 · (2π m_e k_B T / h²)^(3/2) · exp(−E_ion/kT)
        // Factor 2 accounts for the g_i/g_0 = 1 degeneracy ratio times the
        // electron spin degeneracy (2 spin states).
        let saha_prefactor =
            2.0 * (2.0 * std::f64::consts::PI * ELECTRON_MASS * kt / (PLANCK * PLANCK)).powf(1.5);
        let boltzmann_factor = (-e_ion_j / kt).exp();
        let phi = saha_prefactor * boltzmann_factor;

        // Number density [m⁻³]: n = N_A × n_moles / V
        let volume = (4.0 / 3.0) * std::f64::consts::PI * state.radius.powi(3);
        let n_moles = (state.n_gas + state.n_vapor) / AVOGADRO;
        let n_density = n_moles * AVOGADRO / volume.max(1e-30);

        // Solve α² / (1−α) = Φ / n for α:
        // α² + (Φ/n) α − (Φ/n) = 0
        // α = [−(Φ/n) + √((Φ/n)² + 4 Φ/n)] / 2  (positive root)
        let r = phi / n_density.max(1e-30);
        let alpha = r.mul_add(r, 4.0 * r).sqrt().mul_add(0.5, -r * 0.5);
        let alpha = alpha.clamp(0.0, 1.0);

        // Collisional equilibration time scale (Moss et al. 1997)
        const TAU_EQ_S: f64 = 1e-12; // 1 ps

        // Energy absorption rate: α × n_total × E_ion / τ_eq  [W]
        let n_total = state.n_gas + state.n_vapor;
        let power_w = alpha * n_total * e_ion_j / TAU_EQ_S;

        // Sign: endothermic (negative = energy absorbed from bubble)
        Power::new::<watt>(-power_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::cavitation::{
        SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
    };
    use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, C_WATER, DENSITY_WATER};
    use crate::core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
    use crate::physics::acoustics::bubble_dynamics::bubble_state::{
        BubbleParameters, BubbleState, GasSpecies, GasType,
    };

    fn make_params() -> BubbleParameters {
        BubbleParameters {
            r0: 5e-6,
            p0: ATMOSPHERIC_PRESSURE,
            rho_liquid: DENSITY_WATER,
            c_liquid: C_WATER,
            mu_liquid: VISCOSITY_WATER,
            sigma: SURFACE_TENSION_WATER,
            pv: VAPOR_PRESSURE_WATER,
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
            specific_heat_liquid: SPECIFIC_HEAT_WATER,
            accommodation_coeff: 0.35,
            gas_species: GasSpecies::Argon,
            initial_gas_pressure: 101_325.0,
            gas_composition: {
                let mut m = std::collections::HashMap::new();
                m.insert(GasType::N2, 1.0);
                m
            },
            gamma: 5.0 / 3.0,
            t0: crate::core::constants::thermodynamic::ROOM_TEMPERATURE_K,
            driving_frequency: 26_500.0,
            driving_amplitude: 1.5e5,
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }

    fn make_state(temperature: f64) -> BubbleState {
        let params = make_params();
        let mut s = BubbleState::new(&params);
        s.temperature = temperature;
        s
    }

    fn make_calc(enable_plasma: bool) -> EnergyBalanceCalculator {
        EnergyBalanceCalculator::with_options(&make_params(), true, enable_plasma, true)
    }

    /// At T < 10 000 K, plasma effects should return zero (below threshold).
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_plasma_below_threshold_returns_zero() {
        let state = make_state(9_999.0);
        let p = make_calc(true).calculate_plasma_ionization_rate(&state);
        use uom::si::power::watt;
        assert_eq!(p.get::<watt>(), 0.0);
    }

    /// At T = 10 000 K with plasma effects disabled, result is zero.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_plasma_disabled_returns_zero() {
        let state = make_state(20_000.0);
        let p = make_calc(false).calculate_plasma_ionization_rate(&state);
        use uom::si::power::watt;
        assert_eq!(p.get::<watt>(), 0.0);
    }

    /// At high temperature the ionization rate is negative (endothermic) and finite.
    ///
    /// Reference: Moss et al. (1997) Fig. 3 — ionization fraction ≈ 0.01 at 20 000 K.
    /// # Panics
    /// - Panics if assertion fails: `plasma ionization must absorb energy; got {:.3e} W`.
    ///
    #[test]
    fn test_plasma_at_20000k_endothermic_finite() {
        let state = make_state(20_000.0);
        let p = make_calc(true).calculate_plasma_ionization_rate(&state);
        use uom::si::power::watt;
        let watts = p.get::<watt>();
        // Must be endothermic (negative) and non-zero
        assert!(
            watts < 0.0,
            "plasma ionization must absorb energy; got {:.3e} W",
            watts
        );
        assert!(
            watts.is_finite(),
            "power must be finite; got {:.3e} W",
            watts
        );
    }

    /// Saha ionization fraction α grows monotonically with temperature.
    /// # Panics
    /// - Panics if assertion fails: `ionization power must grow in magnitude with T: p1={:.3e}, p2={:.3e}`.
    ///
    #[test]
    fn test_plasma_ionization_monotone_in_temperature() {
        use uom::si::power::watt;
        let calc = make_calc(true);
        let p1 = calc
            .calculate_plasma_ionization_rate(&make_state(15_000.0))
            .get::<watt>();
        let p2 = calc
            .calculate_plasma_ionization_rate(&make_state(20_000.0))
            .get::<watt>();
        // Both are negative (endothermic); higher T → more ionization → larger |p|
        assert!(
            p2 < p1,
            "ionization power must grow in magnitude with T: p1={:.3e}, p2={:.3e}",
            p1,
            p2
        );
    }
}
