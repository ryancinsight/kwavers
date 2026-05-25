use super::gas_dynamics::{GasSpecies, GasType};
use crate::core::constants::acoustic_parameters::AIR_POLYTROPIC_INDEX;
use crate::core::constants::cavitation::{
    INITIAL_BUBBLE_RADIUS, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
};
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, C_WATER, DENSITY_WATER};
use crate::core::constants::thermodynamic::{
    SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER, T_AMBIENT,
};
use std::collections::HashMap;

/// Young-Laplace spherical-bubble surface tension pressure for an arbitrary σ.
///
/// `ΔP = 2σ/R` — Young-Laplace equation for a spherical interface.
/// Reference: Brennen C. E. (1995) *Cavitation and Bubble Dynamics*, §2.2.
///
/// Used by all bubble ODE models; the free-function form accepts variable σ
/// (e.g., Marmottant state-dependent surface tension) without borrowing `self`.
#[inline(always)]
pub fn young_laplace_pressure(sigma: f64, r: f64) -> f64 {
    2.0 * sigma / r
}

/// Liquid-side viscous bubble-wall stress.
///
/// `τ = 4μṘ/R` — viscous damping term in the Rayleigh-Plesset / Keller-Miksis / Gilmore
/// equations.
/// Reference: Brennen C. E. (1995) *Cavitation and Bubble Dynamics*, §2.3, Eq. 2.11.
///
/// Accepts variable `mu_liquid` for generality; all standard models pass `params.mu_liquid`.
#[inline(always)]
pub fn viscous_bubble_wall_stress(mu_liquid: f64, r_dot: f64, r: f64) -> f64 {
    4.0 * mu_liquid * r_dot / r
}

/// Physical parameters for bubble dynamics
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    // Equilibrium properties
    pub r0: f64, // Equilibrium radius [m]
    pub p0: f64, // Ambient pressure [Pa]

    // Liquid properties
    pub rho_liquid: f64, // Liquid density [kg/m³]
    pub c_liquid: f64,   // Sound speed in liquid [m/s]
    pub mu_liquid: f64,  // Dynamic viscosity [Pa·s]
    pub sigma: f64,      // Surface tension [N/m]
    pub pv: f64,         // Vapor pressure [Pa]

    // Thermal properties
    pub thermal_conductivity: f64, // k [W/(m·K)]
    pub specific_heat_liquid: f64, // cp [J/(kg·K)]
    pub accommodation_coeff: f64,  // Thermal accommodation

    // Gas properties
    pub gas_species: GasSpecies,
    pub initial_gas_pressure: f64, // Initial gas pressure [Pa]
    /// Gas composition: maps gas type to mole fraction
    /// Default is air (79% N2, 21% O2)
    pub gas_composition: HashMap<GasType, f64>,
    pub gamma: f64, // Adiabatic index (ratio of specific heats)
    pub t0: f64,    // Ambient temperature [K]

    // Acoustic forcing parameters
    pub driving_frequency: f64, // Driving frequency [Hz]
    pub driving_amplitude: f64, // Pressure amplitude [Pa]

    // Numerical parameters
    pub use_compressibility: bool, // Use Keller-Miksis
    pub use_thermal_effects: bool, // Include heat transfer
    pub use_mass_transfer: bool,   // Include evaporation/condensation
}

impl Default for BubbleParameters {
    fn default() -> Self {
        // Default air composition
        let mut gas_composition = HashMap::new();
        gas_composition.insert(GasType::N2, 0.79);
        gas_composition.insert(GasType::O2, 0.21);

        Self {
            // Water at 20°C with 5 μm air bubble — all water properties from SSOT.
            r0: INITIAL_BUBBLE_RADIUS,
            p0: ATMOSPHERIC_PRESSURE,
            rho_liquid: DENSITY_WATER,
            c_liquid: C_WATER,
            gamma: AIR_POLYTROPIC_INDEX,
            t0: T_AMBIENT, // 20°C in Kelvin (293.15 K)
            mu_liquid: VISCOSITY_WATER,
            sigma: SURFACE_TENSION_WATER,
            pv: VAPOR_PRESSURE_WATER,
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
            specific_heat_liquid: SPECIFIC_HEAT_WATER,
            accommodation_coeff: 0.04,
            gas_species: GasSpecies::Air,
            initial_gas_pressure: ATMOSPHERIC_PRESSURE,
            gas_composition,
            driving_frequency: 26.5e3, // 26.5 kHz (typical medical ultrasound)
            driving_amplitude: 1e5,    // 100 kPa acoustic pressure
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
}

impl BubbleParameters {
    /// Create parameters for pure gas bubble
    #[must_use]
    pub fn with_pure_gas(mut self, gas_type: GasType) -> Self {
        self.gas_composition.clear();
        self.gas_composition.insert(gas_type, 1.0);
        self
    }

    /// Surface tension pressure at the bubble wall from Young-Laplace equation.
    ///
    /// Delegates to [`young_laplace_pressure`] with `self.sigma`.
    #[inline(always)]
    #[must_use]
    pub fn surface_tension_pressure(&self, r: f64) -> f64 {
        young_laplace_pressure(self.sigma, r)
    }

    /// Liquid-side viscous wall stress at the bubble wall.
    ///
    /// Delegates to [`viscous_bubble_wall_stress`] with `self.mu_liquid`.
    #[inline(always)]
    #[must_use]
    pub fn viscous_wall_stress(&self, r_dot: f64, r: f64) -> f64 {
        viscous_bubble_wall_stress(self.mu_liquid, r_dot, r)
    }

    /// Calculate effective Van der Waals constants for gas mixture
    #[must_use]
    pub fn effective_vdw_constants(&self) -> (f64, f64) {
        let mut a_mix = 0.0;
        let mut b_mix = 0.0;

        // Use mixing rules for Van der Waals constants
        // a_mix = (Σ x_i * sqrt(a_i))^2 (geometric mean for a)
        // b_mix = Σ x_i * b_i (arithmetic mean for b)
        for (gas, &fraction) in &self.gas_composition {
            a_mix += fraction * gas.vdw_a().sqrt();
            b_mix += fraction * gas.vdw_b();
        }
        a_mix = a_mix.powi(2);

        (a_mix, b_mix)
    }
}

#[cfg(test)]
mod tests {
    use super::super::gas_dynamics::GasType;
    use super::*;
    use crate::core::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;

    /// Default BubbleParameters matches documented water/air-bubble values.
    ///
    /// Physical reference: 5 µm air bubble in water at 20°C, 1 atm.
    #[test]
    fn default_parameters_match_documented_water_air_bubble() {
        let p = BubbleParameters::default();
        assert_eq!(p.r0, 5e-6, "equilibrium radius = 5 µm");
        assert_eq!(
            p.p0, ATMOSPHERIC_PRESSURE,
            "ambient pressure from SSOT (= 1 atm)"
        );
        assert_eq!(
            p.rho_liquid, DENSITY_WATER,
            "water density from SSOT (= 998.2 kg/m³ at 20 °C)"
        );
        assert_eq!(
            p.sigma, SURFACE_TENSION_WATER,
            "surface tension from SSOT (= 72.8 mN/m at 20 °C)"
        );
        assert!(
            (p.gamma - HEAT_CAPACITY_RATIO_DIATOMIC).abs() < 1e-10,
            "air γ = 1.4 (diatomic)"
        );
    }

    /// Default air composition sums to 1.0 (0.79 N₂ + 0.21 O₂).
    #[test]
    fn default_gas_composition_sums_to_unity() {
        let p = BubbleParameters::default();
        let total: f64 = p.gas_composition.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-14,
            "gas fractions must sum to 1.0 (got {total:.6})"
        );
    }

    /// `with_pure_gas` replaces the mixture with a single-component composition.
    ///
    /// After the call: only one entry in gas_composition with fraction = 1.0.
    #[test]
    fn with_pure_gas_replaces_composition_with_single_species() {
        let p = BubbleParameters::default().with_pure_gas(GasType::Ar);
        assert_eq!(
            p.gas_composition.len(),
            1,
            "must have exactly one gas species"
        );
        let &fraction = p
            .gas_composition
            .get(&GasType::Ar)
            .expect("Ar must be present");
        assert!((fraction - 1.0).abs() < 1e-14, "Ar fraction must be 1.0");
    }

    /// `effective_vdw_constants` returns positive finite values for default air.
    ///
    /// Both a_mix and b_mix are positive for any physically valid composition.
    #[test]
    fn effective_vdw_constants_positive_for_air_mixture() {
        let p = BubbleParameters::default();
        let (a, b) = p.effective_vdw_constants();
        assert!(
            a > 0.0 && a.is_finite(),
            "a_mix must be positive finite (got {a:.6e})"
        );
        assert!(
            b > 0.0 && b.is_finite(),
            "b_mix must be positive finite (got {b:.6e})"
        );
    }
}
