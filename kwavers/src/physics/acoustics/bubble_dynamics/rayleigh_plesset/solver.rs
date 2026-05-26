//! Rayleigh-Plesset equation solver for bubble dynamics

use super::super::bubble_state::{BubbleParameters, BubbleState};
use crate::core::constants::cavitation::{BAR_L2_TO_PA_M6, L_TO_M3};
use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};
use crate::core::constants::numerical::{TWO_PI};

/// Rayleigh-Plesset equation solver (incompressible)
#[derive(Debug)]
pub struct RayleighPlessetSolver {
    params: BubbleParameters,
}

impl RayleighPlessetSolver {
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        Self { params }
    }

    /// Calculate bubble wall acceleration using Rayleigh-Plesset equation
    /// Standard form: ρ(RR̈ + 3/2Ṙ²) = pg - p∞ - 2σ/R - 4μṘ/R
    #[must_use]
    pub fn calculate_acceleration(&self, state: &BubbleState, p_acoustic: f64, t: f64) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;

        // Time-dependent acoustic forcing with proper phase tracking
        let acoustic_phase = TWO_PI * self.params.driving_frequency * t;
        let p_acoustic_instantaneous = p_acoustic * acoustic_phase.sin();
        let p_liquid_far = self.params.p0 + p_acoustic_instantaneous;

        // Direct physics-based pressure calculation.
        // Reference: Rayleigh-Plesset equation, Brennen (1995), "Cavitation and Bubble Dynamics"
        //
        // `calculate_internal_pressure` handles both the non-thermal (polytropic) and thermal
        // (Van der Waals) cases, and correctly separates vapor pressure from non-condensable
        // gas pressure before applying the polytropic scaling:
        //   p_gas = (p0 + 2σ/r0 − pv)·(r0/r)^{3γ} + pv
        // This preserves the equilibrium force balance at r = r0 and gives the physically
        // correct pv floor as r → ∞.
        let p_gas = self.calculate_internal_pressure(state);

        // Debug output for equilibrium testing
        #[cfg(test)]
        if r == self.params.r0 && v == 0.0 && p_acoustic == 0.0 && t == 0.0 {
            println!("Debug calculate_acceleration at equilibrium:");
            println!("  p_gas: {} Pa", p_gas);
            println!("  p_liquid_far: {} Pa", p_liquid_far);
            println!(
                "  Expected p_gas at eq: {} Pa",
                self.params.p0 + self.params.surface_tension_pressure(self.params.r0)
            );
        }

        // Forces on bubble wall (Pa)
        let pressure_diff = p_gas - p_liquid_far;
        let surface_tension = self.params.surface_tension_pressure(r);
        let viscous_stress = self.params.viscous_wall_stress(v, r);

        #[cfg(test)]
        if r == self.params.r0 && v == 0.0 && p_acoustic == 0.0 && t == 0.0 {
            println!("  pressure_diff: {} Pa", pressure_diff);
            println!("  surface_tension: {} Pa", surface_tension);
            println!("  viscous_stress: {} Pa", viscous_stress);
            println!(
                "  net_pressure: {} Pa",
                pressure_diff - surface_tension - viscous_stress
            );
            println!("  denominator: {} kg/m²", self.params.rho_liquid * r);
        }

        // Rayleigh-Plesset equation: ρ(RR̈ + 3/2Ṙ²) = Δp
        // Solving for R̈: R̈ = (Δp/ρR) - (3/2)(Ṙ²/R)
        let net_pressure = pressure_diff - surface_tension - viscous_stress;

        (net_pressure / (self.params.rho_liquid * r)) - (1.5 * v * v / r)
    }

    /// Calculate internal pressure using consistent thermodynamics
    /// This ensures both RP and KM models use the same gas physics
    fn calculate_internal_pressure(&self, state: &BubbleState) -> f64 {
        if !self.params.use_thermal_effects {
            // Polytropic relation for all states (including equilibrium)
            // The formula naturally handles equilibrium when radius = r0
            let gamma = state.gas_species.gamma();
            let p_eq = self.params.p0
                + self.params.surface_tension_pressure(self.params.r0)
                - self.params.pv;
            return p_eq * (self.params.r0 / state.radius).powf(3.0 * gamma) + self.params.pv;
        }

        // For thermal effects, use the same Van der Waals equation as KellerMiksisModel
        let n_total = state.n_gas + state.n_vapor;
        let volume = state.volume();

        // Get effective Van der Waals constants from gas composition
        let (a_mix, b_mix) = self.params.effective_vdw_constants();

        // Convert units to SI
        let a = a_mix * BAR_L2_TO_PA_M6;
        let b = b_mix * L_TO_M3;

        // Van der Waals equation
        let n_moles = n_total / AVOGADRO;

        n_moles * R_GAS * state.temperature / n_moles.mul_add(-b, volume)
            - a * n_moles * n_moles / (volume * volume)
    }
}
