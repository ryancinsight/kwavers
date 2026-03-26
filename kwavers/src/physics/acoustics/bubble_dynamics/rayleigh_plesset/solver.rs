//! Rayleigh-Plesset equation solver for bubble dynamics

use super::super::bubble_state::{BubbleParameters, BubbleState};
use super::super::thermodynamics::{ThermodynamicsCalculator, VaporPressureModel};
use crate::core::constants::cavitation::{BAR_L2_TO_PA_M6, L_TO_M3};
use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};

/// Rayleigh-Plesset equation solver (incompressible)
#[derive(Debug)]
pub struct RayleighPlessetSolver {
    params: BubbleParameters,
    #[allow(dead_code)] // Thermodynamics calculator for bubble modeling
    thermo_calc: ThermodynamicsCalculator,
}

impl RayleighPlessetSolver {
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        // Use same thermodynamics engine as KellerMiksisModel for consistency
        let thermo_calc = ThermodynamicsCalculator::new(VaporPressureModel::Wagner);
        Self {
            params,
            thermo_calc,
        }
    }

    /// Calculate bubble wall acceleration using Rayleigh-Plesset equation
    /// Standard form: ρ(RR̈ + 3/2Ṙ²) = pg - p∞ - 2σ/R - 4μṘ/R
    #[must_use]
    pub fn calculate_acceleration(&self, state: &BubbleState, p_acoustic: f64, t: f64) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;

        // Time-dependent acoustic forcing with proper phase tracking
        let acoustic_phase = 2.0 * std::f64::consts::PI * self.params.driving_frequency * t;
        let p_acoustic_instantaneous = p_acoustic * acoustic_phase.sin();
        let p_liquid_far = self.params.p0 + p_acoustic_instantaneous;

        // Direct physics-based pressure calculation (eliminate approximations)
        // Reference: Rayleigh-Plesset equation, Brennen (1995), "Cavitation and Bubble Dynamics"
        let p_gas = if !self.params.use_thermal_effects {
            // For isothermal bubble dynamics, use polytropic relation: p * V^γ = constant
            // At equilibrium: p_eq * (4/3 * π * r0³)^γ = p * (4/3 * π * r³)^γ
            // Therefore: p = p_eq * (r0/r)^(3γ)

            let gamma = state.gas_species.gamma();

            // The equilibrium pressure is determined by force balance:
            // p_internal = p_external + 2σ/r0 (Young-Laplace equation)
            let p_internal_equilibrium = self.params.p0 + 2.0 * self.params.sigma / self.params.r0;

            // Apply polytropic scaling for current radius
            let radius_ratio = self.params.r0 / r;
            p_internal_equilibrium * radius_ratio.powf(3.0 * gamma)
        } else {
            // Van der Waals equation for thermal effects (literature-validated)
            // Reference: Qin et al. (2023) "Numerical investigation on acoustic cavitation characteristics"
            self.calculate_internal_pressure(state)
        };

        // Debug output for equilibrium testing
        #[cfg(test)]
        if r == self.params.r0 && v == 0.0 && p_acoustic == 0.0 && t == 0.0 {
            println!("Debug calculate_acceleration at equilibrium:");
            println!("  p_gas: {} Pa", p_gas);
            println!("  p_liquid_far: {} Pa", p_liquid_far);
            println!(
                "  Expected p_gas at eq: {} Pa",
                self.params.p0 + 2.0 * self.params.sigma / self.params.r0
            );
        }

        // Forces on bubble wall (Pa)
        let pressure_diff = p_gas - p_liquid_far;
        let surface_tension = 2.0 * self.params.sigma / r;
        let viscous_stress = 4.0 * self.params.mu_liquid * v / r;

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
            let p_eq = self.params.p0 + 2.0 * self.params.sigma / self.params.r0 - self.params.pv;
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

        n_moles * R_GAS * state.temperature / (volume - n_moles * b)
            - a * n_moles * n_moles / (volume * volume)
    }
}
