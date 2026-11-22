//! Rayleigh-Plesset equation solver for bubble dynamics

use super::super::acoustic_pressure::InstantaneousPressure;
use super::super::bubble_state::{BubbleParameters, BubbleState};
use kwavers_core::constants::cavitation::{BAR_L2_TO_PA_M6, L_TO_M3};
use kwavers_core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};

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
    ///
    /// # Acoustic-drive contract
    ///
    /// `p_acoustic` is the INSTANTANEOUS acoustic pressure `p_∞(t) − p0`
    /// [Pa] at time `t`. The Rayleigh–Plesset equation is written in
    /// terms of a general far-field pressure `p_∞(t)` (Brennen 1995,
    /// "Cavitation and Bubble Dynamics"); the sinusoidal form
    /// `p_∞(t) = p0 ± P_A·sin(ωt + φ)` is the standard specialisation
    /// for continuous harmonic drive, not part of the equation. Callers
    /// that have an amplitude synthesise the tone themselves. An earlier
    /// version of this function treated `p_acoustic` as an amplitude and
    /// multiplied by `sin(ωt)` internally, which double-modulated every
    /// caller that already passed an instantaneous pressure. The `t`
    /// argument is retained for interface compatibility.
    #[must_use]
    pub fn calculate_acceleration(
        &self,
        state: &BubbleState,
        p_acoustic: InstantaneousPressure,
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;

        // Formula boundary: the drive arrives typed; the equation below
        // works in pascals.
        let p_acoustic = p_acoustic.as_pascals();

        // Far-field pressure p_∞(t) = p0 + p_acoustic(t); see the contract
        // above. The `t` argument is unused by the RHS.
        let _ = t;
        let p_liquid_far = self.params.p0 + p_acoustic;

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

        // Forces on bubble wall (Pa)
        let pressure_diff = p_gas - p_liquid_far;
        let surface_tension = self.params.surface_tension_pressure(r);
        let viscous_stress = self.params.viscous_wall_stress(v, r);

        // Rayleigh-Plesset equation: ρ(RR̈ + 3/2Ṙ²) = Δp
        // Solving for R̈: R̈ = (Δp/ρR) - (3/2)(Ṙ²/R)
        let net_pressure = pressure_diff - surface_tension - viscous_stress;

        super::wall_acceleration(r, v, self.params.rho_liquid, net_pressure)
    }

    /// Calculate internal pressure using consistent thermodynamics
    /// This ensures both RP and KM models use the same gas physics
    fn calculate_internal_pressure(&self, state: &BubbleState) -> f64 {
        if !self.params.use_thermal_effects {
            // Polytropic relation for all states (including equilibrium)
            // The formula naturally handles equilibrium when radius = r0
            let gamma = state.gas_species.gamma();
            let p_eq = self.params.p0 + self.params.surface_tension_pressure(self.params.r0)
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
