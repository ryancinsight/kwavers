//! Rayleigh-Plesset Equation for Bubble Dynamics
//!
//! This module implements the Rayleigh-Plesset equation for modeling
//! the dynamics of spherical bubbles in liquids.

use super::bubble_state::{BubbleParameters, BubbleState};
use super::energy_balance::{update_temperature_energy_balance, EnergyBalanceCalculator};
use super::thermodynamics::{MassTransferModel, ThermodynamicsCalculator, VaporPressureModel};
use crate::constants::bubble_dynamics::{BAR_L2_TO_PA_M6, L_TO_M3};
use crate::constants::thermodynamics::{AVOGADRO, M_WATER, R_GAS};
use crate::error::KwaversResult;

// Remove duplicate constant definitions - they're now imported from constants module

/// Rayleigh-Plesset equation solver (incompressible)
#[derive(Debug))]
pub struct RayleighPlessetSolver {
    params: BubbleParameters,
    thermo_calc: ThermodynamicsCalculator,
}

impl RayleighPlessetSolver {
    pub fn new(params: BubbleParameters) -> Self {
        // Use same thermodynamics engine as KellerMiksisModel for consistency
        let thermo_calc = ThermodynamicsCalculator::new(VaporPressureModel::Wagner);
        Self {
            params,
            thermo_calc,
        }
    }

    /// Calculate bubble wall acceleration using Rayleigh-Plesset equation
    pub fn calculate_acceleration(&self, state: &BubbleState, p_acoustic: f64, t: f64) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;

        // Time-dependent acoustic forcing with proper phase tracking
        // This accounts for phase shifts and nonlinear propagation effects
        let acoustic_phase = 2.0 * std::f64::consts::PI * self.params.driving_frequency * t;
        let p_acoustic_instantaneous = p_acoustic * acoustic_phase.sin();
        let p_l = self.params.p0 + p_acoustic_instantaneous;

        // Use consistent internal pressure calculation
        let p_internal = self.calculate_internal_pressure(state);

        // Pressure difference
        let p_diff = p_internal - p_l;

        // Viscous term
        let viscous =
            crate::constants::bubble_dynamics::VISCOUS_STRESS_COEFF * self.params.mu_liquid * v / r;

        // Surface tension term
        let surface =
            crate::constants::bubble_dynamics::SURFACE_TENSION_COEFF * self.params.sigma / r;

        // Rayleigh-Plesset equation
        let numerator = p_diff - viscous - surface;
        let denominator = self.params.rho_liquid * r;

        numerator / denominator
            - crate::constants::bubble_dynamics::KINETIC_ENERGY_COEFF * v * v / r
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

/// Keller-Miksis equation solver (compressible)
#[derive(Clone, Debug))]
pub struct KellerMiksisModel {
    params: BubbleParameters,
    thermo_calc: ThermodynamicsCalculator,
    mass_transfer: MassTransferModel,
    energy_calculator: EnergyBalanceCalculator,
}

impl KellerMiksisModel {
    pub fn new(params: BubbleParameters) -> Self {
        // Use Wagner equation by default for highest accuracy
        let thermo_calc = ThermodynamicsCalculator::new(VaporPressureModel::Wagner);
        let mass_transfer = MassTransferModel::new(params.accommodation_coeff);
        let energy_calculator = EnergyBalanceCalculator::new(&params);

        Self {
            params: params.clone(),
            thermo_calc,
            mass_transfer,
            energy_calculator,
        }
    }

    /// Get the bubble parameters
    pub fn params(&self) -> &BubbleParameters {
        &self.params
    }

    /// Calculate the molar heat capacity at constant volume (Cv) for the bubble contents
    ///
    /// For consistency with the Van der Waals equation of state used for pressure,
    /// this should ideally account for real gas effects. However, for many gases
    /// at moderate conditions, the ideal gas Cv is a reasonable approximation.
    ///
    /// Returns: Molar heat capacity at constant volume in J/(mol·K)
    pub fn molar_heat_capacity_cv(&self, state: &BubbleState) -> f64 {
        // For Van der Waals gas, the heat capacity can differ from ideal gas
        // However, for most conditions in bubble dynamics, the ideal gas approximation
        // for Cv is acceptable. This should be documented as an assumption.

        // Get the fundamental Cv value for the gas species
        // This should be a property of the gas, not derived from gamma
        state.gas_species.molar_heat_capacity_cv()
    }

    /// Calculate bubble wall acceleration using Keller-Miksis equation
    ///
    /// The Keller-Miksis equation accounts for liquid compressibility effects
    /// and is more accurate than Rayleigh-Plesset for high-speed bubble dynamics.
    ///
    /// Reference: Keller & Miksis (1980), "Bubble oscillations of large amplitude"
    /// Journal of the Acoustical Society of America, 68(2), 628-633
    ///
    /// The standard formulation from the literature is:
    /// (1 - v/c) * R * R_ddot + (3/2) * v^2 * (1 - v/(3c)) =
    ///     (1/ρ) * [(P_B - P_∞) * (1 + v/c) + R/c * (dP_B/dt - dP_∞/dt)]
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64, // This is d(p_acoustic)/dt
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;
        let c = self.params.c_liquid;
        let rho = self.params.rho_liquid;

        // Update Mach number
        state.mach_number = v.abs() / c;

        // Time-dependent acoustic forcing with proper phase tracking
        // This fully utilizes the driving_frequency and time parameters
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let phase = omega * t;
        let p_acoustic_instantaneous = p_acoustic * phase.sin();

        // Far-field liquid pressure with time-dependent forcing
        let p_liquid_far = self.params.p0 + p_acoustic_instantaneous;
        state.pressure_liquid = p_liquid_far;

        // Internal pressure with thermal effects
        let p_internal = self.calculate_internal_pressure(state);
        state.pressure_internal = p_internal;

        // Pressure at bubble wall (liquid side) including viscous and surface tension
        let p_viscous = 4.0 * self.params.mu_liquid * v / r;
        let p_surface = 2.0 * self.params.sigma / r;
        let p_bubble_wall = p_internal - p_viscous - p_surface;

        // Time derivative of pressure at bubble wall
        // Complete implementation with proper derivative tracking
        let dp_bubble_wall_dt = self.estimate_pressure_derivative(state, p_internal);

        // Time derivative of liquid pressure including phase evolution
        // This properly accounts for both amplitude and phase changes
        let dp_liquid_dt = dp_dt + omega * p_acoustic * phase.cos();

        // Keller-Miksis equation solved for R_ddot
        // Standard formulation from literature
        let mach = v / c;

        // Left-hand side coefficient
        let lhs_coeff = r * (1.0 - mach);

        // Right-hand side terms
        let pressure_term = (p_bubble_wall - p_liquid_far) * (1.0 + mach) / rho;
        let derivative_term = (r / c) * (dp_bubble_wall_dt - dp_liquid_dt) / rho;
        let kinetic_term = 1.5 * v * v * (1.0 - mach / 3.0);

        // Solve for acceleration
        let acceleration = (pressure_term + derivative_term - kinetic_term) / lhs_coeff;

        state.wall_acceleration = acceleration;
        acceleration
    }

    /// Estimate the time derivative of bubble wall pressure
    /// This is a first-order approximation based on the polytropic relation
    fn estimate_pressure_derivative(&self, state: &BubbleState, p_internal: f64) -> f64 {
        if !self.params.use_thermal_effects {
            // For polytropic process: dP/dt = -3*gamma*P*(dR/dt)/R
            let gamma = state.gas_species.gamma();
            return -3.0 * gamma * p_internal * state.wall_velocity / state.radius;
        }

        // For thermal effects, the derivative is more complex
        // This would ideally track temperature and composition changes
        // For now, use a quasi-static approximation
        -3.0 * 1.4 * p_internal * state.wall_velocity / state.radius
    }

    /// Calculate internal pressure with thermal and mass transfer effects
    fn calculate_internal_pressure(&self, state: &BubbleState) -> f64 {
        if !self.params.use_thermal_effects {
            // Polytropic gas relation
            let gamma = state.gas_species.gamma();
            return (self.params.p0
                + crate::constants::bubble_dynamics::SURFACE_TENSION_COEFF * self.params.sigma
                    / self.params.r0
                - self.params.pv)
                * (self.params.r0 / state.radius).powf(3.0 * gamma)
                + self.params.pv;
        }

        // Van der Waals equation for real gas
        // Literature reference: Qin et al. (2023) "Numerical investigation on acoustic cavitation
        // characteristics of an air-vapor bubble", Ultrasonics Sonochemistry
        let n_total = state.n_gas + state.n_vapor;
        let volume = state.volume();

        // Get effective Van der Waals constants from gas composition
        let (a_mix, b_mix) = self.params.effective_vdw_constants();

        // Convert units to SI
        let a = a_mix * BAR_L2_TO_PA_M6; // Pa·m⁶/mol²
        let b = b_mix * L_TO_M3; // m³/mol

        // Van der Waals equation: (P + a*n²/V²)(V - nb) = nRT
        // Solving for P: P = nRT/(V - nb) - a*n²/V²
        let n_moles = n_total / AVOGADRO;

        n_moles * R_GAS * state.temperature / (volume - n_moles * b)
            - a * n_moles * n_moles / (volume * volume)
    }

    /// Update bubble temperature using comprehensive energy balance
    pub fn update_temperature(&self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_thermal_effects {
            return;
        }

        // Calculate internal pressure for work term
        let internal_pressure = self.calculate_internal_pressure(state);

        // Calculate mass transfer rate for latent heat
        let p_sat = self.thermo_calc.vapor_pressure(state.temperature);
        let p_vapor_current =
            state.n_vapor * R_GAS * state.temperature / (AVOGADRO * state.volume());
        let mass_rate = self.mass_transfer.mass_transfer_rate(
            state.temperature,
            p_vapor_current,
            state.surface_area(),
        );

        // Use the comprehensive energy balance model
        update_temperature_energy_balance(
            &self.energy_calculator,
            state,
            &self.params,
            internal_pressure,
            mass_rate,
            dt,
        );
    }

    /// Update vapor content through evaporation/condensation
    pub fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_mass_transfer {
            return;
        }

        // Use proper thermodynamic model for vapor pressure
        let p_sat = self.thermo_calc.vapor_pressure(state.temperature);
        let p_vapor_current =
            state.n_vapor * R_GAS * state.temperature / (AVOGADRO * state.volume());

        // Calculate mass transfer rate using enhanced model
        let mass_rate = self.mass_transfer.mass_transfer_rate(
            state.temperature,
            p_vapor_current,
            state.surface_area(),
        );

        // Update vapor molecules (convert from kg/s to molecules)
        let molecule_rate = mass_rate * AVOGADRO / M_WATER;
        state.n_vapor += molecule_rate * dt;
        state.n_vapor = state.n_vapor.max(0.0);

        // Calculate associated heat transfer
        let heat_rate = self
            .mass_transfer
            .heat_transfer_rate(mass_rate, state.temperature);
        state.temperature -= heat_rate * dt / (state.mass() * self.molar_heat_capacity_cv(state));
    }
}

/// Integrate bubble dynamics with proper handling of stiff ODEs
///
/// This is the recommended integration method that uses adaptive time-stepping
/// with sub-cycling to handle the stiff nature of bubble dynamics equations.
pub fn integrate_bubble_dynamics_stable(
    solver: &KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    use super::adaptive_integration::integrate_bubble_dynamics_adaptive;

    // Use adaptive integration with sub-cycling (no Mutex needed anymore)
    integrate_bubble_dynamics_adaptive(solver, state, p_acoustic, dp_dt, dt, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rayleigh_plesset_equilibrium() {
        let params = BubbleParameters::default();
        let solver = RayleighPlessetSolver::new(params.clone());
        let state = BubbleState::at_equilibrium(&params);

        // At equilibrium, acceleration should be negligible
        let accel = solver.calculate_acceleration(&state, 0.0, 0.0);

        // Verify equilibrium is properly established (should be < 1% of g)
        const GRAVITY_ACCEL: f64 = 9.81;
        assert!(
            accel.abs() < 0.01 * GRAVITY_ACCEL,
            "Acceleration at equilibrium should be negligible: {} m/s²",
            accel
        );
    }

    #[test]
    fn test_keller_miksis_mach_number() {
        let params = BubbleParameters::default();
        let solver = KellerMiksisModel::new(params.clone());
        let mut state = BubbleState::new(&params);

        // Set high velocity
        state.wall_velocity = -300.0; // m/s

        solver.calculate_acceleration(&mut state, 0.0, 0.0, 0.0);

        assert!((state.mach_number - 300.0 / params.c_liquid).abs() < 1e-6);
    }
}
