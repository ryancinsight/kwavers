//! Rayleigh-Plesset Equation for Bubble Dynamics
//!
//! This module implements the Rayleigh-Plesset equation for modeling
//! the dynamics of spherical bubbles in liquids.

use super::bubble_state::{BubbleState, BubbleParameters};
use super::thermodynamics::{MassTransferModel, ThermodynamicsCalculator, VaporPressureModel};
use crate::error::KwaversResult;
use crate::constants::bubble_dynamics::{
    PECLET_SCALING_FACTOR, MIN_PECLET_NUMBER, 
    NUSSELT_BASE, NUSSELT_PECLET_COEFF, NUSSELT_PECLET_EXPONENT,
    N2_FRACTION, O2_FRACTION, VDW_A_N2, VDW_A_O2, VDW_B_N2, VDW_B_O2,
    BAR_L2_TO_PA_M6, L_TO_M3
};
use crate::constants::thermodynamics::{R_GAS, AVOGADRO, M_WATER, T_AMBIENT};
use std::sync::Mutex;

// Remove duplicate constant definitions - they're now imported from constants module

/// Rayleigh-Plesset equation solver (incompressible)
pub struct RayleighPlessetSolver {
    params: BubbleParameters,
}

impl RayleighPlessetSolver {
    pub fn new(params: BubbleParameters) -> Self {
        Self { params }
    }
    
    /// Calculate bubble wall acceleration using Rayleigh-Plesset equation
    pub fn calculate_acceleration(
        &self,
        state: &BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;
        let p_l = self.params.p0 + p_acoustic;
        
        // Internal pressure (polytropic relation)
        let gamma = state.gas_species.gamma();
        let p_gas = (self.params.p0 + 2.0 * self.params.sigma / self.params.r0 - self.params.pv)
            * (self.params.r0 / r).powf(3.0 * gamma);
        
        // Pressure difference
        let p_diff = p_gas + self.params.pv - p_l;
        
        // Viscous term
        let viscous = 4.0 * self.params.mu_liquid * v / r;
        
        // Surface tension term
        let surface = 2.0 * self.params.sigma / r;
        
        // Rayleigh-Plesset equation
        let numerator = p_diff - viscous - surface;
        let denominator = self.params.rho_liquid * r;
        
        numerator / denominator - 1.5 * v * v / r
    }
}

/// Keller-Miksis equation solver (compressible)
#[derive(Clone)]
pub struct KellerMiksisModel {
    params: BubbleParameters,
    thermo_calc: ThermodynamicsCalculator,
    mass_transfer: MassTransferModel,
}

impl KellerMiksisModel {
    pub fn new(params: BubbleParameters) -> Self {
        // Use Wagner equation by default for highest accuracy
        let thermo_calc = ThermodynamicsCalculator::new(VaporPressureModel::Wagner);
        let mass_transfer = MassTransferModel::new(params.accommodation_coeff);
        
        Self { 
            params,
            thermo_calc,
            mass_transfer,
        }
    }
    
    /// Get the bubble parameters
    pub fn params(&self) -> &BubbleParameters {
        &self.params
    }
    
    /// Calculate heat capacity for the bubble contents
    pub fn heat_capacity(&self, state: &BubbleState) -> f64 {
        // Heat capacity depends on gas species and conditions
        // Using constant pressure heat capacity for ideal gas
        const R_GAS: f64 = 8.314; // J/(mol·K)
        let gamma = state.gas_species.gamma();
        let cv = R_GAS / (gamma - 1.0); // Molar heat capacity at constant volume
        let cp = gamma * cv; // Molar heat capacity at constant pressure
        
        // For bubble dynamics, use cv as the bubble volume changes
        cv
    }
    
    /// Calculate bubble wall acceleration using Keller-Miksis equation
    /// 
    /// The Keller-Miksis equation accounts for liquid compressibility effects
    /// and is more accurate than Rayleigh-Plesset for high-speed bubble dynamics.
    /// 
    /// Reference: Keller & Miksis (1980), "Bubble oscillations of large amplitude"
    /// Journal of the Acoustical Society of America, 68(2), 628-633
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let v = state.wall_velocity;
        let c = self.params.c_liquid;
        
        // Update Mach number
        state.mach_number = v.abs() / c;
        
        // Liquid pressure at bubble wall
        let p_l = self.params.p0 + p_acoustic;
        state.pressure_liquid = p_l;
        
        // Internal pressure with thermal effects
        let p_internal = self.calculate_internal_pressure(state);
        state.pressure_internal = p_internal;
        
        // Pressure difference across interface
        let p_diff = p_internal - p_l;
        
        // Viscous stress term
        let viscous = 4.0 * self.params.mu_liquid * v / r;
        
        // Surface tension term
        let surface = 2.0 * self.params.sigma / r;
        
        // Liquid compressibility factor (1 - Mach number)
        let comp_factor = 1.0 - v / c;
        
        // Keller-Miksis equation (corrected formulation)
        // The numerator includes pressure terms and their time derivatives
        let numerator = (p_diff - viscous - surface) * comp_factor / self.params.rho_liquid
            + r * (p_internal - p_l) / (self.params.rho_liquid * c)
            + r * dp_dt / (self.params.rho_liquid * c);
        
        // The denominator accounts for compressibility and kinetic effects
        // Corrected according to Keller & Miksis (1980)
        let denominator = r * comp_factor + 4.0 * self.params.mu_liquid / (self.params.rho_liquid * c);
        
        // The acceleration term includes nonlinear velocity effects
        let acceleration = numerator / denominator - 1.5 * v * v * comp_factor / r;
        
        state.wall_acceleration = acceleration;
        acceleration
    }
    
    /// Calculate internal pressure with thermal and mass transfer effects
    fn calculate_internal_pressure(&self, state: &BubbleState) -> f64 {
        if !self.params.use_thermal_effects {
            // Simple polytropic relation
            let gamma = state.gas_species.gamma();
            return (self.params.p0 + 2.0 * self.params.sigma / self.params.r0 - self.params.pv)
                * (self.params.r0 / state.radius).powf(3.0 * gamma) + self.params.pv;
        }
        
        // Van der Waals equation for real gas
        // Literature reference: Qin et al. (2023) "Numerical investigation on acoustic cavitation 
        // characteristics of an air-vapor bubble", Ultrasonics Sonochemistry
        let n_total = state.n_gas + state.n_vapor;
        let volume = state.volume();
        
        // Van der Waals constants for air (weighted average of N2 and O2)
        let a_air = N2_FRACTION * VDW_A_N2 + O2_FRACTION * VDW_A_O2; // bar·L²/mol²
        let b_air = N2_FRACTION * VDW_B_N2 + O2_FRACTION * VDW_B_O2; // L/mol
        
        // Convert units to SI
        let a = a_air * BAR_L2_TO_PA_M6; // Pa·m⁶/mol²
        let b = b_air * L_TO_M3; // m³/mol
        
        // Van der Waals equation: (P + a*n²/V²)(V - nb) = nRT
        // Solving for P: P = nRT/(V - nb) - a*n²/V²
        let n_moles = n_total / AVOGADRO;
        let pressure = n_moles * R_GAS * state.temperature / (volume - n_moles * b) 
                     - a * n_moles * n_moles / (volume * volume);
        
        pressure
    }
    
    /// Update bubble temperature
    pub fn update_temperature(&self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_thermal_effects {
            return;
        }
        
        let r = state.radius;
        let v = state.wall_velocity;
        
        // Peclet number
        let thermal_diffusivity = self.params.thermal_conductivity / 
            (self.params.rho_liquid * self.params.specific_heat_liquid);
        let peclet = r * v.abs() / thermal_diffusivity;
        
        // Effective polytropic index
        // Based on Prosperetti & Lezzi (1986) heat transfer model
        let gamma = state.gas_species.gamma();
        let gamma_eff = 1.0 + (gamma - 1.0) / (1.0 + PECLET_SCALING_FACTOR / peclet.max(MIN_PECLET_NUMBER));
        
        // Temperature change from compression
        let compression_rate = -v / r;
        let dt_compression = state.temperature * (gamma_eff - 1.0) * compression_rate * dt;
        
        // Heat transfer to liquid using Nusselt correlation
        let nusselt = NUSSELT_BASE + NUSSELT_PECLET_COEFF * peclet.powf(NUSSELT_PECLET_EXPONENT);
        let h = nusselt * self.params.thermal_conductivity / r;
        let area = state.surface_area();
        let mass = state.total_molecules() * state.gas_species.molecular_weight() / AVOGADRO;
        let cv = R_GAS / state.gas_species.molecular_weight() / (gamma - 1.0);
        
        let dt_transfer = -h * area * (state.temperature - T_AMBIENT) / (mass * cv) * dt;
        
        // Update temperature
        state.temperature += dt_compression + dt_transfer;
        state.temperature = state.temperature.max(T_AMBIENT); // Don't go below ambient
        
        // Track maximum
        state.update_max_temperature();
    }
    
    /// Update vapor content through evaporation/condensation
    pub fn update_mass_transfer(&mut self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_mass_transfer {
            return;
        }
        
        // Use proper thermodynamic model for vapor pressure
        let p_sat = self.thermo_calc.vapor_pressure(state.temperature);
        let p_vapor_current = state.n_vapor * R_GAS * state.temperature / 
            (AVOGADRO * state.volume());
        
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
        let heat_rate = self.mass_transfer.heat_transfer_rate(mass_rate, state.temperature);
        state.temperature -= heat_rate * dt / (state.mass() * self.heat_capacity(state));
    }
}

/// Integrate bubble dynamics for one time step
/// 
/// **DEPRECATED**: Use `integrate_bubble_dynamics_adaptive` for better stability
#[deprecated(since = "1.6.0", note = "Use integrate_bubble_dynamics_adaptive for stiff ODE handling")]
pub fn integrate_bubble_dynamics(
    solver: &mut KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) {
    let r0 = solver.params().r0;
    
    // Calculate acceleration
    let acceleration = solver.calculate_acceleration(state, p_acoustic, dp_dt, t);
    
    // Update velocity and position (Velocity Verlet)
    state.wall_velocity += acceleration * dt;
    state.radius += state.wall_velocity * dt + 0.5 * acceleration * dt * dt;
    
    // WARNING: Removed clamping - use adaptive integration for stability
    // The following lines were removed as they violate conservation laws:
    // state.radius = state.radius.max(MIN_RADIUS).min(MAX_RADIUS);
    // state.wall_velocity = state.wall_velocity.max(-1000.0).min(1000.0);
    
    // Update derived quantities
    state.update_compression(r0);
    state.update_collapse_state();
    
    // Update temperature and mass transfer
    solver.update_temperature(state, dt);
    solver.update_mass_transfer(state, dt);
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
    use std::sync::Arc;
    
    // Use adaptive integration with sub-cycling
    integrate_bubble_dynamics_adaptive(
        Arc::new(Mutex::new(solver.clone())),
        state,
        p_acoustic,
        dp_dt,
        dt,
        t,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rayleigh_plesset_equilibrium() {
        let params = BubbleParameters::default();
        let solver = RayleighPlessetSolver::new(params.clone());
        let state = BubbleState::new(&params);
        
        // At equilibrium with no acoustic pressure, acceleration should be zero
        let accel = solver.calculate_acceleration(&state, 0.0, 0.0);
        assert!(accel.abs() < 1e-10);
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