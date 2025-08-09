//! Rayleigh-Plesset equation solver for bubble dynamics
//! 
//! This module implements the Rayleigh-Plesset equation for modeling
//! the dynamics of spherical bubbles in liquids.

use super::bubble_state::{BubbleState, BubbleParameters};
use std::f64::consts::PI;


// Physical constants for bubble dynamics
const MIN_RADIUS: f64 = 1e-9;  // Minimum bubble radius (1 nm)
const MAX_RADIUS: f64 = 1e-2;  // Maximum bubble radius (1 cm)

// Air composition constants
const N2_FRACTION: f64 = 0.79;  // Nitrogen fraction in air
const O2_FRACTION: f64 = 0.21;  // Oxygen fraction in air

// Unit conversion constants
const BAR_L2_TO_PA_M6: f64 = 0.1;   // Convert bar·L²/mol² to Pa·m⁶/mol²
const L_TO_M3: f64 = 1e-3;          // Convert L/mol to m³/mol

// Physical constants
const AVOGADRO: f64 = 6.022e23;     // Avogadro's number (molecules/mol)
const R_GAS: f64 = 8.314;           // Universal gas constant (J/(mol·K))

// Van der Waals constants for gases (from NIST Chemistry WebBook)
const VDW_A_N2: f64 = 1.370;        // bar·L²/mol² for N2
const VDW_B_N2: f64 = 0.0387;       // L/mol for N2
const VDW_A_O2: f64 = 1.382;        // bar·L²/mol² for O2
const VDW_B_O2: f64 = 0.0319;       // L/mol for O2

// Molecular properties
const M_WATER: f64 = 0.018;         // Molecular weight of water (kg/mol)

// Reference conditions
const T_AMBIENT: f64 = 293.15;       // Ambient temperature (K)

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
pub struct KellerMiksisModel {
    params: BubbleParameters,
}

impl KellerMiksisModel {
    pub fn new(params: BubbleParameters) -> Self {
        Self { params }
    }
    
    /// Calculate bubble wall acceleration using Keller-Miksis equation
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
        
        // Pressure difference
        let p_diff = p_internal - p_l;
        
        // Viscous term
        let viscous = 4.0 * self.params.mu_liquid * v / r;
        
        // Surface tension term
        let surface = 2.0 * self.params.sigma / r;
        
        // Compressibility factor
        let comp_factor = 1.0 - v / c;
        
        // Keller-Miksis equation
        let numerator = (p_diff - viscous - surface) / self.params.rho_liquid
            + r * dp_dt / (self.params.rho_liquid * c);
        
        let denominator = r * comp_factor + r * r * v / c;
        
        let acceleration = numerator / denominator - 1.5 * v * v * (1.0 - v / (3.0 * c)) / r;
        
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
        let gamma = state.gas_species.gamma();
        let gamma_eff = 1.0 + (gamma - 1.0) / (1.0 + 10.0 / peclet.max(0.1));
        
        // Temperature change from compression
        let compression_rate = -v / r;
        let dt_compression = state.temperature * (gamma_eff - 1.0) * compression_rate * dt;
        
        // Heat transfer to liquid
        let nusselt = 2.0 + 0.6 * peclet.powf(0.5); // Simplified Nusselt number
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
    pub fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_mass_transfer {
            return;
        }
        
        // Simplified mass transfer model
        let p_sat = self.params.pv * (state.temperature / T_AMBIENT).powf(2.0); // Rough approximation
        let p_vapor_current = state.n_vapor * R_GAS * state.temperature / 
            (AVOGADRO * state.volume());
        
        // Mass transfer rate
        let accommodation = self.params.accommodation_coeff;
        let rate = accommodation * state.surface_area() * 
            (p_sat - p_vapor_current) / (2.0 * PI * M_WATER * R_GAS * state.temperature).sqrt();
        
        // Update vapor molecules
        state.n_vapor += rate * AVOGADRO * dt;
        state.n_vapor = state.n_vapor.max(0.0);
    }
}

/// Integrate bubble dynamics for one time step
pub fn integrate_bubble_dynamics(
    solver: &KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) {
    // Store previous state
    let r0 = solver.params.r0;
    
    // Calculate acceleration
    let acceleration = solver.calculate_acceleration(state, p_acoustic, dp_dt, t);
    
    // Update velocity and position (Velocity Verlet)
    state.wall_velocity += acceleration * dt;
    state.radius += state.wall_velocity * dt + 0.5 * acceleration * dt * dt;
    
    // Enforce physical limits
    state.radius = state.radius.max(MIN_RADIUS).min(MAX_RADIUS);
    state.wall_velocity = state.wall_velocity.max(-1000.0).min(1000.0);
    
    // Update derived quantities
    state.update_compression(r0);
    state.update_collapse_state();
    
    // Update temperature and mass transfer
    solver.update_temperature(state, dt);
    solver.update_mass_transfer(state, dt);
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