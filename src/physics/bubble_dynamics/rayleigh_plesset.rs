//! Rayleigh-Plesset and Keller-Miksis equation solvers
//!
//! Core bubble dynamics equations

use super::bubble_state::{BubbleState, BubbleParameters};
use std::f64::consts::PI;

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
        let r_gas = 8.314; // J/(mol·K)
        let n_total = state.n_gas + state.n_vapor;
        let volume = state.volume();
        
        // Van der Waals constants (simplified)
        let b = 3.0e-5 * n_total / 6.022e23; // Excluded volume
        
        let p_gas = n_total * r_gas * state.temperature / (6.022e23 * (volume - b));
        
        p_gas
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
        let mass = state.total_molecules() * state.gas_species.molecular_weight() / 6.022e23;
        let cv = 8.314 / state.gas_species.molecular_weight() / (gamma - 1.0);
        
        let dt_transfer = -h * area * (state.temperature - 293.15) / (mass * cv) * dt;
        
        // Update temperature
        state.temperature += dt_compression + dt_transfer;
        state.temperature = state.temperature.max(293.15); // Don't go below ambient
        
        // Track maximum
        state.update_max_temperature();
    }
    
    /// Update vapor content through evaporation/condensation
    pub fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) {
        if !self.params.use_mass_transfer {
            return;
        }
        
        // Simplified mass transfer model
        let p_sat = self.params.pv * (state.temperature / 293.15).powf(2.0); // Rough approximation
        let p_vapor_current = state.n_vapor * 8.314 * state.temperature / 
            (6.022e23 * state.volume());
        
        // Mass transfer rate
        let accommodation = self.params.accommodation_coeff;
        let m_water = 0.018; // kg/mol
        let rate = accommodation * state.surface_area() * 
            (p_sat - p_vapor_current) / (2.0 * PI * m_water * 8.314 * state.temperature).sqrt();
        
        // Update vapor molecules
        state.n_vapor += rate * 6.022e23 * dt;
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