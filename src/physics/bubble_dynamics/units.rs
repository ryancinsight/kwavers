//! Unit-Safe Bubble Dynamics Types
//!
//! This module provides type-safe wrappers for bubble dynamics parameters
//! using the uom crate to prevent unit conversion errors at compile time.

use uom::si::f64::*;
use uom::si::pressure::pascal;
use uom::si::length::meter;
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::velocity::meter_per_second;
use uom::si::dynamic_viscosity::pascal_second;
use uom::si::thermal_conductivity::watt_per_meter_kelvin;
use uom::si::specific_heat_capacity::joule_per_kilogram_kelvin;
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::volume::cubic_meter;
use uom::si::area::square_meter;
use uom::si::mass::kilogram;

use std::collections::HashMap;
use super::bubble_state::GasType;

/// Unit-safe bubble parameters
#[derive(Clone, Debug)]
pub struct SafeBubbleParameters {
    /// Initial bubble radius
    pub r0: Length,
    /// Ambient pressure
    pub p0: Pressure,
    /// Liquid density
    pub rho_liquid: MassDensity,
    /// Sound speed in liquid
    pub c_liquid: Velocity,
    /// Dynamic viscosity of liquid
    pub mu_liquid: DynamicViscosity,
    /// Surface tension (N/m) - stored as f64 since uom doesn't have this unit
    pub sigma: f64,
    /// Vapor pressure
    pub pv: Pressure,
    
    // Thermal properties
    /// Thermal conductivity
    pub thermal_conductivity: ThermalConductivity,
    /// Specific heat capacity of liquid
    pub specific_heat_liquid: SpecificHeatCapacity,
    /// Accommodation coefficient (dimensionless)
    pub accommodation_coeff: f64,
    
    // Gas properties
    /// Initial gas pressure
    pub initial_gas_pressure: Pressure,
    /// Gas composition (dimensionless fractions)
    pub gas_composition: HashMap<GasType, f64>,
    
    // Control flags
    pub use_compressibility: bool,
    pub use_thermal_effects: bool,
    pub use_mass_transfer: bool,
}

impl Default for SafeBubbleParameters {
    fn default() -> Self {
        // Default air composition
        let mut gas_composition = HashMap::new();
        gas_composition.insert(GasType::N2, 0.79);
        gas_composition.insert(GasType::O2, 0.21);
        
        Self {
            // Water at 20°C with 5 μm air bubble
            r0: Length::new::<meter>(5e-6),
            p0: Pressure::new::<pascal>(101325.0),
            rho_liquid: MassDensity::new::<kilogram_per_cubic_meter>(998.0),
            c_liquid: Velocity::new::<meter_per_second>(1482.0),
            mu_liquid: DynamicViscosity::new::<pascal_second>(1.002e-3),
            sigma: 0.0728, // N/m
            pv: Pressure::new::<pascal>(2.33e3),
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(0.6),
            specific_heat_liquid: SpecificHeatCapacity::new::<joule_per_kilogram_kelvin>(4182.0),
            accommodation_coeff: 0.04,
            initial_gas_pressure: Pressure::new::<pascal>(101325.0),
            gas_composition,
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
}

/// Unit-safe bubble state
#[derive(Clone, Debug)]
pub struct SafeBubbleState {
    /// Current bubble radius
    pub radius: Length,
    /// Bubble wall velocity (dr/dt)
    pub wall_velocity: Velocity,
    /// Bubble wall acceleration (d²r/dt²)
    pub wall_acceleration: Acceleration,
    /// Gas temperature inside bubble
    pub temperature: ThermodynamicTemperature,
    /// Internal pressure
    pub pressure_internal: Pressure,
    /// Liquid pressure at bubble wall
    pub pressure_liquid: Pressure,
    /// Number of gas molecules (dimensionless)
    pub n_gas: f64,
    /// Number of vapor molecules (dimensionless)
    pub n_vapor: f64,
    /// Mach number (dimensionless)
    pub mach_number: f64,
    /// Compression ratio (dimensionless)
    pub compression_ratio: f64,
    /// Maximum temperature reached
    pub max_temperature: ThermodynamicTemperature,
}

impl SafeBubbleState {
    /// Create new bubble state at equilibrium
    pub fn new(params: &SafeBubbleParameters) -> Self {
        let gas_pressure = params.initial_gas_pressure + 
            Pressure::new::<pascal>(2.0 * params.sigma / params.r0.get::<meter>());
        
        // Estimate molecule count using ideal gas law
        let volume = Volume::new::<cubic_meter>(
            4.0 / 3.0 * std::f64::consts::PI * params.r0.get::<meter>().powi(3)
        );
        let temperature = ThermodynamicTemperature::new::<kelvin>(293.15);
        let n_gas = estimate_molecules(gas_pressure, volume, temperature);
        
        Self {
            radius: params.r0,
            wall_velocity: Velocity::new::<meter_per_second>(0.0),
            wall_acceleration: Acceleration::new::<meter_per_second_squared>(0.0),
            temperature,
            pressure_internal: gas_pressure,
            pressure_liquid: params.p0,
            n_gas,
            n_vapor: 0.0,
            mach_number: 0.0,
            compression_ratio: 1.0,
            max_temperature: temperature,
        }
    }
    
    /// Calculate bubble volume
    pub fn volume(&self) -> Volume {
        let r = self.radius.get::<meter>();
        Volume::new::<cubic_meter>(4.0 / 3.0 * std::f64::consts::PI * r.powi(3))
    }
    
    /// Calculate bubble surface area
    pub fn surface_area(&self) -> Area {
        let r = self.radius.get::<meter>();
        Area::new::<square_meter>(4.0 * std::f64::consts::PI * r.powi(2))
    }
    
    /// Calculate total mass of gas and vapor
    pub fn mass(&self) -> Mass {
        const AVOGADRO: f64 = 6.022e23;
        // Simplified - would need gas species info for accurate calculation
        let molecular_weight = 0.029; // kg/mol for air
        let water_molecular_weight = 0.018; // kg/mol for water
        
        Mass::new::<kilogram>(
            (self.n_gas * molecular_weight + self.n_vapor * water_molecular_weight) / AVOGADRO
        )
    }
}

/// Estimate number of molecules from ideal gas law (unit-safe)
fn estimate_molecules(pressure: Pressure, volume: Volume, temperature: ThermodynamicTemperature) -> f64 {
    
    const R_GAS: f64 = 8.314; // J/(mol·K)
    const AVOGADRO: f64 = 6.022e23;
    
    let p_pa = pressure.get::<pascal>();
    let v_m3 = volume.get::<cubic_meter>();
    let t_k = temperature.get::<kelvin>();
    
    let moles = p_pa * v_m3 / (R_GAS * t_k);
    moles * AVOGADRO
}

/// Calculate Rayleigh-Plesset acceleration (unit-safe)
pub fn calculate_rp_acceleration_safe(
    state: &SafeBubbleState,
    params: &SafeBubbleParameters,
    p_acoustic: Pressure,
) -> Acceleration {
    let r = state.radius;
    let v = state.wall_velocity;
    let p_l = params.p0 + p_acoustic;
    let p_internal = state.pressure_internal;
    
    // Pressure difference
    let p_diff = p_internal - p_l;
    
    // Viscous term: 4μv/r
    let viscous_pa = 4.0 * params.mu_liquid.get::<pascal_second>() * 
                     v.get::<meter_per_second>() / r.get::<meter>();
    let viscous = Pressure::new::<pascal>(viscous_pa);
    
    // Surface tension term: 2σ/r
    let surface = Pressure::new::<pascal>(2.0 * params.sigma / r.get::<meter>());
    
    // Rayleigh-Plesset equation: (P_diff - viscous - surface) / (ρr) - 3v²/2r
    let numerator = p_diff - viscous - surface;
    let denominator = params.rho_liquid * r;
    
    let linear_term = numerator / denominator;
    // Calculate 1.5 * v²/r
    let v_ms = v.get::<meter_per_second>();
    let r_m = r.get::<meter>();
    let nonlinear_accel = 1.5 * v_ms * v_ms / r_m;
    let nonlinear_term = Acceleration::new::<meter_per_second_squared>(nonlinear_accel);
    
    linear_term - nonlinear_term
}

/// Calculate work rate (P dV/dt) in a unit-safe manner
pub fn calculate_work_rate(
    pressure: Pressure,
    radius: Length,
    wall_velocity: Velocity,
) -> Power {
    // dV/dt = 4πr² * dr/dt
    let surface_area = Area::new::<square_meter>(
        4.0 * std::f64::consts::PI * radius.get::<meter>().powi(2)
    );
    let volume_rate = surface_area * wall_velocity;
    
    // Work = -P * dV/dt (negative for compression)
    -pressure * volume_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use uom::si::power::watt;
    
    #[test]
    fn test_unit_safe_parameters() {
        let params = SafeBubbleParameters::default();
        
        // Check that default values are reasonable
        assert_eq!(params.r0.get::<meter>(), 5e-6);
        assert_eq!(params.p0.get::<pascal>(), 101325.0);
        assert_eq!(params.rho_liquid.get::<kilogram_per_cubic_meter>(), 998.0);
    }
    
    #[test]
    fn test_unit_safe_state() {
        let params = SafeBubbleParameters::default();
        let state = SafeBubbleState::new(&params);
        
        // Check equilibrium state
        assert_eq!(state.radius, params.r0);
        assert_eq!(state.wall_velocity.get::<meter_per_second>(), 0.0);
        assert!(state.n_gas > 0.0);
    }
    
    #[test]
    fn test_work_calculation() {
        let pressure = Pressure::new::<pascal>(101325.0);
        let radius = Length::new::<meter>(1e-6);
        let velocity = Velocity::new::<meter_per_second>(-10.0); // Compressing
        
        let work_rate = calculate_work_rate(pressure, radius, velocity);
        
        // Work should be positive during compression (negative velocity)
        assert!(work_rate.get::<watt>() > 0.0);
    }
    
    #[test]
    fn test_unit_conversions() {
        // Test that unit conversions work correctly
        let pressure_pa = Pressure::new::<pascal>(101325.0);
        let pressure_bar = pressure_pa.get::<uom::si::pressure::bar>();
        assert!((pressure_bar - 1.01325).abs() < 1e-4);
        
        let temp_k = ThermodynamicTemperature::new::<kelvin>(293.15);
        let temp_c = temp_k.get::<uom::si::thermodynamic_temperature::degree_celsius>();
        assert!((temp_c - 20.0).abs() < 0.01);
    }
}