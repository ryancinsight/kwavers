//! Parameter types for mechanosensitive channel gating.

use aequitas::systems::si::quantities::{
    Area, ElectricConductance, ElectricPotential, Pressure, SurfaceTension,
};

/// Two-state Boltzmann gating parameters for tension-activated channels.
///
/// # Physical units
///
/// - `gating_area`: in-plane gating area A_gate [m^2]
/// - `half_tension`: half-activation membrane tension T_half [N/m]
/// - `single_channel_conductance`: unitary conductance g_single (S)
/// - `reversal_potential`: reversal (Nernst) potential E_rev (V)
#[derive(Debug, Clone)]
pub struct BoltzmannGatingParams {
    /// In-plane gating area A_gate.
    pub gating_area: Area<f64>,
    /// Membrane tension at half-maximum activation T_half.
    pub half_tension: SurfaceTension<f64>,
    /// Unitary single-channel conductance g_single.
    pub single_channel_conductance: ElectricConductance<f64>,
    /// Reversal potential E_rev.
    pub reversal_potential: ElectricPotential<f64>,
}

/// Pressure-threshold gating parameters for hsTRPA1.
///
/// # Physical units
///
/// - `half_pressure`: radiation pressure at half-maximum activation P_half
/// - `steepness`: sigmoid steepness s
/// - `single_channel_conductance`: unitary conductance
/// - `reversal_potential`: reversal potential
#[derive(Debug, Clone)]
pub struct PressureThresholdParams {
    /// Radiation pressure at half-maximum activation P_half.
    ///
    /// Derived from Ibsen 2015: MI threshold about 0.4 at 1 MHz gives
    /// P_peak about 400 kPa. Radiation pressure
    /// P_rad = P_peak^2 / (2 rho c^2) about 35.6 Pa for water
    /// (rho = 1000 kg/m^3, c = 1500 m/s).
    pub half_pressure: Pressure<f64>,
    /// Sigmoid steepness parameter s.
    pub steepness: Pressure<f64>,
    /// Unitary conductance.
    pub single_channel_conductance: ElectricConductance<f64>,
    /// Reversal potential.
    pub reversal_potential: ElectricPotential<f64>,
}

/// Mechanosensitive channel gating model.
#[derive(Debug, Clone)]
pub enum GatingModel {
    /// Two-state Boltzmann tension-activated gating.
    Boltzmann(BoltzmannGatingParams),
    /// Sigmoidal pressure-threshold gating.
    PressureThreshold(PressureThresholdParams),
}
