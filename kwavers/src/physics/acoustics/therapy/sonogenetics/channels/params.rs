//! Parameter types for mechanosensitive channel gating.

/// Two-state Boltzmann gating parameters for tension-activated channels.
///
/// # Physical units
///
/// - `gating_area_m2`: in-plane gating area A_gate [m^2]
/// - `half_tension_n_per_m`: half-activation membrane tension T_half [N/m]
/// - `single_channel_conductance_s`: unitary conductance g_single [S]
/// - `reversal_potential_v`: reversal (Nernst) potential E_rev [V]
#[derive(Debug, Clone)]
pub struct BoltzmannGatingParams {
    /// In-plane gating area A_gate [m^2].
    pub gating_area_m2: f64,
    /// Membrane tension at half-maximum activation T_half [N/m].
    pub half_tension_n_per_m: f64,
    /// Unitary single-channel conductance g_single [S].
    pub single_channel_conductance_s: f64,
    /// Reversal potential E_rev [V].
    pub reversal_potential_v: f64,
}

/// Pressure-threshold gating parameters for hsTRPA1.
///
/// # Physical units
///
/// - `half_pressure_pa`: radiation pressure at half-maximum activation P_half [Pa]
/// - `steepness_pa`: sigmoid steepness s [Pa]
/// - `single_channel_conductance_s`: unitary conductance [S]
/// - `reversal_potential_v`: reversal potential [V]
#[derive(Debug, Clone)]
pub struct PressureThresholdParams {
    /// Radiation pressure at half-maximum activation P_half [Pa].
    ///
    /// Derived from Ibsen 2015: MI threshold about 0.4 at 1 MHz gives
    /// P_peak about 400 kPa. Radiation pressure
    /// P_rad = P_peak^2 / (2 rho c^2) about 35.6 Pa for water
    /// (rho = 1000 kg/m^3, c = 1500 m/s).
    pub half_pressure_pa: f64,
    /// Sigmoid steepness parameter s [Pa].
    pub steepness_pa: f64,
    /// Unitary conductance [S].
    pub single_channel_conductance_s: f64,
    /// Reversal potential [V].
    pub reversal_potential_v: f64,
}

/// Mechanosensitive channel gating model.
#[derive(Debug, Clone)]
pub enum GatingModel {
    /// Two-state Boltzmann tension-activated gating.
    Boltzmann(BoltzmannGatingParams),
    /// Sigmoidal pressure-threshold gating.
    PressureThreshold(PressureThresholdParams),
}
