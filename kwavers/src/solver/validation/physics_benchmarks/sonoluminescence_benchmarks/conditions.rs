//! Canonical SBSL parameter set (Brenner, Hilgenfeldt & Lohse 2002).

use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;

/// Canonical SBSL parameter set from Brenner, Hilgenfeldt & Lohse (2002).
///
/// These are the "standard" single-bubble sonoluminescence conditions for
/// an air bubble in water at 20°C, driving at 26.5 kHz.
#[derive(Debug, Clone)]
pub struct BrennerSBSLConditions {
    /// Driving frequency (Hz)
    pub freq_hz: f64,
    /// Ambient pressure (Pa)
    pub p0_pa: f64,
    /// Acoustic pressure amplitude (Pa)
    pub p_a_pa: f64,
    /// Equilibrium bubble radius (m)
    pub r0_m: f64,
    /// Water temperature (K)
    pub temperature_k: f64,
    /// Surface tension water at 20°C [N/m]
    pub sigma: f64,
    /// Dynamic viscosity water at 20°C [Pa·s]
    pub mu: f64,
    /// Liquid density [kg/m³]
    pub rho_l: f64,
    /// Polytropic index of air (adiabatic at collapse)
    pub gamma: f64,
    /// Vapour pressure of water at 20°C (Pa)
    pub p_v: f64,
    /// Liquid sound speed (m/s)
    pub c_l: f64,
}

impl Default for BrennerSBSLConditions {
    fn default() -> Self {
        Self {
            freq_hz: 26_500.0,
            p0_pa: ATMOSPHERIC_PRESSURE,
            p_a_pa: 1.35e5,   // 1.35 atm driving (Brenner 2002 Table I)
            r0_m: 5.0e-6,     // 5 µm
            temperature_k: 293.15,
            sigma: 0.0728,
            mu: 1.002e-3,
            rho_l: 998.0,
            gamma: 1.4,   // adiabatic index for air (Yasui 1997 uses γ_eff ≈ 1.4)
            p_v: 2_340.0, // vapour pressure at 20°C [Pa]
            c_l: 1_485.0, // sound speed in water at 20°C [m/s]
        }
    }
}
