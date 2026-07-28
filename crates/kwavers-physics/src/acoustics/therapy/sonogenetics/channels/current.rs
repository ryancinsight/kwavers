//! Ion-current equation for opened mechanosensitive channels.

use aequitas::systems::si::quantities::{ElectricConductance, ElectricCurrent, ElectricPotential};
use leto::Array3;

/// Compute per-voxel ion current from channel open probability.
///
/// # Formula
///
/// `I_inj = g_single * n_channels * P_open * (E_rev - V_m)`
///
/// # Theorem: reversal-potential null current
///
/// For finite `g_single`, `n_channels`, and `P_open`, if `V_m = E_rev`, then
/// `I_inj = g n P (E_rev - V_m) = 0`. Nonselective cation channels with
/// `E_rev > V_m` produce positive depolarizing injected current under the LIF
/// sign convention.
#[must_use]
pub fn ion_current(
    p_open: &Array3<f64>,
    g_single: ElectricConductance<f64>,
    n_channels: f64,
    v_membrane: ElectricPotential<f64>,
    e_rev: ElectricPotential<f64>,
) -> Array3<f64> {
    let current: ElectricCurrent<f64> = g_single * (e_rev - v_membrane) * n_channels;
    let scale = current.into_base();
    p_open.mapv(|p| p * scale)
}
