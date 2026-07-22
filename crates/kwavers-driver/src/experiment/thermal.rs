//! Electro-thermal propagation from the energy budget (Phase 5).
//!
//! Maps per-tile device dissipation (W) through the package thermal-resistance network to a
//! per-tile junction temperature rise (K). Feeds the orchestrator's thermal headroom check and
//! the per-tile copper-area allocation signal.
//!
//! The model is steady-state 0-D (one scalar θ_jc per tile). A more refined transient or 2-D
//! model feeds the same [`ThermalState`] output via a different `propagate_thermal` call; the
//! struct shape is the boundary between the thermal and experiment layers.

use crate::manifest::EnergyBudgetReport;

/// Per-tile thermal state derived from the energy budget + package θ_jc.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalState {
    /// Per-tile junction temperature rise above ambient (K) = `per_tile_device_total_w`i` × θ_jc`.
    pub per_tile_rise_k: Vec<f64>,
    /// Peak tile temperature rise (K) — the thermal-placer's binding constraint.
    pub peak_rise_k: f64,
    /// Thermal headroom (K) = `dt_max_k − peak_rise_k`.
    /// Negative ⇒ over the design thermal budget.
    pub headroom_k: f64,
}

/// Propagate device dissipation from `budget` through package θ_jc (K/W) to a [`ThermalState`].
///
/// * `theta_jc_k_per_w` — junction-to-case thermal resistance (K/W) for the pulser IC package.
///   The HV7355-class default is 40 K/W (same value used by the HV7355 driver baseline).
/// * `dt_max_k` — maximum allowed steady-state junction temperature rise above ambient
///   (K); the design limit the copper area and heat-spreading must satisfy.
///
/// Only device dissipation (`per_tile_device_total_w`) contributes to the junction rise;
/// the series-damping resistor dissipation (`per_tile_resistor_w`) is a board-copper thermal
/// path handled separately.
#[must_use]
pub fn propagate_thermal(
    budget: &EnergyBudgetReport,
    theta_jc_k_per_w: f64,
    dt_max_k: f64,
) -> ThermalState {
    let per_tile_rise_k: Vec<f64> = budget
        .per_tile_device_total_w
        .iter()
        .map(|&w| w * theta_jc_k_per_w)
        .collect();
    let peak_rise_k = per_tile_rise_k.iter().copied().fold(0.0_f64, f64::max);
    ThermalState {
        per_tile_rise_k,
        peak_rise_k,
        headroom_k: dt_max_k - peak_rise_k,
    }
}
