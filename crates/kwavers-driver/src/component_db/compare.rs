//! Candidate-IC comparison and the 96-channel recommendation: filter the catalog to viable parts,
//! run each through the pulser loss model, and rank by a composite cost/power/area/routing score.

use crate::driver::{pulser_dissipation, PulserOp};

use super::catalog::available_pulsers;
use super::pulser_ic::{board_area_per_n_channels_mm2, StockStatus};

/// Comparison result between two pulser ICs for a given operating point.
#[derive(Debug, Clone)]
pub struct PulserComparison {
    /// Manufacturer part number of the evaluated IC.
    pub part_number: &'static str,
    /// Total output channels covered by the selected IC count.
    pub total_channels: usize,
    /// Number of IC packages required to cover all channels.
    pub ics_required: usize,
    /// Aggregate component cost at quantity 1k (USD).
    pub total_cost_usd: f64,
    /// Combined package footprint area for all ICs (mm²).
    pub total_pkg_area_mm2: f64,
    /// Total power dissipated across all ICs at the operating point (W).
    pub total_dissipation_w: f64,
    /// True if the IC includes an integrated beamforming delay block.
    pub beamformer_integrated: bool,
    /// True if the IC includes an integrated T/R switch.
    pub tr_switch_integrated: bool,
    /// True if charge recycling between output levels is supported.
    pub charge_recycling_enabled: bool,
    /// Estimated PCB layer count driven by routing complexity.
    pub pcb_layers_estimate: usize,
    /// Estimated routed via count for the assembled design.
    pub routed_vias_estimate: usize,
    /// Composite optimisation score (higher is better).
    pub score: f64,
}

/// Compare pulser IC candidates for a given channel count and operating point.
/// Returns a list of viable candidates ranked by composite score (lower is better).
#[must_use]
pub fn compare_pulsers(
    n_channels: usize,
    v_drive_v: f64,
    f_drive_mhz: f64,
    duty: f64,
    c_load_pf: f64,
) -> Vec<PulserComparison> {
    let mut results: Vec<PulserComparison> = available_pulsers()
        .iter()
        .filter(|ic| ic.stock_status == StockStatus::Active)
        .filter(|ic| ic.v_max_v >= v_drive_v)
        .filter(|ic| ic.f_max_mhz >= f_drive_mhz)
        .map(|ic| {
            let ics_required = n_channels.div_ceil(ic.channels);
            let op = PulserOp {
                drive_hz: f_drive_mhz * 1.0e6,
                duty,
                c_load_f: c_load_pf * 1.0e-12,
                v_pp: v_drive_v,
                r_on_ohm: ic.r_on_ohm,
                r_series_ohm: 56.0,
                q_g_c: ic.q_g_nc * 1.0e-9,
                v_gate: ic.v_gate_v,
                q_rr_c: 5.0e-9,
            };
            let diss = pulser_dissipation(&op);

            let total_dissipation_w = diss.device_total * ics_required as f64 * ic.channels as f64;
            let total_cost_usd = ic.cost_per_ch_usd * n_channels as f64;
            let total_pkg_area_mm2 =
                ic.package_size_mm.0 * ic.package_size_mm.1 * ics_required as f64;

            let beamformer_integrated = ic.beamforming_mem.is_some();
            let tr_switch_integrated = ic.tr_switch;

            let pcb_layers_estimate = if ic.package.contains("BGA") || ic.channels >= 32 {
                6
            } else if ic.channels >= 8 {
                4
            } else {
                2
            };

            let routed_vias_estimate = n_channels
                * match ic.channels {
                    32 => 4,
                    8 => 8,
                    4 => 6,
                    _ => 10,
                };

            // Board area for scoring uses the per-channel area estimate (which accounts
            // for decoupling, routing keepouts, and channel-density amortisation) rather
            // than raw package area × IC count, which incorrectly rewards many small chips.
            let score_area_mm2 = board_area_per_n_channels_mm2(ic, n_channels);

            let mut score = total_cost_usd
                + total_dissipation_w * 10.0
                + score_area_mm2 * 0.5
                + routed_vias_estimate as f64 * 0.1
                + ics_required as f64 * 5.0; // assembly + handling cost per IC

            if beamformer_integrated {
                score *= 0.85;
            }
            if tr_switch_integrated {
                score *= 0.90;
            }
            if ic.charge_recycling {
                score *= 0.92;
            }

            PulserComparison {
                part_number: ic.part_number,
                total_channels: n_channels,
                ics_required,
                total_cost_usd,
                total_pkg_area_mm2,
                total_dissipation_w,
                beamformer_integrated,
                tr_switch_integrated,
                charge_recycling_enabled: ic.charge_recycling,
                pcb_layers_estimate,
                routed_vias_estimate,
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| a.score.total_cmp(&b.score));
    results
}

/// ICs of interest for the current 96-channel HoloHV design,
/// with a recommended decision based on the trade-off.
#[must_use]
pub fn recommend_96ch_architecture() -> &'static str {
    let results = compare_pulsers(96, 150.0, 2.0, 0.5, 50.0);
    results
        .first()
        .map(|r| r.part_number)
        .unwrap_or("HV7355K6-G")
}
