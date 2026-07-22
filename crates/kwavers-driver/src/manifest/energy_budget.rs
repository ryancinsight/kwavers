//! The kwavers-conformant energy-budget validator: [`EnergyBudgetInputs`] (the routed-board PDN
//! numbers the manifest alone does not carry), [`EnergyBudgetReport`] (the gated scalar set kwavers
//! consumes), and [`DriverManifest::validate_v2_energy_budget`] — the routed-ampacity / per-tile
//! dissipation gate that ties acoustic accounting to the copper place-and-route actually emitted.

use crate::ssot::{CHANNELS_PER_TILE_V2, TX_LANES_V2};

use super::driver_manifest::DriverManifest;
use super::resistor::ResistorPackage;

/// Inputs the kwavers-conformant energy-budget validator needs that the manifest alone does not
/// carry: the transducer load capacitance, the pulser on-resistance / series-damping network, and
/// the **routed board's** CDR/PDN ampacity headroom as measured by [`mod@crate::audit`] or the
/// placement annealing routine. Passing the routed-board number, not the design envelope, is the
/// critical part of the contract: it ties acoustic accounting to the actual copper that the
/// place-and-route emitted.
#[derive(Debug, Clone, Copy)]
pub struct EnergyBudgetInputs {
    /// Total transducer clamped capacitance (F) charged per cycle.
    pub c_load_f: f64,
    /// HV7355 (or per-board equivalent) on-resistance (Ω).
    pub r_on_ohm: f64,
    /// Series damping resistor (Ω) per channel.
    pub r_series_ohm: f64,
    /// Routed-board ampacity headroom (A) summed over every HV supply rail a tile's VPP/VNN must
    /// deliver under worst-case firing.
    pub ampacity_headroom_a: f64,
    /// Footprint chosen for the per-channel series-damping resistor. The validator no longer
    /// rejects on over-rate — the kwavers-side [`crate::validate::validate_against_budget`]
    /// 4th [`crate::validate::Check`] against `KWVERS_MIN_RESISTOR_MARGIN_W`
    /// is the sole gatekeeper. So a 1206 (250 mW) footprint on the article-class 56 Ω / 50 pF
    /// operating point reaches the validator (signed per-tile margin propagates through as
    /// negative entries) and surfaces a failed kwavers-side Check — NOT an early validator
    /// `Err`. The article-class default is the 2512 (1 W) footprint or the 4527 (2 W) footprint
    /// depending on the matching-cap tuning.
    pub damping_footprint: ResistorPackage,
}

/// Aggregated energy-budget scalar set returned by [`DriverManifest::validate_v2_energy_budget`].
/// All numbers share units with the inputs (J·s, A, W); kwavers consumes the scalars as-is for
/// the cross-stack beam-profile propagation step. Every f64 in this struct is a stack-level
/// scalar (or a 4-entry per-tile vector) the validator has already gated against the routed
/// board's PDN envelope, so kwavers can trust the numbers without re-checking boundary inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyBudgetReport {
    /// Stack-aggregated protocol-load proxy (J·s) = `Σ per_tile_protocol_load_j_s`.
    pub total_protocol_load_j_s: f64,
    /// Per-tile protocol-load proxy (J·s); `len() == 4`.
    pub per_tile_protocol_load_j_s: Vec<f64>,
    /// Worst-case unweighted peak current per tile (A) = `c_load · v_pp · f · n_ch`.
    pub peak_i_a: f64,
    /// Worst-case duty-weighted peak current per tile (A) = `peak_i_a · max_frame_duty`,
    /// the figure compared against [`EnergyBudgetInputs::ampacity_headroom_a`].
    pub peak_duty_weighted_i_a: f64,
    /// Largest frame duty across the 4 tile profiles (0..=1).
    pub max_frame_duty: f64,
    /// Per-tile device-side dissipation (W) from [`crate::driver::pulser_dissipation`].
    /// Thermal-placer weight: each HV tile carries exactly this heat in its IC package, so the
    /// per-tile copper-area allocation must supply a thermal-resistance ceiling below
    /// `θ_jc · per_tile_device_total_w` per channel.
    pub per_tile_device_total_w: Vec<f64>,
    /// Per-tile total pulser loss (W) = dynamic + gate + recovery; the dynamic share is split
    /// between device and series damping resistor in proportion to `R_on/(R_on+R_series)`.
    /// The damping-resistor share is the keystone tying board copper width to the v2 budget.
    pub per_tile_pulser_total_w: Vec<f64>,
    /// Per-tile dissipation in the series damping resistor (W) = `dynamic_series_r` from
    /// [`crate::driver::pulser_dissipation`]. Sizing lever for `power_margin_w`: each tile's
    /// 56 Ω damping resistors must fit a 1206 (250 mW) or 2512 (1 W) footprint, and the v2 budget
    /// surfaces the per-tile choice without forcing the consumer to subtract scalar fields.
    pub per_tile_resistor_w: Vec<f64>,
    /// Per-tile resistor power margin (W) under the chosen footprint's IPC-7351 70 °C rating.
    /// SIGNED — `footprint_max_w − per_tile_resistor_w`i`` propagates verbatim: positive
    /// entry ⇒ headroom above the dissipation; negative entry ⇒ footprint under-rates this
    /// tile by `|margin|` W (the kwavers-side 4th [`crate::validate::Check`] against
    /// `KWVERS_MIN_RESISTOR_MARGIN_W` catches the over-rate case). The
    /// inline rejection gate has been **lifted** out of `validate_v2_energy_budget` so the
    /// kwavers consumer reads BOTH the headroom (for footprint bumps `Smd2512 ⇒ Smd4527`) AND
    /// the over-rate magnitude (for matching-cap tightening) without re-deriving the
    /// dissipation. Small ≳ 0 ⇒ tile is at the package limit (cannot absorb further
    /// dissipation on the chosen footprint); deeply negative ⇒ consumer must act.
    /// Mirrors [`crate::validate::KwaversBeamStep::resistor_margin_w`] and
    /// [`crate::validate::KwaversBeamValidation::resistor_margin_w`].
    pub per_tile_resistor_margin_w: Vec<f64>,
    /// Routed-board ampacity headroom (A) supplied by the caller.
    pub ampacity_headroom_a: f64,
    /// Headroom (A) = `ampacity_headroom_a − peak_duty_weighted_i_a` (≥ 0).
    pub headroom_margin_a: f64,
    /// Number of TX lanes validated (always 96 for a full-stack v2 manifest).
    pub lanes: usize,
}

impl DriverManifest {
    /// Track D v2 energy-budget validator. Asserts the 96-lane binding (`tx_nets.len() == 96`),
    /// requires exactly 4 tile profiles, sums the per-tile protocol-load proxy (J·s), and
    /// compares the worst-case duty-weighted peak-tile current `(C_load · V_pp · f · n_ch ·
    /// frame_duty)` against the routed-board ampacity headroom supplied by [`EnergyBudgetInputs`].
    /// Returns the aggregated budget on success, `Err` describing the offending channel count
    /// or current overload.
    pub fn validate_v2_energy_budget(
        &self,
        inputs: EnergyBudgetInputs,
    ) -> Result<EnergyBudgetReport, String> {
        // Boundary guard rail: degenerate inputs trivially pass any positive ampacity. A
        // zero-cap load (no transducer clamped capacitance) or a zero routed ampacity means
        // the routed board cannot be evaluated against the v2 stack, and the caller should
        // know about the operator error rather than silently accepting the result.
        if inputs.c_load_f <= 0.0 {
            return Err(format!(
                "v2 energy budget requires c_load_f > 0 (got {}) on {}",
                inputs.c_load_f, self.hv_board
            ));
        }
        if inputs.ampacity_headroom_a <= 0.0 {
            return Err(format!(
                "v2 energy budget requires ampacity_headroom_a > 0 (got {}) on {}",
                inputs.ampacity_headroom_a, self.hv_board
            ));
        }
        if self.tx_nets.len() != TX_LANES_V2 {
            return Err(format!(
                "v2 energy budget requires {TX_LANES_V2} TX lanes, found {} on {}",
                self.tx_nets.len(),
                self.hv_board
            ));
        }
        if self.tile_profiles.len() != 4 {
            return Err(format!(
                "v2 energy budget requires 4 tile profiles, found {} on {}",
                self.tile_profiles.len(),
                self.hv_board
            ));
        }

        let per_tile_load = self.per_tile_load_j_s();
        let total_load_j_s: f64 = per_tile_load.iter().sum();

        // Per-tile pulser operating point: the `pulser_dissipation` model in [`crate::driver`]
        // splits the dynamic loss between device and series damping resistor in proportion to
        // `R_on/(R_on+R_series)`; that split is the keystone tying thermal accounting to the
        // routed copper.
        let mut peak_i_a = 0.0_f64;
        let mut max_duty = 0.0_f64;
        let mut per_tile_device_w = Vec::with_capacity(self.tile_profiles.len());
        let mut per_tile_pulser_w = Vec::with_capacity(self.tile_profiles.len());
        let mut per_tile_resistor_w = Vec::with_capacity(self.tile_profiles.len());
        for profile in &self.tile_profiles {
            let i_pulse = inputs.c_load_f * profile.vpp_v * self.frequency_hz;
            let i_tile = i_pulse * (CHANNELS_PER_TILE_V2 as f64);
            peak_i_a = peak_i_a.max(i_tile);
            max_duty = max_duty.max(profile.frame_duty());
            let op = crate::driver::PulserOp {
                drive_hz: self.frequency_hz,
                duty: profile.frame_duty(),
                c_load_f: inputs.c_load_f,
                v_pp: profile.vpp_v,
                r_on_ohm: inputs.r_on_ohm,
                r_series_ohm: inputs.r_series_ohm,
                q_g_c: 20e-9, // HV7355 class
                v_gate: 5.0,
                q_rr_c: 5e-9,
            };
            let d = crate::driver::pulser_dissipation(&op);
            per_tile_device_w.push(d.device_total);
            per_tile_pulser_w.push(d.dynamic_total + d.gate + d.recovery);
            per_tile_resistor_w.push(d.dynamic_series_r);
        }
        let peak_duty_weighted_i_a = peak_i_a * max_duty;
        // Stack-level gate fires first: if both the routed-ampacity headroom and the per-tile
        // resistor rating breach, surfacing the global ampacity shortfall lets the operator
        // size the PDN before drilling into the per-tile package — a single design-fix pass
        // addresses all issues. The resistor check then runs after, scoped to the OFFENDING
        // tile index with the IPC-7351 package name, rated max, and over-wattage.
        let headroom_margin_a = inputs.ampacity_headroom_a - peak_duty_weighted_i_a;
        if headroom_margin_a < 0.0 {
            return Err(format!(
                "routed ampacity {:.3} A cannot supply peak {:.3} A (frame duty {} from {} tiles × {} ch)",
                inputs.ampacity_headroom_a, peak_duty_weighted_i_a, max_duty, 4, CHANNELS_PER_TILE_V2
            ));
        }
        // Per-tile resistor power margin (W) under the chosen footprint's
        // IPC-7351 70 °C rating: `footprint_max_w − per_tile_resistor_w[i]`.
        // SIGNED — positive entry ⇒ headroom available for protocol tweaks;
        // negative entry ⇒ chosen footprint under-rates this tile by `|margin|`
        // W (the kwavers-side [`crate::validate::validate_against_budget`]
        // 4th [`crate::validate::Check`] against
        // [`crate::validate::KWVERS_MIN_RESISTOR_MARGIN_W`] is the SOLE
        // gatekeeper; lifting the inline rejection lets the kwavers consumer
        // plan footprint bumps and matching-cap tightening without having the
        // validator truncate its own input).
        let mut per_tile_resistor_margin_w = Vec::with_capacity(per_tile_resistor_w.len());
        for &w in &per_tile_resistor_w {
            per_tile_resistor_margin_w.push(inputs.damping_footprint.power_margin_w(w));
        }

        Ok(EnergyBudgetReport {
            total_protocol_load_j_s: total_load_j_s,
            per_tile_protocol_load_j_s: per_tile_load,
            peak_i_a,
            peak_duty_weighted_i_a,
            max_frame_duty: max_duty,
            per_tile_device_total_w: per_tile_device_w,
            per_tile_pulser_total_w: per_tile_pulser_w,
            per_tile_resistor_w,
            per_tile_resistor_margin_w,
            ampacity_headroom_a: inputs.ampacity_headroom_a,
            headroom_margin_a,
            lanes: TX_LANES_V2,
        })
    }
}
