//! 5-level pulser efficiency model with charge-recycling.
//!
//! # Background
//!
//! A conventional 3-level (class-D) pulser charges the load capacitance `C₀` from
//! `0 → V` and discharges `V → 0` each cycle, dissipating `C₀·V²` per cycle
//! (½CV² per edge, two edges). A 5-level pulser (e.g., STHVUP32, MAX14815)
//! interpolates through intermediate voltage levels (`V/4`, `V/2`, `3V/4`), and
//! critically, *recycles charge* between levels — the charge removed from one
//! phase goes to charge the next phase's output capacitance.
//!
//! The charge-recycling efficiency `η_cr` depends on the level-transition
//! pattern. In the best case (all intermediate levels visited each cycle),
//! the dynamic loss is reduced by a factor of `(n_levels - 1)⁻¹` relative to
//! 3-level drive — i.e., 5-level achieves ~4× reduction vs class-D.
//!
//! # Evidence tier
//!
//! Closed-form circuit physics (charge-redistribution loss model), verified by
//! energy accounting: `E_5lv = E_3lv / (n_levels - 1)` for full charge-recycling.
//! Value-semantic unit tests against hand-computed references.

/// Dynamic loss (W) for an N-level pulser with charge recycling.
///
/// For a conventional 3-level pulser, all `C·V²·f` energy is dissipated each
/// cycle. For N-level with perfect charge recycling, the dissipation is divided
/// by `(N - 1)`. Real charge-recycling efficiency (`cr_efficiency` in 0..1)
/// scales the saving linearly: `P = P_3lv · (1 − cr_eff · (1 − 1/(N-1)))`.
///
/// # Arguments
/// * `p_3lv` — reference class-D dynamic loss (W).
/// * `n_levels` — number of output voltage levels (e.g., 5).
/// * `cr_efficiency` — charge-recycling efficiency, 0 = none, 1 = perfect.
#[must_use]
pub fn nlevel_dynamic_loss_w(p_3lv: f64, n_levels: usize, cr_efficiency: f64) -> f64 {
    if n_levels <= 2 {
        return p_3lv;
    }
    let ideal_factor = 1.0 / (n_levels as f64 - 1.0);
    let actual_factor = 1.0 - cr_efficiency * (1.0 - ideal_factor);
    p_3lv * actual_factor
}

/// Power saving (W) from using N-level drive instead of 3-level.
#[must_use]
pub fn nlevel_power_saving_w(p_3lv: f64, n_levels: usize, cr_efficiency: f64) -> f64 {
    p_3lv - nlevel_dynamic_loss_w(p_3lv, n_levels, cr_efficiency)
}

/// Efficiency gain: ratio of matched vs direct-drive efficiency for
/// N-level pulsers. The N-level provides additional benefit on top of
/// a matching network — it reduces the reactive power that the matching
/// network must handle.
#[must_use]
pub fn nlevel_efficiency(
    p_acoustic_w: f64,
    p_3lv_dynamic_w: f64,
    n_levels: usize,
    cr_efficiency: f64,
) -> f64 {
    let p_loss = nlevel_dynamic_loss_w(p_3lv_dynamic_w, n_levels, cr_efficiency);
    let total = p_acoustic_w + p_loss;
    if total > 0.0 {
        p_acoustic_w / total
    } else {
        0.0
    }
}

/// Charge-recycling efficiency expected for a given architecture.
///
/// * STHVUP32 / MAX14815: ~0.85 (dedicated charge pump between level drivers)
/// * MD1715: ~0.70 (external FETs, less efficient charge recovery)
/// * HV7355 / STHV748S: 0.0 (no charge-recycling)
#[must_use]
pub fn typical_cr_efficiency(part_number: &str) -> f64 {
    match part_number {
        "STHVUP32" => 0.85,
        "MAX14815" => 0.85,
        "MD1715" => 0.70,
        _ => 0.0,
    }
}

/// Energy per cycle (J) for an N-level pulser driving load `C_load` with
/// peak voltage `V_pp`, given charge-recycling efficiency.
#[must_use]
pub fn nlevel_energy_per_cycle_j(
    c_load_f: f64,
    v_pp: f64,
    n_levels: usize,
    cr_efficiency: f64,
) -> f64 {
    let e_3lv = c_load_f * v_pp * v_pp;
    nlevel_dynamic_loss_w(e_3lv, n_levels, cr_efficiency)
}

/// Count of non-supply rails required for an N-level topology, with GND counted as one rail.
///
/// For 3-level class-D (`+VPP / 0 / −VPP`) the only non-supply rail is GND, so this
/// returns 1 — not 0. For 5-level it returns 2 (GND plus one mid-supply level). Higher
/// levels scale as `n_levels − 2`. Use `nlevel_rails(n) − nlevel_rails(3)` to obtain the
/// incremental rail count an N-level driver requires beyond a class-D baseline.
#[must_use]
pub fn nlevel_rails(n_levels: usize) -> usize {
    if n_levels <= 3 {
        1
    } else if n_levels <= 5 {
        2
    } else {
        n_levels - 2
    }
}

/// Summary comparison of 3-level (class-D) vs 5-level drive.
#[derive(Debug, Clone, Copy)]
pub struct LevelComparison {
    /// Number of output voltage levels for this drive topology.
    pub n_levels: usize,
    /// Charge-recycling efficiency improvement over 3-level (fractional).
    pub cr_efficiency: f64,
    /// Total switching-loss power dissipation at the operating point (W).
    pub dynamic_loss_w: f64,
    /// Power saved relative to the 3-level reference (W).
    pub power_saving_w: f64,
    /// Overall drive efficiency as a fraction (0–1).
    pub efficiency: f64,
    /// Number of additional supply rails required beyond the main HV bus.
    pub additional_rails: usize,
}

/// Compare drive topologies for the given operating point.
#[must_use]
pub fn compare_drive_topologies(
    c_load_f: f64,
    v_pp: f64,
    freq_hz: f64,
    duty: f64,
    p_acoustic_w: f64,
) -> Vec<LevelComparison> {
    let p_dyn_3lv = duty * freq_hz * c_load_f * v_pp * v_pp;

    let configs = [
        (3, 0.0, "3-level (class-D)"),
        (5, 0.85, "5-level (STHVUP32/MAX14815)"),
        (5, 0.70, "5-level (MD1715)"),
    ];

    configs
        .iter()
        .map(|&(n, cr, _name)| {
            let loss = nlevel_dynamic_loss_w(p_dyn_3lv, n, cr);
            let saving = p_dyn_3lv - loss;
            let eta = nlevel_efficiency(p_acoustic_w, p_dyn_3lv, n, cr);
            LevelComparison {
                n_levels: n,
                cr_efficiency: cr,
                dynamic_loss_w: loss,
                power_saving_w: saving,
                efficiency: eta,
                additional_rails: nlevel_rails(n) - nlevel_rails(3),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_level_cuts_dynamic_loss_by_4x_under_perfect_recycling() {
        // 50 pF, 150 V, 2 MHz, CW → 2.25 W class-D.
        let p3: f64 = 1.0 * 2.0e6 * 50.0e-12 * 150.0 * 150.0; // 2.25 W
        assert!((p3 - 2.25).abs() < 1e-3);
        // Perfect 5-level: 1/(5-1)=0.25× → 0.5625 W.
        let p5_perfect = nlevel_dynamic_loss_w(p3, 5, 1.0);
        assert!((p5_perfect - p3 / 4.0).abs() < 1e-6);
        // No recycling: same as 3-level.
        let p5_none = nlevel_dynamic_loss_w(p3, 5, 0.0);
        assert!((p5_none - p3).abs() < 1e-6);
    }

    #[test]
    fn typical_cr_for_sthvup32_is_85_percent() {
        assert!((typical_cr_efficiency("STHVUP32") - 0.85).abs() < 1e-6);
        assert!((typical_cr_efficiency("MAX14815") - 0.85).abs() < 1e-6);
        assert_eq!(typical_cr_efficiency("HV7355K6-G"), 0.0);
        assert_eq!(typical_cr_efficiency("STHV748S"), 0.0);
    }

    #[test]
    fn five_level_improves_driver_efficiency() {
        // 50 mW acoustic, 2.25 W dynamic (direct 3-level).
        let eta3 = nlevel_efficiency(0.05, 2.25, 3, 0.0);
        let eta5 = nlevel_efficiency(0.05, 2.25, 5, 0.85);
        assert!(
            eta5 > eta3 * 2.0,
            "5-level with recycling should more than double efficiency: 3={eta3:.4}, 5={eta5:.4}"
        );
    }

    #[test]
    fn comparison_reports_correct_rail_count() {
        assert_eq!(nlevel_rails(3), 1);
        assert_eq!(nlevel_rails(5), 2);
    }

    #[test]
    fn compare_topologies_ranks_five_level_highest() {
        let comp = compare_drive_topologies(50.0e-12, 150.0, 2.0e6, 1.0, 0.05);
        assert_eq!(comp.len(), 3);
        // 5-level with 85% CR should have highest efficiency.
        assert!(
            comp[1].efficiency > comp[0].efficiency,
            "5-level with CR must beat 3-level"
        );
    }

    #[test]
    fn nlevel_energy_per_cycle_is_consistent() {
        let e3 = nlevel_energy_per_cycle_j(50.0e-12, 150.0, 3, 0.0);
        let e5 = nlevel_energy_per_cycle_j(50.0e-12, 150.0, 5, 1.0);
        assert!((e5 - e3 / 4.0).abs() < 1e-12);
    }
}
