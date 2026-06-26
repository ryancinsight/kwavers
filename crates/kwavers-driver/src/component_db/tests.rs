//! Consolidated tests for the `component_db` slice (Phase 4i carve-out): catalog completeness, the
//! candidate-comparison ranking, and the per-IC board-area model. Moved verbatim from the flat
//! `src/component_db.rs` `mod tests` block; the one `for p in &pulsers` adapted to `for p in pulsers`
//! for the `&'static [PulserIc]` slice return. `super::*` resolves the slice facade.

use super::*;

#[test]
fn hvpulser_db_has_all_active_parts() {
    let pulsers = available_pulsers();
    assert!(pulsers.len() >= 6, "expected at least 6 parts in the DB");
    for p in pulsers {
        assert!(!p.part_number.is_empty());
        assert!(p.channels > 0);
        assert!(p.v_max_v > 0.0);
        assert!(p.r_on_ohm > 0.0);
        assert!(!p.package.is_empty());
    }
}

#[test]
fn compare_pulsers_filters_voltage_insufficient() {
    // Only parts rated ≥ 200 V survive a 200 V requirement.
    let results = compare_pulsers(64, 200.0, 2.0, 0.5, 50.0);
    for r in &results {
        assert!(
            r.part_number != "STHVUP32",
            "STHVUP32 is 100V max, should be excluded at 200V"
        );
        assert!(
            r.part_number != "STHV748S",
            "STHV748S is 90V max, should be excluded at 200V"
        );
        assert!(
            r.part_number != "HV7360",
            "HV7360 is 100V max, should be excluded at 200V"
        );
    }
    assert!(!results.is_empty(), "at least MAX14815 and MD1715 survive");
}

#[test]
fn compare_pulsers_scores_sthvup32_highly_for_96ch_100v() {
    let results = compare_pulsers(96, 100.0, 2.0, 0.5, 50.0);
    assert!(!results.is_empty());
    let top = &results[0];
    assert_eq!(
        top.part_number, "STHVUP32",
        "STHVUP32 should rank first for 96ch × 100V: 3 chips, integrated beamforming"
    );
}

#[test]
fn board_area_scales_with_channel_count() {
    let pulsers = available_pulsers();
    let hv7355 = pulsers
        .iter()
        .find(|p| p.part_number == "HV7355K6-G")
        .unwrap();
    let sthvup32 = pulsers
        .iter()
        .find(|p| p.part_number == "STHVUP32")
        .unwrap();

    let area_7355 = board_area_per_n_channels_mm2(hv7355, 96);
    let area_st = board_area_per_n_channels_mm2(sthvup32, 96);

    assert!(
            area_st < area_7355 * 0.6,
            "STHVUP32 (3× BGA) should use less board area than 12× QFN: ST={area_st:.0}, 7355={area_7355:.0}"
        );
}

#[test]
fn cost_is_per_channel_proportional() {
    let results = compare_pulsers(8, 90.0, 2.0, 0.5, 50.0);
    // 8 channels at 90V: HV7355 (2 units) and STHV748S (2 units) compete.
    // STHV748S is cheaper per-channel ($2.80 vs $4.50), so its total should be less.
    let hv7355 = results.iter().find(|r| r.part_number == "HV7355K6-G");
    let st748 = results.iter().find(|r| r.part_number == "STHV748S");
    if let (Some(hv), Some(st)) = (hv7355, st748) {
        assert!(
            st.total_cost_usd < hv.total_cost_usd,
            "STHV748S should cost less at 8ch×90V: ST=${:.0}, HV=${:.0}",
            st.total_cost_usd,
            hv.total_cost_usd
        );
    }
}

#[test]
fn max14815_offers_tr_switch_and_5level() {
    let pulsers = available_pulsers();
    let max15 = pulsers
        .iter()
        .find(|p| p.part_number == "MAX14815")
        .unwrap();
    assert!(max15.tr_switch, "MAX14815 has integrated T/R switch");
    assert_eq!(max15.n_levels, 5);
    assert!(max15.beamforming_mem.is_some());
}

#[test]
fn pulser_comparison_score_prefers_integrated_beamforming() {
    let results = compare_pulsers(32, 100.0, 2.0, 0.5, 50.0);
    let Some(top) = results.first() else {
        panic!("no valid pulser for 32ch×100V");
    };
    assert!(
        top.beamformer_integrated,
        "32ch×100V should recommend STHVUP32 (single IC with beamforming), got {}",
        top.part_number
    );
}
