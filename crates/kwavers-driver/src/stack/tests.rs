//! Consolidated tests for the `stack` slice (Phase 4e carve-out): single-board optimisation,
//! connector mating, and full shield-stack assembly. Moved verbatim from the flat `src/stack.rs`
//! `mod tests` block; `super::*` resolves the slice facade and `DriverManifest` is imported directly.

use super::*;
use crate::manifest::DriverManifest;

fn c() -> StackConstraints {
    StackConstraints {
        dt_max_k: 30.0,
        height_max_mm: 120.0,
        board_pitch_mm: 12.0,
        channel_cap: 16,
    }
}

#[test]
fn board_rise_is_linear_in_channels_and_power() {
    // ΔT = n·p·θ — analytical (passing 1 for single board stack).
    assert!((board_rise_k(16, 0.1, 5.0, 1) - 8.0).abs() < 1e-9);
    assert!((board_rise_k(8, 0.1, 5.0, 1) - 4.0).abs() < 1e-9);
}

#[test]
fn capacity_floors_the_board_count() {
    // 64 channels, 16/tile cap, cool parts → exactly 4 boards (capacity floor), all constraints ok.
    let p = optimize_stack(64, 0.05, 5.0, &c()); // 16·0.05·5·1.45 = 5.8 K ≪ 30
    assert_eq!(p.boards, 4);
    assert_eq!(p.channels_per_tile, 16);
    assert!(p.feasible && p.limiter == "ok");
}

#[test]
fn thermal_forces_more_boards_than_capacity() {
    // Under stack penalty, 0.5 W/channel on 64 channels cannot fit in the 120 mm / 30 K limit.
    // Spreading to more boards increases stack penalty.
    // It ends up finding no feasible solution, stopping at boards = 32 where rise = 28.25 K <= 30 K.
    let p = optimize_stack(64, 0.5, 5.0, &c());
    assert!(!p.feasible, "should be infeasible due to stack penalty");
    assert_eq!(p.limiter, "height");
    assert_eq!(p.boards, 32);
    assert!(p.peak_rise_k <= 30.0);
}

#[test]
fn thermal_forces_more_boards_than_capacity_under_stack_penalty() {
    // Spreading to 8 boards yields rise = 8 * 0.5 * 5 * 2.05 = 41 K <= 42 K (height = 96 mm <= 120 mm).
    let mut constraints = c();
    constraints.dt_max_k = 42.0;
    let p = optimize_stack(64, 0.5, 5.0, &constraints);
    assert!(p.feasible);
    assert_eq!(p.boards, 8);
    assert_eq!(p.channels_per_tile, 8);
    assert!(p.peak_rise_k <= 42.0);
}

#[test]
fn height_makes_an_overcrowded_thermal_solution_infeasible() {
    // Very hot parts needing many boards, but a short enclosure. Thermal needs more height than
    // available ⇒ infeasible, flagged by the height limiter.
    let tight = StackConstraints {
        height_max_mm: 40.0, // only ~3 boards fit
        ..c()
    };
    let p = optimize_stack(64, 1.0, 5.0, &tight); // 5 K/channel ⇒ needs ≥13 boards
    assert!(!p.feasible);
    assert_eq!(p.limiter, "height");
}

fn manifest(role: StackBoardRole, pin_nets: &[&str]) -> StackBoardManifest {
    StackBoardManifest {
        board: role.as_str().into(),
        role,
        connector: "J_STACK".into(),
        board_w_mm: 70.0,
        board_h_mm: 56.0,
        connector_x_mm: 35.0,
        connector_y_mm: 7.0,
        connector_rot_deg: 0.0,
        pin_nets: pin_nets.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn stack_pair_accepts_identical_shield_connector_pinout() {
    let pins = ["VPP", "GND", "P5V", "GND", "P3V3", "GND"];
    let r = verify_stack_pair(
        &manifest(StackBoardRole::Controller, &pins),
        &manifest(StackBoardRole::Driver, &pins),
    );
    assert!(r.pass, "same connector geometry/pinout must pass: {:?}", r);
}

#[test]
fn stack_pair_rejects_swapped_pin() {
    let r = verify_stack_pair(
        &manifest(StackBoardRole::Controller, &["VPP", "GND"]),
        &manifest(StackBoardRole::Driver, &["GND", "VPP"]),
    );
    assert!(!r.pass);
    assert_eq!(r.mismatches.len(), 2);
}

#[test]
fn shield_stack_counts_top_controller_in_height() {
    let p = optimize_shield_stack(64, 0.05, 5.0, &c());
    assert_eq!(p.controller_boards, 1);
    assert_eq!(p.driver_tiles, 4);
    assert_eq!(p.total_boards, 5);
    assert_eq!(p.stack_height_mm, 60.0);
    assert!(p.feasible);
}

#[test]
fn shield_stack_assembly_maps_four_tiles_to_global_channels() {
    let pins = ["VPP", "GND", "BUS_SCLK", "GND"];
    let controller = manifest(StackBoardRole::Controller, &pins);
    let driver = manifest(StackBoardRole::Driver, &pins);
    let c24 = StackConstraints {
        channel_cap: 24,
        ..c()
    };
    let driver_manifest = DriverManifest {
        hv_board: "hv7355_driver_tile.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..24).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG=TCK,TMS,TDI,TDO".into(),
        aperture_m: 4.3e-3 * 23.0 / 15.0,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        // Stack-bus geometry tests don't carry an article-class stimulation program;
        // the cross-manifest compatibility check is board-level, not protocol-level.
        stimulation: None,
        tile_profiles: Vec::new(),
    };
    let plan = optimize_shield_stack(96, 0.05, 5.0, &c24);
    let assembly =
        assemble_shield_stack(&controller, &driver, &driver_manifest, &plan, 96, 12.0, 1.6)
            .expect("96-channel stack must assemble");

    assert_eq!(assembly.total_boards, 5);
    assert_eq!(assembly.boards[0].role, StackBoardRole::Driver);
    assert_eq!(assembly.boards[4].role, StackBoardRole::Controller);
    assert_eq!(assembly.channel_maps.len(), 4);
    assert_eq!(assembly.channel_maps[0].global_tx_nets[0], "TX_0");
    assert_eq!(assembly.channel_maps[0].global_tx_nets[23], "TX_23");
    assert_eq!(assembly.channel_maps[3].global_tx_nets[0], "TX_72");
    assert_eq!(assembly.channel_maps[3].global_tx_nets[23], "TX_95");
    assert!(assembly
        .to_text()
        .contains("format=kicad-routing-shield-stack-assembly-v1"));
}

#[test]
fn shield_stack_assembly_rejects_incomplete_channel_coverage() {
    let pins = ["VPP", "GND"];
    let controller = manifest(StackBoardRole::Controller, &pins);
    let driver = manifest(StackBoardRole::Driver, &pins);
    let c24 = StackConstraints {
        channel_cap: 24,
        ..c()
    };
    let driver_manifest = DriverManifest {
        hv_board: "hv7355_driver_tile.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..23).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG=TCK,TMS,TDI,TDO".into(),
        aperture_m: 4.3e-3 * 22.0 / 15.0,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        // See above: stack-bus tests carry no stimulation program.
        stimulation: None,
        tile_profiles: Vec::new(),
    };
    let plan = optimize_shield_stack(96, 0.05, 5.0, &c24);
    let err = assemble_shield_stack(&controller, &driver, &driver_manifest, &plan, 96, 12.0, 1.6)
        .expect_err("23-channel tile cannot complete a 96-channel stack");
    assert!(err.contains("TX nets"));
}

#[test]
fn recommendations_are_generated_for_stack_plans() {
    let p_cool = optimize_stack(64, 0.05, 5.0, &c());
    let recs_cool = p_cool.recommendations();
    assert!(!recs_cool.is_empty());
    assert!(recs_cool[0].contains("Active cooling"));

    let p_hot = optimize_stack(64, 0.5, 5.0, &c());
    let recs_hot = p_hot.recommendations();
    assert!(!recs_hot.is_empty());
    assert!(recs_hot
        .iter()
        .any(|r| r.contains("Thermal limit exceeded") || r.contains("Stack height exceeds")));
}
