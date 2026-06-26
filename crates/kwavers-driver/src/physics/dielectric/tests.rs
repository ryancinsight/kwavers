//! Phase 3c dielectric slice — lifted tests from the flat `src/dielectric.rs`.
//!
//! Four pinning tests:
//! * `caf_lifetime_grows_with_spacing_and_falls_with_voltage` — analytic scaling
//!   `TTF ∝ spacing²/voltage`.
//! * `ipc2221_spacing_is_monotone_and_covers_150v` — locks the 0.60 mm / 150 V floor AND asserts
//!   `CreepageRule::holohv().hv_clearance` agrees (cross-slice SSOT check on the 150 V rail).
//! * `paschen_minimum_is_near_330v` — verifies the air-breakdown minimum is in `[300, 340]` V
//!   and is a true minimum (both neighbours higher).
//! * `one_fifty_volts_cannot_break_down_air` — the engineering consequence: 150 V is below the
//!   Paschen minimum, so air *cannot* break down at any gap.

use super::*;

#[test]
fn caf_lifetime_grows_with_spacing_and_falls_with_voltage() {
    // Doubling spacing ⇒ 4× life; doubling voltage ⇒ half life.
    assert!((caf_ttf_relative(0.5, 150.0, 1.0, 150.0) - 4.0).abs() < 1e-9);
    assert!((caf_ttf_relative(0.5, 150.0, 0.5, 300.0) - 0.5).abs() < 1e-9);
    assert!((caf_ttf_relative(0.5, 150.0, 0.5, 150.0) - 1.0).abs() < 1e-9);
}

#[test]
fn ipc2221_spacing_is_monotone_and_covers_150v() {
    // 150 V external uncoated needs 0.60 mm; `CreepageRule::holohv` uses this exact value as
    // its HV→LV surface-clearance floor so a single source of truth governs the router
    // hazard gradient, the DRU HV_creepage rule, and `.kicad_dru` emission.
    assert!((ipc2221_min_spacing_mm(150.0) - 0.60).abs() < 1e-6);
    assert!(
        (crate::rules::CreepageRule::holohv().hv_clearance.to_mm() - 0.60).abs() < 1e-6,
        "the routing creepage rule must match IPC-2221B B1 external uncoated for 150 V"
    );
    // Higher rails need more; the spacing table is strictly monotone in voltage.
    assert!(ipc2221_min_spacing_mm(300.0) > ipc2221_min_spacing_mm(150.0));
    assert!(ipc2221_min_spacing_mm(600.0) > 0.5);
}

#[test]
fn paschen_minimum_is_near_330v() {
    let (vmin, pd) = paschen_min_air();
    assert!(
        (300.0..=340.0).contains(&vmin),
        "air Paschen minimum should be ~327 V, got {vmin:.0}"
    );
    assert!(pd > 0.0);
    // It is a minimum: both neighbours are higher.
    assert!(paschen_breakdown_v(pd * 0.5) > vmin);
    assert!(paschen_breakdown_v(pd * 2.0) > vmin);
}

#[test]
fn one_fifty_volts_cannot_break_down_air() {
    // The whole point: 150 V < Paschen minimum ⇒ no air breakdown at any gap.
    assert!(!air_breakdown_possible(150.0));
    // A kilovolt can.
    assert!(air_breakdown_possible(1000.0));
}
