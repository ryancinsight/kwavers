//! Tests lifted from src/place/footprint.rs (rotation + role + footprint escape).
//!
//! `Rot` + `RotationPolicy` resolve via `crate::place::rotation::*` because the
//! carve at Phase 2c extracted them out of `footprint.rs`.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// Section B — Tests lifted from src/place/footprint.rs (rotation + role +
// footprint escape). These tests now resolve `Rot` + `RotationPolicy` via
// `crate::place::rotation::*` because the carve at Phase 2c extracted them out
// of `footprint.rs`.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn rotation_is_order_four_and_preserves_length() {
    let p = Point::new(Nm::from_mm(3.0), Nm::from_mm(1.0));
    assert_eq!(
        Rot::R90.apply(Rot::R90.apply(Rot::R90.apply(Rot::R90.apply(p)))),
        p
    );
    let r = Rot::R90.apply(p);
    assert_eq!(r, Point::new(Nm::from_mm(-1.0), Nm::from_mm(3.0)));
}

#[test]
fn quarter_turn_swaps_courtyard_axes() {
    let s = (Nm::from_mm(8.0), Nm::from_mm(3.0));
    assert_eq!(Rot::R0.apply_size(s), s);
    assert_eq!(Rot::R90.apply_size(s), (Nm::from_mm(3.0), Nm::from_mm(8.0)));
    assert_eq!(Rot::R180.apply_size(s), s);
}

#[test]
fn role_defaults_constrain_rotation() {
    assert_eq!(
        RotationPolicy::for_role(Role::ActiveIc),
        RotationPolicy::Fixed
    );
    assert_eq!(
        RotationPolicy::for_role(Role::Connector),
        RotationPolicy::Fixed
    );
    assert_eq!(
        RotationPolicy::for_role(Role::Decoupling),
        RotationPolicy::HalfTurn
    );
}

#[test]
fn half_turn_policy_preserves_the_floorplanned_axis() {
    assert_eq!(
        Rot::R90.next_allowed(Rot::R90, RotationPolicy::HalfTurn),
        Some(Rot::R270)
    );
    assert_eq!(
        Rot::R270.next_allowed(Rot::R90, RotationPolicy::HalfTurn),
        Some(Rot::R90)
    );
    assert_eq!(Rot::R90.next_allowed(Rot::R90, RotationPolicy::Fixed), None);
    assert_eq!(
        Rot::R90.next_allowed(Rot::R90, RotationPolicy::AnyRightAngle),
        Some(Rot::R180)
    );
}
