//! Component geometry, pad, clearance, and pinmap tests.
//!
//! Sections D and E from the consolidated tests.rs:
//!   D — Tests lifted from src/place/component.rs::tests
//!       (placement, Rect overlap, assembly clearance).
//!   E — Tests lifted from src/place/symbol_import.rs::tests.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// Section B (continued) — fine-pitch escape predicate (footprint geometry)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn min_pad_pitch_is_the_nearest_pad_spacing() {
    assert_eq!(row_fp(0.5, 4).min_pad_pitch(), Some(Nm::from_mm(0.5)));
    assert_eq!(row_fp(0.5, 1).min_pad_pitch(), None);
}

#[test]
fn fine_pitch_triggers_escape_and_coarse_does_not() {
    let thresh = Nm::from_mm(0.7);
    assert!(
        row_fp(0.5, 8).needs_escape(thresh),
        "0.5 mm pitch must escape"
    );
    assert!(
        !row_fp(1.1, 2).needs_escape(thresh),
        "1.1 mm pitch routes on top"
    );
    let bga = FootprintDef::bga("bga", 4, 4, Nm::from_mm(0.8), &[]);
    assert!(bga.needs_escape(thresh), "an explicit BGA always escapes");
}

// ────────────────────────────────────────────────────────────────────────────
// Section D — Tests lifted from src/place/component.rs::tests
// (placement, Rect overlap, assembly clearance). The `lib()` helper lives at
// the top of `tests/mod.rs` (lifted from the inline `mod tests` block).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn courtyard_follows_rotation() {
    let l = lib();
    let mut c = Component {
        fp: 0,
        nets: vec![None],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let r0 = c.courtyard(&l);
    assert_eq!(r0.max.x - r0.min.x, Nm::from_mm(8.0));
    c.placement.rot = Rot::R90;
    let r90 = c.courtyard(&l);
    assert_eq!(r90.max.x - r90.min.x, Nm::from_mm(4.0)); // axes swapped
}

#[test]
fn pad_position_rotates_about_centre() {
    let l = lib();
    let c = Component {
        fp: 0,
        nets: vec![None],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R90,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    // R90 maps the (+3,0) pad to (0,+3): absolute (10, 13).
    assert_eq!(
        c.pad_pos(&l, 0),
        Point::new(Nm::from_mm(10.0), Nm::from_mm(13.0))
    );
}

#[test]
fn overlap_area_zero_when_disjoint() {
    let a = Rect {
        min: Point::new(Nm(0), Nm(0)),
        max: Point::new(Nm(10), Nm(10)),
    };
    let b = Rect {
        min: Point::new(Nm(20), Nm(0)),
        max: Point::new(Nm(30), Nm(10)),
    };
    assert_eq!(a.overlap_area(b), 0.0);
    assert!(a.overlap_area(a) > 0.0);
}

#[test]
fn component_clearance_detects_inflated_courtyard_overlap() {
    use crate::place::component::component_clearance_violations;
    let l = lib();
    let mk = |refdes: &str, x: f64| Component {
        fp: 0,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let touching = vec![mk("U1", 10.0), mk("U2", 19.0)];
    assert!(
        component_clearance_violations(&touching, &l, Nm::from_mm(1.0)).is_empty(),
        "1.0 mm physical gap exactly holds a 1.0 mm courtyard clearance"
    );
    let too_close = vec![mk("U1", 10.0), mk("U2", 18.2)];
    let v = component_clearance_violations(&too_close, &l, Nm::from_mm(1.0));
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].first, "U1");
    assert_eq!(v[0].second, "U2");
    assert!(v[0].overlap_mm2 > 0.0);
}

// ────────────────────────────────────────────────────────────────────────────
// Section E — Tests lifted from src/place/symbol_import.rs::tests.
// The `PinMap` round-trip covers the function-name ↔ pad-number identity the
// schematic gives us; the two byte-tracking pinning tests lock the Phase 1d
// polish contract for unclosed quoted tokens (name + number). Together they
// exhaustively test the module's parse + lookup surface without touching the
// real vendor .kicad_sym fixtures (whose exact pinmap shape is datasheet-
// specific).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn pinmap_name_and_number_round_trip_lookup() {
    // Synthesise the simplified parse output: pairs are in file order; the function
    // name is the first column, the pad number is the second. Round-trip every pair
    // through `number_of(name)` and `name_of(number)` and assert the missing
    // lookups are absent (NOT just empty), so a contributor swapping an
    // implementation for `unwrap_or_default()` is caught.
    let pm = PinMap {
        pins: vec![
            ("VPP".to_string(), "1".to_string()),
            ("GND".to_string(), "2".to_string()),
            ("IN".to_string(), "3".to_string()),
        ],
    };
    assert_eq!(pm.number_of("VPP"), Some("1"));
    assert_eq!(pm.number_of("GND"), Some("2"));
    assert_eq!(pm.number_of("IN"), Some("3"));
    assert_eq!(pm.name_of("1"), Some("VPP"));
    assert_eq!(pm.name_of("2"), Some("GND"));
    assert_eq!(pm.name_of("3"), Some("IN"));
    assert_eq!(pm.number_of("NOPE"), None);
    assert_eq!(pm.name_of("99"), None);
    assert_eq!(pm.len(), 3);
    assert!(!pm.is_empty());
}

#[test]
fn pinmap_numbers_of_includes_all_pads_with_same_name() {
    // A multi-pin function name (many GND pads on a large package) is the use case
    // `numbers_of` exists for. The parser preserves file order, so the returned
    // designators must match the order in the source symbol. A single-pin function
    // name returns one designator; a missing function name returns an empty vec
    // (NOT None, as the missing-as-name case for `number_of` does).
    let pm = PinMap {
        pins: vec![
            ("GND".to_string(), "5".to_string()),
            ("VPP".to_string(), "1".to_string()),
            ("GND".to_string(), "6".to_string()),
            ("GND".to_string(), "7".to_string()),
        ],
    };
    assert_eq!(pm.numbers_of("GND"), vec!["5", "6", "7"]);
    assert_eq!(pm.numbers_of("VPP"), vec!["1"]);
    assert!(pm.numbers_of("NOPE").is_empty());
}

#[test]
fn pinmap_empty_map_reports_empty_state() {
    // `PinMap::default()` is the empty-state constructor. Every lookup must drop
    // through: `len() == 0`, `is_empty()`, `number_of(None)`, `numbers_of([])`.
    // A contributor wrapping these in `Option` would silently change the API.
    let pm = PinMap::default();
    assert_eq!(pm.len(), 0);
    assert!(pm.is_empty());
    assert!(pm.number_of("ANY").is_none());
    assert!(pm.numbers_of("ANY").is_empty());
}

#[test]
fn unclosed_quoted_name_token_reports_byte_offset_of_open_quote() {
    // The trailing `)` would close the `(pin …)` form with a `Vec<(usize, bool,
    // String)>` that contains zero events, so the caller falls through to the
    // `no_pins` early-return — which would mask the real bug. The Phase 1d polish
    // surfaces `Manifest::Parse` from inside `quoted_events` instead. The byte
    // offset carried in the envelope is `qstart - 1`, which lands on the opening
    // `"` of the unclosed name token (the byte just before `qstart = from + idx +
    // plen`).
    use std::io::Write;
    let dir = "target/tmp";
    std::fs::create_dir_all(dir).unwrap_or_else(|_| panic!("create {dir}"));
    let path = format!("{dir}/phase_2c_unclosed_name.kicad_sym");
    // 11 bytes precede the opening `"` of the unclosed name: `(pin (name "`
    let mut f = std::fs::File::create(&path).expect("create symbol file");
    f.write_all(b"(pin (name \"missing-end)").unwrap();
    let err = import_symbol_pinmap(&path).expect_err("unclosed quoted name must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 11,
                "byte offset must point at the opening `\"` of the unclosed name token (got {offset})"
            );
            assert!(
                message.contains("unclosed quoted token"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quoted name token must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn unclosed_quoted_number_token_reports_byte_offset_of_open_quote() {
    // A valid name `"A"` closes cleanly so the parser keeps going; the next
    // `(number "B` opens the byte but never finds the closing `"` before EOF.
    // The envelope must point at the OPENING `"` of the broken number token, NOT
    // at the closed name token above it — the byte-tracking pin prevents a
    // future contributor from accidentally returning the last successfully-parsed
    // position when reporting the failure.
    use std::io::Write;
    let dir = "target/tmp";
    std::fs::create_dir_all(dir).unwrap_or_else(|_| panic!("create {dir}"));
    let path = format!("{dir}/phase_2c_unclosed_number.kicad_sym");
    // 24 bytes precede the opening `"` of the unclosed number:
    // `(pin (name "A") (number "`
    let mut f = std::fs::File::create(&path).expect("create symbol file");
    f.write_all(b"(pin (name \"A\") (number \"B)").unwrap();
    let err = import_symbol_pinmap(&path).expect_err("unclosed quoted number must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 24,
                "byte offset must point at the opening `\"` of the unclosed number token (got {offset})"
            );
            assert!(
                message.contains("unclosed quoted token"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quoted number token must produce Manifest::Parse; got {err:?}"),
    }
}
