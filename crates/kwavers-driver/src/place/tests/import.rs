//! Footprint import and S-expr parsing tests.
//!
//! Tests lifted from src/place/footprint_import.rs::tests
//! (sexpr parser byte-tracking pinning + realshift vendor importer tests).
//!
//! The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) is now at
//! `crate::place::sexpr` (`pub(crate)`); these tests reach it via `use crate::place::sexpr::*`.
//! The DOCS path is preserved verbatim from the original inline module so the
//! committed vendor files stay the differential oracle.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// Section C — Tests lifted from src/place/footprint_import.rs::tests
// (sexpr parser byte-tracking pinning + realshift vendor importer tests).
//
// The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) is now at
// `crate::place::sexpr` (`pub(crate)`); these tests reach it via `use crate::place::sexpr::*`.
// The DOCS path is preserved verbatim from the original inline module so the
// committed vendor files stay the differential oracle.
// ────────────────────────────────────────────────────────────────────────────

// Path to the committed vendor footprints, relative to the crate root.
const DOCS: &str = "docs/cad_models";

#[test]
fn imports_real_xc7a100t_fgg484_bga() {
    // The XC7A100T-2FGG484C AMD vendor footprint is a 484-ball FG(G)BGA, 22×22 grid (with corner
    // depopulation), 1.0 mm pitch. The geometry file gives each ball a letter-number designator
    // ("A1".."AB22"); `pad_index()` lets the netlist wire a net to a specific ball by name.
    let p =
        format!("{DOCS}/XC7A100T_2FGG484C/KiCADv6/footprints.pretty/FGG484ARTIX-7_AMD.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &["A2", "H8"]).unwrap();
    assert_eq!(fp.pads.len(), 484, "real FGG484 pad count");
    assert_eq!(fp.pad_names.len(), 484, "real FGG484 ball designators");
    assert!(fp.pad_index("A1").is_some(), "ball A1 exists");
    assert!(
        fp.pad_index("L11").is_some(),
        "ball L11 exists (centre row/col)"
    );
    assert!(
        fp.pad_index("W22").is_some(),
        "ball W22 exists (max row/col)"
    );
    assert!(fp.pads[fp.pad_index("A2").unwrap()].power_pin);
    assert!(fp.pads[fp.pad_index("H8").unwrap()].power_pin);
    assert!(fp.courtyard.0.to_mm() > 23.0 && fp.courtyard.1.to_mm() > 23.0);
}

#[test]
fn parses_nested_sexpr_with_quotes() {
    let s = parse_sexpr(r#"(footprint "A B" (pad "1" smd (at 1 2)))"#).unwrap();
    assert_eq!(s.head(), Some("footprint"));
    let pad = child(&s, "pad").unwrap();
    assert_eq!(pad.as_list().unwrap()[1].as_atom(), Some("1"));
}

// -------- Phase 1c polish: `Manifest::Parse { offset }` byte-tracking pinning tests --------
//
// These pin the parse_sexpr byte-position contract. The `parse_sexpr` loop iterates
// `char_indices().peekable()` (NOT `chars().enumerate()`); every `Manifest::Parse` carries
// the TRUE UTF-8 byte offset of the offending token, the EOF position when the input ran out,
// or an explicit byte position for nested unclosed forms. The tests use direct pattern
// matching on `crate::error::Error::Manifest(Manifest::Parse {..})` (mirroring the SSOT
// smoke tests at `src/error/manifest.rs::tests::io_at_matches_inline_construction`); the
// aggregator's `#[error(transparent)]` on `Error::Manifest` delegates `source()` straight to
// the inner `Manifest::Parse::source()` (returning `None`, since `Parse` has no `#[source]`
// field), so source-chain walking would NOT find the `Manifest` envelope.

#[test]
fn parse_sexpr_unclosed_paren_offset_points_at_offender() {
    let input = "a)";
    let err = parse_sexpr(input).expect_err("rogue closer must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(*offset, 1, "byte offset must point at the offending `)`");
            assert!(
                message.contains("unexpected closing paren"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("rogue `)` must produce Manifest::Parse; got {err:?}"),
    }
    let s = err.to_string();
    assert!(s.contains("near byte 1"), "Display carries the offset: {s}");
}

#[test]
fn parse_sexpr_unclosed_string_reports_eof_offset() {
    let input = r#"(pad "abc"#;
    let err = parse_sexpr(input).expect_err("unclosed quote must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset,
                input.len(),
                "offset must be src.len() (EOF after the unclosed quote): got {offset}"
            );
            assert!(
                message.contains("unclosed string literal"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quote must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn parse_sexpr_eof_before_top_level_reports_input_len() {
    let input = "(footprint"; // 10 bytes, no closing `)`
    let err = parse_sexpr(input).expect_err("EOF before close must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset,
                input.len(),
                "offset must be src.len() (input exhausted): got {offset}"
            );
            assert!(
                message.contains("input ended before top-level s-expression closed"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("EOF-before-close must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn parse_sexpr_unicode_byte_offset_differs_from_char_offset() {
    let input = "\u{03bc})"; // 3 bytes: [0xCE, 0xBC, 0x29]
    assert_eq!(input.len(), 3, "µ must be 2 bytes in UTF-8");
    let err = parse_sexpr(input).expect_err("rogue `)` after a 2-byte UTF-8 char must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 2,
                "byte offset must be 2 (true UTF-8 byte position), NOT 1 (char ordinal); \
                 this guards against a future contributor reverting to chars().enumerate()"
            );
            assert!(
                message.contains("unexpected closing paren"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn imports_model_offset_and_rotation() {
    let s = parse_sexpr(
        r#"(model "m.step" (offset (xyz 3.0 2.5643 0)) (scale (xyz 1 1 1)) (rotate (xyz 0 0 90)))"#,
    )
    .unwrap();
    assert_eq!(xyz_child(&s, "offset"), Some((3.0, 2.5643, 0.0)));
    assert_eq!(xyz_child(&s, "rotate"), Some((0.0, 0.0, 90.0)));
}

#[test]
fn imported_model_offset_is_recentered_with_pads() {
    let p =
        format!("{DOCS}/430450600/KiCADv6/footprints.pretty/CONN_SD-43045-001_06_MOL.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();

    let pad1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (pad1.offset.x.to_mm() - 3.0).abs() < 1e-4 && (pad1.offset.y.to_mm() + 2.5643).abs() < 1e-4,
        "pad 1 is translated into the courtyard-centred frame"
    );
    let (_, offset, _, _) = fp
        .model
        .as_ref()
        .expect("Molex 0430450600 footprint carries its STEP model");
    assert!(
        (offset.0 - 6.0).abs() < 1e-4 && offset.1.abs() < 1e-4,
        "model offset is translated by the same courtyard-centre shift as the pads"
    );
}

#[test]
fn imports_real_iso7740_footprint() {
    let p = format!("{DOCS}/ISO7740DBQR/KiCADv6/footprints.pretty/DBQ0016A_M.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &["8", "16"]).unwrap();
    assert_eq!(fp.pads.len(), 16, "exact pad count from the vendor file");
    assert_eq!(fp.pad_index("1"), Some(0));
    assert!(fp.pad_index("16").is_some());
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!((p1.offset.x.to_mm() + 2.825).abs() < 1e-3);
    assert!((p1.offset.y.to_mm() + 2.2225).abs() < 1e-3);
    assert!(
        (p1.size.0.to_mm() - 1.65).abs() < 1e-2,
        "rotated long axis (1.65 mm) must lie along X (outward), got {:.3}",
        p1.size.0.to_mm()
    );
    assert!(
        (p1.size.1.to_mm() - 0.4).abs() < 1e-2,
        "rotated short axis (0.4 mm) must lie along the pin pitch (Y), got {:.3}",
        p1.size.1.to_mm()
    );
    let p2 = &fp.pads[fp.pad_index("2").unwrap()];
    let pitch = (p1.offset.y.to_mm() - p2.offset.y.to_mm()).abs();
    assert!(
        p1.size.1.to_mm() < pitch,
        "pad Y-extent {:.3} mm must be below the {:.3} mm pitch (no overlap)",
        p1.size.1.to_mm(),
        pitch
    );
    assert!(fp.pads[fp.pad_index("16").unwrap()].power_pin);
    assert!(fp.courtyard.0.to_mm() > 8.0 && fp.courtyard.1.to_mm() > 5.5);
}

#[test]
fn imports_real_hv7355_qfn56() {
    let p = format!("{DOCS}/HV7355K6_G/KiCADv6/footprints.pretty/QFN56_8X8MC_MCH.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &[]).unwrap();
    assert_eq!(fp.pads.len(), 57, "real QFN56 pad map");
    assert_eq!(fp.pad_names.len(), fp.pads.len());
}

#[test]
fn imports_real_molex_transducer_header_with_board_locks() {
    let p = format!("{DOCS}/430452400/MOLEX_430452400.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();
    assert_eq!(
        fp.pads.len(),
        26,
        "24 signal pins plus two NPTH board-lock holes"
    );
    assert_eq!(fp.pad_index("1"), Some(0));
    assert_eq!(fp.pad_index("24"), Some(23));
    assert!((fp.courtyard.0.to_mm() - 42.69).abs() < 0.01);
    assert!((fp.courtyard.1.to_mm() - 14.78).abs() < 0.01);
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() - 16.5).abs() < 0.01 && (p1.offset.y.to_mm() + 2.8).abs() < 0.01,
        "pad coordinates are centred to the courtyard origin"
    );
    let board_locks = fp.pad_names.iter().filter(|name| name.is_empty()).count();
    assert_eq!(
        board_locks, 2,
        "NPTH holes are retained as mechanical (empty-designator) keepouts"
    );
    assert!(
        fp.pads.len() - board_locks == 24,
        "exactly 24 electrical pins remain"
    );
}

#[test]
fn imports_real_molex_430450400_hv_power_header() {
    let p = format!(
        "{DOCS}/430450400/KiCADv6/footprints.pretty/Molex_Micro-Fit_3.0_43045-0400_2x02_P3.00mm_Horizontal.kicad_mod"
    );
    let fp = import_kicad_mod(&p, Role::Connector, &["1", "2", "3", "4"]).unwrap();
    assert_eq!(fp.pads.len(), 5, "four pins plus one NPTH board lock");
    assert_eq!(fp.pad_index("1"), Some(1));
    assert_eq!(fp.pad_index("4"), Some(4));
    let board_locks = fp.pad_names.iter().filter(|name| name.is_empty()).count();
    assert_eq!(
        board_locks, 1,
        "NPTH board lock is retained as a mechanical keepout"
    );
    assert_eq!(
        fp.pads.len() - board_locks,
        4,
        "exactly four electrical power pins remain"
    );
    assert!((fp.courtyard.0.to_mm() - 11.16).abs() < 0.01);
    assert!((fp.courtyard.1.to_mm() - 13.67).abs() < 0.01);
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() + 1.5).abs() < 0.01 && (p1.offset.y.to_mm() - 2.585).abs() < 0.01,
        "pad 1 is centred to the courtyard coordinate frame"
    );
    assert!(
        ["1", "2", "3", "4"]
            .iter()
            .all(|pin| fp.pads[fp.pad_index(pin).unwrap()].power_pin),
        "all four HV input pins are classified as power-carrying pads"
    );
    let (path, offset, rotate, envelope) = fp
        .model
        .as_ref()
        .expect("Molex 0430450400 footprint carries its STEP model token");
    assert_eq!(path, "${KICAD10_3DMODEL_DIR}/Connector_Molex.3dshapes/Molex_Micro-Fit_3.0_43045-0400_2x02_P3.00mm_Horizontal.step");
    assert!(
        (offset.0 + 1.5).abs() < 1e-6 && (offset.1 - 2.585).abs() < 1e-6 && offset.2 == 0.0,
        "model offset follows the courtyard-centred footprint frame"
    );
    assert_eq!(*rotate, (0.0, 0.0, 0.0));
    assert_eq!(*envelope, None);
}

#[test]
fn imports_real_molex_power_header_model_transform() {
    let p =
        format!("{DOCS}/430450600/KiCADv6/footprints.pretty/CONN_SD-43045-001_06_MOL.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();
    assert_eq!(fp.pads.len(), 7, "six pins plus one NPTH board lock");
    assert_eq!(fp.pad_index("1"), Some(0));
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() - 3.0).abs() < 1e-4 && (p1.offset.y.to_mm() + 2.5643).abs() < 1e-4,
        "pad 1 is centred to the courtyard coordinate frame"
    );
    let (path, offset, rotate, envelope) = fp
        .model
        .as_ref()
        .expect("Molex 0430450600 footprint carries its STEP model");
    assert_eq!(path, "docs/cad_models/430450600_stp/430450600.stp");
    assert!(
        (offset.0 - 6.0).abs() < 1e-4 && offset.1.abs() < 1e-4 && offset.2 == 0.0,
        "model offset follows the courtyard-centred footprint frame"
    );
    assert_eq!(*rotate, (0.0, 0.0, 0.0));
    assert_eq!(*envelope, None);
}
