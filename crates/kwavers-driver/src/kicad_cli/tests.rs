//! Consolidated tests for the `kicad_cli` slice (Phase 4a carve-out): the DRC pass gate, the
//! `pcb drc` argument assembly, the version-tolerant JSON parser, the fab-bundle summary, and the
//! `PATH` probe. Moved verbatim from the flat `src/kicad_cli.rs` `mod tests` block; the slice-private
//! helpers (`drc_args`, `parse_drc_json`, `locate_on_path`) are reached through their `pub(super)`
//! sub-modules.

use std::path::PathBuf;

use super::cli::{drc_args, locate_on_path, DrcOptions};
use super::drc::{parse_drc_json, DrcDefectCount, DrcReport};
use super::fab::FabBundle;

#[test]
fn drc_report_passes_iff_violations_and_unconnected_are_zero() {
    let clean = DrcReport {
        violations: 0,
        unconnected_items: 0,
        warnings: 3,
        ..DrcReport::default()
    };
    assert!(clean.passes());
    let dirty = DrcReport {
        violations: 1,
        ..clean.clone()
    };
    assert!(!dirty.passes());
    let open = DrcReport {
        unconnected_items: 1,
        ..clean
    };
    assert!(!open.passes());
}

#[test]
fn drc_args_include_refill_before_saving_modified_board() {
    let args = drc_args(
        "board.kicad_pcb",
        "drc.json",
        DrcOptions {
            refill_zones: true,
            save_board: true,
        },
    )
    .unwrap();

    assert_eq!(
        args,
        vec![
            "pcb",
            "drc",
            "--refill-zones",
            "--save-board",
            "--format",
            "json",
            "--output",
            "drc.json",
            "board.kicad_pcb",
        ]
    );
}

#[test]
fn drc_args_reject_save_without_refill() {
    let err = drc_args(
        "board.kicad_pcb",
        "drc.json",
        DrcOptions {
            refill_zones: false,
            save_board: true,
        },
    )
    .unwrap_err();

    assert_eq!(err, "KiCad DRC --save-board requires --refill-zones");
}

#[test]
fn parse_drc_json_handles_kicad_format_variants() {
    // KiCad 8/9/10 layout.
    let v8 = r#"{"violations":[{"severity":"error","type":"clearance"}],"unconnected_items":[],"warnings":[]}"#;
    let r = parse_drc_json(v8);
    assert_eq!(r.violations, 1);
    assert_eq!(r.unconnected_items, 0);
    assert_eq!(
        r.defect_counts,
        vec![DrcDefectCount {
            kind: "clearance".to_string(),
            count: 1,
        }]
    );
    let warning_only = r#"{"violations":[{"severity":"warning","type":"track_dangling"}],"unconnected_items":[],"warnings":[]}"#;
    let r = parse_drc_json(warning_only);
    assert_eq!(r.violations, 0);
    assert_eq!(r.warnings, 1);
    assert!(r.passes());
    let typed = r#"{"violations":[{"type":"shorting_items"},{"type":"shorting_items"},{"type":"clearance"}],"unconnected_items":[{"type":"unconnected_items"}],"warnings":[]}"#;
    let r = parse_drc_json(typed);
    assert_eq!(r.violations, 3);
    assert_eq!(r.unconnected_items, 1);
    assert_eq!(
        r.defect_counts,
        vec![
            DrcDefectCount {
                kind: "clearance".to_string(),
                count: 1,
            },
            DrcDefectCount {
                kind: "shorting_items".to_string(),
                count: 2,
            },
            DrcDefectCount {
                kind: "unconnected_items".to_string(),
                count: 1,
            },
        ]
    );
    // Older `violation_count` style.
    let v7 = r#"{"violation_count": 2, "unconnected_items": 3, "warnings": 0}"#;
    let r = parse_drc_json(v7);
    assert_eq!(r.violations, 2);
    assert_eq!(r.unconnected_items, 3);
    assert_eq!(r.warnings, 0);
    // Missing fields yield zero rather than failing the build.
    let empty = "{}";
    let r = parse_drc_json(empty);
    assert_eq!(r.violations, 0);
    assert_eq!(r.unconnected_items, 0);
    assert_eq!(r.warnings, 0);
}

#[test]
fn fab_bundle_summary_lists_every_artifact() {
    let b = FabBundle {
        gerber_dir: PathBuf::from("g"),
        drill_dir: PathBuf::from("d"),
        position_csv: PathBuf::from("p.csv"),
        render_png: Some(PathBuf::from("r.png")),
        bom_csv: Some(PathBuf::from("bom.csv")),
    };
    let s = b.summary_lines();
    assert_eq!(s.len(), 5);
    assert!(s.iter().all(|l| l.starts_with("fab:")));
}

#[test]
fn locate_finds_pwd_relative_binary_or_fails_cleanly() {
    // We can't assume a real kicad-cli in CI; the function must just not panic and return
    // either a path or a clean "not found". Probe a known-good relative path that doesn't
    // exist to assert the negative branch.
    assert!(locate_on_path("definitely-not-a-real-binary-987654321").is_err());
}
