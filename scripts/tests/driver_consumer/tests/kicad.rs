use super::support::{pair, trace};
use kwavers_driver::{
    audit, duplicate_pcb_uuids,
    geom::Nm,
    io::write_kicad_pro,
    parse_kicad_pcb,
    place::{Component, FootprintDef, Placement, Rot},
    rules::{CreepageRule, DesignRules},
    save_kicad_project, save_kicad_sch, write_kicad_pcb, Error, Point,
};
use serde_json::Value;
use std::fs;

#[test]
fn project_and_schematic_preserve_routed_pair() {
    let _trace = trace("two-pin tile interface");
    let (board, rules, directory) = tracing::info_span!("arrange").in_scope(|| {
        (
            pair(&[2_000_000, 8_000_000], &[2_000_000, 4_000_000]),
            DesignRules::holohv(),
            tempfile::tempdir().unwrap(),
        )
    });
    let library = [FootprintDef::two_row_header("PAIR", 1, Nm(1_000_000), &[])];
    let components = [Component {
        refdes: "J1".into(),
        nets: board.nets.iter().map(|n| Some(n.id)).collect(),
        placement: Placement {
            pos: Point::new(Nm(2_000_000), Nm(4_500_000)),
            rot: Rot::R0,
        },
        ..Component::default()
    }];
    let root = directory.path();
    let pcb = root.join("tile.kicad_pcb");
    tracing::info_span!("act").in_scope(|| {
        save_kicad_project(
            &pcb,
            &board,
            &components,
            &library,
            &rules,
            &CreepageRule::holohv(),
        )
        .unwrap();
        save_kicad_sch(&root.join("tile.kicad_sch"), &board, &components, &library).unwrap();
    });
    tracing::info_span!("assert").in_scope(|| {
        let text = fs::read_to_string(pcb).unwrap();
        assert_eq!(text, write_kicad_pcb(&board, &components, &library, &rules));
        assert!(duplicate_pcb_uuids(&text).is_empty());
        let parsed = parse_kicad_pcb(&text).unwrap();
        assert_eq!(parsed.tracks, board.tracks);
        assert_eq!(
            parsed
                .nets
                .iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>(),
            ["DATA_P", "DATA_N"]
        );
        assert_eq!(
            audit(&parsed, &[], &[], &rules).diff_pair_length_mismatch_violations,
            1
        );
        let sch = fs::read_to_string(root.join("tile.kicad_sch")).unwrap();
        for label in [
            "(global_label \"DATA_P\"",
            "(global_label \"DATA_N\"",
            "\"J1\"",
        ] {
            assert!(sch.contains(label));
        }
        let dru = fs::read_to_string(root.join("tile.kicad_dru")).unwrap();
        assert!(dru.contains("(constraint track_width (min 0.150mm))"));
        let pro: Value =
            serde_json::from_slice(&fs::read(root.join("tile.kicad_pro")).unwrap()).unwrap();
        assert_eq!(pro["meta"]["filename"], "tile.kicad_pro");
        assert_eq!(
            pro["board"]["design_settings"]["rules"]["min_clearance"],
            0.13
        );
        assert!(matches!(parse_kicad_pcb("("), Err(Error::Manifest(_))));
        let error = save_kicad_project(
            &root.join("absent/tile.kicad_pcb"),
            &board,
            &components,
            &library,
            &rules,
            &CreepageRule::holohv(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    });
}

#[test]
fn persisted_skew_keeps_one_nanometre_over_limit() {
    let _trace = trace("exported routing decision");
    let (board, rules) = tracing::info_span!("arrange").in_scope(|| {
        (
            pair(&[2_000_000, 8_000_000], &[2_000_000, 8_500_001]),
            DesignRules::holohv(),
        )
    });
    let parsed = tracing::info_span!("act")
        .in_scope(|| parse_kicad_pcb(&write_kicad_pcb(&board, &[], &[], &rules)).unwrap());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            audit(&board, &[], &[], &rules).diff_pair_length_mismatch_violations,
            1
        );
        let faults = audit(&parsed, &[], &[], &rules).diff_pair_length_mismatch_violations;
        assert_eq!(
            faults, 1,
            "re-imported N endpoint {:?}",
            parsed.tracks[1].end
        );
        assert_eq!(parsed.tracks, board.tracks);
    });
}

#[test]
fn project_drill_and_outer_diameter_follow_separate_rules() {
    let _trace = trace("project fabrication minima");
    for drill in [200_000, 250_000] {
        let mut rules = tracing::info_span!("arrange").in_scope(DesignRules::holohv);
        rules.min_via_drill = Nm(drill);
        let json = tracing::info_span!("act").in_scope(|| write_kicad_pro("tile", &rules));
        let pro: Value = serde_json::from_str(&json).unwrap();
        let limits = &pro["board"]["design_settings"]["rules"];
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                limits["min_through_hole_diameter"],
                drill as f64 / 1_000_000.0
            );
            assert_eq!(
                limits["min_via_diameter"],
                (drill + 2 * rules.min_annular.0) as f64 / 1_000_000.0
            );
        });
    }
}

#[test]
fn project_preserves_the_caller_supplied_basename() {
    let _trace = trace("JSON filename");
    let name = tracing::info_span!("arrange").in_scope(|| "tile \"A\"\\rev\t2");
    let json =
        tracing::info_span!("act").in_scope(|| write_kicad_pro(name, &DesignRules::holohv()));
    tracing::info_span!("assert").in_scope(|| {
        let pro: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(pro["meta"]["filename"], format!("{name}.kicad_pro"));
    });
}
