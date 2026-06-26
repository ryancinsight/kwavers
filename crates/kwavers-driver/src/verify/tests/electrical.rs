//! Electrical/logical verification tests: decoupling proximity, ERC, BOM, schematic isolation,
//! and parasitic AC-coupling.

use super::*;
use crate::board::{Board, NetClassKind, Track};
use std::collections::HashMap;

#[test]
fn decoupling_proximity_measures_to_the_nearest_power_pin() {
    let ic = FootprintDef::new(
        "U",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
            size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    );
    let cap = FootprintDef::new(
        "C",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![],
    );
    let lib = vec![ic, cap];
    let comps = vec![
        comp_at(0, "U1", 10.0, 10.0, None),
        comp_at(1, "C_NEAR", 10.0, 10.0, Some(0)),
        comp_at(1, "C_FAR", 13.0, 10.0, Some(0)),
    ];
    let r = decoupling_proximity(&comps, &lib, 3.0);
    assert_eq!(
        r.far_caps.len(),
        1,
        "only the far cap fails (measured to the power pin)"
    );
    assert_eq!(r.far_caps[0].0, "C_FAR");
    assert!((r.far_caps[0].1 - 5.0).abs() < 1e-6);
    assert!(!r.pass);
}

#[test]
fn erc_flags_unconnected_pad_and_floating_net() {
    let mut b = Board::new(spec());
    let a = b.add_net("A", NetClassKind::Signal);
    let lib = vec![two_pad_fp()];
    let comps = vec![comp(0, "R1", vec![Some(a), None])];
    let symbol_pin_maps = HashMap::new();
    let r = erc(&b, &comps, &lib, &symbol_pin_maps);
    assert_eq!(r.unconnected_pads, vec![("R1".into(), 1)]);
    assert_eq!(r.floating_nets, vec!["A".to_string()]);
    assert!(!r.pass);
}

#[test]
fn erc_flags_power_pin_on_signal_net() {
    let mut b = Board::new(spec());
    let sig = b.add_net("CTRL", NetClassKind::Signal);
    let mut fp = two_pad_fp();
    fp.pads[0].power_pin = true;
    let lib = vec![fp];
    let comps = vec![comp(0, "U1", vec![Some(sig), Some(sig)])];
    let symbol_pin_maps = HashMap::new();
    let r = erc(&b, &comps, &lib, &symbol_pin_maps);
    assert_eq!(r.power_pin_on_signal, vec![("U1".into(), 0, "CTRL".into())]);
    assert!(!r.pass);
}

#[test]
fn bom_flags_duplicate_refdes() {
    let lib = vec![two_pad_fp()];
    let comps = vec![
        comp(0, "R1", vec![None, None]),
        comp(0, "R1", vec![None, None]),
        comp(0, "R2", vec![None, None]),
    ];
    let r = bom(&comps, &lib);
    assert_eq!(r.part_count, 3);
    assert_eq!(r.duplicate_refdes, vec!["R1".to_string()]);
    assert!(!r.pass);
}

#[test]
fn schematic_isolation_bfs_detects_leakage() {
    let mut b = Board::new(spec());
    let sclk = b.add_net("BUS_SCLK", NetClassKind::Signal);
    let sclk_p = b.add_net("SCLK_P", NetClassKind::Signal);

    let lib = vec![two_pad_fp()];
    let comps_clean = vec![
        comp(0, "J_STACK", vec![Some(sclk)]),
        comp(0, "ISO1", vec![Some(sclk), Some(sclk_p)]),
        comp(0, "U1", vec![Some(sclk_p)]),
    ];
    let r_clean = schematic_isolation_bfs(&b, &comps_clean, &lib);
    assert!(r_clean.pass);

    let comps_dirty = vec![
        comp(0, "J_STACK", vec![Some(sclk)]),
        comp(0, "ISO1", vec![Some(sclk), Some(sclk_p)]),
        comp(0, "R_ERR", vec![Some(sclk), Some(sclk_p)]),
        comp(0, "U1", vec![Some(sclk_p)]),
    ];
    let r_dirty = schematic_isolation_bfs(&b, &comps_dirty, &lib);
    assert!(!r_dirty.pass);
    assert_eq!(r_dirty.violations.len(), 1);
    assert_eq!(r_dirty.violations[0].control_net, "BUS_SCLK");
    assert_eq!(r_dirty.violations[0].hv_net, "SCLK_P");
    assert_eq!(
        r_dirty.violations[0].path,
        vec![
            "BUS_SCLK".to_string(),
            "R_ERR".to_string(),
            "SCLK_P".to_string()
        ]
    );
}

#[test]
fn ac_coupling_flags_low_impedance() {
    let mut b = Board::new(spec());
    let trig = b.add_net("TRIG_0", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(51.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: trig,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.5)),
        end: Point::new(Nm::from_mm(51.0), Nm::from_mm(5.5)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: gnd,
    });

    let r = parasitic_ac_coupling_check(&b);
    assert!(r.pass);

    b.tracks.clear();
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: trig,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.5)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(5.5)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: gnd,
    });

    let r2 = parasitic_ac_coupling_check(&b);
    assert!(!r2.pass);
    assert_eq!(r2.violations.len(), 1);
    assert_eq!(r2.violations[0].switching_net, "TRIG_0");
}
