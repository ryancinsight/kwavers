use super::super::*;

#[test]
fn detects_differential_pair_keepout() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
    use crate::place::rotation::Rot;

    let track = |b: &mut Board, net, y: f64| {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut signal = board();
    let p = signal.add_net("DATA_P", NetClassKind::Signal);
    let n = signal.add_net("DATA_N", NetClassKind::Signal);
    let other = signal.add_net("OTHER", NetClassKind::Signal);
    track(&mut signal, p, 4.0);
    track(&mut signal, n, 4.6);
    track(&mut signal, other, 5.35);
    let r = audit(&signal, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "an unrelated signal inside the 30 mil differential-pair keepout is flagged once"
    );

    let mut clock = board();
    let p = clock.add_net("CLK_P", NetClassKind::Signal);
    let n = clock.add_net("CLK_N", NetClassKind::Signal);
    let other = clock.add_net("OTHER", NetClassKind::Signal);
    track(&mut clock, p, 4.0);
    track(&mut clock, n, 4.6);
    track(&mut clock, other, 5.55);
    let r = audit(&clock, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "clock pairs use the wider 50 mil keepout"
    );

    let mut pair_to_pair = board();
    let ap = pair_to_pair.add_net("A_P", NetClassKind::Signal);
    let an = pair_to_pair.add_net("A_N", NetClassKind::Signal);
    let bp = pair_to_pair.add_net("B_P", NetClassKind::Signal);
    let bn = pair_to_pair.add_net("B_N", NetClassKind::Signal);
    track(&mut pair_to_pair, ap, 4.0);
    track(&mut pair_to_pair, an, 4.6);
    track(&mut pair_to_pair, bp, 5.2);
    track(&mut pair_to_pair, bn, 5.8);
    let r = audit(&pair_to_pair, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "adjacent differential pairs closer than 5W are flagged once per pair relationship"
    );

    let mut component_blocked = board();
    let p = component_blocked.add_net("USB_P", NetClassKind::Signal);
    let n = component_blocked.add_net("USB_N", NetClassKind::Signal);
    let quiet = component_blocked.add_net("GPIO", NetClassKind::Signal);
    track(&mut component_blocked, p, 4.0);
    track(&mut component_blocked, n, 4.6);
    let lib = vec![FootprintDef::new(
        "R_0402",
        (Nm::from_mm(0.8), Nm::from_mm(0.8)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let blocker = Component {
        fp: 0,
        nets: vec![Some(quiet)],
        refdes: "R1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(4.3)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let blocked = audit(&component_blocked, &[blocker], &lib, &DesignRules::holohv());
    assert_eq!(
        blocked.diff_pair_violations, 1,
        "an unrelated component courtyard between P/N members creates a differential-pair obstruction"
    );
    assert!(
        !blocked.hard_drc_clean(),
        "component intrusion between differential-pair members rejects optimizer clean-board selection"
    );
}

#[test]
fn detects_asymmetric_diff_pair_coupling_caps() {
    use crate::place::component::Placement;
    use crate::place::footprint::PadDef;
    use crate::place::rotation::Rot;

    let mut matched = board();
    let p = matched.add_net("MGT_P", NetClassKind::Signal);
    let n = matched.add_net("MGT_N", NetClassKind::Signal);

    for (net, y) in [(p, 4.0), (n, 5.0)] {
        matched.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let cap_fp = FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.25), Nm(0)),
                size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.25), Nm(0)),
                size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![cap_fp];
    let cap = |net, refdes: &str, x, y| Component {
        fp: 0,
        nets: vec![Some(net), None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let matched_comps = vec![cap(p, "C_AC_P", 6.0, 4.0), cap(n, "C_AC_N", 6.3, 5.0)];
    let clean = audit(&matched, &matched_comps, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_coupling_cap_symmetry_violations, 0,
        "P/N coupling caps whose pair-axis stations differ by 0.3 mm stay inside tolerance"
    );

    let shifted_comps = vec![cap(p, "C_AC_P", 6.0, 4.0), cap(n, "C_AC_N", 8.0, 5.0)];
    let shifted = audit(&matched, &shifted_comps, &lib, &DesignRules::holohv());
    assert_eq!(
        shifted.diff_pair_coupling_cap_symmetry_violations, 1,
        "P/N coupling caps shifted by more than 0.5 mm along the pair axis are flagged"
    );

    let one_sided = audit(
        &matched,
        &[cap(p, "C_AC_P", 6.0, 4.0)],
        &lib,
        &DesignRules::holohv(),
    );
    assert_eq!(
        one_sided.diff_pair_coupling_cap_symmetry_violations, 1,
        "a coupling capacitor on only one leg is not symmetric"
    );
}

#[test]
fn detects_oversized_diff_pair_coupling_cap_packages() {
    use crate::place::component::Placement;
    use crate::place::footprint::PadDef;
    use crate::place::rotation::Rot;

    let mut b = board();
    let p = b.add_net("USB_P", NetClassKind::Signal);
    let n = b.add_net("USB_N", NetClassKind::Signal);

    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let cap_fp = |name: &str, w_mm: f64, h_mm: f64| {
        FootprintDef::new(
            name,
            (Nm::from_mm(w_mm), Nm::from_mm(h_mm)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.25), Nm(0)),
                    size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.25), Nm(0)),
                    size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        )
    };
    let cap = |fp, net, refdes: &str, x, y| Component {
        fp,
        nets: vec![Some(net), None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    let clean_lib = vec![cap_fp("C0603", 1.6, 0.8)];
    let clean_comps = vec![cap(0, p, "C_AC_P", 6.0, 4.0), cap(0, n, "C_AC_N", 6.0, 5.0)];
    let clean = audit(&b, &clean_comps, &clean_lib, &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_coupling_cap_symmetry_violations, 0,
        "the symmetric fixture isolates package size from placement symmetry"
    );
    assert_eq!(
        clean.diff_pair_coupling_cap_package_violations, 0,
        "0603-class coupling capacitors stay inside the 1.7 mm package budget"
    );

    let dirty_lib = vec![cap_fp("C0805", 2.0, 1.25)];
    let dirty_comps = vec![cap(0, p, "C_AC_P", 6.0, 4.0), cap(0, n, "C_AC_N", 6.0, 5.0)];
    let dirty = audit(&b, &dirty_comps, &dirty_lib, &DesignRules::holohv());
    assert_eq!(
        dirty.diff_pair_coupling_cap_symmetry_violations, 0,
        "oversized but symmetric coupling capacitors are not a symmetry violation"
    );
    assert_eq!(
        dirty.diff_pair_coupling_cap_package_violations, 2,
        "both 0805-class coupling capacitors exceed the 0603-class courtyard budget"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "oversized differential-pair coupling capacitors must reject optimizer clean-board selection"
    );
}
