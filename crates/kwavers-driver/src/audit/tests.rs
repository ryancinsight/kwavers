use super::*;
use crate::board::{Board, LayerId, NetClassKind, NetId, Pad, Track, Via, ViaKind, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::audit::detect_power::{detect_charge_recycling_violations_board, detect_pulse_skip_violations};
use crate::rules::DesignRules;

    fn board() -> Board {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        Board::new(spec)
    }

    #[test]
    fn hard_drc_clean_covers_optimizer_rejection_fields() {
        let clean = FaultReport::default();
        assert!(
            clean.hard_drc_clean(),
            "a default report has no hard DRC faults"
        );

        type FaultMarker = fn(&mut FaultReport);
        let dirty_fields: [(&str, FaultMarker); 64] = [
            ("clearance violations", |r: &mut FaultReport| {
                r.clearance_violations = 1;
            }),
            ("via adjacency", |r: &mut FaultReport| {
                r.via_adjacency = 1;
            }),
            ("acid traps", |r: &mut FaultReport| {
                r.acid_traps = 1;
            }),
            ("dangling", |r: &mut FaultReport| r.dangling = 1),
            ("sharp bend", |r: &mut FaultReport| r.sharp_bends = 1),
            ("track crossing", |r: &mut FaultReport| {
                r.track_crossing_violations = 1;
            }),
            ("hole clearance", |r: &mut FaultReport| {
                r.hole_clearance_violations = 1;
            }),
            ("serpentine spacing", |r: &mut FaultReport| {
                r.serpentine_spacing_violations = 1;
            }),
            ("serpentine length", |r: &mut FaultReport| {
                r.serpentine_length_violations = 1;
            }),
            ("serpentine compensation distance", |r: &mut FaultReport| {
                r.serpentine_compensation_distance_violations = 1;
            }),
            ("different-net via spacing", |r: &mut FaultReport| {
                r.via_spacing_violations = 1;
            }),
            ("plane hotspot via spacing", |r: &mut FaultReport| {
                r.plane_hotspot_via_spacing_violations = 1;
            }),
            ("diff-pair violation", |r: &mut FaultReport| {
                r.diff_pair_violations = 1;
            }),
            ("diff-pair layer mismatch", |r: &mut FaultReport| {
                r.diff_pair_layer_mismatch_violations = 1;
            }),
            (
                "diff-pair interface layer mismatch",
                |r: &mut FaultReport| {
                    r.diff_pair_interface_layer_mismatch_violations = 1;
                },
            ),
            (
                "diff-pair interface via count mismatch",
                |r: &mut FaultReport| {
                    r.diff_pair_interface_via_count_mismatch_violations = 1;
                },
            ),
            ("diff-pair via count", |r: &mut FaultReport| {
                r.diff_pair_via_count_violations = 1;
            }),
            ("diff-pair length mismatch", |r: &mut FaultReport| {
                r.diff_pair_length_mismatch_violations = 1;
            }),
            (
                "diff-pair segment length mismatch",
                |r: &mut FaultReport| {
                    r.diff_pair_segment_length_mismatch_violations = 1;
                },
            ),
            ("parallel bus length mismatch", |r: &mut FaultReport| {
                r.parallel_bus_length_mismatch_violations = 1;
            }),
            ("diff-pair spacing variation", |r: &mut FaultReport| {
                r.diff_pair_spacing_variation_violations = 1;
            }),
            ("diff-pair via symmetry", |r: &mut FaultReport| {
                r.diff_pair_via_symmetry_violations = 1;
            }),
            ("diff-pair coupling cap symmetry", |r: &mut FaultReport| {
                r.diff_pair_coupling_cap_symmetry_violations = 1;
            }),
            ("diff-pair coupling cap package", |r: &mut FaultReport| {
                r.diff_pair_coupling_cap_package_violations = 1;
            }),
            ("diff-pair stitching cap symmetry", |r: &mut FaultReport| {
                r.diff_pair_stitching_cap_symmetry_violations = 1;
            }),
            ("diff-pair pad entry mismatch", |r: &mut FaultReport| {
                r.diff_pair_pad_entry_mismatch_violations = 1;
            }),
            ("diff-pair pad entry length", |r: &mut FaultReport| {
                r.diff_pair_pad_entry_length_violations = 1;
            }),
            ("diff-pair keepout", |r: &mut FaultReport| {
                r.diff_pair_keepout_violations = 1;
            }),
            ("high-speed edge", |r: &mut FaultReport| {
                r.high_speed_edge_violations = 1;
            }),
            ("high-speed component edge", |r: &mut FaultReport| {
                r.high_speed_component_edge_violations = 1;
            }),
            ("high-speed termination placement", |r: &mut FaultReport| {
                r.high_speed_termination_placement_violations = 1;
            }),
            ("high-speed parallel spacing", |r: &mut FaultReport| {
                r.high_speed_parallel_spacing_violations = 1;
            }),
            (
                "high-speed adjacent-layer parallel",
                |r: &mut FaultReport| {
                    r.high_speed_adjacent_layer_parallel_violations = 1;
                },
            ),
            ("reference plane margin", |r: &mut FaultReport| {
                r.reference_plane_margin_violations = 1;
            }),
            ("reference plane absence", |r: &mut FaultReport| {
                r.reference_plane_absence_violations = 1;
            }),
            (
                "inner layer dual ground reference",
                |r: &mut FaultReport| {
                    r.inner_layer_dual_ground_reference_violations = 1;
                },
            ),
            ("power reference stitching cap", |r: &mut FaultReport| {
                r.power_reference_stitching_cap_violations = 1;
            }),
            ("reference plane intrusion", |r: &mut FaultReport| {
                r.reference_plane_intrusion_violations = 1;
            }),
            ("ground plane fragmentation", |r: &mut FaultReport| {
                r.ground_plane_fragmentation_violations = 1;
            }),
            ("split-domain reference", |r: &mut FaultReport| {
                r.split_domain_reference_violations = 1;
            }),
            ("mixed-domain shared reference", |r: &mut FaultReport| {
                r.mixed_domain_shared_reference_violations = 1;
            }),
            ("virtual split crossing", |r: &mut FaultReport| {
                r.virtual_split_crossing_violations = 1;
            }),
            ("high-speed stub", |r: &mut FaultReport| {
                r.high_speed_stub_violations = 1;
            }),
            ("high-speed transition return", |r: &mut FaultReport| {
                r.high_speed_transition_ground_via_violations = 1;
            }),
            (
                "diff-pair transition return symmetry",
                |r: &mut FaultReport| {
                    r.diff_pair_transition_ground_via_symmetry_violations = 1;
                },
            ),
            ("high-speed terminal ground via", |r: &mut FaultReport| {
                r.high_speed_terminal_ground_via_violations = 1;
            }),
            ("high-speed via pad proximity", |r: &mut FaultReport| {
                r.high_speed_via_pad_proximity_violations = 1;
            }),
            ("high-speed via diameter", |r: &mut FaultReport| {
                r.high_speed_via_diameter_violations = 1;
            }),
            ("blind/buried via drill", |r: &mut FaultReport| {
                r.blind_buried_via_drill_violations = 1;
            }),
            ("microvia aspect", |r: &mut FaultReport| {
                r.microvia_aspect_violations = 1;
            }),
            ("decoupling ground via", |r: &mut FaultReport| {
                r.decoupling_ground_via_violations = 1;
            }),
            ("decoupling power layer", |r: &mut FaultReport| {
                r.decoupling_power_layer_violations = 1;
            }),
            ("decoupling loop area", |r: &mut FaultReport| {
                r.decoupling_loop_area_violations = 1;
            }),
            ("active IC power plane", |r: &mut FaultReport| {
                r.active_ic_power_plane_violations = 1;
            }),
            ("charge reservoir", |r: &mut FaultReport| {
                r.charge_reservoir_violations = 1;
            }),
            ("high-speed via stub", |r: &mut FaultReport| {
                r.high_speed_via_stub_violations = 1;
            }),
            ("unfilled via in pad", |r: &mut FaultReport| {
                r.unfilled_via_in_pad_violations = 1;
            }),
            ("surge suppressor via", |r: &mut FaultReport| {
                r.surge_suppressor_via_violations = 1;
            }),
            ("split-plane crossing", |r: &mut FaultReport| {
                r.split_plane_crossings = 1;
            }),
            ("transmission line length", |r: &mut FaultReport| {
                r.transmission_line_length_violations = 1;
            }),
            ("decoupling cap distance", |r: &mut FaultReport| {
                r.decoupling_cap_distance_violations = 1;
            }),
            ("antenna impedance", |r: &mut FaultReport| {
                r.antenna_impedance_violations = 1;
            }),
            ("ground via stitching", |r: &mut FaultReport| {
                r.ground_via_stitching_violations = 1;
            }),
            ("via return count", |r: &mut FaultReport| {
                r.high_speed_via_return_count_violations = 1;
            }),
        ];

        for (name, mark_dirty) in dirty_fields {
            let mut report = FaultReport::default();
            mark_dirty(&mut report);
            assert!(
                !report.hard_drc_clean(),
                "{name} must reject optimizer clean-board selection"
            );
        }
    }

    /// Meta-test: every `self.X == 0` clause inside `hard_drc_clean` (the optimizer-rejection
    /// gate) must have a matching `r.X = N` set inside the `dirty_fields` array of
    /// `hard_drc_clean_covers_optimizer_rejection_fields`. Parses `audit.rs` at build time via
    /// `include_str!` so adding a clause to `hard_drc_clean` -- or removing one from
    /// `dirty_fields` -- fails this assertion even if every other test passes.
    ///
    /// Hand-rolled prefix scans rather than regex to keep the test robust against audit.rs
    /// formatting drift. The two scans are intentionally tolerant: any line of hard_drc_clean's
    /// body beginning with `&& self.` or `self.` followed by an identifier and ` == 0`; any
    /// line inside the dirty_fields array containing `r.` followed by an identifier and `=`. The
    /// failure message names every missing field so the maintainer can extend dirty_fields in
    /// one shot.
    #[test]
    fn dirty_fields_mirrors_every_hard_drc_clean_clause() {
        use std::collections::BTreeSet;
        const SRC: &str = include_str!("fault_report.rs");
        // Post-carve, the dirty_fields array lives in tests.rs itself — meta-test scans its own
        // source so a future file move doesn't silently bypass the gate.
        const SRC_DF: &str = include_str!("tests.rs");

        // Step 1: every `self.X == 0` clause inside the hard_drc_clean body.
        let lines: Vec<&str> = SRC.lines().collect();
        let mut hard_fields = BTreeSet::new();
        let mut in_hard_drc = false;
        let mut brace_depth: i32 = 0;
        for line in &lines {
            let t = line.trim();
            if !in_hard_drc {
                if t.starts_with("pub(crate) fn hard_drc_clean(")
                    || t.starts_with("pub fn hard_drc_clean(")
                {
                    in_hard_drc = true;
                    brace_depth += t.matches('{').count() as i32 - t.matches('}').count() as i32;
                }
                continue;
            }
            brace_depth += t.matches('{').count() as i32 - t.matches('}').count() as i32;
            if brace_depth <= 0 {
                break;
            }
            if brace_depth != 1 {
                continue; // ignore nested-closure clauses inside hard_drc_clean
            }
            if t.starts_with("//") || t.starts_with("/*") {
                continue; // ignore Rust comments (line or block) inside hard_drc_clean body
            }
            if !t.contains(" == 0") {
                continue;
            }
            // First clause: `self.X == 0`. Continuation: `&& self.X == 0`.
            let rest = if let Some(r) = t.strip_prefix("&& self.") {
                r
            } else if let Some(r) = t.strip_prefix("self.") {
                r
            } else {
                continue;
            };
            let end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            let field = rest[..end].trim_end_matches([' ', '\t']);
            if !field.is_empty() {
                hard_fields.insert(field.to_string());
            }
        }

        // Step 2: every `r.X = N` set inside the dirty_fields array body. Post-carve the array
        // lives in `tests.rs` (this file), so we re-read SRC_DF instead of `fault_report.rs`.
        let df_lines: Vec<&str> = SRC_DF.lines().collect();
        let start = df_lines
            .iter()
            .position(|l| {
                let t = l.trim_start();
                (t.starts_with("let ") || t.starts_with("let mut "))
                    && t.contains("dirty_fields:")
                    && t.contains("= [")
            })
            .expect("dirty_fields array exists");
        let end = df_lines
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, l)| l.trim() == "];")
            .map(|(i, _)| i)
            .expect("dirty_fields terminates with ];");
        let array_text = df_lines[start..=end + 1].join("\n");

        let mut dirty_field_sets = BTreeSet::new();
        for line in array_text.lines() {
            let t_line = line.trim_start();
            if t_line.starts_with("//") {
                continue;
            }
            // Find the first assignment target substring. This handles one-line closures such as
            // `("dangling", |r| r.dangling = 1)` and multi-line bodies with `r.X = 1;`.
            let Some(rel_idx) = line.find("r.") else {
                continue;
            };
            let after_r = &line[rel_idx + 2..];
            let Some(eq_idx) = after_r.find('=') else {
                continue;
            };
            let before = after_r[..eq_idx].trim_end_matches([' ', '\t']);
            let end = before
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(before.len());
            let field = before[..end].trim_end_matches([' ', '\t']);
            if !field.is_empty() {
                dirty_field_sets.insert(field.to_string());
            }
        }

        // Step 3: diff and report any missing entries.
        let missing: Vec<String> = hard_fields
            .iter()
            .filter(|f| !dirty_field_sets.contains(*f))
            .cloned()
            .collect();
        assert!(
            missing.is_empty(),
            "every `self.X == 0` clause in hard_drc_clean must have a matching `r.X = N` set              inside dirty_fields (this file, near `hard_drc_clean_covers_optimizer_rejection_fields`)              so adding a clause to hard_drc_clean cannot silently bypass the optimizer-rejection              test. Missing entries: {missing:?}. Extend dirty_fields in that test to add one entry              per missing field, setting that field to 1."
        );
    }

    #[test]
    fn serpentine_spacing_uses_edge_gap_not_centerline_gap() {
        let mut b = board();
        let sig = b.add_net("SERP", NetClassKind::Signal);
        let make_segment = |y_mm| Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y_mm)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y_mm)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: sig,
        };

        b.tracks.push(make_segment(4.0));
        b.tracks.push(make_segment(4.675));
        let edge_violation = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            edge_violation.serpentine_spacing_violations, 1,
            "0.675 mm centerline spacing is 4.5W, but the 0.525 mm copper edge gap is below 4W"
        );

        b.tracks[1] = make_segment(4.75);
        let edge_clear = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            edge_clear.serpentine_spacing_violations, 0,
            "0.75 mm centerline spacing gives the required 0.6 mm copper edge gap for 0.15 mm traces"
        );
    }

    #[test]
    fn serpentine_compensation_must_stay_near_bend_root() {
        let spec =
            GridSpec::cover(Nm::from_mm(70.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut remote = Board::new(spec);
        let net = remote.add_net("SERP_REMOTE", NetClassKind::Signal);
        let h = |x0, x1, y| Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        };
        let v = |x, y0, y1| Track {
            start: Point::new(Nm::from_mm(x), Nm::from_mm(y0)),
            end: Point::new(Nm::from_mm(x), Nm::from_mm(y1)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        };
        remote.tracks.push(h(2.0, 50.0, 10.0));
        remote.tracks.push(v(2.0, 10.0, 10.8));
        remote.tracks.push(h(2.0, 50.0, 10.8));
        remote.tracks.push(v(50.0, 10.0, 10.8));

        let remote_report = audit(&remote, &[], &[], &DesignRules::holohv());
        assert_eq!(
            remote_report.serpentine_spacing_violations, 0,
            "0.65 mm edge gap is wider than the 4W spacing budget for 0.15 mm traces"
        );
        assert_eq!(
            remote_report.serpentine_compensation_distance_violations, 1,
            "the parallel compensation midpoint is 24 mm from the nearest bend, above the 15 mm guide budget"
        );

        let mut local = Board::new(spec);
        let net = local.add_net("SERP_LOCAL", NetClassKind::Signal);
        let h = |x0, x1, y| Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        };
        let v = |x, y0, y1| Track {
            start: Point::new(Nm::from_mm(x), Nm::from_mm(y0)),
            end: Point::new(Nm::from_mm(x), Nm::from_mm(y1)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        };
        local.tracks.push(h(2.0, 20.0, 10.0));
        local.tracks.push(v(2.0, 10.0, 10.8));
        local.tracks.push(h(2.0, 20.0, 10.8));
        local.tracks.push(v(20.0, 10.0, 10.8));

        let local_report = audit(&local, &[], &[], &DesignRules::holohv());
        assert_eq!(
            local_report.serpentine_compensation_distance_violations, 0,
            "the local compensation midpoint is 9 mm from a bend, inside the 15 mm guide budget"
        );
    }

    #[test]
    fn sharp_bend_detection_rejects_acute_bends_and_accepts_one_thirty_five() {
        let mut acute = board();
        let net = acute.add_net("TX_ACUTE", NetClassKind::Signal);
        acute.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        acute.tracks.push(Track {
            start: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        let acute_report = audit(&acute, &[], &[], &DesignRules::holohv());
        assert_eq!(
            acute_report.sharp_bends, 1,
            "a 45 degree same-net bend is sharper than the guide's 135 degree routing geometry"
        );

        let mut obtuse = board();
        let net = obtuse.add_net("TX_OBTUSE", NetClassKind::Signal);
        obtuse.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        obtuse.tracks.push(Track {
            start: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        let obtuse_report = audit(&obtuse, &[], &[], &DesignRules::holohv());
        assert_eq!(
            obtuse_report.sharp_bends, 0,
            "a 135 degree same-net bend follows the guide's preferred bend geometry"
        );
    }

    #[test]
    fn track_crossing_detection_rejects_opposed_diagonals_inside_one_cell() {
        let mut dirty = board();
        let a = dirty.add_net("A", NetClassKind::Signal);
        let b = dirty.add_net("B", NetClassKind::Signal);
        dirty.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(3.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: a,
        });
        dirty.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(3.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: b,
        });

        let dirty_report = audit(&dirty, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty_report.track_crossing_violations, 1,
            "different-net diagonal tracks crossing inside a grid cell are a KiCad DRC failure"
        );
        assert!(
            !dirty_report.hard_drc_clean(),
            "track crossings must reject optimizer clean-board selection"
        );

        let mut clean = board();
        let a = clean.add_net("A", NetClassKind::Signal);
        let b = clean.add_net("B", NetClassKind::Signal);
        clean.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(3.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: a,
        });
        clean.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(3.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: b,
        });

        let clean_report = audit(&clean, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean_report.track_crossing_violations, 0,
            "crossing geometry on different copper layers is not a same-layer DRC crossing"
        );
    }

    #[test]
    fn emi_hotspots_flags_only_hv_to_lv_pairs() {
        let mut b = board();
        let hv = b.add_net("VPP", NetClassKind::Hv);
        let lv = b.add_net("CTRL", NetClassKind::Signal);
        let lv2 = b.add_net("CLK", NetClassKind::Signal);
        let pad = |b: &mut Board, x, y, n| {
            b.add_pad(Pad {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                layers: vec![LayerId(0)],
                net: Some(n),
            })
        };
        pad(&mut b, 5.0, 5.0, hv);
        pad(&mut b, 7.0, 5.0, lv); // 2 mm from HV ⇒ one HV↔LV hotspot (within 6 mm)
        pad(&mut b, 8.0, 5.0, lv2); // LV↔LV with the other LV ⇒ ignored; 3 mm from HV ⇒ also a hotspot
        pad(&mut b, 19.0, 19.0, lv); // far from HV ⇒ no hotspot
        let pts = emi_hotspots(&b, Nm::from_mm(6.0));
        // HV pad pairs with the two near LV pads (2 mm, 3 mm) ⇒ 2 hotspots; LV↔LV and the far LV none.
        assert_eq!(pts.len(), 2, "only HV↔LV pairs within 6 mm count");
    }

    #[test]
    fn copper_imbalance_is_symmetric_pair_and_counts_planes() {
        use crate::board::{Zone, ZoneFill};
        // 4-layer board; a big plane polygon (a square) we can drop on chosen layers.
        let mk = |spec: GridSpec, gl: u16, vl: u16| {
            let mut b = Board::new(spec);
            let g = b.add_net("GND", NetClassKind::Ground);
            let v = b.add_net("VPP", NetClassKind::Hv);
            let sq = vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
            ];
            for (net, layer) in [(g, gl), (v, vl)] {
                b.zones.push(Zone {
                    net,
                    layer: LayerId(layer),
                    polygon: sq.clone(),
                    fill: ZoneFill::ThermalRelief,
                });
            }
            b
        };
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
        // Planes on the symmetric inner pair (1,2): pairs (0,3) empty↔empty and (1,2) plane↔plane —
        // both matched ⇒ balanced.
        assert!(
            copper_imbalance(&mk(spec, 1, 2)) < 0.01,
            "symmetric plane placement is warp-balanced"
        );
        // Planes on adjacent layers (2,3): pair (0,3) empty↔plane and (1,2) empty↔plane — both
        // one-sided ⇒ imbalanced.
        assert!(
            copper_imbalance(&mk(spec, 2, 3)) > 0.9,
            "planes on a non-symmetric pair are imbalanced"
        );
    }

    #[test]
    fn copper_imbalance_flags_single_layer_routing() {
        use crate::board::{NetClassKind, Track};
        let mut b = board();
        let n = b.add_net("N", NetClassKind::Signal);
        // All copper on layer 0 ⇒ strong imbalance (other layers empty).
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
            width: Nm::from_mm(0.3),
            layer: LayerId(0),
            net: n,
        });
        assert!(
            copper_imbalance(&b) > 0.9,
            "single-layer copper must read as imbalanced"
        );
        // Mirror it onto layer 1 ⇒ balanced.
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
            width: Nm::from_mm(0.3),
            layer: LayerId(1),
            net: n,
        });
        assert!(
            copper_imbalance(&b) < 0.01,
            "mirrored copper must read as balanced"
        );
    }

    #[test]
    fn detects_crossing_flight_lines() {
        let mut b = board();
        let a = b.add_net("A", NetClassKind::Signal);
        let c = b.add_net("B", NetClassKind::Signal);
        let pad = |b: &mut Board, x, y, n| {
            b.add_pad(Pad {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                layers: vec![LayerId(0)],
                net: Some(n),
            })
        };
        // Net A: left→right across the middle. Net B: bottom→top across the middle. They cross.
        pad(&mut b, 0.0, 10.0, a);
        pad(&mut b, 20.0, 10.0, a);
        pad(&mut b, 10.0, 0.0, c);
        pad(&mut b, 10.0, 20.0, c);
        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.crossings, 1,
            "the two flight lines must cross exactly once"
        );
    }

    #[test]
    fn detects_near_short_and_weights_hv() {
        let mut b = board();
        let hv = b.add_net("VPP", NetClassKind::Hv);
        let lv = b.add_net("CTRL", NetClassKind::Signal);
        // Two pads of different nets 0.2 mm apart (< 3*0.13 = 0.39 mm margin).
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(hv),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(5.2), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(lv),
        });
        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(r.near_shorts, 1, "the close HV/LV pad pair is a near-short");
        assert!(r.risk_score > 0.0);
    }

    #[test]
    fn detects_crossing_diagonal_tracks_of_different_nets() {
        // The dominant real-board failure: two 45° tracks of different nets crossing on one layer.
        // Their centre-lines intersect ⇒ edge-gap is negative ⇒ a hard clearance violation that
        // kicad-cli flags as `tracks_crossing` + `clearance`. The audit must see it too, else the
        // cooptimize judge is blind to it and never optimises it out.
        let mut b = board();
        let a = b.add_net("A", NetClassKind::Signal);
        let c = b.add_net("C", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(6.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: a,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(5.0), Nm::from_mm(6.0)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: c,
        });
        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert!(
            r.clearance_violations >= 1,
            "crossing diagonal tracks of different nets must register as a clearance violation, got {}",
            r.clearance_violations
        );
    }

    #[test]
    fn detects_via_hole_too_close_to_foreign_track() {
        // A via whose drilled barrel passes within the hole-to-copper clearance of a foreign-net
        // track on a layer the barrel spans — kicad-cli's `hole_clearance` class, previously
        // unmodelled by the audit so escape-via boards passed internally but failed externally.
        let mut b = board();
        let n = b.add_net("N", NetClassKind::Signal);
        let f = b.add_net("F", NetClassKind::Signal);
        let rules = DesignRules::holohv();
        // Via barrel F.Cu..In1 at (5,5), drill 0.3 mm (radius 0.15).
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            net: n,
            from: LayerId(0),
            to: LayerId(1),
            kind: crate::board::ViaKind::Blind,
            filled: false,
        });
        // Foreign track on In1 (layer 1, in the barrel span) passing 0.2 mm from the via centre:
        // hole-edge gap = 0.2 − 0.15(drill) − 0.075(half-width) = −0.025 mm < clearance ⇒ violation.
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.2)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.2)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: f,
        });
        let r = audit(&b, &[], &[], &rules);
        assert_eq!(
            r.hole_clearance_violations, 1,
            "via hole within clearance of a foreign track on a spanned layer is a violation"
        );
        // A same-net track at the same spot is the via's own connection — never a hole violation.
        let mut b2 = board();
        let n2 = b2.add_net("N", NetClassKind::Signal);
        b2.vias.push(Via {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            net: n2,
            from: LayerId(0),
            to: LayerId(1),
            kind: crate::board::ViaKind::Blind,
            filled: false,
        });
        b2.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.2)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.2)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: n2,
        });
        assert_eq!(
            audit(&b2, &[], &[], &rules).hole_clearance_violations,
            0,
            "same-net copper at the via is its own connection, not a hole violation"
        );
    }

    #[test]
    fn detects_dangling_track_end() {
        let mut b = board();
        let n = b.add_net("N", NetClassKind::Signal);
        // A track whose start is on a pad but whose end floats in space.
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            layers: vec![LayerId(0)],
            net: Some(n),
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });
        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(r.dangling, 1, "the floating track end is an antenna");
    }

    #[test]
    fn detects_unsplit_track_body_junction_as_dangling() {
        let mut b = board();
        let n = b.add_net("N", NetClassKind::Signal);
        for pos in [
            Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(3.0), Nm::from_mm(4.0)),
        ] {
            b.add_pad(Pad {
                pos,
                layers: vec![LayerId(0)],
                net: Some(n),
            });
        }
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(3.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });

        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.dangling, 1,
            "an endpoint on an unsplit same-net track body still emits a KiCad track_dangling warning"
        );
    }

    #[test]
    fn detects_high_speed_stub_branch() {
        let mut branched = board();
        let tx = branched.add_net("TX_STUB", NetClassKind::Signal);
        let node = Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0));
        for (start, end) in [
            (Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)), node),
            (node, Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0))),
            (node, Point::new(Nm::from_mm(8.0), Nm::from_mm(12.0))),
        ] {
            branched.tracks.push(Track {
                start,
                end,
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net: tx,
            });
        }
        let r = audit(&branched, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.high_speed_stub_violations, 1,
            "one TX T-junction is a connected high-speed stub"
        );

        let mut daisy = board();
        let tx2 = daisy.add_net("TX_DAISY", NetClassKind::Signal);
        daisy.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx2,
        });
        daisy.tracks.push(Track {
            start: Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx2,
        });
        let clean = audit(&daisy, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.high_speed_stub_violations, 0,
            "a two-segment daisy-chain has no branch node"
        );
    }

    #[test]
    fn detects_differential_pair_layer_and_via_mismatch() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let p = b.add_net("CLK_P", NetClassKind::Signal);
        let n = b.add_net("CLK_N", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: p,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: n,
        });
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(4.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net: p,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });

        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.diff_pair_layer_mismatch_violations, 1,
            "pair members routed on different layer sets must be flagged"
        );
        assert_eq!(
            r.diff_pair_via_count_violations, 1,
            "pair members with different via counts must be flagged"
        );
    }

    #[test]
    fn detects_same_interface_diff_pair_layer_mismatch() {
        use crate::board::{Via, ViaKind};

        let mut clean = board();
        let d0p = clean.add_net("MIPI_D0_P", NetClassKind::Signal);
        let d0n = clean.add_net("MIPI_D0_N", NetClassKind::Signal);
        let d1p = clean.add_net("MIPI_D1_P", NetClassKind::Signal);
        let d1n = clean.add_net("MIPI_D1_N", NetClassKind::Signal);
        let auxp = clean.add_net("HDMI_AUX_P", NetClassKind::Signal);
        let auxn = clean.add_net("HDMI_AUX_N", NetClassKind::Signal);
        for (net, y) in [
            (d0p, 4.0),
            (d0n, 5.0),
            (d1p, 7.0),
            (d1n, 8.0),
            (auxp, 11.0),
            (auxn, 12.0),
        ] {
            clean.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(8.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let clean_report = audit(&clean, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean_report.diff_pair_layer_mismatch_violations, 0,
            "each pair is internally routed on one layer set"
        );
        assert_eq!(
            clean_report.diff_pair_interface_layer_mismatch_violations, 0,
            "same-interface MIPI_D0/D1 pairs share the same layer set"
        );
        assert_eq!(
            clean_report.diff_pair_interface_via_count_mismatch_violations, 0,
            "same-interface MIPI_D0/D1 pairs use the same total via count"
        );

        let mut dirty = clean.clone();
        for track in dirty
            .tracks
            .iter_mut()
            .filter(|track| track.net == d1p || track.net == d1n)
        {
            track.layer = LayerId(1);
        }
        let dirty_report = audit(&dirty, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty_report.diff_pair_layer_mismatch_violations, 0,
            "each individual pair still has matched P/N layer sets"
        );
        assert_eq!(
            dirty_report.diff_pair_interface_layer_mismatch_violations, 1,
            "MIPI_D1 routed on a different layer set than MIPI_D0 violates same-interface routing"
        );
        assert_eq!(
            dirty_report.diff_pair_interface_via_count_mismatch_violations, 0,
            "the layer-mismatch fixture keeps total via counts matched"
        );
        assert!(
            !dirty_report.hard_drc_clean(),
            "same-interface differential-pair layer mismatch rejects clean-board selection"
        );

        let mut via_mismatched = clean;
        for (net, x, y) in [(d1p, 4.0, 7.0), (d1n, 4.0, 8.0)] {
            via_mismatched.vias.push(Via {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                drill: Nm::from_mm(0.2),
                diameter: Nm::from_mm(0.46),
                net,
                from: LayerId(0),
                to: LayerId(1),
                kind: ViaKind::Micro,
                filled: false,
            });
        }
        let via_report = audit(&via_mismatched, &[], &[], &DesignRules::holohv());
        assert_eq!(
            via_report.diff_pair_layer_mismatch_violations, 0,
            "each individual pair still has matched P/N layer sets"
        );
        assert_eq!(
            via_report.diff_pair_via_count_violations, 0,
            "D1 P/N members use the same via count, so the per-pair via rule stays clean"
        );
        assert_eq!(
            via_report.diff_pair_interface_layer_mismatch_violations, 0,
            "all interface pairs still route on the same layer set"
        );
        assert_eq!(
            via_report.diff_pair_interface_via_count_mismatch_violations, 1,
            "MIPI_D1 using more total vias than MIPI_D0 violates same-interface matching"
        );
        assert!(
            !via_report.hard_drc_clean(),
            "same-interface differential-pair via-count mismatch rejects clean-board selection"
        );
    }

    #[test]
    fn detects_differential_pair_via_station_mismatch() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let p = b.add_net("MIPI_P", NetClassKind::Signal);
        let n = b.add_net("MIPI_N", NetClassKind::Signal);
        for (net, y) in [(p, 4.0), (n, 5.0)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        for (net, x, y) in [(p, 6.0, 4.0), (n, 6.3, 5.0)] {
            b.vias.push(Via {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                drill: Nm::from_mm(0.2),
                diameter: Nm::from_mm(0.46),
                net,
                from: LayerId(0),
                to: LayerId(1),
                kind: ViaKind::Micro,
                filled: false,
            });
        }
        let matched = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            matched.diff_pair_via_count_violations, 0,
            "both pair members use one via"
        );
        assert_eq!(
            matched.diff_pair_via_symmetry_violations, 0,
            "0.3 mm via station mismatch stays inside the 0.5 mm tolerance"
        );

        b.vias[1].pos.x = Nm::from_mm(8.0);
        let shifted = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            shifted.diff_pair_via_count_violations, 0,
            "equal via count alone is not enough to prove via symmetry"
        );
        assert_eq!(
            shifted.diff_pair_via_symmetry_violations, 1,
            "P/N vias at different routing stations are flagged"
        );
    }

    #[test]
    fn detects_differential_pair_length_mismatch() {
        let mut mismatched = board();
        let p = mismatched.add_net("DATA_P", NetClassKind::Signal);
        let n = mismatched.add_net("DATA_N", NetClassKind::Signal);
        mismatched.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: p,
        });
        mismatched.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });
        let dirty = audit(&mismatched, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.diff_pair_length_mismatch_violations, 1,
            "pair members whose routed lengths differ by more than tolerance are flagged"
        );

        let mut matched = board();
        let p2 = matched.add_net("ADDR_P", NetClassKind::Signal);
        let n2 = matched.add_net("ADDR_N", NetClassKind::Signal);
        for (net, y, end_x) in [(p2, 4.0, 8.0), (n2, 5.0, 8.3)] {
            matched.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(end_x), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let clean = audit(&matched, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.diff_pair_length_mismatch_violations, 0,
            "0.3 mm intra-pair mismatch stays inside the 0.5 mm tolerance"
        );
    }

    #[test]
    fn detects_differential_pair_segment_length_mismatch() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let p = b.add_net("LANE_P", NetClassKind::Signal);
        let n = b.add_net("LANE_N", NetClassKind::Signal);
        for (net, y, layer0_end_x, layer1_start_x) in [(p, 4.0, 8.0, 8.0), (n, 5.0, 4.0, 4.0)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(layer0_end_x), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(layer1_start_x), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(1),
                net,
            });
            b.vias.push(Via {
                pos: Point::new(Nm::from_mm(layer0_end_x), Nm::from_mm(y)),
                drill: Nm::from_mm(0.2),
                diameter: Nm::from_mm(0.46),
                net,
                from: LayerId(0),
                to: LayerId(1),
                kind: ViaKind::Micro,
                filled: false,
            });
        }

        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.diff_pair_length_mismatch_violations, 0,
            "total routed length remains matched"
        );
        assert_eq!(
            r.diff_pair_via_count_violations, 0,
            "both pair members use the same via count"
        );
        assert_eq!(
            r.diff_pair_segment_length_mismatch_violations, 1,
            "per-layer differential-pair segment skew is flagged even when total length matches"
        );
    }

    #[test]
    fn detects_parallel_bus_length_mismatch() {
        let mut b = board();
        let d0 = b.add_net("BUS_D0", NetClassKind::Signal);
        let d1 = b.add_net("BUS_D1", NetClassKind::Signal);
        let tx0 = b.add_net("TX_0", NetClassKind::Signal);
        let tx1 = b.add_net("TX_1", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(2.0)),
            end: Point::new(Nm::from_mm(9.0), Nm::from_mm(2.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: d0,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(11.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: d1,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(8.0)),
            end: Point::new(Nm::from_mm(3.0), Nm::from_mm(8.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx0,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx1,
        });

        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.parallel_bus_length_mismatch_violations, 0,
            "BUS_D0/BUS_D1 differ by exactly the 2 mm bus-skew budget, and TX_0/TX_1 are not bus-grouped"
        );

        b.tracks[1].end = Point::new(Nm::from_mm(11.5), Nm::from_mm(4.0));
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.parallel_bus_length_mismatch_violations, 1,
            "BUS_D0/BUS_D1 differ by 2.5 mm, exceeding the configured parallel-bus skew budget"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "parallel bus skew must reject optimizer clean-board selection"
        );
    }

    #[test]
    fn detects_differential_pair_pad_entry_mismatch() {
        let mut b = board();
        let p = b.add_net("PAD_ENTRY_P", NetClassKind::Signal);
        let n = b.add_net("PAD_ENTRY_N", NetClassKind::Signal);
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            layers: vec![LayerId(0)],
            net: Some(p),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(n),
        });

        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.2), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: p,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.4), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(10.2), Nm::from_mm(5.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.diff_pair_pad_entry_mismatch_violations, 0,
            "0.2 mm P/N pad-entry mismatch stays inside the 0.5 mm local budget"
        );
        assert_eq!(
            clean.diff_pair_length_mismatch_violations, 0,
            "the fixture keeps total routed P/N length matched"
        );

        b.tracks[1].start = Point::new(Nm::from_mm(3.0), Nm::from_mm(5.0));
        b.tracks[1].end = Point::new(Nm::from_mm(10.8), Nm::from_mm(5.0));
        let mismatched = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            mismatched.diff_pair_pad_entry_mismatch_violations, 1,
            "1.0 mm versus 0.2 mm pad-entry breakout is flagged independently of total length"
        );
        assert_eq!(
            mismatched.diff_pair_length_mismatch_violations, 0,
            "the over-budget pad-entry mismatch is not a total route-length mismatch"
        );
    }

    #[test]
    fn detects_differential_pair_pad_entry_length() {
        let mut b = board();
        let p = b.add_net("LONG_ENTRY_P", NetClassKind::Signal);
        let n = b.add_net("LONG_ENTRY_N", NetClassKind::Signal);
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            layers: vec![LayerId(0)],
            net: Some(p),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(n),
        });
        for (net, y) in [(p, 4.0), (n, 5.0)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(3.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.diff_pair_pad_entry_mismatch_violations, 0,
            "equal 1 mm pad entries are symmetric"
        );
        assert_eq!(
            clean.diff_pair_pad_entry_length_violations, 0,
            "1 mm pad entries stay inside the 2 mm local breakout budget"
        );

        for track in &mut b.tracks {
            track.start.x = Nm::from_mm(5.0);
            track.end.x = Nm::from_mm(12.0);
        }
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.diff_pair_pad_entry_mismatch_violations, 0,
            "equal 3 mm pad entries are still symmetric"
        );
        assert_eq!(
            dirty.diff_pair_pad_entry_length_violations, 1,
            "matched-but-long 3 mm pad entries violate the 2 mm local breakout budget once per pair"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "overlength differential-pair pad entries must reject optimizer clean-board selection"
        );
    }

    #[test]
    fn detects_differential_pair_spacing_variation() {
        let mut clean = board();
        let p = clean.add_net("USB_P", NetClassKind::Signal);
        let n = clean.add_net("USB_N", NetClassKind::Signal);
        for (net, y0, y1) in [(p, 4.0, 4.0), (n, 4.6, 4.8)] {
            clean.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y0)),
                end: Point::new(Nm::from_mm(6.0), Nm::from_mm(y0)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
            clean.tracks.push(Track {
                start: Point::new(Nm::from_mm(8.0), Nm::from_mm(y1)),
                end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y1)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let within = audit(&clean, &[], &[], &DesignRules::holohv());
        assert_eq!(
            within.diff_pair_spacing_variation_violations, 0,
            "0.2 mm pair-spacing variation stays inside the 0.25 mm tolerance"
        );

        let mut dirty = clean.clone();
        dirty.tracks[3].start.y = Nm::from_mm(5.2);
        dirty.tracks[3].end.y = Nm::from_mm(5.2);
        let widened = audit(&dirty, &[], &[], &DesignRules::holohv());
        assert_eq!(
            widened.diff_pair_spacing_variation_violations, 1,
            "pair spacing that opens by more than tolerance is flagged"
        );
    }

    #[test]
    fn detects_differential_pair_keepout() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

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
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

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
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

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

    #[test]
    fn detects_high_speed_active_ic_near_board_edge() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

        let mut b = board();
        let tx = b.add_net("TX_CLK", NetClassKind::Signal);
        let ctrl = b.add_net("CTRL", NetClassKind::Signal);

        let lib = vec![
            FootprintDef::new(
                "ACTIVE",
                (Nm::from_mm(4.0), Nm::from_mm(4.0)),
                Role::ActiveIc,
                vec![PadDef {
                    offset: Point::new(Nm(0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                }],
            ),
            FootprintDef::new(
                "EDGE_CONN",
                (Nm::from_mm(4.0), Nm::from_mm(4.0)),
                Role::Connector,
                vec![PadDef {
                    offset: Point::new(Nm(0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                }],
            ),
        ];
        let comp = |fp, net, refdes: &str, x, y| Component {
            fp,
            nets: vec![Some(net)],
            refdes: refdes.into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let comps = vec![
            comp(0, tx, "U_EDGE", 3.0, 10.0),
            comp(0, tx, "U_CENTER", 10.0, 10.0),
            comp(0, ctrl, "U_CTRL_EDGE", 3.0, 15.0),
            comp(1, tx, "J_EDGE", 3.0, 3.0),
        ];

        let r = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            r.high_speed_component_edge_violations, 1,
            "only the active IC carrying a high-speed net inside the 3 mm edge keepout is flagged"
        );
    }

    #[test]
    fn detects_high_speed_termination_far_from_active_ic() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

        let mut b = board();
        let tx = b.add_net("TX_TERM", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        let lib = vec![
            FootprintDef::new(
                "ACTIVE",
                (Nm::from_mm(4.0), Nm::from_mm(4.0)),
                Role::ActiveIc,
                vec![PadDef {
                    offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                }],
            ),
            FootprintDef::new(
                "R0402",
                (Nm::from_mm(1.2), Nm::from_mm(0.6)),
                Role::Passive,
                vec![
                    PadDef {
                        offset: Point::new(Nm::from_mm(-0.4), Nm(0)),
                        size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                        layers: vec![LayerId(0)],
                        power_pin: false,
                    },
                    PadDef {
                        offset: Point::new(Nm::from_mm(0.4), Nm(0)),
                        size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                        layers: vec![LayerId(0)],
                        power_pin: false,
                    },
                ],
            ),
        ];
        let comp = |fp, nets, refdes: &str, x, y| Component {
            fp,
            nets,
            refdes: refdes.into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let comps = vec![
            comp(0, vec![Some(tx)], "U1", 10.0, 10.0),
            comp(1, vec![Some(tx), Some(gnd)], "R_NEAR", 11.7, 10.0),
            comp(1, vec![Some(tx), Some(gnd)], "R_FAR", 17.0, 10.0),
            comp(1, vec![Some(tx), Some(gnd)], "C_IGNORE", 17.0, 12.0),
        ];

        let r = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            r.high_speed_termination_placement_violations, 1,
            "only the resistor-like high-speed terminator outside the 2 mm active-pad budget is flagged"
        );
    }

    #[test]
    fn detects_signal_track_intruding_on_reference_plane() {
        use crate::board::{Zone, ZoneFill};

        let mut b = board();
        let tx = b.add_net("TX_PLANE_CUT", NetClassKind::Signal);
        let ctrl = b.add_net("CTRL_OTHER_LAYER", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.zones.push(Zone {
            net: gnd,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: tx,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(12.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(12.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: ctrl,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(14.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(14.0)),
            width: Nm::from_mm(0.25),
            layer: LayerId(1),
            net: gnd,
        });

        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.reference_plane_intrusion_violations, 1,
            "only the non-plane signal track routed through the reference-plane zone is flagged"
        );
    }

    #[test]
    fn detects_signal_crossing_opposite_split_ground_domain() {
        use crate::board::{Zone, ZoneFill};

        let mut b = board();
        let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
        let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
        let agnd = b.add_net("AGND", NetClassKind::Ground);
        let dgnd = b.add_net("DGND", NetClassKind::Ground);
        let zone = |net, x0: f64, x1: f64| Zone {
            net,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
                Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        };
        b.zones.push(zone(agnd, 2.0, 10.0));
        b.zones.push(zone(dgnd, 10.0, 18.0));
        let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
            board.tracks.push(Track {
                start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(1),
                net,
            });
        };

        let mut clean = b.clone();
        route(&mut clean, analog, 3.0, 9.0, 6.0);
        route(&mut clean, digital, 11.0, 17.0, 8.0);
        let same_domain = audit(&clean, &[], &[], &DesignRules::holohv());
        assert_eq!(
            same_domain.split_domain_reference_violations, 0,
            "analog over AGND and digital over DGND stay inside their split-ground domains"
        );

        let mut crossed = b;
        route(&mut crossed, digital, 3.0, 9.0, 6.0);
        route(&mut crossed, analog, 11.0, 17.0, 8.0);
        let opposite_domain = audit(&crossed, &[], &[], &DesignRules::holohv());
        assert_eq!(
            opposite_domain.split_domain_reference_violations, 2,
            "digital over AGND and analog over DGND both violate split-domain reference routing"
        );
    }

    #[test]
    fn detects_mixed_domain_shared_ground_return_overlap() {
        let mut b = board();
        let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
        let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.zones.push(Zone {
            net: gnd,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        let route = |board: &mut Board, net, y: f64| {
            board.tracks.push(Track {
                start: Point::new(Nm::from_mm(3.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(17.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        };

        let mut separated = b.clone();
        route(&mut separated, analog, 6.0);
        route(&mut separated, digital, 8.0);
        let clean = audit(&separated, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.mixed_domain_shared_reference_violations, 0,
            "analog and digital returns sharing GND but separated by more than the sensitive keepout are clean"
        );

        route(&mut b, analog, 6.0);
        route(&mut b, digital, 6.8);
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.mixed_domain_shared_reference_violations, 1,
            "analog and digital tracks closer than the sensitive keepout over one GND plane overlap return-current corridors"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "mixed-domain return-current overlap must reject optimizer clean-board selection"
        );
    }

    #[test]
    fn detects_virtual_split_domain_crossing() {
        let mut b = board();
        let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
        let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
        for (net, x) in [(analog, 4.0), (digital, 16.0)] {
            b.add_pad(Pad {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(4.0)),
                layers: vec![LayerId(0)],
                net: Some(net),
            });
            b.add_pad(Pad {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(16.0)),
                layers: vec![LayerId(0)],
                net: Some(net),
            });
        }
        let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
            board.tracks.push(Track {
                start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        };

        let mut clean_board = b.clone();
        route(&mut clean_board, analog, 3.0, 8.0, 6.0);
        route(&mut clean_board, digital, 12.0, 17.0, 14.0);
        let clean = audit(&clean_board, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.virtual_split_crossing_violations, 0,
            "analog and digital tracks staying on their centroid-derived virtual sides are clean"
        );

        route(&mut b, analog, 3.0, 12.0, 6.0);
        route(&mut b, digital, 12.0, 17.0, 14.0);
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.virtual_split_crossing_violations, 1,
            "the analog track crossing the inferred x=10 mm virtual split line is flagged"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "virtual split crossing must reject optimizer clean-board selection"
        );
    }

    #[test]
    fn detects_fragmented_same_net_ground_plane() {
        let mut b = board();
        let gnd = b.add_net("GND", NetClassKind::Ground);
        let island = |x0: f64, x1: f64| Zone {
            net: gnd,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
                Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
                Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        };

        b.zones.push(island(2.0, 18.0));
        let continuous = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            continuous.ground_plane_fragmentation_violations, 0,
            "one ground pour on a layer is a continuous reference plane"
        );

        b.zones.push(island(20.0, 24.0));
        let fragmented = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            fragmented.ground_plane_fragmentation_violations, 1,
            "two same-net ground pour islands on one layer fragment the reference plane"
        );

        b.zones[1].fill = ZoneFill::Solid;
        let teardrop_like = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            teardrop_like.ground_plane_fragmentation_violations, 0,
            "solid same-net reinforcement zones are not counted as plane-pour fragmentation"
        );
    }

    #[test]
    fn detects_high_speed_track_without_adjacent_reference_plane() {
        use crate::board::{Zone, ZoneFill};

        let mut b = board();
        let tx = b.add_net("TX_REF", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx,
        });

        let missing = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            missing.reference_plane_absence_violations, 1,
            "a high-speed segment with no adjacent reference zone is flagged"
        );

        b.zones.push(Zone {
            net: gnd,
            layer: LayerId(0),
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        let same_layer = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            same_layer.reference_plane_absence_violations, 1,
            "same-layer copper is not an adjacent reference plane"
        );

        b.zones.clear();
        b.zones.push(Zone {
            net: gnd,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(10.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(10.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        let partial = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            partial.reference_plane_absence_violations, 1,
            "adjacent reference coverage must cover the segment start, middle, and end"
        );

        b.zones[0].polygon = vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ];
        let covered = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            covered.reference_plane_absence_violations, 0,
            "a full adjacent ground zone supplies the high-speed reference plane"
        );
    }

    #[test]
    fn detects_inner_high_speed_track_without_dual_ground_reference() {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
        let mut b = Board::new(spec);
        let tx = b.add_net("TX_INNER", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: tx,
        });
        let reference = |layer| Zone {
            net: gnd,
            layer,
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        };

        b.zones.push(reference(LayerId(0)));
        let one_sided = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            one_sided.reference_plane_absence_violations, 0,
            "one adjacent ground zone satisfies the generic adjacent-reference rule"
        );
        assert_eq!(
            one_sided.inner_layer_dual_ground_reference_violations, 1,
            "an inner high-speed layer requires ground reference zones on both adjacent layers"
        );

        b.zones.push(reference(LayerId(2)));
        let dual = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dual.inner_layer_dual_ground_reference_violations, 0,
            "ground zones on both adjacent layers clear the inner-layer high-speed reference rule"
        );
    }

    #[test]
    fn detects_power_plane_reference_without_stitching_caps() {
        use crate::board::{Zone, ZoneFill};
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

        let mut b = board();
        let tx = b.add_net("TX_PWR_REF", NetClassKind::Signal);
        let pwr = b.add_net("VDD", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx,
        });
        b.zones.push(Zone {
            net: pwr,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        let cap_fp = FootprintDef::new(
            "C0402",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        );
        let mk_cap = |refdes: &str, x: f64| Component {
            fp: 0,
            nets: vec![Some(pwr), Some(gnd)],
            refdes: refdes.into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let lib = vec![cap_fp];

        let missing = audit(&b, &[], &lib, &DesignRules::holohv());
        assert_eq!(
            missing.power_reference_stitching_cap_violations, 1,
            "a power-plane-referenced high-speed track needs endpoint stitching capacitors"
        );

        let one_ended = audit(&b, &[mk_cap("C_SRC", 4.5)], &lib, &DesignRules::holohv());
        assert_eq!(
            one_ended.power_reference_stitching_cap_violations, 1,
            "stitching only the source endpoint is insufficient"
        );

        let stitched = audit(
            &b,
            &[mk_cap("C_SRC", 4.5), mk_cap("C_SINK", 15.5)],
            &lib,
            &DesignRules::holohv(),
        );
        assert_eq!(
            stitched.power_reference_stitching_cap_violations, 0,
            "source and sink stitching capacitors clear the power-reference requirement"
        );

        b.zones[0].net = gnd;
        let ground_referenced = audit(&b, &[], &lib, &DesignRules::holohv());
        assert_eq!(
            ground_referenced.power_reference_stitching_cap_violations, 0,
            "a ground-plane reference does not require power-reference stitching capacitors"
        );
    }

    #[test]
    fn detects_asymmetric_diff_pair_power_reference_stitching_caps() {
        use crate::board::{Zone, ZoneFill};
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

        let mut b = board();
        let p = b.add_net("MGT_P", NetClassKind::Signal);
        let n = b.add_net("MGT_N", NetClassKind::Signal);
        let vref = b.add_net("VREF", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        b.zones.push(Zone {
            net: vref,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                Point::new(Nm::from_mm(14.0), Nm::from_mm(0.0)),
                Point::new(Nm::from_mm(14.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(0.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        for (net, y) in [(p, 4.0), (n, 5.0)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
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
        let cap = |refdes: &str, x, y| Component {
            fp: 0,
            nets: vec![Some(vref), Some(gnd)],
            refdes: refdes.into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let lib = vec![cap_fp];
        let matched = vec![
            cap("C_PS0", 2.0, 3.7),
            cap("C_PS1", 10.0, 3.7),
            cap("C_NS0", 2.0, 5.3),
            cap("C_NS1", 10.0, 5.3),
        ];
        let clean = audit(&b, &matched, &lib, &DesignRules::holohv());
        assert_eq!(
            clean.power_reference_stitching_cap_violations, 0,
            "each power-referenced pair endpoint has a local stitching capacitor"
        );
        assert_eq!(
            clean.diff_pair_stitching_cap_symmetry_violations, 0,
            "P/N stitching capacitors at matching stations are symmetric"
        );

        let shifted = vec![
            cap("C_PS0", 2.0, 3.7),
            cap("C_PS1", 10.0, 3.7),
            cap("C_NS0", 2.8, 5.3),
            cap("C_NS1", 10.8, 5.3),
        ];
        let dirty = audit(&b, &shifted, &lib, &DesignRules::holohv());
        assert_eq!(
            dirty.power_reference_stitching_cap_violations, 0,
            "shifted capacitors are still local to the signal endpoints"
        );
        assert_eq!(
            dirty.diff_pair_stitching_cap_symmetry_violations, 1,
            "0.8 mm P/N stitching-cap station mismatch exceeds the 0.5 mm symmetry tolerance"
        );
    }

    #[test]
    fn split_plane_stitching_cap_must_be_local_to_crossing() {
        use crate::board::{Zone, ZoneFill};
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};

        let mut b = board();
        let tx = b.add_net("TX_SPLIT", NetClassKind::Signal);
        let pwr = b.add_net("VDD", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        b.zones.push(Zone {
            net: gnd,
            layer: LayerId(0),
            polygon: vec![
                Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                Point::new(Nm::from_mm(10.0), Nm::from_mm(0.0)),
                Point::new(Nm::from_mm(10.0), Nm::from_mm(20.0)),
                Point::new(Nm::from_mm(0.0), Nm::from_mm(20.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(9.0), Nm::from_mm(10.0)),
            end: Point::new(Nm::from_mm(11.0), Nm::from_mm(10.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx,
        });

        let cap_fp = FootprintDef::new(
            "C0402",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        );
        let cap_at = |x: f64, y: f64| Component {
            fp: 0,
            nets: vec![Some(pwr), Some(gnd)],
            refdes: "C_SPLIT".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let lib = vec![cap_fp];

        let far = audit(&b, &[cap_at(14.0, 10.0)], &lib, &DesignRules::holohv());
        assert_eq!(
            far.split_plane_crossings, 1,
            "a stitching capacitor 4 mm from the split crossing is outside the 2 mm path budget"
        );

        let near = audit(&b, &[cap_at(10.0, 11.0)], &lib, &DesignRules::holohv());
        assert_eq!(
            near.split_plane_crossings, 0,
            "a stitching capacitor within 1 mm of the split crossing supplies a local return path"
        );

        let local_signal_cap = Component {
            nets: vec![Some(tx), Some(gnd)],
            ..cap_at(10.0, 11.0)
        };
        let not_stitched = audit(&b, &[local_signal_cap], &lib, &DesignRules::holohv());
        assert_eq!(
            not_stitched.split_plane_crossings, 1,
            "a local capacitor must bridge the crossed reference zone to another reference net"
        );
    }

    #[test]
    fn detects_asymmetric_diff_pair_transition_ground_vias() {
        let mut b = board();
        let p = b.add_net("LANE_P", NetClassKind::Signal);
        let n = b.add_net("LANE_N", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        for (net, y) in [(p, 4.0), (n, 5.0)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let via_at = |net, x: f64, y: f64| Via {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        };
        b.vias.push(via_at(p, 6.0, 4.0));
        b.vias.push(via_at(n, 6.0, 5.0));
        b.vias.push(via_at(gnd, 6.0, 3.7));
        b.vias.push(via_at(gnd, 6.0, 5.3));

        let matched = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            matched.high_speed_transition_ground_via_violations, 0,
            "both P/N layer transitions have local ground transition vias"
        );
        assert_eq!(
            matched.diff_pair_transition_ground_via_symmetry_violations, 0,
            "P/N ground transition vias at the same pair-axis station are symmetric"
        );

        b.vias[3].pos = Point::new(Nm::from_mm(7.0), Nm::from_mm(5.3));
        let shifted = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            shifted.high_speed_transition_ground_via_violations, 0,
            "the shifted ground via is still within the local return-via distance"
        );
        assert_eq!(
            shifted.diff_pair_transition_ground_via_symmetry_violations, 1,
            "a 1 mm P/N ground-via station mismatch exceeds the 0.5 mm symmetry tolerance"
        );
    }

    #[test]
    fn detects_unrelated_high_speed_parallel_spacing() {
        let mut b = board();
        let a = b.add_net("TX_A", NetClassKind::Signal);
        let c = b.add_net("TX_C", NetClassKind::Signal);
        for (net, y) in [(a, 4.0), (c, 4.55)] {
            b.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.high_speed_parallel_spacing_violations, 1,
            "unrelated long parallel TX traces closer than 3W are flagged"
        );

        let mut generic = board();
        let tx0 = generic.add_net("TX_0", NetClassKind::Signal);
        let tx1 = generic.add_net("TX_1", NetClassKind::Signal);
        for (net, y) in [(tx0, 4.0), (tx1, 5.0)] {
            generic.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let generic_report = audit(&generic, &[], &[], &DesignRules::holohv());
        assert_eq!(
            generic_report.high_speed_parallel_spacing_violations, 0,
            "a 0.85 mm edge gap clears generic 3W spacing for non-clock high-speed traces"
        );

        let mut clock = board();
        let clk = clock.add_net("TX_CLK", NetClassKind::Signal);
        let data = clock.add_net("TX_DATA", NetClassKind::Signal);
        for (net, y) in [(clk, 4.0), (data, 5.0)] {
            clock.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let clock_report = audit(&clock, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clock_report.high_speed_parallel_spacing_violations, 1,
            "the same 0.85 mm edge gap violates the 1.27 mm clock keepout"
        );

        let mut diff_pair = board();
        let p = diff_pair.add_net("LANE_P", NetClassKind::Signal);
        let n = diff_pair.add_net("LANE_N", NetClassKind::Signal);
        for (net, y) in [(p, 4.0), (n, 4.6)] {
            diff_pair.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let paired = audit(&diff_pair, &[], &[], &DesignRules::holohv());
        assert_eq!(
            paired.high_speed_parallel_spacing_violations, 0,
            "true P/N differential mates are exempt from unrelated-trace crosstalk spacing"
        );

        let mut short = board();
        let s0 = short.add_net("TX_S0", NetClassKind::Signal);
        let s1 = short.add_net("TX_S1", NetClassKind::Signal);
        for (net, y) in [(s0, 4.0), (s1, 4.6)] {
            short.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(4.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer: LayerId(0),
                net,
            });
        }
        let short_overlap = audit(&short, &[], &[], &DesignRules::holohv());
        assert_eq!(
            short_overlap.high_speed_parallel_spacing_violations, 0,
            "short pad-entry adjacency below the coupled-length threshold is not counted"
        );
    }

    #[test]
    fn detects_adjacent_layer_high_speed_parallelism() {
        let mut broadside = board();
        let tx0 = broadside.add_net("TX_TOP", NetClassKind::Signal);
        let tx1 = broadside.add_net("TX_INNER", NetClassKind::Signal);
        for (net, layer) in [(tx0, LayerId(0)), (tx1, LayerId(1))] {
            broadside.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
                width: Nm::from_mm(0.15),
                layer,
                net,
            });
        }
        let broadside_report = audit(&broadside, &[], &[], &DesignRules::holohv());
        assert_eq!(
            broadside_report.high_speed_parallel_spacing_violations, 0,
            "same-layer parallel spacing is not responsible for adjacent-layer coupling"
        );
        assert_eq!(
            broadside_report.high_speed_adjacent_layer_parallel_violations, 1,
            "overlapping adjacent-layer high-speed runs should be routed orthogonally or separated"
        );

        let mut orthogonal = board();
        let x = orthogonal.add_net("TX_X", NetClassKind::Signal);
        let y = orthogonal.add_net("TX_Y", NetClassKind::Signal);
        orthogonal.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: x,
        });
        orthogonal.tracks.push(Track {
            start: Point::new(Nm::from_mm(6.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(9.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: y,
        });
        let orthogonal_report = audit(&orthogonal, &[], &[], &DesignRules::holohv());
        assert_eq!(
            orthogonal_report.high_speed_adjacent_layer_parallel_violations, 0,
            "adjacent-layer high-speed crossings are orthogonal, not broadside parallel"
        );

        let mut separated = board();
        let s0 = separated.add_net("TX_S0", NetClassKind::Signal);
        let s1 = separated.add_net("TX_S1", NetClassKind::Signal);
        for (net, layer, y) in [(s0, LayerId(0), 4.0), (s1, LayerId(1), 5.0)] {
            separated.tracks.push(Track {
                start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
                end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
                width: Nm::from_mm(0.15),
                layer,
                net,
            });
        }
        let separated_report = audit(&separated, &[], &[], &DesignRules::holohv());
        assert_eq!(
            separated_report.high_speed_adjacent_layer_parallel_violations, 0,
            "0.85 mm planar edge offset clears the adjacent-layer 3W broadside budget"
        );
    }

    #[test]
    fn detects_same_net_non_ground_via_plane_hotspot_spacing() {
        let mut b = board();
        let tx = b.add_net("TX_CLUSTER", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        let via_at = |net, x| Via {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(8.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        };

        b.vias.push(via_at(tx, 5.0));
        b.vias.push(via_at(tx, 5.7));
        let clustered = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clustered.via_spacing_violations, 0,
            "same-net vias are outside the different-net via-spacing DRC"
        );
        assert_eq!(
            clustered.plane_hotspot_via_spacing_violations, 1,
            "0.24 mm outer gap violates the 15 mil plane-hotspot spacing budget"
        );

        b.vias[1].pos = Point::new(Nm::from_mm(6.0), Nm::from_mm(8.0));
        let spaced = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            spaced.plane_hotspot_via_spacing_violations, 0,
            "0.54 mm outer gap clears the 15 mil plane-hotspot spacing budget"
        );

        b.vias.clear();
        b.vias.push(via_at(gnd, 5.0));
        b.vias.push(via_at(gnd, 5.7));
        let ground_stitching = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            ground_stitching.plane_hotspot_via_spacing_violations, 0,
            "close same-net ground stitching vias are intentional return-plane ties"
        );
    }

    #[test]
    fn detects_high_speed_transition_without_ground_via() {
        let mut b = board();
        let tx = b.add_net("TX_LAYER", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(6.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net: tx,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.high_speed_transition_ground_via_violations, 1,
            "a high-speed layer transition without a local ground transition via is flagged"
        );

        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(7.0), Nm::from_mm(6.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net: gnd,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.high_speed_transition_ground_via_violations, 0,
            "a nearby ground transition via supplies the local return path"
        );
    }

    #[test]
    fn detects_high_speed_terminal_without_ground_return() {
        let mut b = board();
        let tx = b.add_net("TX_TERM", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        let p0 = Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0));
        let p1 = Point::new(Nm::from_mm(12.0), Nm::from_mm(4.0));
        for p in [p0, p1] {
            b.add_pad(Pad {
                pos: p,
                layers: vec![LayerId(0)],
                net: Some(tx),
            });
        }

        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.high_speed_terminal_ground_via_violations, 2,
            "both high-speed source/sink pads lack local ground return copper"
        );

        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(4.0)),
            layers: vec![LayerId(0)],
            net: Some(gnd),
        });
        let one_sided = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            one_sided.high_speed_terminal_ground_via_violations, 1,
            "only the terminal with a nearby ground feature is cleared"
        );

        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(12.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(gnd),
        });
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.high_speed_terminal_ground_via_violations, 0,
            "source and sink terminals both have local return copper"
        );
    }

    #[test]
    fn detects_high_speed_via_far_from_same_net_pad() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let tx = b.add_net("TX_PAD_VIA", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(tx),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(6.0)),
            layers: vec![LayerId(0)],
            net: Some(gnd),
        });

        let via_at = |board: &mut Board, net: NetId, x_mm: f64, y_mm: f64| {
            board.vias.push(Via {
                pos: Point::new(Nm::from_mm(x_mm), Nm::from_mm(y_mm)),
                drill: Nm::from_mm(0.2),
                diameter: Nm::from_mm(0.46),
                net,
                from: LayerId(0),
                to: LayerId(1),
                kind: ViaKind::Micro,
                filled: false,
            });
        };

        via_at(&mut b, tx, 6.5, 5.0);
        via_at(&mut b, gnd, 6.5, 6.0);
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.high_speed_via_pad_proximity_violations, 0,
            "a high-speed via 1.5 mm from its same-net pad stays inside the 2 mm budget"
        );
        assert_eq!(
            clean.high_speed_transition_ground_via_violations, 0,
            "the focused test keeps the high-speed transition return path locally satisfied"
        );

        b.vias.clear();
        via_at(&mut b, tx, 8.0, 5.0);
        via_at(&mut b, gnd, 8.0, 6.0);
        let far = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            far.high_speed_via_pad_proximity_violations, 1,
            "a high-speed via 3.0 mm from its same-net pad is outside the 2 mm budget"
        );
        assert_eq!(
            far.high_speed_transition_ground_via_violations, 0,
            "the violation is via-to-pad placement, not a missing ground transition via"
        );
    }

    #[test]
    fn detects_unfilled_via_in_non_ground_smd_pad() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let sig = b.add_net("BGA_SIG", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        let smd = Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0));
        let thermal = Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0));
        let drilled = Point::new(Nm::from_mm(11.0), Nm::from_mm(5.0));
        b.add_pad(Pad {
            pos: smd,
            layers: vec![LayerId(0)],
            net: Some(sig),
        });
        b.add_pad(Pad {
            pos: thermal,
            layers: vec![LayerId(0)],
            net: Some(gnd),
        });
        b.add_pad(Pad {
            pos: drilled,
            layers: vec![LayerId(0), LayerId(1)],
            net: Some(sig),
        });
        let via = |net, pos, filled| Via {
            pos,
            drill: Nm::from_mm(0.1),
            diameter: Nm::from_mm(0.25),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled,
        };

        b.vias.push(via(sig, smd, false));
        b.vias.push(via(gnd, thermal, false));
        b.vias.push(via(sig, drilled, false));
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.unfilled_via_in_pad_violations, 1,
            "only the unfilled via inside the non-ground SMD pad violates VIPPO filling"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "unfilled via-in-pad must reject optimizer clean-board selection"
        );

        b.vias[0].filled = true;
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.unfilled_via_in_pad_violations, 0,
            "filled VIPPO plus unfilled ground thermal-pad/drilled-pad vias are accepted"
        );
    }

    #[test]
    fn detects_oversized_high_speed_via_diameter() {
        use crate::board::{Via, ViaKind};

        let mut b = board();
        let tx = b.add_net("TX_VIA_SIZE", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.pads.push(Pad {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            layers: vec![LayerId(0)],
            net: Some(tx),
        });
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(5.5), Nm::from_mm(5.0)),
            drill: Nm::from_mm(0.2),
            diameter: DesignRules::holohv().via_diameter(),
            net: tx,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(5.5), Nm::from_mm(5.5)),
            drill: Nm::from_mm(0.2),
            diameter: DesignRules::holohv().via_diameter(),
            net: gnd,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });

        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.high_speed_via_pad_proximity_violations, 0,
            "the high-speed via is within the same-net pad proximity budget"
        );
        assert_eq!(
            clean.high_speed_transition_ground_via_violations, 0,
            "the high-speed via has a local ground transition via"
        );
        assert_eq!(
            clean.high_speed_via_diameter_violations, 0,
            "the default rule-sized high-speed via is accepted"
        );

        b.vias[0].diameter = Nm::from_mm(0.7);
        let oversized = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            oversized.high_speed_via_pad_proximity_violations, 0,
            "oversizing the via does not change its same-net pad distance"
        );
        assert_eq!(
            oversized.high_speed_transition_ground_via_violations, 0,
            "oversizing the via does not remove the local ground transition via"
        );
        assert_eq!(
            oversized.high_speed_via_diameter_violations, 1,
            "a high-speed via larger than the selected rule diameter is flagged"
        );
    }

    #[test]
    fn detects_oversized_blind_and_buried_via_drills() {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
        let mut b = Board::new(spec);
        let sig = b.add_net("SIG_BLIND_BURIED_DRILL", NetClassKind::Signal);

        let via = |kind: ViaKind, drill_mm: f64, x_mm: f64| Via {
            pos: Point::new(Nm::from_mm(x_mm), Nm::from_mm(5.0)),
            drill: Nm::from_mm(drill_mm),
            diameter: Nm::from_mm(drill_mm + 0.15),
            net: sig,
            from: match kind {
                ViaKind::Buried => LayerId(1),
                _ => LayerId(0),
            },
            to: match kind {
                ViaKind::Buried => LayerId(2),
                ViaKind::Blind => LayerId(2),
                ViaKind::Micro | ViaKind::Through => LayerId(3),
            },
            kind,
            filled: false,
        };

        b.vias.push(via(ViaKind::Blind, 0.15, 4.0));
        b.vias.push(via(ViaKind::Buried, 0.15, 6.0));
        b.vias.push(via(ViaKind::Through, 0.30, 8.0));
        let clean = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            clean.blind_buried_via_drill_violations, 0,
            "rule-sized blind/buried vias and larger through vias are accepted"
        );

        b.vias.push(via(ViaKind::Blind, 0.16, 10.0));
        b.vias.push(via(ViaKind::Buried, 0.20, 12.0));
        let dirty = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            dirty.blind_buried_via_drill_violations, 2,
            "only oversized blind and buried via drills violate the 0.15 mm fabrication limit"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "oversized blind/buried vias must reject optimizer clean-board selection"
        );
    }

    #[test]
    fn detects_decoupling_cap_without_local_ground_via() {
        use crate::board::{Via, ViaKind};
        use crate::place::component::Placement;

        let mut b = board();
        let pwr = b.add_net("VDD", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        let lib = vec![FootprintDef::new(
            "C0402",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Decoupling,
            vec![
                crate::place::PadDef {
                    offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                crate::place::PadDef {
                    offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        )];
        let comps = vec![Component {
            fp: 0,
            nets: vec![Some(pwr), Some(gnd)],
            refdes: "C1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: crate::place::Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        }];

        let dirty = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            dirty.decoupling_ground_via_violations, 1,
            "an SMD decoupling ground pad without a local ground via is flagged"
        );

        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(14.0), Nm::from_mm(10.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net: gnd,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        let far = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            far.decoupling_ground_via_violations, 1,
            "a ground via outside the 1 mm decoupling budget is not local"
        );

        b.vias[0].pos = Point::new(Nm::from_mm(10.5), Nm::from_mm(10.0));
        let clean = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            clean.decoupling_ground_via_violations, 0,
            "a nearby ground via clears the decoupling local-return requirement"
        );
    }

    #[test]
    fn detects_decoupling_power_pin_on_opposite_layer() {
        use crate::board::{Via, ViaKind};
        use crate::place::component::Placement;

        let mut b = board();
        let pwr = b.add_net("VDD", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(6.2), Nm::from_mm(5.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.45),
            net: gnd,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        let cap_fp = FootprintDef::new(
            "C0402",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Decoupling,
            vec![
                crate::place::PadDef {
                    offset: Point::new(Nm::from_mm(-0.2), Nm(0)),
                    size: (Nm::from_mm(0.2), Nm::from_mm(0.2)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                crate::place::PadDef {
                    offset: Point::new(Nm::from_mm(0.2), Nm(0)),
                    size: (Nm::from_mm(0.2), Nm::from_mm(0.2)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        );
        let ic_fp = |power_layer| {
            FootprintDef::new(
                "U",
                (Nm::from_mm(4.0), Nm::from_mm(4.0)),
                Role::ActiveIc,
                vec![crate::place::PadDef {
                    offset: Point::new(Nm(0), Nm(0)),
                    size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                    layers: vec![power_layer],
                    power_pin: true,
                }],
            )
        };
        let comps = vec![
            Component {
                fp: 1,
                nets: vec![Some(pwr)],
                refdes: "U1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
                    rot: crate::place::Rot::R0,
                },
                assoc_ic: None,
                locked: false,
                ..Default::default()
            },
            Component {
                fp: 0,
                nets: vec![Some(pwr), Some(gnd)],
                refdes: "C1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.0)),
                    rot: crate::place::Rot::R0,
                },
                assoc_ic: Some(0),
                locked: false,
                ..Default::default()
            },
        ];

        let clean_lib = vec![cap_fp.clone(), ic_fp(LayerId(0))];
        let clean = audit(&b, &comps, &clean_lib, &DesignRules::holohv());
        assert_eq!(
            clean.decoupling_ground_via_violations, 0,
            "the local ground via is present"
        );
        assert_eq!(
            clean.decoupling_power_layer_violations, 0,
            "cap power pad and IC power pin share F.Cu"
        );

        let dirty_lib = vec![cap_fp, ic_fp(LayerId(1))];
        let dirty = audit(&b, &comps, &dirty_lib, &DesignRules::holohv());
        assert_eq!(
            dirty.decoupling_ground_via_violations, 0,
            "the failing case is not missing its ground return via"
        );
        assert_eq!(
            dirty.decoupling_power_layer_violations, 1,
            "the cap power pad cannot reach the associated IC power pin without a layer change"
        );
    }

    #[test]
    fn detects_oversized_decoupling_commutation_loop_area() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

        let mut b = board();
        let vdd = b.add_net("VDD_LOOP", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        let lib = vec![
            FootprintDef::new(
                "U_LOOP",
                (Nm::from_mm(8.0), Nm::from_mm(8.0)),
                Role::ActiveIc,
                vec![
                    PadDef {
                        offset: Point::new(Nm::from_mm(-2.0), Nm::from_mm(0.0)),
                        size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                        layers: vec![LayerId(0)],
                        power_pin: true,
                    },
                    PadDef {
                        offset: Point::new(Nm::from_mm(2.0), Nm::from_mm(0.0)),
                        size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                        layers: vec![LayerId(0)],
                        power_pin: true,
                    },
                ],
            ),
            FootprintDef::new(
                "C_LOOP",
                (Nm::from_mm(2.0), Nm::from_mm(1.0)),
                Role::Decoupling,
                vec![
                    PadDef {
                        offset: Point::new(Nm::from_mm(-0.5), Nm::from_mm(0.0)),
                        size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                        layers: vec![LayerId(0)],
                        power_pin: true,
                    },
                    PadDef {
                        offset: Point::new(Nm::from_mm(0.5), Nm::from_mm(0.0)),
                        size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                        layers: vec![LayerId(0)],
                        power_pin: true,
                    },
                ],
            ),
        ];
        let ic = Component {
            fp: 0,
            nets: vec![Some(vdd), Some(gnd)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let cap = |y_mm| Component {
            fp: 1,
            nets: vec![Some(vdd), Some(gnd)],
            refdes: "C1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(y_mm)),
                rot: Rot::R0,
            },
            assoc_ic: Some(0),
            locked: false,
            ..Default::default()
        };

        let near = vec![ic.clone(), cap(12.0)];
        let clean = audit(&b, &near, &lib, &DesignRules::holohv());
        assert_eq!(
            clean.decoupling_loop_area_violations, 0,
            "a 5 mm² commutation loop is inside the configured 10 mm² budget"
        );

        let far = vec![ic, cap(18.0)];
        let dirty = audit(&b, &far, &lib, &DesignRules::holohv());
        assert_eq!(
            dirty.decoupling_loop_area_violations, 1,
            "a 20 mm² commutation loop exceeds the configured loop-area budget"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "oversized decoupling commutation loops must reject clean-board selection"
        );
    }

    #[test]
    fn detects_active_ic_power_pad_without_internal_plane() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

        let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
        let mut b = Board::new(spec);
        let vdd = b.add_net("VDD_CORE", NetClassKind::Power);
        let sig = b.add_net("GPIO0", NetClassKind::Signal);

        let lib = vec![FootprintDef::new(
            "U_THERMAL_POWER_PAD",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(1.5), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        )];
        let comps = vec![Component {
            fp: 0,
            nets: vec![Some(vdd), Some(sig)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        }];
        b.zones.push(Zone {
            net: vdd,
            layer: LayerId(1),
            polygon: vec![
                Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
                Point::new(Nm::from_mm(12.0), Nm::from_mm(12.0)),
                Point::new(Nm::from_mm(8.0), Nm::from_mm(12.0)),
            ],
            fill: ZoneFill::ThermalRelief,
        });

        let clean = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            clean.active_ic_power_plane_violations, 0,
            "a same-net internal plane under the active IC power pad satisfies the thermal-plane rule"
        );

        b.zones[0].net = sig;
        let dirty = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            dirty.active_ic_power_plane_violations, 1,
            "a signal zone under the power pad is not a same-net internal power plane"
        );
        assert!(
            !dirty.hard_drc_clean(),
            "missing active-IC internal power-plane support must reject clean-board selection"
        );
    }

    #[test]
    fn detects_high_speed_via_stub() {
        use crate::board::{Via, ViaKind};

        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
        let mut dirty = Board::new(spec);
        let tx = dirty.add_net("TX_STUBBED_VIA", NetClassKind::Signal);
        let p = Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0));
        dirty.tracks.push(Track {
            start: Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)),
            end: p,
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx,
        });
        dirty.tracks.push(Track {
            start: p,
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net: tx,
        });
        dirty.vias.push(Via {
            pos: p,
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net: tx,
            from: LayerId(0),
            to: LayerId(3),
            kind: ViaKind::Through,
            filled: false,
        });
        let r = audit(&dirty, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.high_speed_via_stub_violations, 1,
            "a full-stack via used only on layers 0..1 leaves a high-speed via stub"
        );

        let mut clean = dirty.clone();
        clean.vias[0].to = LayerId(1);
        clean.vias[0].kind = ViaKind::Micro;
        let r = audit(&clean, &[], &[], &DesignRules::holohv());
        assert_eq!(
            r.high_speed_via_stub_violations, 0,
            "a via whose physical span matches the used signal layers has no stub"
        );
    }

    #[test]
    fn clean_board_has_no_faults() {
        let mut b = board();
        let n = b.add_net("N", NetClassKind::Signal);
        let p0 = Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0));
        let p1 = Point::new(Nm::from_mm(8.0), Nm::from_mm(2.0));
        b.add_pad(Pad {
            pos: p0,
            layers: vec![LayerId(0)],
            net: Some(n),
        });
        b.add_pad(Pad {
            pos: p1,
            layers: vec![LayerId(0)],
            net: Some(n),
        });
        b.tracks.push(Track {
            start: p0,
            end: p1,
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: n,
        });
        let r = audit(&b, &[], &[], &DesignRules::holohv());
        assert_eq!(
            (r.crossings, r.near_shorts, r.dangling),
            (0, 0, 0),
            "single connected net is clean"
        );
    }

    #[test]
    fn detects_via_in_surge_suppressor_connector_path() {
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};
        use crate::place::Placement;

        let lib = vec![
            FootprintDef::new(
                "J",
                (Nm::from_mm(3.0), Nm::from_mm(3.0)),
                Role::Connector,
                vec![PadDef {
                    offset: Point::new(Nm(0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                }],
            ),
            FootprintDef::new(
                "TVS",
                (Nm::from_mm(1.0), Nm::from_mm(0.6)),
                Role::Passive,
                vec![PadDef {
                    offset: Point::new(Nm(0), Nm(0)),
                    size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                }],
            ),
        ];
        let mut clean = board();
        let incoming = clean.add_net("USB_IN", NetClassKind::Signal);
        let comps = vec![
            Component {
                fp: 0,
                nets: vec![Some(incoming)],
                refdes: "J1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: None,
                locked: false,
                ..Default::default()
            },
            Component {
                fp: 1,
                nets: vec![Some(incoming)],
                refdes: "TVS1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: None,
                locked: false,
                ..Default::default()
            },
        ];
        clean.vias.push(Via {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.45),
            net: incoming,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
        let clean_report = audit(&clean, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            clean_report.surge_suppressor_via_violations, 0,
            "a same-net via beyond the suppressor is not in the connector-to-clamp path"
        );

        let mut dirty = clean;
        dirty.vias[0].pos = Point::new(Nm::from_mm(5.0), Nm::from_mm(10.0));
        let dirty_report = audit(&dirty, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            dirty_report.surge_suppressor_via_violations, 1,
            "a via in the incoming connector-to-suppressor segment adds parasitic inductance"
        );
        assert!(
            !dirty_report.hard_drc_clean(),
            "connector-to-suppressor vias must reject clean-board selection"
        );
    }

    #[test]
    fn audit_detects_isolation_and_ac_coupling() {
        use crate::board::{LayerId, Pad, Track};
        use crate::place::footprint::Role;
        use crate::place::rotation::Rot;
        use crate::place::Placement;

        let mut b = board();
        // Setup net classes
        let hv = b.add_net("TRIG_HV", NetClassKind::Hv);
        let lv = b.add_net("CTRL_LV", NetClassKind::Signal);
        let gnd = b.add_net("GND", NetClassKind::Ground);

        // Add pads for the nets to give them coordinates
        let p_hv = Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0));
        let p_lv = Point::new(Nm::from_mm(10.0), Nm::from_mm(5.0));
        b.add_pad(Pad {
            pos: p_hv,
            layers: vec![LayerId(0)],
            net: Some(hv),
        });
        b.add_pad(Pad {
            pos: p_lv,
            layers: vec![LayerId(0)],
            net: Some(lv),
        });

        // Add components that bridge the isolation boundary
        let lib = vec![FootprintDef {
            name: "R_0603".to_string(),
            pads: vec![],
            courtyard: (Nm::from_mm(1.6), Nm::from_mm(0.8)),
            role: Role::Passive,
            rotation_policy: crate::place::rotation::RotationPolicy::HalfTurn,
            pad_names: vec![],
            model: None,
            ball_pitch: None,
            i_dd_a: 0.0,
            capacitance_f: 0.0,
        }];

        // Low-voltage control net source component (connector J2)
        let j2 = Component {
            fp: 0,
            nets: vec![Some(lv)],
            refdes: "J2".to_string(),
            placement: Placement {
                pos: p_lv,
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };

        // Component U1 (the HV pulser) connected to TRIG_HV
        let u1 = Component {
            fp: 0,
            nets: vec![Some(hv)],
            refdes: "U1".to_string(),
            placement: Placement {
                pos: p_hv,
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };

        // Resistor directly bridging them (R1)
        let r1 = Component {
            fp: 0,
            nets: vec![Some(hv), Some(lv)],
            refdes: "R1".to_string(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(7.5), Nm::from_mm(5.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };

        let comps = vec![j2, u1, r1];

        // Parallel switching tracks causing AC coupling
        // Track 1: switching net
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(1.0)),
            width: Nm::from_mm(0.2),
            layer: LayerId(0),
            net: hv,
        });
        // Track 2: GND net, adjacent coplanar spacing = 0.3mm (within 2mm spacing limit)
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.5)),
            end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(1.5)),
            width: Nm::from_mm(0.2),
            layer: LayerId(0),
            net: gnd,
        });

        let r = audit(&b, &comps, &lib, &DesignRules::holohv());

        assert_eq!(
            r.isolation_violations, 1,
            "direct LV to HV bridge should violate isolation"
        );
        assert_eq!(
            r.ac_coupling_violations, 1,
            "close parallel HV to GND track should violate AC coupling"
        );
        assert!(
            !r.hotspots.is_empty(),
            "hotspots must contain violation coordinates"
        );
        assert!(
            r.risk_score > 60.0,
            "risk score must include isolation and AC coupling penalties"
        );
    }

    /// Charge-reservoir sufficiency: an active IC with `I_dd = 1.0` A demand, driven by a
    /// single 100 pF cap, clearly cannot meet its datasheet rating under 3.3 V / 5 ns
    /// switching — `I_per_cap = 100e-12 * 3.3 / 5e-9 = 0.066 A << 1.0 A` ⇒ violation. The vacuous path
    /// (`i_dd_a = 0.0`) and the just-balanced path (`i_dd_a = 0.066`) must read clean: this
    /// proves the detector correctly distinguishes "no rating set" / "just enough" from
    /// "under-provisioned".
    #[test]
    fn charge_reservoir_violations_fire_on_under_provisioned_ic() {
        use crate::board::{LayerId, NetClassKind};
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};
        let ic_fp = FootprintDef::new(
            "U",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::ActiveIc,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        )
        .with_i_dd_a(0.0); // per-path override below
        let cap_fp = FootprintDef::new(
            "C",
            (Nm::from_mm(2.0), Nm::from_mm(1.0)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        )
        .with_capacitance_f(100e-12); // 100 pF → I_supply = 100e-12 * 3.3 / 5e-9 = 0.066 A

        let mk = |ic_i_dd: f64| {
            let spec =
                GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
            let mut b = Board::new(spec);
            let vcc = b.add_net("+3V3", NetClassKind::Power);
            let gnd = b.add_net("GND", NetClassKind::Ground);
            // IC has VCC on pad 0, GND on pad 1.
            let mut pwic_fp = ic_fp.clone();
            pwic_fp.i_dd_a = ic_i_dd;
            let pwic_lib = vec![pwic_fp, cap_fp.clone()];
            let ic = Component {
                fp: 0,
                nets: vec![Some(vcc), Some(gnd)],
                refdes: "U1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: None,
                locked: false,
                ..Default::default()
            };
            let cap = Component {
                fp: 1,
                nets: vec![Some(vcc), Some(gnd)],
                refdes: "C1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(9.5), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: Some(0),
                locked: false,
                ..Default::default()
            };
            (b, vec![ic, cap], pwic_lib)
        };

        // Path 1 — under-provisioned: I_dd = 1.0 A, single 100 pF cap delivers 0.066 A.
        let (b, comps, lib) = mk(1.0);
        let r = audit(&b, &comps, &lib, &DesignRules::holohv());
        assert_eq!(
            r.charge_reservoir_violations, 1,
            "I_dd = 1.0 A and one 100 pF cap (I_supply = 0.066 A) must underflow"
        );
        assert!(
            !r.hard_drc_clean(),
            "hard_drc_clean must trip on a single charge-reservoir violation"
        );
        assert!(
            r.risk_score >= 20.0,
            "risk_score must fold the charge-reservoir tier (20.0/violation)"
        );
        let ic_x_nm: i64 = Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0;
        assert!(
            r.hotspots.iter().any(|p| p.x.0 == ic_x_nm),
            "hotspot must mark the IC's VPP pad at x = 8 mm − 2 mm = 6 mm, not the cap"
        );

        // Path 2 — vacuous: i_dd_a = 0.0 ⇒ no rating set, detector skips silently.
        let (b2, comps2, lib2) = mk(0.0);
        let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
        assert_eq!(
            r2.charge_reservoir_violations, 0,
            "an IC with no datasheet I_dd must read vacuous (the validate-style pattern)"
        );

        // Path 3 — just balanced: i_dd_a = 0.066 A ⇒ supply exactly meets demand (no violation).
        let (b3, comps3, lib3) = mk(0.066);
        let r3 = audit(&b3, &comps3, &lib3, &DesignRules::holohv());
        assert_eq!(
            r3.charge_reservoir_violations, 0,
            "exactly balanced (I_supply ≋ I_dd) must not flag a violation"
        );

        // Path 4 — with no caps: i_supply = 0, but i_dd > 0 ⇒ still a violation.
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b4 = Board::new(spec);
        let vcc = b4.add_net("+3V3", NetClassKind::Power);
        let gnd = b4.add_net("GND", NetClassKind::Ground);
        let pads = vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ];
        let ic_only_fp = FootprintDef::new(
            "U",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::ActiveIc,
            pads,
        )
        .with_i_dd_a(1.0);
        let lib4 = vec![ic_only_fp];
        let comps4 = vec![Component {
            fp: 0,
            nets: vec![Some(vcc), Some(gnd)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        }];
        let r4 = audit(&b4, &comps4, &lib4, &DesignRules::holohv());
        assert_eq!(
            r4.charge_reservoir_violations, 1,
            "an IC with no associated cap must still violate — sum is 0, demand > 0"
        );
    }

    #[test]
    fn charge_reservoir_violations_fire_on_under_provisioned_buck() {
        // Mirror of `charge_reservoir_violations_fire_on_under_provisioned_ic`:
        // same `assoc_ic`-tied `Role::Decoupling` cap pool and
        // `I_supply = C · dV / dt` math (dv = 3.3 V, dt = 5 ns from `holohv()`);
        // the only delta is `Role::Power` (buck converter) at the consumer side.
        // Vacuous semantics (`i_dd_a ≤ 0.0`), the `+ 1e-12` slack, the first
        // power-pin pad hotspot, and `risk_score`-fold weight all match by construction
        // — the only new coverage is that the detector recognises the buck.
        use crate::board::{LayerId, NetClassKind};
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};
        let buck_fp = FootprintDef::new(
            "BUCK",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::Power,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        );
        let cap_fp = FootprintDef::new(
            "C",
            (Nm::from_mm(2.0), Nm::from_mm(1.0)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        )
        .with_capacitance_f(100e-12); // 100 pF → I_supply = 100e-12 · 3.3 / 5e-9 = 0.066 A
        let mk = |buck_i_dd_a: f64, cap_c_f: f64| {
            let spec =
                GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
            let mut b = Board::new(spec);
            let vin = b.add_net("+12V_IN", NetClassKind::Power);
            let gnd = b.add_net("GND", NetClassKind::Ground);
            // Buck has V_IN on pad 0 (power_pin = true), GND on pad 1.
            let mut ppwr_fp = buck_fp.clone();
            ppwr_fp.i_dd_a = buck_i_dd_a;
            let mut pcap_fp = cap_fp.clone();
            pcap_fp.capacitance_f = cap_c_f;
            let pwic_lib = vec![ppwr_fp, pcap_fp];
            let buck = Component {
                fp: 0,
                nets: vec![Some(vin), Some(gnd)],
                refdes: "BUCK1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: None,
                locked: false,
                ..Default::default()
            };
            let cap = Component {
                fp: 1,
                nets: vec![Some(vin), Some(gnd)],
                refdes: "C1".into(),
                placement: Placement {
                    pos: Point::new(Nm::from_mm(9.5), Nm::from_mm(10.0)),
                    rot: Rot::R0,
                },
                assoc_ic: Some(0), // ties the cap to the buck at fp 0.
                locked: false,
                ..Default::default()
            };
            (b, vec![buck, cap], pwic_lib)
        };

        // Path 1 — sized: cap 100 pF, buck I_dd = 0.066 A (≈ I_supply, exact
        // balance after the detector's `+ 1e-12` slack). Demand sat.
        let (b1, comps1, lib1) = mk(0.066, 100e-12);
        let r1 = audit(&b1, &comps1, &lib1, &DesignRules::holohv());
        assert_eq!(
            r1.charge_reservoir_violations, 0,
            "buck with I_dd == 0.066 A and one 100 pF cap (I_supply = 0.066 A) must not violate",
        );
        // No `hard_drc_clean()` assertion here — the bare 20 mm × 20 mm fixture
        // has no tracks, so other audit clauses (clearance/dangling/...) still
        // legitimately flag the empty-board baseline. The relevant guarantee
        // is just the charge-reservoir count, matching the IC test.

        // Path 2 — vacuous: i_dd_a = 0.0 ⇒ no datasheet rating set; the
        // detector silently skips the buck (the validate-style vacuous pattern
        // that mirrors the IC test's Path 2).
        let (b2, comps2, lib2) = mk(0.0, 100e-12);
        let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
        assert_eq!(
            r2.charge_reservoir_violations, 0,
            "a buck with no datasheet I_dd must read vacuous — detector skips silently",
        );

        // Path 3 — under-sized caps: same I_dd = 0.066 A, but cap 1 pF
        // (I_supply = 1e-12 · 3.3 / 5e-9 = 6.6e-4 A ≪ 0.066 A).
        let (b3, comps3, lib3) = mk(0.066, 1e-12);
        let r3 = audit(&b3, &comps3, &lib3, &DesignRules::holohv());
        assert_eq!(
            r3.charge_reservoir_violations, 1,
            "buck with I_dd = 0.066 A and one 1 pF cap must underflow (sum = 0.00066 A)",
        );
        assert!(
            !r3.hard_drc_clean(),
            "hard_drc_clean must trip on a single buck charge-reservoir violation",
        );
        assert!(
            r3.risk_score >= 20.0,
            "risk_score must fold the charge-reservoir tier (20.0/violation)",
        );
        // Hotspot at the buck's first power-pin pad: pad 0 at offset (-2.0, 0) mm
        // from the buck footprint centre (8.0, 10.0) mm → hotspot
        // p.x.0 == Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0 (i.e. 6.0 mm), not the cap.
        let buck_vin_x_nm: i64 = Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0;
        assert!(
            r3.hotspots.iter().any(|p| p.x.0 == buck_vin_x_nm),
            "hotspot must mark the buck's V_IN pad at x = 6.0 mm, not the cap",
        );

        // Path 4 — no caps: i_supply = 0 (no `Role::Decoupling` entries tied
        // to the buck via `assoc_ic`), but buck I_dd > 0 ⇒ the detector still
        // trips on its own. Mirrors the IC test's Path 4: `lib` has only the
        // consumer footprint; `comps` has only the consumer component (no cap
        // with `assoc_ic = Some(0)`). Sum = 0, demand > 0 ⇒ underflow.
        let spec4 =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b4 = Board::new(spec4);
        let vin4 = b4.add_net("+12V_IN", NetClassKind::Power);
        let gnd4 = b4.add_net("GND", NetClassKind::Ground);
        let pads_buck = vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ];
        let buck_only_fp = FootprintDef::new(
            "BUCK",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::Power,
            pads_buck,
        )
        .with_i_dd_a(0.066); // demands 0.066 A with no caps ⇒ sum = 0 ⇒ underflow
        let lib4 = vec![buck_only_fp];
        let comps4 = vec![Component {
            fp: 0,
            nets: vec![Some(vin4), Some(gnd4)],
            refdes: "BUCK1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        }];
        let r4 = audit(&b4, &comps4, &lib4, &DesignRules::holohv());
        assert_eq!(
            r4.charge_reservoir_violations, 1,
            "buck with no associated cap must still violate — sum is 0, demand > 0",
        );
    }

    #[test]
    fn charge_recycling_fires_on_nlevel_ic_without_cr_bus() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};
        let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
        let board = Board::new(spec); // no CR_* nets

        let fp = FootprintDef::new(
            "MD1715-DB",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        );
        let comp = Component {
            refdes: "U1".into(),
            fp: 0,
            nets: vec![],
            placement: Placement {
                pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let (count, pts) = detect_charge_recycling_violations_board(&board, &[comp], &[fp]);
        assert_eq!(
            count, 1,
            "one N-level IC without a CR bus net → 1 violation"
        );
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn charge_recycling_passes_when_cr_net_present() {
        use crate::place::component::Placement;
        use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};
        let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
        let mut board = Board::new(spec);
        board.add_net("CHR_BUS", NetClassKind::Signal);

        let fp = FootprintDef::new(
            "MAX14815-AAE",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        );
        let comp = Component {
            refdes: "U1".into(),
            fp: 0,
            nets: vec![],
            placement: Placement {
                pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let (count, _) = detect_charge_recycling_violations_board(&board, &[comp], &[fp]);
        assert_eq!(
            count, 0,
            "N-level IC with a CR bus net present → no violation"
        );
    }

    #[test]
    fn pulse_skip_fires_when_error_exceeds_tolerance() {
        let spec =
            GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
        let mut board = Board::new(spec);
        // 4 TX nets — with 80% skip fraction, rms_err = sqrt(0.8/4) = 0.447, >> 5% tol
        for i in 0..4 {
            board.add_net(format!("TX_{i}"), NetClassKind::Hv);
        }
        let mut rules = DesignRules::holohv();
        rules.max_skip_fraction = 0.8;
        rules.pressure_error_tol = 0.05;
        let (count, _) = detect_pulse_skip_violations(&board, &rules);
        assert_eq!(count, 1, "high skip fraction on few channels → violation");
    }

    #[test]
    fn pulse_skip_passes_within_tolerance() {
        let spec =
            GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
        let mut board = Board::new(spec);
        // 96 TX nets — with 20% skip: rms_err = sqrt(0.2/96) ≈ 0.046, just under 5% tol
        for i in 0..96 {
            board.add_net(format!("TX_{i}"), NetClassKind::Hv);
        }
        let mut rules = DesignRules::holohv();
        rules.max_skip_fraction = 0.2;
        rules.pressure_error_tol = 0.05;
        let (count, _) = detect_pulse_skip_violations(&board, &rules);
        assert_eq!(count, 0, "20% skip on 96 channels is within 5% error tol");
    }
