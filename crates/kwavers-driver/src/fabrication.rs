//! Fabrication-readiness verification.
//!
//! Lesson learned: a board can pass DRC, ERC, LVS, assembly, and the physics suite yet **not be
//! fabrication-complete**, because it is placed and routed on *synthesized abstraction* footprints
//! (a generic pad ring, a routed subset of pins) rather than the exact manufacturer footprint. A
//! green check set then *hides* that limitation. This axis makes it explicit: it reports, per
//! component, whether it sits on a real imported footprint (one carrying pin identifiers, populated by
//! [`crate::place::footprint_import::import_kicad_mod`]) or an abstraction, and gates a separate
//! `ready` flag so the limitation is presented rather than buried under `all_pass`.

use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

/// Fabrication-readiness findings over the placed components.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FabReadiness {
    /// Components placed on a real (imported, pin-named) manufacturer footprint.
    pub exact: usize,
    /// Components placed on a synthesized abstraction footprint (no pin map).
    pub abstraction: usize,
    /// Distinct abstraction footprint names still in use (the parts that block fabrication).
    pub abstraction_footprints: Vec<String>,
    /// True only when **every** component is on a real footprint — i.e. the board is fabrication-ready
    /// (gerbers would assemble the genuine parts), not merely DRC-clean against abstractions.
    pub ready: bool,
}

/// A footprint is treated as the **exact** manufacturer part when it carries pin identifiers
/// (`pad_names`), which only the `.kicad_mod` importer populates; a synthesized abstraction
/// (`perimeter_pads`, `bga`, `two_row_header`) leaves them empty.
#[must_use]
pub fn is_exact_footprint(fp: &FootprintDef) -> bool {
    !fp.pad_names.is_empty()
}

/// Assess whether the placed design is fabrication-ready: every component on a real imported footprint.
/// Reports the count and the distinct abstraction footprints that still need replacing.
#[must_use]
pub fn fabrication_readiness(comps: &[Component], lib: &[FootprintDef]) -> FabReadiness {
    let mut r = FabReadiness::default();
    // Deduplicate by footprint library index (integer, no heap alloc) so the name is cloned
    // at most once per distinct abstraction footprint rather than twice per occurrence.
    let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for c in comps {
        let fp = &lib[c.fp];
        if is_exact_footprint(fp) {
            r.exact += 1;
        } else {
            r.abstraction += 1;
            if seen.insert(c.fp) {
                r.abstraction_footprints.push(fp.name.clone());
            }
        }
    }
    r.ready = r.abstraction == 0 && r.exact > 0;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Nm, Point};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

    fn fp_named(name: &str, with_names: bool) -> FootprintDef {
        let mut fp = FootprintDef::new(
            name,
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![crate::board::LayerId(0)],
                power_pin: false,
            }],
        );
        if with_names {
            fp.pad_names = vec!["1".into()]; // as the importer would populate
        }
        fp
    }

    fn comp(fp: usize) -> Component {
        Component {
            fp,
            nets: vec![None],
            refdes: "U".into(),
            placement: Placement {
                pos: Point::new(Nm(0), Nm(0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        }
    }

    #[test]
    fn flags_abstraction_footprints_as_not_ready() {
        let lib = vec![
            fp_named("HV7355-abstract", false),
            fp_named("ISO7740-real", true),
        ];
        let comps = vec![comp(0), comp(0), comp(1)];
        let r = fabrication_readiness(&comps, &lib);
        assert_eq!(r.exact, 1);
        assert_eq!(r.abstraction, 2);
        assert_eq!(
            r.abstraction_footprints,
            vec!["HV7355-abstract".to_string()]
        );
        assert!(
            !r.ready,
            "any abstraction footprint ⇒ not fabrication-ready"
        );
    }

    #[test]
    fn all_exact_is_ready() {
        let lib = vec![fp_named("HV7355-real", true)];
        let comps = vec![comp(0), comp(0)];
        let r = fabrication_readiness(&comps, &lib);
        assert!(r.ready && r.abstraction == 0 && r.exact == 2);
    }
}
