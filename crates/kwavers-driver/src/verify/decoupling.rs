//! Decoupling proximity — PDN constraint.
use crate::place::component::Component;
use crate::place::footprint::{FootprintDef, Role};

/// Decoupling-placement findings: bypass capacitors too far from the IC they decouple.
#[derive(Debug, Clone, Default)]
pub struct DecouplingReport {
    /// `(cap refdes, distance mm, budget mm)` for each decoupling cap beyond its proximity budget.
    pub far_caps: Vec<(String, f64, f64)>,
    /// Whether every decoupling cap is within its budget of its associated IC.
    pub pass: bool,
}

/// Check that each bypass capacitor associated with an IC sits within
/// `max_dist_mm` centre-to-centre of the IC it bypasses — the PDN constraint that keeps the
/// connection-loop inductance low enough for the cap to stay effective through its target band (see
/// [`crate::physics::pdn::max_decoupling_distance_mm`] for the budget derivation). Caps with no association are
/// skipped (a bulk/rail cap, not pin-bypassing).
#[must_use]
pub fn decoupling_proximity(
    comps: &[Component],
    lib: &[FootprintDef],
    max_dist_mm: f64,
) -> DecouplingReport {
    let mut far_caps = Vec::new();
    for c in comps {
        if !matches!(lib[c.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic) = c.assoc_ic else { continue };
        let ic_comp = &comps[ic];
        let ic_fp = &lib[ic_comp.fp];
        // The decoupling loop runs cap → the IC's nearest *power pin*, not its centre — a large IC's
        // supply pads sit far from centre, so a centre measure would be unsatisfiable. Fall back to
        // the IC centre only if the footprint marks no power pin.
        let mut d_nm = f64::INFINITY;
        for (k, pad) in ic_fp.pads.iter().enumerate() {
            if pad.power_pin {
                d_nm = d_nm.min(c.placement.pos.euclid(ic_comp.pad_pos(lib, k)));
            }
        }
        if !d_nm.is_finite() {
            d_nm = c.placement.pos.euclid(ic_comp.placement.pos);
        }
        let d = d_nm * 1.0e-6; // nm → mm
        if d > max_dist_mm {
            far_caps.push((c.refdes.clone(), d, max_dist_mm));
        }
    }
    far_caps.sort_by(|a, b| a.0.cmp(&b.0));
    DecouplingReport {
        pass: far_caps.is_empty(),
        far_caps,
    }
}
