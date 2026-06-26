//! Assembly — component-to-component overlap / courtyard spacing.
use crate::board::LayerId;
use crate::geom::Nm;
use crate::place::component::{component_clearance_violations, Component};
use crate::place::footprint::FootprintDef;

/// A footprint whose 3D model exceeds its courtyard: `(footprint name, model (w,h) mm,
/// courtyard (w,h) mm)`.
pub type OversizedModel = (String, (f64, f64), (f64, f64));

/// Component mounted in a way that adds avoidable assembly process steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblySideViolation {
    /// Reference designator of the offending component.
    pub refdes: String,
    /// Footprint name.
    pub footprint: String,
    /// Violated side-placement rule.
    pub reason: String,
}

/// Assembly/manufacturing findings over placed component bodies.
#[derive(Debug, Clone, Default)]
pub struct AssemblyReport {
    /// Component pairs whose clearance-inflated courtyards overlap.
    pub clearance_violations: Vec<crate::place::component::ComponentClearanceViolation>,
    /// Footprints whose 3D model body is larger than the declared courtyard — the overlap check runs
    /// on courtyards, so an oversized model can collide in the render while the check passes.
    pub oversized_models: Vec<OversizedModel>,
    /// Components whose pad layer set violates the top-side assembly policy.
    pub side_violations: Vec<AssemblySideViolation>,
    /// Whether every assembly check passed.
    pub pass: bool,
}

/// Parse a `…<w>x<h>mm…` body size out of a KiCad 3D-model filename (e.g.
/// `LQFP-100_14x14mm_P0.5mm` → `(14.0, 14.0)`, `…_5.0x3.2mm` → `(5.0, 3.2)`). Returns the first
/// `AxBmm` token found, or `None` if the name carries no size.
///
/// Marked `pub(super)` so the slice-wide [`crate::verify::tests`] (a sibling module of `assembly`
/// under the [`crate::verify`] facade) can drive the unit test
/// `model_dims_parsed_from_kicad_filenames` against the same parser this module uses. Keeping the
/// implementation here means a future change to the parsing rules (e.g. accepting comma decimals) is
/// reflected in production and tests simultaneously.
pub(super) fn model_dims_mm(path: &str) -> Option<(f64, f64)> {
    // Drop the model file extension so a size token sitting right before it (e.g. `…_5.0x3.2mm.step`)
    // is still a clean `AxBmm` token after splitting on the path/name separators.
    let stem = path
        .strip_suffix(".step")
        .or_else(|| path.strip_suffix(".stp"))
        .or_else(|| path.strip_suffix(".wrl"))
        .unwrap_or(path);
    for token in stem.split(['_', '/', '-', '\\']) {
        if let Some(dims) = token.strip_suffix("mm") {
            if let Some((a, b)) = dims.split_once('x') {
                if let (Ok(w), Ok(h)) = (a.parse::<f64>(), b.parse::<f64>()) {
                    return Some((w, h));
                }
            }
        }
    }
    None
}

/// Validate package assembly: (1) courtyard spacing using the same clearance the placer optimizes,
/// and (2) that each footprint's **courtyard encloses its 3D model body** (parsed from the model
/// filename) in some 90° orientation — otherwise a courtyard-based overlap check is unsound (the body
/// can collide in the render while the spacing check passes, the oscillator-over-SOIC failure).
#[must_use]
pub fn assembly(
    comps: &[Component],
    lib: &[FootprintDef],
    courtyard_clearance: Nm,
) -> AssemblyReport {
    let clearance_violations = component_clearance_violations(comps, lib, courtyard_clearance);
    let mut oversized_models = Vec::new();
    let mut side_violations = Vec::new();
    for fp in lib {
        if let Some((path, _, _, envelope)) = &fp.model {
            if let Some((mw, mh)) = envelope.or_else(|| model_dims_mm(path)) {
                let (cw, ch) = (fp.courtyard.0.to_mm(), fp.courtyard.1.to_mm());
                // The model fits if it fits in either courtyard orientation (the body may be rotated).
                let fits =
                    (cw + 1e-9 >= mw && ch + 1e-9 >= mh) || (cw + 1e-9 >= mh && ch + 1e-9 >= mw);
                if !fits {
                    oversized_models.push((fp.name.clone(), (mw, mh), (cw, ch)));
                }
            }
        }
    }
    for c in comps {
        let fp = &lib[c.fp];
        if fp.pads.is_empty() {
            continue;
        }
        let is_smd = fp.pads.iter().all(|pad| pad.layers.len() == 1);
        let is_through_hole = fp.pads.iter().any(|pad| pad.layers.len() > 1);
        if is_smd
            && fp
                .pads
                .iter()
                .any(|pad| pad.layers.first() != Some(&LayerId(0)))
        {
            side_violations.push(AssemblySideViolation {
                refdes: c.refdes.clone(),
                footprint: fp.name.clone(),
                reason: "SMD pads must be on the top assembly side".into(),
            });
        }
        if is_through_hole
            && fp
                .pads
                .iter()
                .filter(|pad| pad.layers.len() > 1)
                .any(|pad| !pad.layers.contains(&LayerId(0)))
        {
            side_violations.push(AssemblySideViolation {
                refdes: c.refdes.clone(),
                footprint: fp.name.clone(),
                reason: "through-hole pads must include the top assembly side".into(),
            });
        }
    }
    AssemblyReport {
        pass: clearance_violations.is_empty()
            && oversized_models.is_empty()
            && side_violations.is_empty(),
        clearance_violations,
        oversized_models,
        side_violations,
    }
}
