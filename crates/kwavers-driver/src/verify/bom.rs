//! BOM — bill-of-materials sanity.
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

/// Bill-of-materials sanity findings.
#[derive(Debug, Clone, Default)]
pub struct BomReport {
    /// Number of distinct placed parts.
    pub part_count: usize,
    /// Reference designators used more than once.
    pub duplicate_refdes: Vec<String>,
    /// Parts whose footprint carries no 3D model attachment (a render/assembly gap, not fatal).
    pub missing_model: Vec<String>,
    /// Whether the BOM is internally consistent (no duplicate refdes).
    pub pass: bool,
}

/// Validate the bill of materials: every part has a unique reference designator, and flag parts
/// whose footprint has no 3D model (an assembly/render completeness gap).
#[must_use]
pub fn bom(comps: &[Component], lib: &[FootprintDef]) -> BomReport {
    use std::collections::HashMap;
    let mut r = BomReport {
        part_count: comps.len(),
        ..Default::default()
    };
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for c in comps {
        *seen.entry(c.refdes.as_str()).or_insert(0) += 1;
        if lib[c.fp].model.is_none() {
            r.missing_model.push(c.refdes.clone());
        }
    }
    for (rd, n) in seen {
        if n > 1 {
            r.duplicate_refdes.push(rd.to_string());
        }
    }
    r.duplicate_refdes.sort();
    r.pass = r.duplicate_refdes.is_empty();
    r
}
