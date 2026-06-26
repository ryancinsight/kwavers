use crate::board::{Board, NetId};
use crate::geom::Nm;

/// Copper thickness in metres for a weight in ounces (1 oz ≈ 34.8 µm).
#[inline]
#[must_use]
pub fn copper_thickness_m(copper_oz: f64) -> f64 {
    copper_oz * 34.8e-6
}

/// Minimum IPC-2221 track width for `current_a` at a `dt_c` °C rise.
#[must_use]
pub fn ipc2221_min_width(current_a: f64, dt_c: f64, copper_oz: f64, internal: bool) -> Nm {
    if current_a <= 0.0 || dt_c <= 0.0 {
        return Nm(0);
    }
    let k = if internal { 0.024 } else { 0.048 };
    let area_mil2 = (current_a / (k * dt_c.powf(0.44))).powf(1.0 / 0.725);
    let thickness_mil = 1.378 * copper_oz; // 1 oz ≈ 1.378 mil
    let width_mil = area_mil2 / thickness_mil;
    Nm::from_mm(width_mil * 0.0254) // mil → mm
}

/// Current density (A/mm²) in a track of `width_m` and copper weight `oz`.
#[must_use]
pub fn current_density_a_per_mm2(current_a: f64, width_m: f64, copper_oz: f64) -> f64 {
    let area_mm2 = (width_m * 1.0e3) * (copper_thickness_m(copper_oz) * 1.0e3);
    if area_mm2 <= 0.0 {
        return f64::INFINITY;
    }
    current_a / area_mm2
}

/// A net whose routed track width is below the IPC-2221 minimum for its current.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmpacityDeficit {
    /// The under-sized net.
    pub net: NetId,
    /// Narrowest routed width on the net (mm).
    pub actual_mm: f64,
    /// Required width for the net's current (mm).
    pub required_mm: f64,
}

/// Check every routed net against its ampacity requirement; return the deficits.
///
/// `current_a(net)` supplies the net's design (RMS) current; `dt_c` the allowed rise.
pub fn ampacity_check(
    board: &Board,
    current_a: impl Fn(NetId) -> f64,
    dt_c: f64,
    copper_oz: f64,
) -> Vec<AmpacityDeficit> {
    // Narrowest routed width per net.
    let mut narrowest: std::collections::BTreeMap<u32, i64> = std::collections::BTreeMap::new();
    for t in &board.tracks {
        let e = narrowest.entry(t.net.0).or_insert(i64::MAX);
        *e = (*e).min(t.width.0);
    }
    let mut out = Vec::new();
    for (net_id, w) in narrowest {
        let net = NetId(net_id);
        let need = ipc2221_min_width(current_a(net), dt_c, copper_oz, false);
        if w < need.0 {
            out.push(AmpacityDeficit {
                net,
                actual_mm: Nm(w).to_mm(),
                required_mm: need.to_mm(),
            });
        }
    }
    out
}
