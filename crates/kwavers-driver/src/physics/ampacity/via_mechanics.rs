/// Annular ring (mm) of a via: radial copper around the drill, `(diameter − drill)/2`. Must meet
/// the fab's minimum (DFM) or the drill can break out of the pad.
#[must_use]
pub fn annular_ring_mm(diameter_mm: f64, drill_mm: f64) -> f64 {
    (diameter_mm - drill_mm) / 2.0
}

/// Plated-through-hole aspect ratio (board thickness / drill). Standard fabs reliably plate up to
/// ~10:1; higher risks voids — a fab DFM limit on the smallest usable via for a given stack.
#[must_use]
pub fn pth_aspect_ratio(board_thickness_m: f64, drill_m: f64) -> f64 {
    if drill_m <= 0.0 {
        return f64::INFINITY;
    }
    board_thickness_m / drill_m
}
