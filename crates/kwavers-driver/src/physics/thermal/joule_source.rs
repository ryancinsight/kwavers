//! Joule-heating source term from routed tracks: `f = q/k_eff` where the per-track power is
//! `I²R` (DC resistance from [`crate::physics::ampacity::track_resistance()`]). The output feeds
//! the electro-thermal solver in [`super::electrothermal::solve_electrothermal`].

use crate::geom::GridSpec;

/// Joule-heating source `f = q/k_eff` from routed tracks: each track dissipates `I²R` over the
/// cells it crosses, coupling the electrical solution into the thermal field (electro-thermal
/// co-analysis). `current_a(net)` is the net's RMS current; `copper_oz` the copper weight.
#[must_use]
pub fn joule_source(
    spec: GridSpec,
    board: &crate::board::Board,
    current_a: impl Fn(crate::board::NetId) -> f64,
    copper_oz: f64,
    k_eff: f64,
    thickness: f64,
) -> Vec<f64> {
    let mut f = vec![0.0f64; spec.nx * spec.ny];
    let width_m = thickness; // conducting thickness reused for volumetric spread
    let cell_area = (spec.pitch.to_mm() * 1.0e-3).powi(2);
    let vol = cell_area * width_m;
    for t in &board.tracks {
        let i = current_a(t.net);
        if i <= 0.0 {
            continue;
        }
        let len_m = t.start.euclid(t.end) * 1.0e-9; // nm → m
        if len_m <= 0.0 {
            continue;
        }
        let r = crate::physics::ampacity::track_resistance(
            len_m,
            t.width.to_mm() * 1.0e-3,
            copper_oz,
        );
        let power = i * i * r; // I²R over this segment
        // Spread over the two endpoint cells.
        for end in [t.start, t.end] {
            let (ix, iy) = spec.cell_of(end);
            f[iy * spec.nx + ix] += (power / 2.0) / vol / k_eff;
        }
    }
    f
}
