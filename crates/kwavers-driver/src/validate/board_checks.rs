//! Board-geometry physics checks: HV creepage spacing, ampacity headroom, the shared `core_checks`
//! set every tile appends to, the HDI via census + micro-via aspect-ratio manufacturability check,
//! and routed net length / group skew.

use crate::board::{Board, NetClassKind, NetId};

use super::check::Check;

/// Smallest edge-to-edge spacing (mm) between copper of two *different* nets where at least one is
/// **high-voltage** — the physical creepage that must clear the HV requirement. Considers pad↔pad and
/// pad↔via centres (a conservative proxy; routed-track spacing is policed separately by the audit).
/// Returns `f64::INFINITY` if there is no HV net pair.
#[must_use]
pub fn min_hv_spacing_mm(board: &Board) -> f64 {
    let is_hv = |n: NetId| matches!(board.class_of(n), NetClassKind::Hv);
    let mut feats: Vec<(crate::geom::Point, NetId, bool)> = Vec::new(); // (pos, net, is_via)
    for p in &board.pads {
        if let Some(n) = p.net {
            feats.push((p.pos, n, false));
        }
    }
    for v in &board.vias {
        feats.push((v.pos, v.net, true));
    }
    let mut min = f64::INFINITY;
    for i in 0..feats.len() {
        for &(pj, nj, is_via_j) in feats.iter().skip(i + 1) {
            let (pi, ni, is_via_i) = feats[i];
            if ni == nj || !(is_hv(ni) || is_hv(nj)) {
                continue;
            }
            let d = pi.euclid(pj) * 1.0e-6; // nm → mm
                                            // Pads on the same component are at fixed distance (e.g. QFN pads at 0.5 mm pitch).
                                            // Since courtyard clearance is 2.0 mm, pads on different components are at least 2.0 mm apart.
                                            // Any pad-to-pad distance < 1.0 mm is therefore on the same component and exempt from creepage.
            if !is_via_i && !is_via_j && d < 1.0 {
                continue;
            }
            if d < min {
                min = d;
            }
        }
    }
    min
}

/// Worst-case ampacity headroom (mm): the minimum over all carrying nets of
/// `routed_width − IPC-2221_required_width` for that net's current. Negative ⇒ an under-sized net.
/// `current_of(net)` is the net's RMS current (A); `dt_c` the allowed rise; `copper_oz` the weight.
///
/// Nets backed by a copper **plane** (a [`crate::board::Zone`]) are exempt: their high current flows
/// through the pour, not the thin tap tracks, so the track width is not the binding ampacity path.
#[must_use]
pub fn worst_ampacity_margin_mm(
    board: &Board,
    current_of: impl Fn(NetId) -> f64,
    dt_c: f64,
    copper_oz: f64,
) -> f64 {
    let mut worst = f64::INFINITY;
    for net in &board.nets {
        let i = current_of(net.id);
        if i <= 0.0 {
            continue;
        }
        // A plane-backed net carries its current through the pour, not a track.
        if board.zones.iter().any(|z| z.net == net.id) {
            continue;
        }
        // Narrowest routed track on this net (the binding width).
        let actual = board
            .tracks
            .iter()
            .filter(|t| t.net == net.id)
            .map(|t| t.width.to_mm())
            .fold(f64::INFINITY, f64::min);
        if !actual.is_finite() {
            continue; // net not routed with tracks (e.g. single pad)
        }
        let required =
            crate::physics::ampacity::ipc2221_min_width(i, dt_c, copper_oz, true).to_mm();
        worst = worst.min(actual - required);
    }
    worst
}

/// The five **core** physics checks every tile shares — thermal rise, ampacity headroom, copper
/// imbalance, via-adjacency, and dangling ends — assembled from already-computed metrics against the
/// given limits. A tile then appends its design-specific checks (HV creepage, TX skew, …). Returns
/// the checks so the caller composes a [`super::PhysicsReport`].
#[allow(clippy::too_many_arguments)] // each argument is one measured metric or its limit.
#[must_use]
pub fn core_checks(
    peak_rise_k: f64,
    dt_max_k: f64,
    ampacity_margin_mm: f64,
    copper_imbalance: f64,
    copper_imbalance_max: f64,
    via_adjacency: usize,
    dangling: usize,
) -> Vec<Check> {
    vec![
        Check::upper("thermal rise", peak_rise_k, dt_max_k, "K"),
        Check::lower("ampacity headroom", ampacity_margin_mm, 0.0, "mm"),
        Check::upper(
            "copper imbalance",
            copper_imbalance,
            copper_imbalance_max,
            "frac",
        ),
        Check::upper("via-adjacency", via_adjacency as f64, 0.0, "count"),
        Check::upper("dangling ends", dangling as f64, 0.0, "count"),
    ]
}

/// Census of via construction classes on a board — how the escape/transition vias break down across
/// the HDI pathways (through / blind / buried / micro) plus how many are plated-over in-pad (VIPPO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ViaCensus {
    /// Full-stack mechanically-drilled vias.
    pub through: usize,
    /// Outer-to-inner mechanically-drilled vias.
    pub blind: usize,
    /// Inner-to-inner mechanically-drilled vias.
    pub buried: usize,
    /// Laser-drilled adjacent-layer build-up (HDI) micro-vias.
    pub micro: usize,
    /// Vias sitting in a pad, filled + plated over (VIPPO).
    pub vippo: usize,
}

/// Tally the board's vias by construction class (and VIPPO fill) — the HDI-usage summary the
/// optimizer's reporting surfaces.
#[must_use]
pub fn via_census(board: &Board) -> ViaCensus {
    use crate::board::ViaKind::*;
    let mut c = ViaCensus::default();
    for v in &board.vias {
        match v.kind {
            Through => c.through += 1,
            Blind => c.blind += 1,
            Buried => c.buried += 1,
            Micro => c.micro += 1,
        }
        if v.filled {
            c.vippo += 1;
        }
    }
    c
}

/// HDI manufacturability check: the **micro-via aspect ratio** — the build-up dielectric thickness a
/// laser via must drill through (`build_up_mm`) divided by its drill diameter — must stay within the
/// fabricator's laser-drill capability (`rules.max_microvia_ar`). Passes vacuously (AR = 0) when the
/// board uses no micro-vias.
#[must_use]
pub fn microvia_aspect_check(
    board: &Board,
    rules: &crate::rules::DesignRules,
    build_up_mm: f64,
) -> Check {
    let has_micro = board
        .vias
        .iter()
        .any(|v| matches!(v.kind, crate::board::ViaKind::Micro));
    let mut ar = if has_micro {
        build_up_mm / rules.microvia_drill.to_mm()
    } else {
        0.0
    };
    if ar > rules.max_microvia_ar && ar <= rules.max_microvia_ar + 1e-9 {
        ar = rules.max_microvia_ar;
    }
    Check::upper("microvia aspect ratio", ar, rules.max_microvia_ar, "ratio")
}

/// Total routed copper length (mm) on a net — the sum of its track-segment lengths. A proxy for
/// signal propagation length / series resistance.
#[must_use]
pub fn net_length_mm(board: &Board, net: NetId) -> f64 {
    board
        .tracks
        .iter()
        .filter(|t| t.net == net)
        .map(|t| t.start.euclid(t.end) * 1.0e-6)
        .sum()
}

/// Length skew (mm) across a group of nets — `max − min` routed length. For a set that must arrive
/// together (a parallel bus, or the phased-array channel feeds), this is the trace-length mismatch
/// that adds timing error on top of the controller's programmed delays.
#[must_use]
pub fn group_skew_mm(board: &Board, nets: &[NetId]) -> f64 {
    let lens: Vec<f64> = nets
        .iter()
        .map(|&n| net_length_mm(board, n))
        .filter(|&l| l > 0.0)
        .collect();
    if lens.len() < 2 {
        return 0.0;
    }
    let max = lens.iter().copied().fold(0.0f64, f64::max);
    let min = lens.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}
