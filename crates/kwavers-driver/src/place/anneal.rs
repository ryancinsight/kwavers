//! Simulated-annealing placer.
//!
//! Minimises [`energy`] over component placements with translation moves plus footprint-policy
//! constrained rotation moves. The PRNG is a seeded SplitMix64 so a given seed yields a
//! bit-identical placement — placement is reproducible and testable, never `rand`-driven.

use crate::geom::{Nm, Point};
use crate::place::component::Component;
use crate::place::energy::{energy, EnergyTerms, PlaceConfig};
use crate::place::footprint::{FootprintDef, Role};

/// Seeded SplitMix64 — small, dependency-free, deterministic.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// A uniform step in `[-span, span]` mm, as `Nm`.
    fn step(&mut self, span_mm: f64) -> Nm {
        Nm::from_mm((self.unit() * 2.0 - 1.0) * span_mm)
    }
}

/// Annealing schedule.
#[derive(Debug, Clone, Copy)]
pub struct AnnealParams {
    /// Number of proposed moves.
    pub steps: usize,
    /// Initial temperature.
    pub t0: f64,
    /// Geometric cooling factor per step (`< 1`).
    pub cooling: f64,
    /// Initial translation step (mm); decays with temperature.
    pub step_mm: f64,
    /// Probability of a rotation move (vs a translation). The selected footprint's
    /// `crate::place::rotation::RotationPolicy` still gates whether a proposed rotation is legal.
    pub rot_prob: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for AnnealParams {
    fn default() -> Self {
        AnnealParams {
            steps: 20_000,
            t0: 50.0,
            cooling: 0.9995,
            step_mm: 8.0,
            rot_prob: 0.15,
            seed: 0x1234_5678,
        }
    }
}

/// Clamp a component centre so its courtyard stays inside the board keep-in margin.
fn clamp_inside(c: &mut Component, lib: &[FootprintDef], cfg: &PlaceConfig) {
    let (w, h) = c.placement.rot.apply_size(lib[c.fp].courtyard);
    let hw = Nm(w.0 / 2);
    let hh = Nm(h.0 / 2);
    let lo_x = cfg.margin.0 + hw.0;
    let hi_x = cfg.board.0 .0 - cfg.margin.0 - hw.0;
    let lo_y = cfg.margin.0 + hh.0;
    let hi_y = cfg.board.1 .0 - cfg.margin.0 - hh.0;
    let x = c.placement.pos.x.0.clamp(lo_x.min(hi_x), hi_x.max(lo_x));
    let y = c.placement.pos.y.0.clamp(lo_y.min(hi_y), hi_y.max(lo_y));
    c.placement.pos = Point::new(Nm(x), Nm(y));
}

/// Compute the force vector acting on component `idx` (in mm) due to overlap, wirelength,
/// decoupling, and termination constraints.
fn compute_force(
    idx: usize,
    comps: &[Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
) -> (f64, f64) {
    let mut fx = 0.0;
    let mut fy = 0.0;
    let c = &comps[idx];
    let pos_mm_x = c.placement.pos.x.to_mm();
    let pos_mm_y = c.placement.pos.y.to_mm();
    let rect_clear = c.courtyard(lib).inflate(Nm(cfg.courtyard_clearance.0 / 2));

    // 1. Repulsive overlap force
    let w_overlap = cfg.weights.overlap;
    for (j, other) in comps.iter().enumerate() {
        if j == idx {
            continue;
        }
        let other_rect = other
            .courtyard(lib)
            .inflate(Nm(cfg.courtyard_clearance.0 / 2));
        let area = rect_clear.overlap_area(other_rect);
        if area > 0.0 {
            let area_mm2 = area * 1.0e-12;
            let ox = pos_mm_x - other.placement.pos.x.to_mm();
            let oy = pos_mm_y - other.placement.pos.y.to_mm();
            let d = (ox.powi(2) + oy.powi(2)).sqrt();
            if d > 1.0e-6 {
                fx += w_overlap * area_mm2 * (ox / d);
                fy += w_overlap * area_mm2 * (oy / d);
            } else {
                let angle = (idx + j) as f64 * 0.1;
                fx += w_overlap * area_mm2 * angle.cos();
                fy += w_overlap * area_mm2 * angle.sin();
            }
        }
    }

    // 2. Attractive wirelength force (HPWL-like spring pulling to net centroid)
    let w_hpwl = cfg.weights.hpwl;
    for net_opt in &c.nets {
        if let Some(net) = *net_opt {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut count = 0;
            for (other_idx, other) in comps.iter().enumerate() {
                if other_idx == idx {
                    continue;
                }
                for (other_pos, _layers, other_net) in other.placed_pads(lib) {
                    if other_net == Some(net) {
                        sum_x += other_pos.x.to_mm();
                        sum_y += other_pos.y.to_mm();
                        count += 1;
                    }
                }
            }
            if count > 0 {
                let cx = sum_x / count as f64;
                let cy = sum_y / count as f64;
                fx += w_hpwl * (cx - pos_mm_x);
                fy += w_hpwl * (cy - pos_mm_y);
            }
        }
    }

    // 3. Decoupling attraction force
    let w_dec = cfg.weights.decoupling;
    if matches!(lib[c.fp].role, Role::Decoupling) {
        if let Some(ic) = c.assoc_ic {
            let ic_c = &comps[ic];
            let mut nearest_pin = None;
            let mut nearest_dist = f64::INFINITY;
            for (k, pad) in lib[ic_c.fp].pads.iter().enumerate() {
                if pad.power_pin {
                    let pin_pos = ic_c.pad_pos(lib, k);
                    let d = c.placement.pos.euclid(pin_pos) * 1.0e-6;
                    if d < nearest_dist {
                        nearest_dist = d;
                        nearest_pin = Some(pin_pos);
                    }
                }
            }
            if let Some(pin_pos) = nearest_pin {
                fx += w_dec * (pin_pos.x.to_mm() - pos_mm_x);
                fy += w_dec * (pin_pos.y.to_mm() - pos_mm_y);
            }
        }
    } else {
        for other in comps {
            if matches!(lib[other.fp].role, Role::Decoupling) && other.assoc_ic == Some(idx) {
                let cap_pos = other.placement.pos;
                let mut nearest_pin = None;
                let mut nearest_dist = f64::INFINITY;
                for (k, pad) in lib[c.fp].pads.iter().enumerate() {
                    if pad.power_pin {
                        let pin_pos = c.pad_pos(lib, k);
                        let d = cap_pos.euclid(pin_pos) * 1.0e-6;
                        if d < nearest_dist {
                            nearest_dist = d;
                            nearest_pin = Some(pin_pos);
                        }
                    }
                }
                if let Some(pin_pos) = nearest_pin {
                    fx += w_dec * (cap_pos.x.to_mm() - pin_pos.x.to_mm());
                    fy += w_dec * (cap_pos.y.to_mm() - pin_pos.y.to_mm());
                }
            }
        }
    }

    // 4. Termination attraction force.
    let w_term = cfg.weights.termination;
    if w_term > 0.0
        && c.fp < lib.len()
        && matches!(lib[c.fp].role, Role::Passive)
        && c.refdes.starts_with('R')
    {
        let mut nearest_pad = None;
        let mut nearest_dist = f64::INFINITY;
        for (term_pos, _term_layers, term_net) in c.placed_pads(lib) {
            let Some(net) = term_net else { continue };
            for active in comps {
                if active.fp >= lib.len() || !matches!(lib[active.fp].role, Role::ActiveIc) {
                    continue;
                }
                for (active_pos, _active_layers, active_net) in active.placed_pads(lib) {
                    if active_net == Some(net) {
                        let d = term_pos.euclid(active_pos) * 1.0e-6;
                        if d < nearest_dist {
                            nearest_dist = d;
                            nearest_pad = Some(active_pos);
                        }
                    }
                }
            }
        }
        if let Some(active_pos) = nearest_pad {
            fx += w_term * (active_pos.x.to_mm() - pos_mm_x);
            fy += w_term * (active_pos.y.to_mm() - pos_mm_y);
        }
    }

    (fx, fy)
}

/// Anneal `comps` in place; returns the final energy breakdown. Movable components are those whose
/// index is in `movable` (others — e.g. a fixed connector — stay put but still contribute energy).
pub fn anneal(
    comps: &mut [Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
    movable: &[usize],
    params: &AnnealParams,
    congestion: Option<&crate::place::energy::CongestionField>,
) -> EnergyTerms {
    if movable.is_empty() {
        return energy(comps, lib, cfg, congestion);
    }
    let mut rng = Rng(params.seed);
    let mut cur = energy(comps, lib, cfg, congestion).total;
    let mut t = params.t0;
    let initial_rot: Vec<_> = comps.iter().map(|c| c.placement.rot).collect();

    for _ in 0..params.steps {
        let idx = movable[rng.below(movable.len())];
        let prev = comps[idx].placement;

        let mut translated = true;
        if rng.unit() < params.rot_prob {
            let policy = lib[comps[idx].fp].rotation_policy;
            if let Some(rot) = comps[idx]
                .placement
                .rot
                .next_allowed(initial_rot[idx], policy)
            {
                comps[idx].placement.rot = rot;
                translated = false;
            }
        }
        if translated {
            let span = params.step_mm * (t / params.t0).sqrt().max(0.05);
            let (fx, fy) = compute_force(idx, comps, lib, cfg);
            let f_mag = (fx.powi(2) + fy.powi(2)).sqrt();
            let (fnx, fny) = if f_mag > 1.0e-6 {
                (fx / f_mag, fy / f_mag)
            } else {
                (0.0, 0.0)
            };

            let rx = rng.step(span).to_mm();
            let ry = rng.step(span).to_mm();

            // Apply a force-directed proposal bias (alpha = 0.3)
            let alpha = 0.2;
            let sx = (1.0 - alpha) * rx + alpha * fnx * span;
            let sy = (1.0 - alpha) * ry + alpha * fny * span;

            comps[idx].placement.pos = Point::new(
                comps[idx].placement.pos.x + Nm::from_mm(sx),
                comps[idx].placement.pos.y + Nm::from_mm(sy),
            );
        }
        clamp_inside(&mut comps[idx], lib, cfg);

        let cand = energy(comps, lib, cfg, congestion).total;
        let de = cand - cur;
        if de <= 0.0 || rng.unit() < (-de / t).exp() {
            cur = cand; // accept
        } else {
            comps[idx].placement = prev; // reject
        }
        t *= params.cooling;
    }
    energy(comps, lib, cfg, congestion)
}
