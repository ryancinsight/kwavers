//! IR-drop solver on a routed power net — relocated from `crate::pdn` at Phase 3b so the
//! electro-thermal coupling chain (`ir_drop` → [`super::joule_source::joule_source`] →
//! [`super::electrothermal::solve_electrothermal`]) all sits in `crate::physics::thermal`.
//!
//! The acoustic output of a 150 V pulser scales with the delivered rail voltage, so resistive
//! voltage drop on VPP/GND between the supply connector and each device sets the channel-to-channel
//! **amplitude uniformity** of the array. This estimates the worst-case IR drop along the routed
//! power nets as a resistor network: each track segment is a conductance `g = 1/R` (R from
//! [`crate::physics::ampacity::track_resistance()`]); the supply pad is the voltage reference and device pads
//! draw current. Node voltages solve the same Laplace system as the thermal field —
//! `∇·(σ∇V) = −J` — by Gauss–Seidel over the routed graph.

use std::collections::HashMap;

use crate::board::{Board, NetId};

/// Worst-case IR drop on a power net.
#[derive(Debug, Clone, Copy)]
pub struct IrDrop {
    /// The power net analysed.
    pub net: NetId,
    /// Largest node-to-supply voltage drop (V).
    pub max_drop_v: f64,
}

/// Quantise a point to a node key (track endpoints already coincide at grid-cell centres).
fn key(p: crate::geom::Point) -> (i64, i64) {
    (p.x.0, p.y.0)
}

/// Estimate IR drop on `net`: `supply` is the source node, total `load_current_a` is split evenly
/// across the net's leaf nodes (devices). `copper_oz` sets sheet resistance.
///
/// Returns the largest leaf-to-supply drop, or `None` if the net has no routed tracks.
pub fn ir_drop(
    board: &Board,
    net: NetId,
    supply: crate::geom::Point,
    load_current_a: f64,
    copper_oz: f64,
    iters: usize,
) -> Option<IrDrop> {
    // Build the conductance graph from this net's track segments.
    let mut nodes: HashMap<(i64, i64), usize> = HashMap::new();
    let idx = |k: (i64, i64), n: &mut HashMap<(i64, i64), usize>| -> usize {
        let len = n.len();
        *n.entry(k).or_insert(len)
    };
    // edges: (a, b, conductance)
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let mut degree: HashMap<usize, u32> = HashMap::new();
    for t in &board.tracks {
        if t.net != net {
            continue;
        }
        let len_m = t.start.euclid(t.end) * 1.0e-9;
        if len_m <= 0.0 {
            continue;
        }
        let r =
            crate::physics::ampacity::track_resistance(len_m, t.width.to_mm() * 1.0e-3, copper_oz);
        if r <= 0.0 {
            continue;
        }
        let a = idx(key(t.start), &mut nodes);
        let b = idx(key(t.end), &mut nodes);
        edges.push((a, b, 1.0 / r));
        *degree.entry(a).or_default() += 1;
        *degree.entry(b).or_default() += 1;
    }
    if nodes.is_empty() {
        return None;
    }
    let n = nodes.len();
    let supply_node = *nodes.get(&key(supply))?;

    // Leaves (degree 1, not the supply) are the current-draw points.
    let leaves: Vec<usize> = (0..n)
        .filter(|&i| i != supply_node && degree.get(&i).copied().unwrap_or(0) <= 1)
        .collect();
    let per_leaf = if leaves.is_empty() {
        0.0
    } else {
        load_current_a / leaves.len() as f64
    };
    let mut inj = vec![0.0f64; n]; // current injected (A); supply absorbs the rest
    for &l in &leaves {
        inj[l] = -per_leaf; // load draws current
    }

    // Node conductance sums + adjacency.
    let mut gsum = vec![0.0f64; n];
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(a, b, g) in &edges {
        gsum[a] += g;
        gsum[b] += g;
        adj[a].push((b, g));
        adj[b].push((a, g));
    }

    // Gauss–Seidel: V_i = (Σ_j g_ij V_j + I_i) / Σ_j g_ij, supply pinned at 0 (reference).
    let mut v = vec![0.0f64; n];
    for _ in 0..iters {
        for i in 0..n {
            if i == supply_node || gsum[i] <= 0.0 {
                continue;
            }
            let mut acc = inj[i];
            for &(j, g) in &adj[i] {
                acc += g * v[j];
            }
            v[i] = acc / gsum[i];
        }
    }
    // Drop = how far below the supply reference (most negative node).
    let max_drop = v.iter().copied().fold(0.0f64, |m, x| m.max(-x));
    Some(IrDrop {
        net,
        max_drop_v: max_drop,
    })
}
