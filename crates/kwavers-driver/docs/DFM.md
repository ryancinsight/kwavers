# Layout & routing best practices enforced by kicad-routing

The engine bakes these in as *costs and keepouts* (they shape place/route), not as post-hoc DRC.
Status: ✅ implemented · 🔜 planned (next increment) · 📋 ruleset-only (DRC-checked).

**Audit (2026-06-23)** — every 🔜 item below was re-checked against `src/place/energy.rs` and
`src/place/footprint_import.rs` and the route/audit stack. Line ranges cited are the
grounded source-of-truth closure: the item is closed iff that term still fires on a
representative placement.

## Board-edge & outline
- ✅ **Edge clearance** — copper/tracks keep ≥ `DesignRules.edge_clearance` (0.5 mm) from the
  outline. `Grid::reserve_edge` blocks the perimeter ring on all layers, so the router never
  reaches the routed/V-scored edge.
- ✅ **Components inside the edge margin** — `PlaceConfig.margin` (default 1.5 mm) feeds into
  `energy.t.edge` (`src/place/energy.rs:251-254`), which is near-hard (default weight 50.0).
  Annealing cannot accept any courtyard that crosses `[margin, w - margin] × [margin, h - margin]`.

## Placement convention (your guidance: chip centered, I/O at periphery)
- ✅ **I/O at the periphery** — `Role::Connector` ⇒ `energy.t.periphery += distance_to_nearest_edge`
  (`src/place/energy.rs:259-267`); locked connectors ARE exempt (`to_edge == 0` when margin
  reaches the board rim, e.g. J1 in `examples/fpga_tile.rs`).
- ✅ **Active ICs centred** — `Role::ActiveIc` ⇒ `energy.t.periphery += d(centre, board_centre)`
  (`src/place/energy.rs:269-281`); verified by `annealing_removes_overlap_and_peripheralises_connectors`
  test (`src/place/mod.rs`).
- ✅ **Decoupling caps adjacent to their IC power pins** — `Role::Decoupling` ⇒
  `energy.t.decoupling += d(cap, nearest IC power pin)` (`src/place/energy.rs:309-323`) with
  default weight 18.0 (well above HPWL's 0.5 so loop inductance dominates).
- ✅ **Courtyard non-overlap** — `energy.t.overlap += rect_clear_area × 1e-12` with default
  weight 50.0; part of the simulated-annealing hard core.
- ✅ **Domain grouping + LV↔HV isolation axis** — the placer now recognises a domain axis
  and a drift-cost term that locks LV parts to one side of a board-median axis and HV parts
  to the other. Per-component tagging is via
  [`Component::isolation_domain: IsolationDomain`](src/place/component.rs), defaulting to
  [`IsolationDomain::Lv`](src/place/footprint.rs). `PlaceConfig::isolation_axis: Axis`
  ([`Axis::X` | `Axis::Y`](src/place/energy.rs)) chooses which board axis carries the median,
  and the per-rail weight
  [`PlaceWeights::isolation_drift`](src/place/energy.rs:65-91) scales the
  [`EnergyTerms::isolation_drift`](src/place/energy.rs:191-203) contribution that gets
  summed into the [total energy](src/place/energy.rs:945-955). The term is gated by
  [`has_lv && has_hv`](src/place/energy.rs:881-895) so a single-domain design stays inert;
  it fires only when both domains are co-located in one placement (e.g. HV drive tile +
  LV FPGA controller cooptimise). The `regional` subcircuit bounding-box objectives are
  retained for the schematic-cluster use-case (`src/place/energy.rs:412-471`).

## High-voltage (150 V)
- ✅ **Creepage gradient** — HV nets pay rising cost near LV features (and vice-versa);
  `PhysicsCost` from `CreepageRule::holohv()` in `src/rules.rs` (now bumped to **0.60 mm** per
  IPC-2221B Table 6-1 B1 external uncoated ≤150 V, see `src/dielectric.rs::ipc2221_min_spacing_mm`).
- ✅ **Isolation barrier as a placement axis** — closed by the `t.isolation_drift` term above;
  the placer now actively pushes LV components toward `axis_min_along` and HV components toward
  `axis_max_along` ([`Axis::X` ⇒ `proj = x`](src/place/energy.rs:898-921)), with the cost
  proportional to the absolute mm-distance from the wrong edge. The IPC-2221B 0.60 mm creepage
  cost (PhysicsCost from `CreepageRule::holohv()` in `src/rules.rs`) remains the *routing-time*
  enforcer; the placement term is a *placement-time* enforcer. They complement rather than
  collide: the placer pre-positions domains for clear routing corridors, while the rule set
  polices trace-level creepage.

## Thermal
- 🔜 **Power-device heat-spread** — the code shapes **placement** spread (`thermal` floor +
  `ic_spread` continuous, `src/place/energy.rs:283-296` + similar pair-energy loop) but does
  not reserve bottom-side copper area under the HV7355s. The article's "ground planes in
  mid-layers act as the heat spreader" is `thermal.rs` territory, not a `place` term.
  Genuine 🔜: add `PlaceConfig::keepout_rects[name]` for under-board thermal copper reservations.
- 📋 Ground planes in mid-layers act as the heat spreader and shield (per the article).

## Power & signal integrity
- ✅ **Short, wide high-current paths** — `energy.t.hpwl` (`src/place/energy.rs:355-371`) with
  default weight 0.5 prefers minimal wirelength without sacrificing correctness; track width is
  governed by `DesignRules::hv_track` (0.25 mm) and routed on layer 0/1 by `via_cost`
  discourages needless layer changes for HV (`src/route/cost.rs`). The "wide" half is rule-set;
  the "short" half is HPWL + functional-region tightening.
- ✅ **Layer affinity** — HV → outer copper (surface creepage policeable), control → inner.
  Enforced in `src/route/cost.rs::layer_affinity_cost`.
- ✅ **Via minimization** — `via_cost` discourages needless layer changes; `audit.rs` flags
  layer-transition vias without ground stitching.
- 📋 **Via stitching / plane fill** — planes + stitching vias on emit (IO increment).

## Components (sourcing best practice)
- 📋 **DigiKey availability is a hard filter** — every MPN orderable (in stock / sane lead).
  `src/component_db.rs::available_pulsers` lists six in-stock 2026 candidates; the optimiser's
  comparison function feeds the score. Downloaded CAD ⇒ assumed available; alternatives are
  surfaced by MPN for the user to confirm.

> Add anything missing here and it becomes an engine constraint, not a review note.
