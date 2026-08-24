# kwavers-driver

Physics-guided, manufacturing-aware driver-electronics design for
[kwavers](https://github.com/ryancinsight/kwavers).

The crate designs the high-voltage ultrasound driver boards that feed a kwavers transducer
model: it places components under a physics and DFM objective, routes them with a
negotiated-congestion router, verifies the result against fabrication and electrical rules,
and emits a deterministic artifact set that ties the generated board back to
[`kwavers-transducer`](https://docs.rs/kwavers-transducer)'s beam-propagation pipeline.

## Why not a sequential A\* router

A maze router commits each net greedily, so early nets wall off later ones. In a congested
region — a shared SPI daisy-chain, a BGA escape, a high-voltage fan-in — no mechanism lets
one net step aside for a more constrained one, and the solution becomes order-dependent:
on the dense multi-channel tiles it does not converge. The geometry is the wall, not the
effort.

## The algorithm

`route` implements **negotiated-congestion routing** (PathFinder — McMurchie & Ebeling,
*FPGA '95*). Every net routes every iteration, temporary overlap allowed; a node's cost is

```text
cost(n) = (base(n) + history(n)) * (1 + overuse(n) * present_factor)
```

`present_factor` escalates each iteration and `history(n)` accumulates on persistently
overused nodes, so nets *negotiate* shared resources until the routing is **legal** — no
node over capacity. This is order-independent, the property A\* lacks.

Multi-terminal nets grow as trees, not as pairwise connections, and the growth strategy is
chosen by net class: power and ground expand Prim-style from any terminal, while signal and
high-voltage nets use chain-tip growth — the search source moves to each reached terminal —
so a multi-terminal high-speed net forms a daisy chain (degree ≤ 2) instead of branching
stubs.

## Physics guidance

`cost::RoutingCost` is the extension seam: `base(n)` folds the physics and manufacturing
constraints into the search instead of checking them after the fact. `cost::PhysicsCost`
implements

- **high-voltage creepage** as a spatial hazard gradient — an HV net pays a rising cost as
  it approaches low-voltage features, and vice versa, so the clearance rule *shapes* the
  route instead of failing DRC afterwards; and
- **layer affinity** — HV to outer copper, since creepage is a surface phenomenon; control
  to inner copper, shielded between planes.

A thermal- or impedance-aware cost is a new `impl RoutingCost`, not a router change.

## Public surface

`use kwavers_driver::prelude::*;` brings the unit newtypes, geometry, board model, and
physics facade into scope.

| Module | Responsibility |
|---|---|
| `units` | Unit newtypes (`Nm`, `Hz`, `Ohm`, `Volt`, `Amp`, `Watt`, `Kelvin`, …) — type-level unit safety at zero runtime cost |
| `geom` | Exact nanometre geometry (`Nm`, `Point`) and the `GridSpec` lattice |
| `board` | Pure domain model: nets, pads, tracks, vias, `NetClassKind`, split domains |
| `rules` | `DesignRules` (fabrication floors) and the high-voltage creepage rule |
| `cost` | `RoutingCost` seam and `PhysicsCost` |
| `route` | Grid resource model, multi-source search, the PathFinder loop, copper emission |
| `place` | Physics/DFM simulated-annealing placer (rotation, congestion and weakness feedback) |
| `pipeline` | Place→route bridge and the place↔route co-optimization loop |
| `dfm` | Post-routing manufacturability: teardrops, mitring, via dedup, ampacity widening, ground pour |
| `optim` | Joint driver–thermal–acoustic co-optimization |
| `verify` | ERC, DRC, LVS, assembly, keep-in, BOM, isolation BFS, AC coupling — `verify_all` |
| `audit` | Adversarial DFM/SI critic: crossings, clearance, near-short risk, crosstalk, via adjacency, antenna |
| `fabrication` | Fabrication-readiness verification |
| `component_db` / `component_accuracy` | Sourced part database and CAD/footprint accuracy manifest |
| `driver` | Driver power loss, efficiency, and matching-network physics |
| `five_level` / `pulse_skip` / `tr_switch` | Pulser topology, adaptive pulse skipping, T/R switch models |
| `stack` | Multi-tile stack planning: board count, channels per board, connector and thermal compatibility |
| `manifest` | Driver-to-simulation manifest — the handoff to the acoustic pipeline |
| `validate` | Whole-design physics validation across the per-domain models |
| `experiment` | End-to-end driver + transducer experiment orchestration (`run_experiment`, `build_beam_report`) |
| `render` | Native SVG renderer for the routed board |
| `io` / `kicad_cli` | KiCad file emission and the external `kicad-cli` wrapper (feature `io`) |
| `error` | Per-slice error hierarchy |
| `ssot` | Cross-cutting constants and string literals |

## Physics and analysis suite

Each axis under `physics` is a standard or first-principles model with a value-semantic
test — an analytical oracle, a published reference point, or an independent cross-check —
and feeds the adversarial place↔route loop where it informs the layout.

| Slice | Physics |
|---|---|
| `physics::thermal` | 2-D heat conduction (validated against a manufactured solution) · electro-thermal Joule coupling · thermal vias · transient τ |
| `physics::ampacity` | IPC-2221 width and resistance · skin depth / AC resistance · current density · electromigration (Black) · plated-through-hole aspect ratio |
| `physics::emi` | Commutation-loop inductance · trace inductance · capacitive drive current · L·dI/dt overshoot · switching, gate, and recovery loss |
| `physics::pdn` | IR drop (Gauss–Seidel resistor network) · target impedance · hold-up capacitance · decoupling SRF |
| `physics::dielectric` | Paschen air breakdown · IPC-2221 voltage spacing · CAF time-to-failure |
| `physics::si` | Microstrip impedance · propagation delay · skew budget |
| `physics::acoustic` | Wavelength · grating-lobe steering · BVD resonance · near-field / f-number · element directivity · tissue attenuation |

Cross-checks fall out of the models rather than being asserted into them: the BVD series
resonance of the modeled element lands on the drive frequency
(`bvd_resonance_matches_2mhz_drive`), and the commutation-loop model reproduces the
pulser's rated peak drive current from its loop inductance and node capacitance.

## Evidence tier

Negotiated-congestion convergence (legality on a congested instance with no single-layer
solution) and the creepage-gradient effect are covered by value-semantic unit tests in
their modules — property and empirical tier, not a machine-checked proof. The thermal
solver is verified against a manufactured solution at its stated order of accuracy.

## Status

`cargo nextest run -p kwavers-driver` runs **494 tests, all passing**, with `cargo fmt`
and Clippy clean.

The crate is the product of a phased refactor from `kicad-routing` (see
`docs/MIGRATION.md` and `docs/ARCHITECTURE.md`). Phases 0–5 have landed: the unit
newtypes, prelude, per-slice error hierarchy, geometry, cost, route, place, physics, and
output slices are migrated, and the `experiment` subtree orchestrates end-to-end driver +
transducer runs. **Phase 6 remains** — public-API examples and a docs backfill.

There is no `examples/` directory in this repository: the original per-tile and stack
example programs carry proprietary board geometry and are excluded from version control
(see the repository `.gitignore`). Phase 6 adds public-surface examples that exercise the
library without that geometry. Until then the library API and its tests are the executable
documentation.

## Build

```sh
cargo nextest run -p kwavers-driver
cargo clippy -p kwavers-driver --all-targets -- -D warnings
cargo doc -p kwavers-driver --no-deps
```

## Features

- `io` (default) — KiCad `.kicad_pcb` / `.kicad_dru` emission and the `kicad-cli` wrapper.
- `kwavers` — enables `experiment::KwaversSim`, the real acoustic simulator backed by
  `kwavers-transducer`. Without it, `experiment` uses the in-crate fallback simulator.

## Documentation

- API reference: <https://docs.rs/kwavers-driver>
- Architecture and refactor plan: `docs/ARCHITECTURE.md`, `docs/MIGRATION.md`
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
