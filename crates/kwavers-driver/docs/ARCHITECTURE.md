# Architecture — `kwavers-driver`

**Status**: Phase 0 scaffold landed; architecture locked against the phased plan in
`docs/MIGRATION.md`.

This document is the SSOT for the vertical-slice tree shape, the DIP trait catalogue, the
SSOT module layout, and the principles applied to every module under this crate.
It is locked against `docs/MIGRATION.md`; the two read together.

## Target tree shape

```text
crates/kwavers-driver/
├── Cargo.toml                          ← workspace member; kwavers-transducer is feature-gated
├── README.md                           ← crate-level intro + migration pointer
├── src/
│   ├── lib.rs                          ← top-level entry: registers modules, declares SSOT surface
│   ├── prelude.rs                      ← SSOT idiomatic re-export surface for downstream crates
│   ├── error.rs                        ← Sealed error hierarchy + `Result` alias
│   ├── units.rs                        ← ZST unit newtypes (zero-cost Compile-time unit safety)
│   ├── ssot.rs                         ← SSOT cross-cutting constants + sidecar format strings
│   ├── geometry/                       ← vertical slice: distances, newtypes, lattice, hulls
│   │   ├── mod.rs
│   │   ├── newtype.rs                  ← Nm, Mm, Point, GridSpec (phantom-typed)
│   │   ├── distance.rs                 ← dist_point_seg, dist_seg_seg, orient
│   │   ├── hull.rs                     ← convex_hull
│   │   └── tests.rs
│   ├── board/                          ← vertical slice: PCB domain (SRP: types vs topology)
│   │   ├── mod.rs
│   │   ├── domain.rs                   ← Board + Net + Pad + Track + Via + Zone + LayerId
│   │   ├── netclass.rs                 ← NetClassKind + SplitDomain + IsolationDomain
│   │   ├── topology.rs                 ← multi-terminal Steiner tree + zones
│   │   └── tests.rs
│   ├── rules/                          ← vertical slice: DFM / design rules
│   │   ├── mod.rs
│   │   ├── creepage.rs                 ← CreepageRule (HV surface tracking)
│   │   ├── design_rules.rs             ← DesignRules (fab floors) + ViaPolicy
│   │   └── tests.rs
│   ├── physics/                        ← vertical slice: ALL physics axes (each its own subtree)
│   │   ├── mod.rs
│   │   ├── ampacity/
│   │   │   ├── mod.rs                  ← IPC-2221 width + Black electromigration + skin depth
│   │   │   └── tests.rs
│   │   ├── dielectric/
│   │   │   ├── mod.rs                  ← Paschen + IPC-2221 voltage + CAF TTF
│   │   │   └── tests.rs
│   │   ├── thermal/
│   │   │   ├── mod.rs                  ← 2-D heat conduction + electro-thermal + thermal vias
│   │   │   └── tests.rs
│   │   ├── emi/
│   │   │   ├── mod.rs                  ← commutation-loop inductance + EMI losses
│   │   │   └── tests.rs
│   │   ├── pdn/
│   │   │   ├── mod.rs                  ← IR drop + target impedance + decoupling SRF
│   │   │   └── tests.rs
│   │   ├── si/
│   │   │   ├── mod.rs                  ← microstrip impedance + propagation delay + skew
│   │   │   └── tests.rs
│   │   └── acoustic/                   ← DIP-seam slice (delegates to kwavers-transducer)
│   │       ├── mod.rs
│   │       ├── transducer.rs           ← AcousticSimulator trait + KwaversSim impl
│   │       ├── model.rs                ← InCrateAcousticSim impl (in-crate fallback)
│   │       └── tests.rs
│   ├── cost/                           ← DIP seam: RoutingCost trait + PhysicsCost impl
│   │   ├── mod.rs
│   │   ├── routing_cost.rs             ← the trait (monomorphised at call site)
│   │   ├── physics.rs                  ← PhysicsCost impl (creepage gradient + layer affinity)
│   │   ├── adapter.rs                  ← zero-cost adapter for IO traversal
│   │   └── tests.rs
│   ├── route/                          ← PathFinder algorithm
│   │   ├── mod.rs
│   │   ├── grid.rs                     ← resource model: occupancy + history + ownership
│   │   ├── search.rs                   ← multi-source Dijkstra tree-growth
│   │   ├── pathfinder.rs               ← the negotiation loop
│   │   ├── tree.rs                     ← multi-terminal Steiner tree
│   │   ├── emission.rs                 ← negotiated copper emit
│   │   └── tests.rs
│   ├── place/                          ← SA placer
│   │   ├── mod.rs
│   │   ├── anneal.rs
│   │   ├── energy.rs
│   │   ├── footprint.rs
│   │   ├── import.rs                   ← kicad_mod + symbol_import (feature-gated)
│   │   ├── rotation.rs                 ← `Rot` as a ZST enum marker
│   │   └── tests.rs
│   ├── stack/                          ← multi-tile board stacking (manifest + assembly + plan)
│   │   ├── mod.rs
│   │   ├── instance.rs
│   │   ├── assembly.rs
│   │   ├── compat.rs
│   │   ├── plan.rs
│   │   └── tests.rs
│   ├── manifest/                       ← typed sidecar + v2 stim + per-tile schedule
│   │   ├── mod.rs
│   │   ├── driver.rs                   ← DriverManifest
│   │   ├── profile.rs                  ← StimulationProgram + TileStimulationProfile
│   │   ├── resistor.rs                 ← ResistorPackage sealed enum + power_margin_w
│   │   ├── energy.rs                   ← EnergyBudgetInputs + EnergyBudgetReport
│   │   ├── validate.rs                 ← validate_v2_energy_budget (lifted-rejection gate)
│   │   ├── sidecar.rs                  ← kv emit + parse
│   │   ├── kwavers.rs                  ← KwaversBeamStep + KwaversBeamValidation (typed contract)
│   │   └── tests.rs
│   ├── io/                             ← kicad_io (feature-gated)
│   │   ├── mod.rs
│   │   ├── pcb.rs                      ← write_kicad_pcb, save_kicad_pcb
│   │   ├── sch.rs                      ← write_kicad_sch, save_kicad_sch
│   │   ├── drc.rs                      ← write_kicad_dru
│   │   ├── proj.rs                     ← write_kicad_pro (project sidecar)
│   │   ├── cli.rs                      ← KiCadCli wrapper
│   │   └── tests.rs
│   ├── verify/                         ← ERC / DRC / LVS / assembly / keep-in / BOM
│   │   ├── mod.rs
│   │   ├── erc.rs
│   │   ├── drc.rs
│   │   ├── lvs.rs
│   │   ├── assembly.rs
│   │   ├── keepin.rs
│   │   ├── bom.rs
│   │   ├── isolation.rs
│   │   ├── coupled.rs                  ← AcCoupling
│   │   ├── orchestration.rs            ← verify_all
│   │   └── tests.rs
│   ├── audit/                          ← adversarial DFM/SI critic
│   │   ├── mod.rs
│   │   ├── crosstalk.rs
│   │   ├── shortcuts.rs
│   │   ├── faults.rs
│   │   ├── antenna.rs
│   │   ├── pulse_skip.rs
│   │   ├── tr_switch.rs
│   │   ├── five_level.rs
│   │   ├── dfm.rs                      ← ground_pour, teardrops, miter_right_angle_corners, etc.
│   │   ├── component_db.rs             ← PulserComparison + StockStatus
│   │   ├── component_accuracy.rs
│   │   └── tests.rs
│   ├── render/                         ← svg emit
│   │   ├── mod.rs
│   │   ├── svg.rs
│   │   └── tests.rs
│   ├── place_route/                    ← the pipeline (cross-cutting algorithm stack)
│   │   ├── mod.rs
│   │   ├── co_optimize.rs
│   │   ├── terminal.rs                 ← terminal derivation from placement
│   │   ├── clearance.rs                ← pad-clearance halo derivation
│   │   └── tests.rs
│   └── experiment/                     ← NEW: end-to-end driver-side experiment simulation
│       ├── mod.rs                      ← Experiment trait + ExperimentReport + ExperimentRunner
│       ├── stimulus.rs                 ← Stimulus trait + DefaultStimulus impl
│       ├── acoustic.rs                 ← AcousticSimulator trait + KwaversSim impl + InCrateAcousticSim impl
│       ├── thermal.rs                  ← electro-thermal propagation through the experiment
│       ├── dispatch.rs                 ← per-tile transducer dispatch + 96-lane binding
│       ├── metrics.rs                  ← focal pressure + MI + ISPPA + lateral/axial 6 dB extents
│       ├── recorder.rs                 ← deterministic artifact emit (.kv + .npz + position-fixed BMP)
│       ├── runner.rs                   ← the Experiment::run orchestrator
│       └── tests.rs                    ← end-to-end test
├── examples/                           ← PUBLIC-API examples (no confidential surface)
│   ├── manifest_round_trip.rs          ← synthesise manifest → kv → parse
│   ├── beam_propagation.rs             ← stim → sim → metrics via kwavers-transducer
│   ├── resistor_triad.rs               ← Smd1206 / 2512 / 2512-HE / 4527 triad
│   ├── placement_demo.rs               ← component list → place → metrics
│   └── routing_demo.rs                 ← netlist → route → DRC output
├── benches/                            ← CONFIDENTIAL proprietary benches (relocated from old examples/)
│   ├── hv7355_tile.rs                  ← proprietary per-tile HV layout demo
│   ├── fpga_tile.rs                    ← proprietary per-tile FPGA layout demo
│   ├── v2_per_tile_stim.rs             ← proprietary per-tile stimulation schedule
│   ├── stack_model.rs                  ← proprietary stack-geometry model
│   ├── real_finepitch_demo.rs          ← proprietary fine-pitch escape demo
│   ├── beamforming_results.rs          ← proprietary beamforming figure generation
│   ├── fpga_tile_exact.rs              ← proprietary exact-footprint FPGA demo (when ready)
│   ├── hv7355_32ch_tile.rs             ← proprietary HV-7355 32-channel tile demo
│   ├── real_footprint_demo.rs          ← proprietary real-footprint import demo
│   └── emit_demo.rs                    ← proprietary end-to-end emit demo
└── docs/
    ├── ADR-001-architecture.md         ← updated to reflect kwavers-driver rename
    ├── component_cad_inventory.md
    ├── connector_cad_inventory.md
    ├── DFM.md
    ├── article_vs_current_stack.md     ← updated at Phase 6
    ├── MIGRATION.md                    ← (this dir's companion)
    └── ARCHITECTURE.md                 ← THIS FILE
```

## DIP-seam trait catalogue

Every cross-cutting abstraction lives behind a trait whose concrete impl is the SSOT for
"what the orchestrator actually runs". Each trait family is monomorphised at the call site
(Rust zero-cost abstractions + generics), with `Cow<'_, T>` for owned-or-borrowed cross-
module references and ZST markers for type-level state.

| Trait | Module | Generic params | Concrete impl(s) |
|---|---|---|---|
| `RoutingCost` | `crate::cost::routing_cost` | `<C: CostContext>` | `PhysicsCost`, `AdapterCost` |
| `Stimulus` | `crate::experiment::stimulus` | `<S: StimulusState>` | `DefaultStimulus` |
| `AcousticSimulator` | `crate::experiment::acoustic` | feature-gated on `kwavers` | `KwaversSim` (real call) + `InCrateAcousticSim` (fallback) |
| `Experiment` | `crate::experiment::runner` | `<S: Stimulus, A: AcousticSimulator>` | the orchestrator's `run()` method |
| `Ledger` (future) | reserved | TBD | TBD |

Each trait has **one** concrete impl that's the SSOT. Adding a new impl never changes the
trait, only the type-param at the call site — that is the **OCP-as-DIP** posture (open
for extension via new impls, closed for modification of existing call sites).

## SSOT module layout

`src/ssot.rs` is the SINGLE home of:

* **Safety bounds** — `KWVERS_MIN_FOCAL_PRESSURE_PA`, `KWVERS_MI_CAVITATION_CEILING`,
  `KWVERS_MIN_GRATING_FREE_STEER_DEG`, `KWVERS_MIN_RESISTOR_MARGIN_W`.
* **Check names** — `RESISTOR_MARGIN_CHECK_NAME` + the 3 kwavers pre-step check names
  (focal-pressure floor, MI ceiling, grating-lobe-free).
* **Sidecar format strings** — `MANIFEST_FORMAT_V1`, `MANIFEST_FORMAT_V2`,
  `MANIFEST_KEY_*` (one per sidecar key).
* **Layout constants** — `TX_LANES_V2 = 96`, `CHANNELS_PER_TILE_V2 = 24`, etc.
* **Article-preset defaults** — `ARTICLE_FOCAL_PRESSURE_PER_AMP_PA`,
  `WATER_Z0_RAYL`, etc.

Each constant gets a dedicated line and a `///` doc that names: source, semantic
("real headroom BUDGET" vs "bare ceiling"), expected test references. The discipline
mirrors the existing `RESISTOR_MARGIN_CHECK_NAME` SSOT, generalised to the full crate.

## Principles applied

| Principle | Application in `kwavers-driver` |
|---|---|
| **Documentation** | `#![deny(missing_docs)]` enforced at the crate level. Every `pub` item carries a `///` doc, every mod carries a `//!` module-level doc with the source-of-truth pointer. |
| **Redundancy-free** | No two paths for the same idea. Every magic number migrated to `ssot.rs`. |
| **Deep vertical hierarchical tree** | Two-level minimum nesting per slice (e.g. `physics/acoustic/` not `physics_acoustic_rs`). Each slice's directory owns its types + impls + tests. |
| **SRP** | One responsibility per module: `manifest/driver.rs` is the manifest data struct, `manifest/validate.rs` is the manifest validator, `manifest/sidecar.rs` is the kv emit/parse. |
| **SOC** | Physics is its own tree, layout its own tree, IO its own tree. Cross-cutting orchestration lives in `experiment/` and `place_route/`. |
| **SSOT** | One home for every magic value. `src/ssot.rs`. Cross-check test: a future contributor who adds a literal in any module fails `cargo test` via the SSOT migration test (Phase 1). |
| **DIP** | High-level orchestration depends on traits (RoutingCost, AcousticSimulator, Experiment), not on concrete kwavers-transducer calls. |
| **DRY** | No multi-path implementations. The kv parser lives in `manifest/sidecar.rs`; nothing else parses kv. |
| **Zero copy** | Read paths use `&[T]`, `&str`, `Cow<'_, T>`; ownership variants only at API boundaries. |
| **Zero cost abstractions** | Trait dispatch is monomorphised; generics + const params over `Box<dyn>`. |
| **Monomorphization** | `cost::RoutingCost` and `experiment::AcousticSimulator` carry their type-param through to the call site; a future cost model is a new impl, no Box<dyn>, no virtual call. |
| **Zero-sized types** | Marker types for type-level state: `pub struct Sealed;` (seals a trait), `pub struct ReadOnly;` (capability marker), and `pub enum Rot { R0, R90, R180, R270 }` (zero-byte rotation marker). |
| **COW (`Cow<'_, T>`)** | Owned-or-borrowed strings for IO paths, sidecar kv parsing, and cross-module references where ownership is conditional. |

## Layered dependency direction

```text
                         ┌─────────────────────────────────┐
                         │  experiment::runner            │  ← orchestrator
                         │  (Experiment::run)             │
                         └──────────┬──────────────────────┘
                                    │ depends on traits (DIP)
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
   experiment::stimulus  experiment::acoustic  experiment::dispatch
                                            │
                                            │ feature-gated:
                                            │   `kwavers = ["dep:kwavers-transducer"]`
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │  kwavers-transducer   │  ← external crate (NOT in src/)
                                │  ::propagate_focused… │
                                └───────────────────────┘

                  ┌─────────────────────────────────────┐
                  │  validate / physics / board / cost  │  ← physics + validate (lower layer)
                  │  route / place / place_route        │     also depends on traits
                  └────────┬────────────────────────────┘
                           │ depends on ...
                  ┌────────▼────────────────────────────┐
                  │  geometry / units / ssot / error   │  ← foundation layer
                  └────────────────────────────────────┘
```

Higher layers depend on lower layers through traits (DIP). Lower layers never depend on
higher layers (acyclic). The foundation — `geometry/units/ssot/error` — is the only
layer that the rest of the crate can build upon.

## Why this tree

The crate does many things. Each leaf of the tree does ONE thing; each leaf has its own
tests; the leaves compose through traits. When a future contributor wants to add a new
physics axis, they add a new `physics/<axis>/mod.rs` subtree — they DON'T touch the
routing kernel, the placement, the IO, or the sidecar. When a kwavers-side
consumer wants to swap the acoustic simulator, they provide a new `AcousticSimulator`
impl — they DON'T touch `src/board.rs` or the kv parser. The dependency direction +
the trait-catalogue enforce that.

## Risk register

| Risk | Mitigation |
|---|---|
| Over-engineering (ZST + Cow where plain primitives work) | Phase 0 only adds the placeholders; concrete ZST/Cow usage lands in the slice that actually needs it. |
| Vertical slicing fragments simple math utilities | The foundation layer keeps utility math (e.g. `dist_point_seg`) co-located with its types. Sub-submodularity is for cross-cutting concerns only. |
| Trait explosion (a trait for everything) | The DIP-ser trait catalogue is gated to ~5 trait families. Adding a new trait requires a migration doc update + a deliberate Phase. |
| Kwavers-transducer path-binding fragility | The `path = "../kwavers-transducer"` is checked at `cargo build --features kwavers`; the default build does not need the path to exist. |
| Confidential surface leak through `benches/` | `benches/` runs via `cargo bench`, not `cargo run --example`. The public-API `examples/` is the source of truth for the public surface. |
