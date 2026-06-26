# kwavers-driver

Physics-guided, manufacturing-aware driver electronics design for the **kwavers** ecosystem
— a Rust crate that builds the holohv high-voltage ultrasound-driver boards (negotiated-
congestion PathFinder autorouter, simulated-annealing physics-guided placer, per-domain
physics models, the `experiment` simulation framework that orchestrates end-to-end driver
+: transducer experiments, and the deterministic artifact emit that ties the generated
boards to `kwavers-transducer`'s beam-propagation pipeline).

> **Migration in progress.** This crate was renamed from `kicad-routing` ∑ `kicad_routing` to
> `kwavers-driver` ∑ `kwavers_driver` at Phase 0 (see **docs/MIGRATION.md**). The next
> phases refactor the flat-module hierarchy into a deeply-nested vertical-slice tree per
> the SSOT/DIP/SRP principles in **docs/ARCHITECTURE.md**.

## Why this exists

The KiCadRoutingTools A\* maze router is sequential and order-dependent: it commits each net
greedily, so on the 16/24-channel tiles the dense SPI/HIN control bus and the HV creepage
constraints drive it into an unresolvable state (the converge loop gets *stuck* — the geometry,
not the effort, is the wall). This crate replaces that core with the algorithm class that solves
congested, order-sensitive routing.

## Algorithm

**Negotiated-congestion routing (PathFinder — McMurchie & Ebeling, FPGA '95).** All nets route
every iteration allowing temporary overlap; a node's cost is

```text
cost(n) = (base(n) + history(n)) · (1 + overuse(n) · present_factor)
```

`present_factor` escalates each iteration and `history(n)` accumulates on persistently overused
nodes, so nets *negotiate* shared resources until the routing is **legal** (no node over
capacity). This is order-independent — the property A\* lacks.

Multi-terminal nets grow as a Prim-style tree (a rectilinear-Steiner approximation), so the
shared SPI bus and power stars route as trees, not pairwise.

## Physics guidance

`base(n)` is supplied by the `RoutingCost` trait — the extension seam. `PhysicsCost` folds the
design intent directly into the search:

- **HV creepage** as a spatial hazard gradient: HV nets pay a rising cost as they approach
  low-voltage features (and vice versa), so the 0.5 mm `HV_creepage` rule *shapes* the route
  instead of failing DRC after the fact.
- **Layer affinity**: HV → outer copper (creepage is a surface phenomenon), control → inner
  copper (shielded between planes).

A future thermal- or impedance-aware cost is a new `impl RoutingCost`, not a router change.

## Module map

| Module | Responsibility |
|---|---|
| `geom` | Exact nm geometry (`Nm`, `Point`) and the `GridSpec` lattice |
| `board` | Pure domain model: nets, pads, tracks, vias, `NetClassKind` |
| `rules` | `DesignRules` (fab floors) + `CreepageRule` (HV) |
| `cost` | `RoutingCost` trait + `PhysicsCost` |
| `route::grid` | Resource model: occupancy, history, via-column ownership, adjacency |
| `route::search` | Multi-source Dijkstra tree-growth under the negotiated cost |
| `route::pathfinder` | The PathFinder negotiation loop + copper emission |
| `place` | Physics/DFM simulated-annealing placer (rotation, congestion+weakness feedback) |
| `pipeline` | Place→route bridge: terminals + pad-clearance halos |
| `io` | `.kicad_pcb` + `.kicad_dru` emission |
| `audit` | Adversarial DFM/SI critic: flight-line crossings, hard clearance, near-short risk, crosstalk, via-adjacency, antenna |
| `stack` | Stack-board manifest, shield connector compatibility, thermal/height stack planning |

## Physics & analysis suite

Each axis is a real standard or first-principles model with a validation test (analytical oracle,
reference point, or independent cross-check), and feeds the adversarial place↔route loop where it
informs the layout.

| Module | Physics |
|---|---|
| `thermal` | 2-D heat conduction (MMS-validated) · electro-thermal Joule coupling · thermal vias · transient τ |
| `ampacity` | IPC-2221 width & resistance · skin depth/AC resistance · current density · electromigration (Black) · PTH aspect ratio |
| `emi` | Commutation-loop inductance · trace inductance · capacitive drive current · L·dI/dt overshoot · switching/gate/recovery loss |
| `pdn` | IR-drop (Gauss–Seidel resistor network) · target impedance · hold-up C · decoupling SRF |
| `dielectric` | Paschen air-breakdown · IPC-2221 voltage spacing · CAF time-to-failure |
| `si` | Microstrip impedance · propagation delay · skew budget |
| `acoustic` | Wavelength · grating-lobe steering · BVD resonance · near-field/f-number · element directivity · tissue attenuation |

Cross-checks that fall out of the model: the BVD series resonance comes to **2.08 MHz** (the drive
frequency), and charging the 50 pF load at 150 V in 5 ns gives **1.5 A** (the HV7355's peak rating).

## Status

**113 tests, fmt and nextest clean.** The 16-channel HV7355 tile (`examples/hv7355_tile.rs`)
and FPGA controller tile (`examples/fpga_tile.rs`) run the full co-optimised place→route loop and
emit internally verified KiCad boards. They are separate examples because the article architecture is
a stack: the FPGA controller board mates to the HV driver board through the shared `J_STACK` pinout.
Running both creates the full driver artifact set. The current FPGA component is the `XC7A-QFP`
example abstraction using a 100-pin QFP body so the stack bus, JTAG, configuration flash, routing,
and DRC checks are executable today; it is not yet the article-referenced `XC7A200T` 484-ball FBGA
footprint.

The HV board is one 16-channel tile: it instantiates two `HV7355K6-G` devices, routes `TX_0..TX_15`,
and daisy-chains the two HV7355 serial shift registers as one 16-bit load path. Generated boards also
emit `component_accuracy_hv.kv` and `component_accuracy_fpga.kv`; those manifests currently report
`exact_complete=false` because exact downloaded `.kicad_mod` import is still open. See
`docs/component_cad_inventory.md`.

The examples use a 0.5 mm routing lattice with explicit pad/via keepouts so KiCad's continuous
clearance rules, not only grid-cell occupancy, are satisfied. Component courtyards and explicit 3D
model envelopes carry a 2.0 mm assembly-clearance gate, and `verify_all` reports assembly failures
alongside ERC/DRC/LVS/BOM. The HV tile emits `output/full_driver/driver_manifest.kv` with the
transducer connector and 16 routed `TX_*` nets; the FPGA tile updates that manifest with JTAG,
configuration-flash, and stack-bus programming evidence. `examples/beamforming_results.rs` requires
that generated manifest and renders deterministic pulsed focal-envelope pictures for -45°, 0°, and
+45° using the article's 16-element, 4.3 mm aperture-axis assumption, 10 mm focus, and 5 ns
quantised delay profile. The simulation geometry matches the `kwavers-transducer` center-spanned
linear-array convention; the planned extraction name is `kwavers-drivers` because this crate owns
driver electronics, PCB generation, and hardware validation while `kwavers-transducer` owns acoustic
transducer models. The current renderer is an empirical near-field diagnostic; full wave propagation
validation should move to `kwavers-transducer`/kwavers before treating the images as article-grade
acoustic evidence.
`examples/stack_model.rs` consumes the clean board manifests in `output/full_driver/` and emits the
stack model artifacts for one top FPGA board plus four 24-channel HV shields.

**Roadmap** (see `backlog.md`): exact-pinout footprint import, exact FPGA selection/import, direct
kwavers workspace integration, and external KiCad CLI verification in CI.

## Build

```sh
cargo nextest run     # tests
cargo clippy --all-targets -- -D warnings
cargo run --release --example stack_model -- output/full_driver
cargo run --release --example beamforming_results -- output/beamforming output/full_driver/driver_manifest.kv
```
