# kicad-routing backlog

Change-class tags per SemVer. Sprint target: 0.1.0 (Foundation).

## Done — 0.1.0 Foundation
- [x] [minor] **Physics vertical-slice migration (Phase 3a + 3b)** — `src/ampacity.rs` carved
      into `src/physics/ampacity/` (Phase 3a); `src/thermal.rs` carved into
      `src/physics/thermal/` AND `IrDrop` + `ir_drop` promoted out of `src/pdn.rs` into the
      thermal slice (Phase 3b). `pub mod ampacity; pub mod thermal;` declared at
      `src/physics/mod.rs`. `track_resistance` lives at `crate::physics::ampacity::track_resistance`.
      Electro-thermal coupling chain (`ir_drop` → `joule_source` → `solve_electrothermal`) now
      sits in one crate plane. `src/pdn.rs` keeps the decoupling/resonance/impedance half of PDN.
      9 thermal/IR-drop tests consolidated into `src/physics/thermal/tests.rs`. Forward-tracking
      closed at `src/physics/mod.rs` docstring cut-over status block.
- [x] [minor] Core domain model (`geom`, `board`, `rules`) — exact nm geometry, pure board model.
- [x] [minor] `RoutingCost` seam + `PhysicsCost` (creepage gradient + layer affinity).
- [x] [minor] Negotiated-congestion router (`route::grid/search/pathfinder`): PathFinder loop,
      Prim tree growth, copper emission. Tested: crossing-net legalisation (the A\*-deadlock
      case), physics steering, value-semantic geometry/grid invariants.
- [x] [patch] Split hard DRC clearance from soft adversarial near-short risk; complete the HV7355
      article-replication example with a clearance-safe routing lattice and 5 ns focused-beamforming
      delay validation for the 16-element, 4.3 mm aperture, 10 mm focus, ±45° steering case.
- [x] [patch] Generate full-driver artifacts by running both stackable boards: FPGA controller tile
      plus HV7355 driver tile. KiCad CLI DRC reports zero violations and zero unconnected items for
      both outputs.
- [x] [patch] Add deterministic Rust beamforming image generation for the article's -45°, 0°, and
      +45° steering cases using the 5 ns quantised focused delay profile.
- [x] [patch] Replace the first beamforming image renderer's continuous-wave ridge plot with a
      focal-arrival pulse-envelope diagnostic. The regenerated broadside image shows a compact focal
      spot at 10 mm and records 6 dB lateral/axial widths in `beamforming_metrics.csv`.
- [x] [patch] Promote component courtyard spacing to a hard assembly gate and tighten pad/via
      keepouts against KiCad's continuous clearance DRC. Full-driver outputs render without package
      intersections and KiCad CLI DRC reports zero violations.
- [x] [patch] Tie beamforming simulation to the generated full-driver artifacts. The HV tile emits a
      manifest with the transducer connector and 16 routed TX nets; the FPGA tile records JTAG,
      configuration-flash, and stack-bus programming evidence; beamforming renders require that
      manifest as input.
- [x] [patch] Add explicit 3D model envelopes for connector bodies whose KiCad model filenames do not
      encode body dimensions, so package-body clearance is checked by assembly verification and not
      only by visual render inspection.
- [x] [patch] Inventory downloaded connector CAD files and selected production connector part numbers
      in `docs/connector_cad_inventory.md`.
- [x] [patch] Add stack-board manifests, FPGA/HV connector compatibility checking, and shield-stack
      thermal/height planning. Generated stack pinout passes across outline, connector geometry, and
      normalized 24-pin stack-bus pinout.
- [x] [patch] Extract downloaded component CAD archives into `docs/cad_models`, add
      `docs/component_cad_inventory.md`, and emit per-board component accuracy manifests that
      distinguish 16-channel electrical completeness from exact-footprint completeness.
- [x] [patch] Extract downloaded FPGA option CAD for `XC7A100T-2FGG484C` and
      `XC7A35T-1FTG256C`; record their exact footprint paths and pad counts in the component CAD
      inventory.
- [x] [patch] Complete the stack-level assembly manifest. `stack_model` now emits
      `shield_stack_assembly.kv`, proving the five-board stack order, controller/HV slots, stack-bus
      pinout, and four 16-channel HV tile maps covering global `TX_0..TX_63`.

## Next increments
- [ ] [patch] **TC8020 nsf board clearance cleanup**. Retain the exact local `TC8020K6_G` QFN56
      footprint/symbol flow and reduce the regenerated `nsf_neuromod_phased_array` board from
      `clearance=18`, `crossings=302` to hard internal DRC clean, then run external KiCad DRC when
      `kicad-cli` is available. Acceptance: board remains `complete=true`, `legal=true`,
      `lvs_pass=true`, no downgraded or synthesized TC8020 footprint path, internal hard DRC clean,
      and generated board is no longer labelled inspection-only.
- [ ] [minor] **`.kicad_pcb` IO boundary** (`io` module, feature-gated). S-expression read of
      footprints/pads/nets and write of tracks/vias. This is what lets the engine run on the real
      holohv8t/16t/24t tiles. The DIP boundary is already in place (the core has zero infra deps),
      so this is additive. Acceptance: round-trip a tile board; route it; DRC shows 0 shorts where
      `route_v2.sh` (A\*) left residual control-bus shorts.
- [ ] [minor] **Downloaded connector footprint import**. Parse or adapt the exact Molex/Samtec
      `.kicad_mod` files under `docs/cad_models` into `FootprintDef` without hand-porting pad
      geometry. Acceptance: replace the demo `J_PWR`/`J_TX` connector abstractions with exact
      Micro-Fit footprints and retain internal LVS PASS plus KiCad DRC 0 violations.
- [x] [minor] **Exact HV power-input reroute**. Replace the remaining HV `J4` demo power/input
      footprint with the exact Molex `0430450400` footprint from
      `docs/cad_models/430450400/MOLEX_430450400.kicad_mod` and reroute to the new through-hole pad
      centers. Acceptance: KiCad CLI render shows the exact `430450400.step` body over exact
      through-hole pads, and KiCad DRC reports 0 violations and 0 unconnected items.
- [ ] [minor] **Exact FPGA footprint selection/import**. Replace the `XC7A-QFP` routable controller
      abstraction with the selected production FPGA footprint and escape model. Acceptance: the FPGA
      board still emits JTAG/config-flash/stack-bus programming evidence, KiCad DRC reports 0
      violations/unconnected items, and `verify_stack_pair` still passes against the HV tile.
- [ ] [minor] **Exact HV7355 footprint import**. Replace the 24-pad `HV7355K6-G` routed abstraction
      with the exact local `QFN56_8X8MC_MCH.kicad_mod` land pattern for both U1 and U2. Acceptance:
      `component_accuracy_hv.kv` reports `exact_complete=true` for the HV pulser rows while
      preserving two HV7355 devices, 16 `TX_*` nets, LVS PASS, and KiCad DRC 0 violations.
- [x] [minor] **kwavers beam propagation validation** — Track D v2 plus the 2026-06-29 propagation
      close-out closes this accept criterion. Reproduce end-to-end with
      `cargo run --release --example v2_per_tile_stim`. Deliverables:
      - Close-out 2026-06-29: the driver validation path now executes real
        `kwavers-transducer` focused propagation. `kwavers-transducer` owns
        `propagate_focused_linear_array`, and `KwaversSim::simulate` plus
        feature-enabled `validate_against_budget` consume its pressure map.
        Regenerated `output/beamforming` from `output/manifests/v2_per_tile_stim.kv`
        reports 96 channels, 11.027 MPa focal pressure, MI 7.797, 4108 W/cm2
        ISPPA, 0.500 mm lateral width, 2.074 mm axial width, grating-lobe-free
        true, far-field false, and all four validation checks passing.
      - Sidecar: `output/v2_per_tile_stim.kv` carrying the 96-lane per-tile schedule.
        Each HV tile emits 10 keys — `stim_tile_{i}_prf_hz` / `shift_s` / `phase_deg`
        / `ramp_s` plus the inherited `tbd_s` / `sd_s` / `isi_s` / `tt_s` / `vpp_v` /
        `dead_time_s` — paired with a flat `tx_nets=TX_0,TX_1,…,TX_95` binding
        (=4 tiles × 24 channels). Deterministic `.12e` float serialization; the
        lib round-trip test `manifest_v2_tile_form_round_trips_four_distinct_profiles`
        guarantees no kwavers-side precision loss.
      - Scalar contract: `DriverManifest::validate_v2_energy_budget` returns
        `EnergyBudgetReport { total_protocol_load_j_s, per_tile_protocol_load_j_s,
        peak_i_a, peak_duty_weighted_i_a, max_frame_duty, per_tile_device_total_w,
        per_tile_pulser_total_w, per_tile_resistor_w, ampacity_headroom_a,
        headroom_margin_a, lanes=96 }` — the kwavers-transducer propagation step
        reads these straight off the API, no re-derivation needed.
      Reproduce end-to-end: 1) cargo run --release --example v2_per_tile_stim (sidecar);
      2) cargo test --lib validate::tests (9 new tests verify the pre-step
      adapter + validation happy path, gate rejection, lane mismatch, smoke
      run, sensible bounds, grating-lobe boolean, TX_LANES_V2 cross-check).
      Deliverables for the kwavers consumer: pre-step struct
      KwaversBeamStep { lanes, aperture_m, frequency_hz, sound_speed_m_s,
      focal_m, timing_step_s, pitch_m, wavelength_m, f_number }; scalar contract
      EnergyBudgetReport; output struct KwaversBeamValidation { step,
      focal_pressure_pa, grating_lobe_free, in_far_field, isppa_w_cm2,
      mechanical_index, axial_extent_mm, lateral_extent_mm, report }. With the
      `kwavers` feature enabled, `validate_against_budget` fills those scalars
      from `kwavers_transducer::propagate_focused_linear_array`; the default
      build keeps the analytical fallback for dependency-size control.
      Sealed at the kwavers seam: the **ResistorPackage** contract (`Smd1206 = 250 mW`,
      `Smd2512 = 1 W`, `Smd4527 = 2 W` at IPC-7351 70 °C ambient) closes the per-tile
      dissipation-vs-footprint gap. New fourth input `damping_footprint: ResistorPackage`
      on `EnergyBudgetInputs` drives an inline `power_rating_check` inside
      `validate_v2_energy_budget` (ampacity gate first, then per-tile rating so a single
      design-fix pass addresses both). Per-tile margin is propagated to the kwavers
      consumer: `EnergyBudgetReport::per_tile_resistor_margin_w` →
      `KwaversBeamStep::resistor_margin_w` →
      `KwaversBeamValidation::resistor_margin_w`, and a 4th kwavers-side safety check
      `resistor margin (per-tile min) ≥ 0 W` locks the post-rejection invariant at the
      seam. The propagated kwavers path reads `step.resistor_margin_w[i]` to plan footprint bumps
      (`      Smd2512 -> Smd4527`) and matching-cap tightening without re-deriving
      `pulser_dissipation`. Smd4527 covers the article-class envelope (50 pF / 150 V /
      +50 Hz per-tile PRF stagger: tile[0] ~0.98 W, tile[3] ~1.13 W, Smd4527
      worst-case margin ~0.87 W). The example fixture uses a bigger +150 Hz
      per-tile stagger demonstration (build_four_tile_per_tile_manifest in
      examples/v2_per_tile_stim.rs: tile[3] ~1.43 W, Smd4527 margin ~0.57 W).
      355 lib tests green. Reproduce `cargo test --lib` and
      `cargo run --release --example v2_per_tile_stim` (sidecar
      output/v2_per_tile_stim.kv + stdout per-tile dissipation carrying the
      per-tile margin scalar into the kwavers seam contract).
      Liftout of the inline rejection gate (Track D v2 follow-up):
      `ResistorPackage::power_rating_check(...) -> Result<f64, ResistorRatingError>`
      is rename-and-replaced by `ResistorPackage::power_margin_w(self, dissipation_w) -> f64`
      returning SIGNED margins; the `ResistorRatingError` struct + `over_w()` accessor
      are deleted. The `DriverManifest::validate_v2_energy_budget` no longer rejects on
      over-rate -- signed per-tile margins propagate through
      `EnergyBudgetReport::per_tile_resistor_margin_w` ->
      `KwaversBeamStep::resistor_margin_w` ->
      `KwaversBeamValidation::resistor_margin_w` verbatim. The kwavers-side 4th
      `Check` against the new safety constant `KWVERS_MIN_RESISTOR_MARGIN_W = 0.05 W`
      is the **SOLE gatekeeper** on the per-tile resistor rating (no longer
      redundant -- it can actually fail now when a chosen footprint under-rates the
      article-class envelope; the prior post-rejection safety-net redundancy is
      gone). Sole-gatekeeper regression lock: `Smd1206` on 50 pF / 150 V / +50 Hz
      per-tile PRF stagger -- validator returns Ok with negative per-tile margins
      (worst-case tile[3] ~~ -0.88 W), then `validate_against_budget` returns Ok
      with `report.all_pass == false` on the 4th Check (focal-pressure floor /
      MI ceiling / grating-lobe-free steer still pass). 358 lib tests green (13 in
      manifest + 17 in validate). Reproduce: `cargo test --lib` and `cargo run
      --release --example v2_per_tile_stim`.


- **Close-out — 2026 Q2 follow-up.** `KWVERS_MIN_RESISTOR_MARGIN_W` tilted from a bare 0.0
  IPC-7351 ceiling to a `0.05 W` (50 mW) SLACK FLOOR for a real headroom BUDGET above the
  IPC-7351 70 °C AMBIENT rating. The 4th `Check` now catches "fits by exactly its rating"
  choices before stack-temperature drift crosses the IPC ceiling in the field. Triad
  demonstrated end-to-end in `examples/v2_per_tile_stim.rs`:
  step 4-5 (Smd4527 comfortable ~+0.57-1.02 W margin per tile over the +150 Hz PRF stagger
  envelope), step 5b (Smd2512He tight-but-still-fits ~+0.07-0.52 W margin per tile, just
  above the 0.05 W slack floor on the binding tile), step 5c (Smd2512 under-rated
  tile[3] margin `+0.000859375 W` FP64-exact; kwavers 4th `Check` SHUTS the fixture while
  the other 3 (focal-pressure / MI / grating-lobe-free) STILL pass, n_failing == 1).
- **Close-out — 2026-06-28.** `examples/beamforming_results.rs` is upgraded from a standalone
  near-field diagnostic to a kwavers-backed validation example. It reads the generated full-stack
  v2 manifest, runs `run_experiment(..., &KwaversSim, ...)`, writes `beamforming_validation.kv`,
  `tile_geometry.csv`, `beamforming_metrics.csv`, and deterministic BMP visualizations, and documents
  the contract in `docs/beamforming_validation.md`. The geometry adapter now preserves the 96-lane
  manifest span by converting driver center-span aperture to kwavers pitch-cell aperture.
- [ ] [minor] **SA placement** (`place` module). Reuse `PhysicsCost`'s hazard field as the
      placement energy (isolation-barrier cost, courtyard overlap, HPWL) — one cost SSOT for place
      and route. Port from `gen_place.py`. Acceptance: places holohv16t with 0 courtyard overlap +
      improved routability vs the Python SA.
- [x] [patch] **Scratch reuse across the present-factor loop** — `Scratch` is per-`route()` call;
      confirmed no grid-sized realloc inside the iteration loop (epoch mechanism + single alloc).
- [x] [minor] **Rip-up ordering by criticality** — HV/Signal nets routed before Power/Ground;
      within class descending terminal-count tiebreaker routes harder nets first each iteration.
- [x] [minor] **Targeted rip-up (partial re-route)** — PathFinder now computes `overuse_bitset()`
      at the start of each iteration (iter > 0) and skips legal nets. Eliminates redundant Dijkstra
      searches in later iterations; convergence accelerated for dense boards like the 484-ball BGA.
      Evidence: unit tests in `grid::tests` + all 269 routing tests pass.
- [x] [patch] **History decay** — `accumulate_history` now takes `decay: f32`; stale congestion
      penalties decay by 5 %/iter on cleared nodes, preventing permanent bias away from channels
      that have since de-congested.
- [x] [patch] **Stall-break adaptive schedule** — consecutive iterations with no overuse improvement
      trigger a 2× present-factor boost every 3 stall iterations (capped 8×), breaking congestion
      equilibria where two nets keep exchanging the same overloaded resource.
- [x] [patch] **Diagonal (45°) moves** — 8-neighbourhood enabled by default in `DesignRules::holohv()`;
      `ACUTE_ANGLE_PENALTY = 0.5` in `search.rs` discourages acid-trap geometries at routing time.

## Residual risk / honest limits
- All 358 unit/integration tests pass. DFM metrics (acid_traps, clearance_violations, sharp_bends)
  for the FPGA and HV7355 tile examples have last measured at: FPGA acid=42, clearance=5,
  sharp_bends=148; external KiCad DRC=0. The pad-aware miter skip (2026-06-23) is expected to
  reduce clearance_violations and some sharp_bends in the next long routing run.
- The router's *legality* metric is grid-node capacity, not full polygonal DRC. Pad/via keepouts now
  include a continuous-geometry guard and example outputs are DRC-verified in KiCad, but arbitrary
  imported boards still need external KiCad DRC as the empirical manufacturing oracle.
- Single-net reachability failures (a fully walled-off terminal) are reported via
  `NetRoute.connected = false`, surfaced in `RouteOutcome.complete` — never silently dropped.
- DigiKey availability is a live procurement property. The example names modern replacement part
  families, but final procurement must re-check stock, package, voltage/current ratings, and lifecycle
  on the order date.
- The FPGA board currently uses the `XC7A-QFP` executable abstraction, not an exact production FPGA
  footprint. The article-referenced `XC7A200T` FBGA remains a candidate class, but exact FPGA
  selection/import is still open.
- The current shield-stack assembly is complete at the stack-manifest level: one controller plus four
  16-channel HV tiles, 12 mm board pitch, and 60 mm total height. It still inherits the per-board
  exact-footprint gaps recorded in `component_accuracy_*.kv`.
- The current beam images are deterministic visualizations of kwavers-backed validation metrics and
  realized channel geometry, not fabricated acoustic measurements. The validation sidecars
  (`beamforming_validation.kv`, `tile_geometry.csv`, `beamforming_metrics.csv`) are the evidence
  artifacts for article-replication work.
- Direct kwavers dependency integration is deferred while the driver subtree remains confidential and
  standalone. The intended crate name is `kwavers-drivers`, paired with `kwavers-transducer`; revisit
  naming at extraction time if the package scope broadens beyond driver electronics.
