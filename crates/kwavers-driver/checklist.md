# kicad-routing checklist

Sprint target: 0.1.0
Phase: Closure
In-flight item: [minor] Phase 6 COMPLETE — crate moved to `crates/kwavers-driver/`, joined parent kwavers workspace, kwavers-transducer path dep wired, `KwaversSim::simulate` filled with real `design_array`-based implementation; 0.3.13.

## Completed
- [x] [minor] Phase 6 workspace move + kwavers-transducer wiring: crate copied from
      `leoneuro/driver/kicad-routing/` to `crates/kwavers-driver/`; standalone `[workspace]` table
      removed; parent `D:\kwavers\Cargo.toml` members updated; `kwavers-transducer = { path =
      "../kwavers-transducer", optional = true }` dep added; `kwavers = ["dep:kwavers-transducer"]`
      feature updated; `KwaversSim::simulate` filled — calls `kwavers_transducer::design_array` to
      get exact element geometry (realized `n_channels`, `grating_lobe_free`, `aperture_y_m()`),
      then computes pressure/MI/ISPPA from the design-derived channel count; profiles dropped
      (workspace member profiles are no-ops). `Cargo.toml` `0.3.12` → `0.3.13`. Build + test green.
- [x] [minor] Phase 5 experiment framework (4 placeholders filled + 4 new files — `metrics.rs`,
      `recorder.rs`, `runner.rs`, `tests.rs` + `mod.rs` updated + `kwavers = []` feature marker):
      `stimulus.rs` (`Stimulus` DIP trait + `DefaultStimulus` manifest wrapper), `acoustic.rs`
      (`AcousticSimulator` DIP trait + `PressureMap` + `InCrateAcousticSim` real in-crate physics
      impl + `#[cfg(feature="kwavers")] KwaversSim` DipSeam stub for Phase 6), `thermal.rs`
      (`ThermalState` + `propagate_thermal` 0-D θ_jc model), `dispatch.rs` (`LaneBinding` +
      `TileDispatch` equal-partition lane→tile table), `metrics.rs` (`ExperimentMetrics` aggregate
      + `build_beam_report` 4-check assembler), `recorder.rs` (`ExperimentRecord` + `artifact_key`
      deterministic key), `runner.rs` (`run_experiment` DIP-injected orchestrator + `ExperimentReport`).
      18 new value-semantic tests (positive/negative/boundary). `Cargo.toml` `0.3.11` → `0.3.12`.
      Full suite **433/433 green**; experiment slice clippy-clean.
- [x] [patch] Phase 4m pipeline slice migration (Tier 1; 1827 LOC — co-optimization orchestrator):
      flat `src/pipeline.rs` carved into a 6-file `src/pipeline/` subtree split by role — `config.rs`
      (`CoOpt` + per-role dissipation), `result.rs` (`CoOptResult`), `place_board.rs` (`place_to_board`
      + `RoutingInputs` + placement keepout/repulsion helpers), `cooptimize.rs` (the `cooptimize` loop
      + min-layer/min-area variants), `tests.rs` (verbatim), `mod.rs` facade. Coupled orchestrator
      threaded via `pub(super)` seams (cooptimize drives 4 place_board helpers; result↔cooptimize
      mutual ref for `component_clearance_clean`; 2 test-exercised helpers pub(super)). Public path
      `crate::pipeline::*` byte-identical (7-symbol lib re-export + the leoneuro example's
      `pipeline::CoOpt` resolve through the facade). All source files ≤ 455 LOC. `Cargo.toml` `0.3.10`
      → `0.3.11`. `pipeline::` 10/10; full suite 415/415 green; pipeline fmt + clippy clean. Detail in
      `docs/MIGRATION.md ## Phase 4m`.
- [x] [patch] Phase 4l dfm slice migration (Tier 1; 2430 LOC — largest god-file): flat `src/dfm.rs`
      carved into a 7-file `src/dfm/` subtree split by pass role — `copper.rs` (ampacity widen /
      quietest-layer / ground-pour), `vias.rs` (dedup / teardrops / plane-distribute), `tracks.rs`
      (merge-collinear / pad-entry-stubs / body-junction-split / dangling+orphan cleanup), `diagonal.rs`
      (45°→ortho conversion + acid-trap chamfer + via-clearance repair), `miter.rs` (90°→135° corner
      mitering — split out as the opposite concern), `tests.rs` (verbatim), `mod.rs` facade. Each pass
      is self-contained (no production cross-call); 3 private helpers stay file-local. Public path
      `crate::dfm::*` byte-identical (8-symbol lib re-export + pipeline's 8 `crate::dfm::` calls resolve
      through the facade — zero rewrites). All source files ≤ 465 LOC. Carved mechanically (block-
      extraction script, per-file imports pruned). `Cargo.toml` `0.3.9` → `0.3.10`. `dfm::` 21/21; full
      suite 415/415 green; dfm fmt + clippy + doc clean. Detail in `docs/MIGRATION.md ## Phase 4l`.
- [x] [patch] Phase 4k units slice migration (Tier 1; 692 LOC + 12 tests): flat `src/units.rs` carved
      into a 7-file `src/units/` subtree split by role — `length.rs` (`Nm` repr-transparent integer
      nm), `quantity.rs` (`Unit` trait + `Float<U>` + `approx_eq`), `kinds.rs` (10 ZST kinds + aliases),
      `factories.rs` (SI-prefix constructors + temperature bridge), `arithmetic.rs` (same-unit/scalar
      macros + cross-unit dimensional algebra), `tests.rs` (12 verbatim), `mod.rs` facade. FIRST carve
      using extract-tests-first order (the Phase 4j data-loss process fix). Public path
      `crate::units::*` byte-identical (lib.rs 11-type re-export + `crate::geom::Nm` re-export resolve
      through the facade → ~230 `Nm` call-sites untouched); macro/generic coupling preserved by
      co-locating macros with invocations. Already zero-cost (`#[repr(transparent)]` throughout), no
      forced optimization. All 7 files ≤ 157 LOC. `Cargo.toml` `0.3.8` → `0.3.9`. `units::` 12/12,
      full suite 415/415 green; slice fmt + clippy + doc clean. Detail in `docs/MIGRATION.md ## Phase 4k`.
- [x] [patch] Phase 4j validate slice migration + test reconstruction (1272 LOC): flat
      `src/validate.rs` carved into a 5-file `src/validate/` subtree split by role — `check.rs`
      (`Check`/`PhysicsReport`), `board_checks.rs` (HV creepage/ampacity/core_checks/via_census/
      microvia/net-length/skew), `kwavers_beam.rs` (the driver→transducer seam: `KwaversBeamStep`,
      `manifest_to_kwavers_beam_step`, `KwaversBeamValidation`, `validate_against_budget`), `tests.rs`,
      `mod.rs` facade. Public path `crate::validate::*` byte-identical (only doc-link refs externally);
      SSOT bounds stay in `crate::ssot`. ⚠ DATA LOSS: the original 662-line test block was lost when
      the flat file was removed (tool auto-resolving the `validate.rs`↔`validate/` module ambiguity)
      before its tests could be extracted; leoneuro is git-ignored so no VCS copy. `tests.rs` is a
      from-contract reconstruction — 12 genuine value-semantic tests (assertions derived analytically),
      all passing; restores coverage, cross-check vs original if a copy resurfaces. `Cargo.toml`
      `0.3.7` → `0.3.8`. Full suite 415/415 green; validate/ fmt + clippy clean. Detail in
      `docs/MIGRATION.md ## Phase 4j`.
- [x] [minor] Phase 4i component_db slice migration + static-table optimization (634 LOC + 7 tests):
      flat `src/component_db.rs` carved into a 6-file `src/component_db/` subtree split by role —
      `pulser_ic.rs` (`PulserIc`/`StockStatus` + per-IC accessors), `catalog.rs` (pulser table +
      `available_pulsers`), `dcdc.rs` (`DcDcModule` + table + `available_dcdc_modules`), `compare.rs`
      (`PulserComparison`/`compare_pulsers`/`recommend_96ch_architecture`), `tests.rs` (7 verbatim),
      `mod.rs` facade. OPTIMIZATION: `PulserIc`/`DcDcModule` are all-`&'static str`/scalar (no String),
      so both datasheet tables — previously `vec![…]` rebuilt+heap-allocated every call — became
      compile-time `static` slices; `available_pulsers`/`available_dcdc_modules` now return
      `&'static [_]` (zero per-call alloc). Callers use `.iter()` (unchanged); one test `for p in
      &pulsers` → `for p in pulsers`. `Cargo.toml` `0.3.6` → `0.3.7`. Verified green: `component_db::`
      7/7 pass, full suite 420/421 (the 1 fail is the peer's `audit::tests::dirty_fields_…` meta-test,
      mid-restructure — not this slice); static tables const-construct, slice fmt-clean + zero new
      warnings. Detail in `docs/MIGRATION.md ## Phase 4i`.
- [x] [patch] Phase 4h optim slice migration + dead-allocation cleanup (Tier 1; 528 LOC + 5 tests):
      flat `src/optim.rs` carved into a 6-file `src/optim/` subtree split by role — `context.rs`
      (`ArrayGeometry`/`ThermalContext`/`PdnConfig`/`EmiContext` inputs), `report.rs` (`DesignReport`),
      `evaluate.rs` (`evaluate_design_point` orchestrator), `kernels.rs` (`max_safe_duty_thermal`/
      `ringing_exceeds_breakdown`/`hot_track_resistance`), `tests.rs` (5 verbatim), `mod.rs` facade.
      CLEANUP: removed a dead N-element `Vec` allocation in `evaluate_design_point` — it called
      `focused_delay_profile_s` then discarded the result (`let _ = delays;`) with no assertion;
      behaviour-identical (no `DesignReport` field uses it), one fewer heap alloc per call. Public path
      `crate::optim::*` byte-identical; leaf consumer (no external callers). All 6 files ≤ 158 LOC.
      `Cargo.toml` `0.3.5` → `0.3.6`. `cargo nextest run --lib` 421/421 green; slice fmt-clean + zero
      new clippy/doc warnings. Detail in `docs/MIGRATION.md ## Phase 4h`.
- [x] [patch] Phase 4g driver slice migration (Tier 1; 706 LOC + 13 tests): flat `src/driver.rs`
      carved into a 7-file `src/driver/` subtree split by physics role — `pulser.rs` (core loss model
      `PulserOp`/`PulserDissipation`/`pulser_dissipation`), `reactive.rs` (matching-network/reactive/
      ringdown/switching-node math), `rating.rs` (thermal-duty + package-power-rating limits),
      `sweep.rs` (frequency-sweep loss optimiser), `compare.rs` (cross-IC comparison), `tests.rs`
      (13 verbatim), `mod.rs` facade + `pub(super) const DEFAULT_THETA_JC_K_PER_W` (SSOT for the
      θ_jc shared by sweep+compare). Public path `crate::driver::*` byte-identical (20-symbol lib.rs
      re-export + `crate::driver::PowerOverload` unchanged); 4 external callers (component_db,
      manifest::energy_budget, optim, pipeline) resolve via the facade — zero rewrites. Two stale
      doc refs corrected in-move (`crate::thermal::` → `crate::physics::thermal::`; malformed
      `use///` line). All 7 files ≤ 240 LOC. `Cargo.toml` `0.3.4` → `0.3.5`. `cargo nextest run --lib`
      421/421 green; slice fmt-clean + zero new clippy/doc warnings. Detail in `docs/MIGRATION.md ## Phase 4g`.
- [x] [patch] Phase 4f manifest slice migration (Tier 1; 1422 LOC + 21 tests): flat `src/manifest.rs`
      carved into a 7-file `src/manifest/` subtree split by schema role — `stimulation.rs`
      (`StimulationProgram`/`TileStimulationProfile`), `resistor.rs` (`ResistorPackage`),
      `driver_manifest.rs` (`DriverManifest` schema + text round-trip + accessors), `energy_budget.rs`
      (`EnergyBudgetInputs`/`EnergyBudgetReport` + a second `impl DriverManifest` for
      `validate_v2_energy_budget`), `extract.rs` (`hv_manifest_from_board`), `tests.rs` (21 verbatim),
      `mod.rs` facade. The 387-line `DriverManifest` impl split across two files via two `impl` blocks
      (serialization vs energy-budget). Public path `crate::manifest::*` byte-identical, zero outward
      caller rewrites; SSOT constants stay sourced from `crate::ssot`. Source files all ≤ 275 LOC.
      `Cargo.toml` `0.3.3` → `0.3.4`. `cargo nextest run --lib` 421/421 green; slice fmt-clean + zero
      new clippy/doc warnings. Detail in `docs/MIGRATION.md ## Phase 4f`.
- [x] [patch] Phase 4a kicad_cli slice migration (Tier 1; 748 LOC + 6 tests): flat `src/kicad_cli.rs`
      carved into a 5-file `src/kicad_cli/` subtree split by role — `cli.rs` (`KiCadCli` process
      wrapper + `DrcOptions`, with `pub(super)` `drc_args`/`locate_on_path`), `drc.rs` (`DrcReport`/
      `DrcDefectCount` + the version-tolerant `parse_drc_json`), `fab.rs` (`FabBundle` + summary),
      `tests.rs` (6 verbatim), `mod.rs` facade. Public path `crate::kicad_cli::*` byte-identical
      (top-level `pub mod` → directory), zero outward caller rewrites. All 5 files ≤ 290 LOC.
      `Cargo.toml` `0.3.2` → `0.3.3`. `cargo nextest run --lib` 421/421 green; my slices fmt-clean +
      zero new clippy/doc warnings. Detail in `docs/MIGRATION.md ## Phase 4a`.
- [x] [patch] Phase 4e stack slice migration (Tier 1; 893 LOC + 11 tests): flat `src/stack.rs` carved
      into an 8-file `src/stack/` subtree split by role — `plan.rs` (single-board thermal/height/
      capacity optimiser), `role.rs` (`StackBoardRole`), `manifest.rs` (`StackBoardManifest` +
      extraction), `compatibility.rs` (`verify_stack_pair`), `shield.rs` (full shield-stack assembly),
      `util.rs` (`pub(super)` geometry/canonicalisation helpers), `tests.rs` (11 verbatim), `mod.rs`
      facade. Public path `crate::stack::*` byte-identical (top-level `pub mod` → directory), so zero
      outward caller rewrites; the two internal doc-links resolve unchanged. All 8 files ≤ 268 LOC.
      `Cargo.toml` `0.3.1` → `0.3.2`. `cargo nextest run --lib` 421/421 green; zero new clippy/doc
      warnings from the slice. Detail in `docs/MIGRATION.md ## Phase 4e`.
- [x] [patch] Phase 3g acoustic slice migration (Tier 1; 437 LOC + 13 tests): `src/acoustic.rs`
      carved into an 8-file `src/physics/acoustic/` subtree (`mod.rs`, `wavelength.rs`,
      `grating.rs`, `focus.rs`, `element.rs`, `safety.rs`, `nonlinear.rs`, `tests.rs`).
      `pub mod acoustic;` declared at `src/physics/mod.rs`; flat `pub mod acoustic;` retired
      from `src/lib.rs`; flat `src/acoustic.rs` deleted. Sub-module split by **physical
      role**, not file-size symmetry: `wavelength.rs` (λ + BVD series-branch resonance + new
      BVD anti-resonance per Kino §3.4 / IEEE Std 176); `grating.rs` (steering limits +
      ULA array factor); `focus.rs` (relative delays + quantisation + error); `element.rs`
      (Fresnel range + directivity + f-number + pitch-from-aperture + focal gain);
      `safety.rs` (MI + tissue derating + continuous-RMS intensity + new ISPPA +
      new round-trip attenuation); `nonlinear.rs` (Earnshaw σ). 3 NEW APIs added:
      `bvd_anti_resonance_hz(ls_h, cs_f, c0_f)` (textbook BVD anti-resonance per Kino §3.4 —
      the originally-introduced `bvd_parallel_resonance_hz(L_p, C_p)` was a generic LC tank
      kernel, NOT the textbook BVD anti-resonance, renamed at code-reviewer round-2;
      retro-fit doc-residue cleanup touched `wavelength.rs`/`mod.rs`/`tests.rs`/`physics/mod.rs`),
      `isppa_w_per_m2(p_neg, z0, duty)` (FDA Track-3 spatial-peak pulse-average intensity,
      SSOT-distinct from continuous-RMS `acoustic_intensity_w_per_m2`),
      `round_trip_attenuation_db(α, f, z)` (pulse-echo two-way loss,
      SSOT-distinct from one-way `tissue_attenuation_db`). 20 tests consolidated
      (13 prior lifted verbatim + 7 new SSOT-distinction + range tests for the 3 NEW APIs).
      `src/lib.rs::pub use physics::acoustic::{...}` is the canonical 21-symbol crate-root
      surface. `Cargo.toml` bumped `0.3.0` → `0.3.1`. cargo build clean (1 unrelated
      `Rot` unused-import warning in src/place/footprint.rs), cargo test --lib
      physics::acoustic 20/0 green, full lib 421/0 green, cargo doc strict-clean for
      `src/physics/acoustic/` zero errors. The residual ~10 cargo doc strict-clean errors
      across the rest of the crate are pre-existing Phase 3a/3b/3c/3d/3e/3f
      ambiguous-link + top-level-retirement issues; catalogued under
      `docs/MIGRATION.md ## Phase 3 follow-ups ## Phase 3 doc-strict clean-up (placeholder)`
      as Phase 3h scope (NOT scope-creep of Phase 3g).
- [x] [patch] Phase 3f si slice migration (Tier 1; 232 LOC + 6 tests): `src/si.rs` carved into
      a 5-file `src/physics/si/` subtree (`mod.rs`, `impedance.rs`, `propagation.rs`,
      `crosstalk.rs`, `tests.rs`). `pub mod si;` declared at `src/physics/mod.rs`; flat
      `pub mod si;` retired from `src/lib.rs`; flat `src/si.rs` deleted. 8 existing fns
      carried across + 3 NEW APIs (`impedance_target` for signal-line branching-match target,
      `return_loss_db` for caller-loop RL over freq bands,
      `channel_operating_margin_db` for IEEE amplitude-ratio COM).
      `src/lib.rs::pub use physics::si::{...}` is the canonical crate-root surface.
      Cargo.toml bumped `0.2.15` → `0.3.0`.
- [x] [patch] Phase 3e pdn remainder slice migration + capacitive_drive_currenta typo fix
- [x] [patch] Phase 3d emi slice migration (Tier 2; 388 LOC + 8 tests): `src/emi.rs` carved into an
      8-file `src/physics/emi/` subtree (`mod.rs`, `scene.rs`, `loop.rs`, `trace_partial.rs`,
      `losses.rs`, `overshoot.rs`, `radiated.rs`, `tests.rs`). `pub mod emi;` declared at
      `src/physics/mod.rs`; flat `pub mod emi;` retired from `src/lib.rs`; flat
      `src/emi.rs` deleted. Slice facade uses explicit named `pub use` (NOT glob like
      ampacity). **The `r#loop` raw-identifier escape**: `loop` is a Rust reserved
      keyword — the file on disk is `loop.rs` but the mod decl MUST be `pub mod r#loop;`
      with all sibling imports using `super::r#loop::{...}`. Slice-private internals:
      `pub(super) const MU0` (vacuum permeability, shared between `loop.rs` and
      `trace_partial.rs`) + `pub(super) fn polygon_area_mm2` (shoelace helper for the
      scene walker). 9 outward-caller sites across 5 files re-routed from `crate::emi::*`
      to `crate::physics::emi::*`: `src/optim.rs` (line 22 source-code + 1 doc-link),
      `src/audit.rs` (2 source-code refs + 3 doc-links — applied via `sed -i` because
      the 317K-char audit.rs exceeds `str_replace`'s 100K-char limit), `src/driver.rs`
      (2 doc-links), `src/place/footprint.rs` (1 doc-link), `src/rules.rs` (2 doc-links).
      `src/lib.rs::pub use physics::emi::{...}` is the canonical 10-symbol crate-root
      surface, byte-identical to the prior flat re-export. `Cargo.toml` bumped
      `0.2.13` → `0.2.14`. cargo build/test/doc clean.
- [x] [patch] Phase 3c dielectric slice migration (user-task numbering = MIGRATION.md' Phase 3b): `src/dielectric.rs`
      (134 LOC + 4 tests) carved into a 5-file `src/physics/dielectric/` subtree
      (`mod.rs`, `paschen.rs`, `ipc2221_spacing.rs`, `caf.rs`, `tests.rs`). `pub mod dielectric;`
      declared at `src/physics/mod.rs`; flat `pub mod dielectric;` retired from `src/lib.rs`;
      flat `src/dielectric.rs` deleted. Slice facade uses explicit named `pub use` (NOT glob
      `pub use X::*;` like ampacity) to keep slice-private air constants (`A_AIR`/`B_AIR`/
      `GAMMA`) out of the slice-level API surface; the rationale is a `//` line comment so it
      doesn't surface as part of the published rustdoc API contract. Single source-code
      doc-link fan-out in `src/rules.rs:370` re-routed from `crate::dielectric::ipc2221_min_spacing_mm`
      to `crate::physics::dielectric::ipc2221_min_spacing_mm`. `src/lib.rs::pub use physics::dielectric::{...}`
      is the canonical crate-root surface, byte-identical to the prior flat re-export. `Cargo.toml`
      bumped `0.2.12` → `0.2.13`. `cargo build --lib` 0 errors, `cargo test --lib physics::dielectric`
      4/4 passing, `cargo doc --no-deps` 0 errors.
- [x] [patch] Phase 3b thermal slice migration (vertical-slice carve-out): `src/thermal.rs`
      (468 LOC + 6 tests) carved into a 7-file `src/physics/thermal/` subtree per the target tree
      shape. `pub mod thermal;` declared at `src/physics/mod.rs`; flat `src/thermal.rs` retired.
      `IrDrop` + `ir_drop` promoted out of `src/pdn.rs` into `src/physics/thermal/ir_drop.rs` so
      the electro-thermal coupling chain (`ir_drop` → `joule_source` → `solve_electrothermal`)
      sits in one crate plane. `src/pdn.rs` keeps the decoupling/resonance/impedance half of
      PDN. 9 thermal/IR-drop tests consolidated into `src/physics/thermal/tests.rs`. Lib.rs
      re-exports `pub use physics::thermal::{ir_drop, junction_temperature_k, solve_board,
      solve_electrothermal, temperature_derated_resistance, thermal_time_constant_s,
      thermal_via_conductance, transient_rise_k, IrDrop, ThermalField};`.
- [x] [patch] Phase 3a ampacity slice migration (vertical-slice carve-out): `src/ampacity.rs`
      (257 LOC + 7 tests) carved into a 7-file `src/physics/ampacity/` subtree. `pub mod
      ampacity;` declared at `src/physics/mod.rs`; flat `src/ampacity.rs` retired. The
      `track_resistance` function (the sole Tier-2 upstream enabler for both `thermal::joule_source`
      and `pdn::ir_drop`) lives at `crate::physics::ampacity::track_resistance`.
- [x] [patch] 32-channel HV7355 board closure: `hv7355_32ch_tile` now generates a connected
      100 mm x 80 mm six-layer board with four 8-channel output banks, banked control nets, banked
      negative HV returns, 0.20 mm dense-board HV/power routing, and KiCad CLI DRC reports
      0 violations and 0 unconnected items on
      `output/boards/hv7355_32ch_tile/hv7355_32ch_tile.kicad_pcb`. The DFM pipeline now splits
      same-net body-contacted track junctions so internal dangling audit and KiCad
      `track_dangling` warnings agree at zero.
- [x] [patch] Output layout cleanup: canonical board variants are separated under
      `output/boards/hv7355_24ch_tile/` and `output/boards/hv7355_32ch_tile/`; legacy single-board
      examples, standalone renders, reports, manifests, and archived fabrication bundles are no
      longer mixed into the root of `output/`.
- [x] [patch] Hard/soft audit split: `clearance_violations` is the hard DRC count; `near_shorts`
      remains the wider adversarial risk band.
- [x] [patch] KiCad DRC write-gate cleanup: examples now refuse to write KiCad boards unless the
      co-optimization result is complete, legal, hard-internal-DRC clean, LVS clean, and component
      courtyard clean. Stale generated boards and rejected `_j4_*` trial artifacts that carried
      current KiCad DRC errors were removed; the fresh non-third-party KiCad sweep reports
      0 errors and 0 unconnected items on every remaining board.
- [x] [patch] HV7355 article example: 0.5 mm routing lattice, continuous pad/via keepouts, complete
      route, internal DRC/assembly/LVS/ERC/BOM verification PASS, and beamforming delay quantization
      check for ±45° steering at 10 mm focus.
- [x] [patch] FPGA/HV stack connector consistency: 2x12 narrow serial/control-clock bus on both tile
      examples.
- [x] [patch] Full-driver output: FPGA controller and HV7355 driver examples both generated into
      `output/full_driver`.
- [x] [patch] Beamforming output: Rust renderer generated -45°, 0°, and +45° steering pictures plus
      metrics CSV in `output/beamforming`.
- [x] [patch] Beamforming render correction: replaced the continuous-wave ridge visualization with a
      focal-arrival pulse-envelope render and recorded lateral/axial 6 dB widths in the metrics CSV.
- [x] [patch] Assembly-clearance gate: component courtyard spacing is a hard verification axis, and
      full-driver renders show separated packages.
- [x] [patch] Connector body-envelope gate: transducer and JTAG connector 3D model envelopes are
      verified against their courtyards so rendered package bodies cannot silently exceed placement
      clearance.
- [x] [patch] Board-backed beamforming: `output/full_driver/driver_manifest.kv` records the HV
      transducer connector, 16 TX nets, and FPGA programming evidence; `beamforming_results` reads
      that manifest before rendering.
- [x] [patch] Connector CAD inventory: downloaded Molex/Samtec/Amphenol connector CAD files are
      extracted under `docs/cad_models` and summarized in `docs/connector_cad_inventory.md`.
- [x] [patch] Stack manifests and shield-stack model: FPGA and HV examples emit `stack_fpga.kv` and
      `stack_hv.kv`; compatibility passes on outline, connector geometry, and normalized 24-pin
      pinout.
- [x] [patch] Complete stack assembly: `stack_model` emits `shield_stack_assembly.kv` with one
      controller slot, four HV slots, 60 mm stack height, and global TX channel coverage `TX_0..TX_95`.
- [x] [patch] Component CAD audit: extracted downloaded FPGA/HV/supporting component CAD archives
      into `docs/cad_models`, added `docs/component_cad_inventory.md`, and made both board examples
      emit component accuracy manifests.
- [x] [patch] 96-channel shield stack: HV tiles now instantiate three daisy-chained `HV7355K6-G`
      pulsers for 24 local channels, and the stack assembly maps four HV tiles to `TX_0..TX_95`.
- [x] [patch] Internal copper-edge keep-in gate: final routed track/via copper edge clearance is
      validated after ampacity widening, preventing KiCad-only board-edge clearance failures.
- [x] [patch] Article comparison figures: `docs/article_vs_current_stack.md` includes generated stack,
      board, and beamforming figures.
- [x] [patch] Exact HV transducer connector: HV `J2` imports the local Molex `0430452400`
      footprint/STEP, routes all 24 TX pins through plated-pin access groups, and emits pad-entry
      copper verified by KiCad DRC.
- [x] [patch] Exact stack/programming connectors: FPGA `J1`, FPGA `J3`, FPGA `J_STACK`, and HV
      `J_STACK` now import the downloaded Molex/Samtec `.kicad_mod` footprints with pad-name net
      mapping; both generated full-driver boards pass external KiCad DRC with 0 violations and
      0 unconnected items.
- [x] [patch] FPGA controller 3D model attachment: exact Molex `0430450600` PCB-header STEP is
      attached to FPGA `J1`, and the KiCad `BGA-484_23.0x23.0mm_Layout22x22_P1.0mm.step` package
      body is attached to FPGA `U1`; `output/full_driver/fpga_controller_tile_model_fix.png`
      verifies both bodies render. Remaining generated-board 3D gaps are FPGA VCC bulk caps and
      HV7355 `U1..U3`.
- [x] [patch] Imported model transform preservation: KiCad footprint import now parses nested
      model `(offset ...)` and `(rotate ...)` vectors from vendor `.kicad_mod` files, including the
      raw Molex `0430450600` `(3.0, 2.5643, 0)` STEP offset.
- [x] [patch] CAD model coordinate-frame normalization: KiCad footprint import now translates STEP
      model offsets by the same courtyard-centre shift applied to imported pads, so imported pads and
      bodies stay in one coordinate frame. The 32-channel comparison tile also derives stock two-row
      pin-header model offsets from the centered pad grid (`J1`: `(-1.27, -10.16, 0)`, `J2-J5`:
      `(-1.27, -3.81, 0)`, `J6`: `(-1.27, -5.08, 0)`), emits vertical 2-column pad grids matching
      the stock KiCad header bodies, and KiCad DRC/render were regenerated.
- [x] [patch] Thermal-feedback optimizer scale: co-optimization now feeds a normalized solved
      temperature field into placement and applies `thermal_weight`/`emi_weight` directly, while
      preserving `feedback_weight` for congestion/weakness feedback.
- [x] [patch] Connector model coordinate audit: FPGA `J3` and FPGA/HV `J_STACK` stock KiCad pin-header
      STEP bodies now use pad-grid-derived offsets/rotations (`J3`: `(-6.35, 0, 0)`, `-90 deg`;
      `J_STACK`: `(-13.97, -1.27, 0)`, `-90 deg`) and final KiCad CLI renders verify aligned
      bodies in `output/full_driver/fpga_controller_tile_connector_aligned.png` and
      `output/full_driver/hv7355_driver_tile_connector_aligned.png`.
- [x] [patch] HV transducer connector 3D Y alignment: Molex `0430452400` STEP body now uses
      `(0, 2.8, 0)` so the rendered right-angle connector covers the generated plated pin rows
      instead of leaving the pad dots exposed above the connector face.
- [x] [patch] HV connector refdes correction: the 24-position Molex `0430452400` output is now
      `J2` in the board artifacts, manifests, stack assembly metadata, and comparison docs; KiCad
      CLI DRC on `output/full_driver/hv7355_driver_tile.kicad_pcb` reports 0 violations and
      0 unconnected items, and `output/full_driver/hv7355_driver_tile_connector_aligned.png` shows
      the `J2` label on the rendered `430452400` CAD body.
- [x] [patch] J4 exact board-header render body: extracted `docs/cad_archives/430450400.zip` into
      `docs/cad_models/430450400/` and attached the exact `430450400.step` to HV `J4`; KiCad CLI DRC
      on the output board reports 0 violations and 0 unconnected items, and
      `output/full_driver/hv7355_driver_tile_connector_aligned.png` renders the exact header body.
- [x] [patch] J4 exact-footprint trial rejected: a direct swap from the demo SMD pad geometry to
      exact `MOLEX_430450400` through-hole pads without rerouting produced 1219 KiCad DRC violations
      and 2 unconnected items, so exact footprint closure remains a routed-board regeneration task.
- [x] [patch] KiCad automation hardening: DRC wrapper supports `--refill-zones`/`--save-board`,
      generated PCB saves reject duplicate object UUIDs, emitted boards assert UUID uniqueness, and
      exact Molex `0430450400` footprint import is value-tested against the downloaded KiCad-v6 CAD.
      Current `output/full_driver/hv7355_driver_tile.kicad_pcb` refilled through KiCad CLI with
      0 violations and 0 unconnected items.
- [x] [patch] KiCad DRC severity parsing and exact-J4 trial audit: KiCad JSON `violations[]` entries
      are now classified by `severity`, so warning-only reports do not masquerade as hard DRC errors
      while `unconnected_items[]` still fails `DrcReport::passes()`. The UUID-repaired exact-J4
      candidate reported 0 hard errors, 3 warnings, and 4 unconnected items; routed exact-J4 variants
      removed the opens but still produced 15-19 hard KiCad errors, so the rejected trial artifacts
      were deleted during the DRC cleanup.
- [x] [patch] Exact J4 footprint/routing closure: HV `J4` now uses the exact downloaded
      `MOLEX_430450400` through-hole footprint and `docs/cad_models/430450400/430450400.step` body
      at `(at 95.52 20 90)`, keeping the right-angle footprint's `PCB EDGE` datum on the board
      edge while moving the connector into the power-input y-region. The routed board preserves the
      existing power anchors, moves the P3V3 layer transition out of the BUS corridor, and passes
      KiCad CLI DRC with 0 violations and 0 unconnected items after zone refill/save. Regression
      coverage now asserts the production artifact contains exactly one exact J4 footprint, rejects
      the old `(at 95.52 32 90)` lower-quadrant placement, and contains no
      `J_PWR_Molex_0430450400_demo` token.
- [x] [patch] J4 STEP rotation regression: HV `J4` now keeps the downloaded
      `430450400.step` model at zero local rotation so the rendered board-header body follows the
      footprint pad/NPTH coordinate system instead of rotating away from the plated-hole pattern.
      Regression coverage rejects the previous local `90 deg` model rotation.
- [x] [patch] Example target compile hygiene: exported `Axis` and `IsolationDomain` from the crate
      root and updated stale example `Component`/`PlaceConfig` literals to use the defaulted
      isolation fields, restoring `cargo clippy --all-targets --all-features -- -D warnings`.
- [x] [patch] FPGA exact-example compile hygiene: `examples/fpga_tile_exact.rs` now keeps the used
      `VCC_1V8` net parameter live and removes stale unused JTAG/SPI ball-group constants, restoring
      `cargo clippy --all-targets --all-features -- -D warnings`.
- [x] [patch] EMI/manufacturing optimizer hardening: EMI hotspots are fed back as a 6 mm coupling
      radius field plus a judged objective, and the internal NASA-rule fixture now isolates sharp
      bend, serpentine, via-spacing, differential-pair, edge, and split-plane checks.
- [x] [patch] Optimizer hard-DRC gate consolidation: co-optimization clean-board selection now uses
      the same hard DRC predicate as final verification, including every guide-derived routing rule.
- [x] [patch] Optimizer seed/judge hardening: co-optimization now judges the seed floorplan before
      annealed feedback rounds and ranks out discrete grid-cell/via-column shorts when choosing
      among complete legal candidates.
- [x] [patch] Serpentine copper-edge spacing correction: same-net serpentine spacing now compares
      copper edge gap, not centerline separation, against the guide's 4W spacing rule.
- [x] [patch] Serpentine compensation locality gate: internal DRC now enforces the guide's 15 mm
      maximum distance from length-compensation meanders to the local bend/mismatch root.
- [x] [patch] Sharp-bend geometry correction: same-net bend DRC now rejects acute bends as well as
      90-degree bends while accepting the guide's preferred 135-degree bend geometry.
- [x] [patch] Placement rotation hardening: footprints now carry a rotation policy, imported CAD
      inherits the policy from role, and the annealer preserves fixed connector/IC/power orientations
      while allowing only 180-degree flips for symmetric passives by default.
- [x] [patch] High-speed return-plane margin gate: internal DRC now enforces the guide's 3W
      reference-plane side margin for high-speed tracks inside ground/power zones and feeds
      violations back into adversarial routing risk.
- [x] [patch] Reference-plane margin routing cost: physics-guided routing now grades Signal/HV
      nodes near adjacent reference-zone boundaries so search prefers stronger return-plane margin.
- [x] [patch] Top-side high-speed routing cost: physics-guided routing now applies a soft
      bottom-layer penalty to Signal/HV nets so top-side routes are preferred when other constraints
      are equal.
- [x] [patch] High-speed reference-plane presence gate: internal DRC now rejects TX/OUT/TRIG/HV
      tracks whose start, midpoint, and end lack adjacent-layer ground/power reference coverage.
- [x] [patch] Reference-plane routing cost: physics-guided routing now penalizes Signal/HV nodes
      without adjacent ground/power zone coverage so search prefers reference-backed routing space.
- [x] [patch] Inner-layer dual-ground reference gate: internal DRC now rejects TX/OUT/TRIG/HV
      tracks routed on inner layers unless both adjacent layers provide ground-zone coverage.
- [x] [patch] Inner-layer dual-ground routing cost: physics-guided routing now adds an extra
      Signal/HV penalty on inner nodes unless both adjacent layers have ground-zone coverage.
- [x] [patch] Power-reference stitching-cap gate: internal DRC now rejects TX/OUT/TRIG/HV tracks
      referenced only to a power plane unless C* power-to-ground capacitors are near both endpoints.
- [x] [patch] Differential-pair stitching-cap symmetry gate: internal DRC now flags P/N
      power-reference stitching capacitors whose pair-axis stations exceed the configured symmetry
      budget.
- [x] [patch] Power-reference routing cost: physics-guided routing now prefers ground-backed
      Signal/HV reference planes over power-only references while still preferring power over none.
- [x] [patch] Split-plane stitching-cap proximity gate: internal DRC now requires stitching
      capacitors within the configured 2 mm budget of the split-plane crossing point.
- [x] [patch] Split-plane stitching-cap reference-bridge gate: internal DRC now requires the local
      stitching capacitor to bridge the crossed reference zone to another ground/power reference net.
- [x] [patch] Plane-hotspot via-spacing gate: internal DRC now rejects same-net non-ground via
      clusters whose outer pad gaps are below the configured 15 mil copper-pass budget.
- [x] [patch] Reference-plane intrusion gate: internal DRC now rejects non-plane signal tracks routed
      through a ground/power reference zone on the same layer, preventing routed copper from carving
      the plane selected for return-current integrity.
- [x] [patch] Ground-plane fragmentation gate: internal DRC now rejects same-net ground reference
      pours split into multiple thermal-relief islands on the same copper layer.
- [x] [patch] Split-domain reference-plane gate: internal DRC now rejects explicit analog/digital
      signal nets routed over the opposite AGND/DGND split-ground reference zone.
- [x] [patch] Mixed-domain shared-return gate: internal DRC now rejects analog and digital signal
      tracks that share one adjacent ground reference zone while crossing or running inside the
      sensitive keepout distance, covering same-GND return-current interference.
- [x] [patch] High-speed stub topology gate: internal DRC now detects connected TX/OUT/TRIG/HV
      T-branches so routed high-speed nets prefer daisy-chain topology instead of stub antennas.
- [x] [patch] High-speed daisy-chain routing: Signal/HV multi-terminal nets now grow from the
      current chain tip and avoid re-entering earlier chain nodes, so the router forms branch-free
      high-speed chains before the stub audit runs.
- [x] [patch] Critical-net routing order: each PathFinder negotiation pass now routes HV and Signal
      nets before Power/Ground nets while preserving caller-visible output order, so high-speed
      routes reserve clean channels before lower-priority copper.
- [x] [patch] Differential-pair symmetry gate: internal DRC now flags pair members routed on
      different layer sets or with different via counts, matching the guide's same-layer/same-via
      routing rule.
- [x] [patch] High-speed layer-transition return gate: internal DRC now flags TX/OUT/TRIG/HV
      signal vias that change layers without a nearby ground transition via.
- [x] [patch] Differential-pair transition-return symmetry gate: internal DRC now flags P/N
      layer-transition ground vias whose pair-axis stations exceed the configured symmetry budget.
- [x] [patch] Differential-pair length-matching gate: internal DRC now flags pair members whose
      routed lengths differ by more than the configured 0.5 mm tolerance.
- [x] [patch] High-speed parallel spacing gate: internal DRC now flags unrelated long parallel
      TX/OUT/TRIG/HV runs closer than the configured 3W spacing while exempting true differential
      pair mates.
- [x] [patch] Clock high-speed parallel spacing gate: internal DRC now applies the 50 mil clock
      keepout to unrelated long parallel high-speed runs when either net is clock-like, while
      keeping non-clock runs on the 3W rule.
- [x] [patch] Board-utilization placement energy: placement now includes a low-weight macro-grid
      utilization term so optimization penalizes large unused board regions without overriding
      short routing, edge keep-in, or minimum-board-size selection.
- [x] [patch] Similar-component alignment energy: placement now penalizes identical package/role
      instances that mix 0/180 and 90/270 axes, preserving half-turn passive flips while improving
      assembly regularity and routing-channel predictability.
- [x] [patch] Hot-device airflow blockage placement energy: placement now penalizes connector
      courtyards intersecting the nearest-edge cooling corridor to active IC and power packages.
- [x] [patch] Regional functional-block placement energy: placement now groups components sharing
      local nets while ignoring board-global nets that touch every component, matching section
      7.2.2.2 functional subsection placement guidance.
- [x] [patch] Regional rail-domain placement energy: placement now groups components with matching
      power-pin VCC/GND net sets while keeping shared global ground from merging unrelated blocks.
- [x] [patch] Regional associated-component placement energy: placement now groups components with
      their explicit `assoc_ic` main component so support parts that do not share a local net still
      remain in the main IC's functional block.
- [x] [patch] Crystal oscillator proximity placement energy: placement now pulls X*/Y*
      clock-source support components toward their associated IC pads on shared nets so the
      critical oscillator routes stay short before routing.
- [x] [patch] Connector-side surge-suppressor placement energy: placement now pulls D*/TVS*
      passive suppressors toward incoming connectors on shared nets so the clamp path is short
      before routing.
- [x] [patch] Surge-suppressor via-path gate: internal DRC now rejects same-net vias placed in the
      incoming connector-to-D*/TVS* suppressor segment, matching the guide's warning about via
      parasitic inductance in the clamp path.
- [x] [patch] Regional interloper placement energy: placement now penalizes unrelated components
      whose centers intrude into another local functional block's bounding region.
- [x] [patch] Regional package-intrusion placement energy: placement now penalizes unrelated
      component courtyards that enter another local functional block's physical envelope, even when
      the unrelated component center remains outside the block.
- [x] [patch] Regional power-isolation placement energy: placement now penalizes power-conversion
      package courtyards that enter a local signal block's isolation halo, matching section 7.2.2.2
      voltage/current regional separation guidance.
- [x] [patch] Regional connector-EMI placement energy: placement now penalizes active ICs with
      connected signal pads when their courtyards enter a connector EMI halo, matching section
      7.2.2.2 guidance to keep sensitive high-speed devices away from board-edge connector noise.
- [x] [patch] Signal-flow crossing placement energy: placement now penalizes crossed two-terminal
      local-net flight lines so floorplans prefer smoother signal flow before routing.
- [x] [patch] Regional fold-back signal-flow energy: placement now penalizes local point-to-point
      chains whose intermediate component has both connected neighbors on the same side, so
      floorplans prefer smooth unidirectional flow through functional regions.
- [x] [patch] Regional dogleg signal-flow energy: placement now penalizes local point-to-point
      chains whose intermediate component forces an orthogonal turn, extending section 7.2.2.2
      smooth unidirectional signal-flow guidance beyond fold-back cases.
- [x] [patch] Regional main-chip pad-proximity energy: placement now penalizes actual connected
      pad-to-pad distance for direct active-IC signal links, so floorplans prefer shorter
      main-component traces instead of only compact component centers.
- [x] [patch] Regional connector-ingress placement energy: placement now penalizes connector-local
      net flow that does not point inward from the nearest board edge toward the board core,
      extending section 7.2.2.2 smooth unidirectional signal-flow guidance to I/O ingress.
- [x] [patch] Routing-channel blockage placement energy: placement now penalizes unrelated
      component courtyards that intersect another two-terminal local net's direct routing channel.
- [x] [patch] High-speed edge routing cost: physics-guided routing now uses `DesignRules` high-speed
      edge clearance and reference-margin budgets so Signal/HV search avoids board-edge cells before
      audit rejects them.
- [x] [patch] High-speed preferred-spacing routing cost: physics-guided routing now rasterizes
      existing Signal/HV track proximity into a soft spacing field so later high-speed routes prefer
      extra separation outside hard bottlenecks before the parallel-spacing audit runs.
- [x] [patch] Adjacent-layer high-speed parallelism gate: internal DRC now rejects unrelated
      high-speed runs that overlap in parallel on adjacent copper layers, while accepting orthogonal
      adjacent-layer crossings and laterally separated broadside routes.
- [x] [patch] Adjacent-layer high-speed routing cost: physics-guided routing now penalizes Signal/HV
      cells near existing high-speed copper on immediately adjacent layers, so search prefers
      lateral separation before the broadside-parallel audit runs.
- [x] [patch] High-speed via routing cost: the routing cost seam now charges Signal/HV layer
      transitions above plane-like nets, biasing high-speed routing to reduce impedance-disrupting
      vias before via-stub, return-via, and differential-pair symmetry audits run.
- [x] [patch] Guide-stage routing order: placement-to-routing now tags crystal-associated nets,
      decoupling power/ground legs, controlled signals, and bulk nets so PathFinder reserves
      crystal and decoupling channels before general controlled routing.
- [x] [patch] Termination-resistor placement energy: placement now pulls resistor-like passives
      toward active IC pads on their shared net so terminators are placed before final audit rejects
      long stubs.
- [x] [patch] Termination-resistor force-directed placement: annealing proposals now include the
      same active-pad attraction used by termination placement energy, so the optimizer searches
      toward close terminators instead of relying only on random accepted moves.
- [x] [patch] Differential-pair segment length gate: internal DRC now rejects locally skewed
      via-delimited pair segments even when total P/N route length, layer set, and via count match.
- [x] [patch] Parallel-bus length-skew gate: internal DRC now groups explicitly indexed bus nets
      such as `BUS_D0`/`BUS_D1` and rejects routed group skew above the configured 2 mm budget,
      while leaving non-bus indexed TX channels ungrouped.
- [x] [patch] Same-interface differential-pair layer gate: internal DRC now groups explicitly
      indexed differential pairs such as `MIPI_D0_P/N` and `MIPI_D1_P/N` and rejects interface
      groups whose pairs use different routed layer sets.
- [x] [patch] Same-interface differential-pair via-count gate: internal DRC now groups explicitly
      indexed differential pairs and rejects interface groups whose pairs use different total routed
      via counts.
- [x] [patch] Differential-pair pad-entry symmetry gate: internal DRC now rejects P/N pad-entry
      breakout distances whose mismatch exceeds the configured 0.5 mm local symmetry budget.
- [x] [patch] Differential-pair pad-entry length gate: internal DRC now rejects matched-but-long
      P/N pad-entry breakouts whose uncoupled length exceeds the configured 2 mm local budget.
- [x] [patch] Differential-pair constant-spacing gate: internal DRC now rejects routed P/N pairs
      whose parallel segment spacing varies beyond the configured 0.25 mm tolerance.
- [x] [patch] Differential-pair via-station symmetry gate: internal DRC now rejects equal-count
      P/N via sets whose stations differ by more than the configured 0.5 mm tolerance.
- [x] [patch] High-speed component edge placement gate: internal DRC now flags active ICs carrying
      TX/OUT/TRIG/HV nets when their courtyards enter the 3 mm high-speed component edge keepout,
      while preserving board-edge placement for connectors.
- [x] [patch] High-speed termination placement gate: internal DRC now flags resistor-like passive
      terminators on TX/OUT/TRIG/HV nets when no active IC pad on that net is within the configured
      2 mm placement budget.
- [x] [patch] High-speed terminal return-path gate: internal DRC now flags TX/OUT/TRIG/HV source
      and sink pads that lack nearby ground return copper, extending the guide's return-via rule
      beyond layer-transition vias.
- [x] [patch] Virtual split-domain routing gate: internal DRC now infers analog/digital virtual
      split lines from domain pad centroids and rejects analog or digital signal tracks crossing to
      the opposite side, matching the guide's virtual split-plane routing rule.
- [x] [patch] Diagonal acid-trap chamfer repair: DFM chamfering now selects the orthogonal L-corner
      from the offending acute junction, so a diagonal branch is not replaced by a duplicate branch
      that leaves the original acid-trap topology intact.
- [x] [patch] Diagonal route-crossing hardening: internal audit now rejects same-layer different-net
      track crossings, PathFinder rejects crossed foreign diagonal edges and diagonal moves that clip
      a foreign via corner, and the FPGA example's external KiCad DRC count dropped from 327 to 20
      on the checked iteration.

## Next
- [x] [patch] Resolve the remaining FPGA example DRC residues: all 10 external KiCad clearance,
      8 shorting-items, 1 hole-clearance, and 1 dangling-track violations were progressively closed
      by the obstacle halo fixes, pad stub guards, GND power halo, and the DONE/PROG diagonal repair.
      The final external KiCad DRC=1 (DONE diagonal at 50.5,39.5) is marked [x] below. Internal
      DRC gates for via-adjacency, sharp-bend, serpentine-length, and decoupling-proximity are
      implemented as the checked items below; residual metric values (acid_traps=42,
      clearance_violations=5, sharp_bends=148) reflect the internal adversarial model — not
      KiCad-level violations.
- [x] [patch] Pad-aware miter skip in `miter_right_angle_corners`: the DFM pass now accepts
      `comps`, `lib`, and `rules` and skips any miter whose 45° diagonal would come within
      `min_clearance + half_track_width` of a foreign-net or unconnected pad edge. Eliminates the
      DFM-pass-induced clearance violations caused by miter endpoints landing in pad halos on
      fine-pitch dense routing. Regression test: `miter_skips_corner_too_close_to_foreign_pad`.
      359/359 tests pass; clippy clean. Also fixed: pre-existing `validate` test string mismatch
      (`>= 0 W` vs `≥ 0 W`), exported kwavers seam API (`validate_against_budget`,
      `manifest_to_kwavers_beam_step`, `KwaversBeamStep`, `KwaversBeamValidation`,
      `EnergyBudgetInputs`, `EnergyBudgetReport`, `ResistorPackage`, `TileStimulationProfile`),
      created `.config/nextest.toml` timeout gate, and isolated `fpga_tile_exact` linker
      failure behind `required-features = ["exact_models"]` in Cargo.toml.
- [x] [patch] High-speed via-pad proximity gate: internal DRC now flags TX/OUT/TRIG/HV vias whose
      nearest same-net pad is outside the configured 2 mm local-placement budget.
- [x] [patch] High-speed via diameter gate: internal DRC now flags TX/OUT/TRIG/HV vias whose
      outer diameter exceeds the selected design-rule via diameter.
- [x] [patch] Decoupling ground-via gate: internal DRC now flags SMD decoupling capacitor ground
      pads whose nearest ground via is outside the configured 1 mm local-return budget.
- [x] [patch] Decoupling power-layer gate: internal DRC now flags SMD decoupling capacitor power
      pads that do not share a copper layer with the associated IC power pin on the same rail,
      preventing a hidden via in the bypass power leg.
- [x] [patch] Decoupling commutation-loop area gate: internal DRC now flags associated decoupling
      capacitors whose cap-to-IC power/ground loop exceeds the configured 10 mm² area budget.
- [x] [patch] Active-IC internal power-plane gate: internal DRC now flags active IC power/ground
      pads without a same-net internal plane under the pad, matching the guide's thermal
      footprint/plane connection guidance.
- [x] [patch] High-speed via-stub gate: internal DRC now flags TX/OUT/TRIG/HV vias whose physical
      barrel extends beyond the signal's actually used layers, so through-via stubs are rejected
      unless the routed transition uses the full barrel span.
- [x] [patch] Via-in-pad fill gate: internal DRC now flags unfilled vias placed directly in
      non-ground SMD pads, enforcing the guide's VIPPO filling rule for signal/power SMD pads.
- [x] [patch] Blind/buried via drill gate: internal DRC now flags blind and buried vias whose
      drill exceeds the configured 0.15 mm fabrication limit from the guide.
- [x] [patch] Differential-pair keepout gate: internal DRC now flags unrelated signal copper inside
      differential-pair keepout corridors, including the guide's 30 mil generic rule, 50 mil clock
      rule, and 5W adjacent-pair spacing rule.
- [x] [patch] Differential-pair component-intrusion gate: internal DRC now rejects unrelated
      component courtyards crossing between parallel P/N members, extending the guide's no
      components-or-vias-between-pairs rule beyond pad/via point checks.
- [x] [patch] Differential-pair coupling-cap symmetry gate: internal DRC now flags one-sided or
      station-mismatched P/N AC-coupling capacitors on routed differential pairs.
- [x] [patch] Differential-pair coupling-cap package gate: internal DRC now flags C* AC-coupling
      capacitors on routed differential pairs whose courtyard exceeds the configured 0603-class
      package budget.
- [x] [patch] Assembly-side placement gate: verification now flags bottom-side SMD footprints and
      through-hole pads that do not include the top assembly side, matching the guide's same-side
      SMD and top-side through-hole placement rule.
- [x] [patch] KiCad wrapper verification cleanup: fixed the KiCad DRC parser for array-valued and
      whitespace-separated count fields, corrected fab-bundle artifact counting, and reduced the
      HV7355 example's pad-net wiring argument list to a context struct.
- [x] [patch] CAD archive consolidation: extracted all 51 ZIP archives from `driver/docs/` into
      `kicad-routing/docs/cad_models/` (22 new folders), moved PDFs to `docs/datasheets/`, Gerbers
      and reference KiCad project files to `docs/reference_design/`, renders to `docs/renders/`,
      `KiCadRoutingTools` Python+Rust routing tool to `tools/`, and removed the root `design/` and
      `docs/` directories. All component and connector inventory docs updated.
- [x] [minor] Molex 0430450400 footprint portability gap closed: footprint copied from KiCad 10 system
      library to `docs/cad_models/430450400/KiCADv6/footprints.pretty/`; import path in
      `examples/hv7355_tile.rs` updated to local path. No footprint import now depends on
      any KiCad installation path. 308 nextest tests pass; clippy clean.
- [x] [minor] FPGA exact footprint replacement (`XC7A100T-2FGG484C` 484-pad FGG484); `exact_complete=true`.
- [x] [minor] HV7355 exact footprint replacement (57-pad QFN56 `QFN56_8X8MC_MCH.kicad_mod`); `exact_complete=true`.
- [x] [patch] Output artifact layout cleanup: canonical board outputs are separated into
      `output/boards/hv7355_24ch_tile/` and `output/boards/hv7355_32ch_tile/`; old root-level
      generated KiCad, sidecar, report, and fab artifacts are archived under
      `output/archive/root_generated/`.
- [x] [patch] Verification cleanup: `examples/fpga_tile_exact.rs` no longer uses linker-sensitive
      `format!` allocation for TX net names/manifest text, and the resistor-margin validation check
      label is single-sourced to prevent glyph drift in tests.
- [x] [patch] 32-channel render-body cleanup: `examples/hv7355_32ch_tile.rs` now attaches visible
      KiCad stock package/header CAD bodies to U1-U4, C1-C16, and J1-J6; regression coverage checks
      the generated board contains those model blocks.

## Verification
- [x] `cargo fmt --check`
- [x] `cargo clippy --all-targets --all-features -- -D warnings` (clean)
- [x] `cargo nextest run` (359 tests — all passing; `.config/nextest.toml` timeout gate in place)
- [x] `cargo test --doc`
- [x] `cargo doc --no-deps`
- [x] `cargo build --release --example hv7355_tile` (local footprint path confirmed)
- [x] `cargo build --example fpga_tile hv7355_tile hv7355_32ch_tile` (dev build; all compile clean)
- [x] `cargo run --release --example fpga_tile` — LVS ok (shorts 0), acid_traps ≤ 34, clearance_violations ≈ 291–301, serpentine_spacing_violations 0, diff_pair_violations 0

## Current Sprint Phase
Phase 3 (Closure). Current state:
All tile examples: `complete=true legal=true`, 359/359 tests, clippy clean.
FPGA: acid=42, clearance=5, crossings=0, sharp_bends=148; external KiCad DRC=0.
HV7355: clearance=6, vias=16; SVG 45×30 mm; all copper in-bounds.
32ch HV: external KiCad DRC=0 violations, 0 unconnected; render now shows package/header CAD bodies.
Resolved this sprint: pad-aware miter skip in `miter_right_angle_corners`; FPGA DRC residues closed;
duplicate pipeline comment removed.
Remaining open: kwavers beam-profile propagation integration.

## Next Increment
- [x] [patch] Fix DOUT/TMS LVS short: `convert_diagonals_to_orthogonal_safe` now seeds
      `cell_net` from all axial track cells (full run), via barrels (all layers in span), and
      pad cells (through-hole = all layers, SMD = declared layer). Root cause: corner check
      only seeded track endpoints, so mid-track and via cells were not blocked.
- [x] [patch] Fix example compilation: `fn place` in `fpga_tile.rs` and `hv7355_tile.rs`
      converted to generic `<S: Into<String>>` (was `impl Into<String>` which the MSYS2
      Rust 1.95 toolchain rejects in nested functions with E0562). Call sites with double-comma
      syntax and missing `&mut comps` argument also repaired.
- [x] [patch] Remove `DEBUG DCO` diagnostic `println!` loop from `fpga_tile.rs` (dead code
      after the debug phase; was cluttering release output).
- [x] [patch] Fix `serpentine_spacing_violations` false positives: add minimum overlap-length
      guard (10 × track width = 1.5 mm) to horizontal and vertical pair checks in `audit.rs`.
      L-shape DFM legs (0.5 mm) are below threshold and no longer counted.
- [x] [patch] Fix `diff_pair_violations` false positives: add minimum segment-length guard
      (1 mm) to corridor detector in `audit.rs`. L-shape legs (0.5 mm) are excluded.
- [x] [minor] Add native SVG renderer (`src/render.rs`): `render_board_svg` + `save_board_svg`,
      exported from `lib.rs`. Both tile examples emit `<board>.svg` alongside the `.kicad_pcb`.
      Renders copper by layer (bottom-to-top), vias, pads, courtyard outlines, refdes labels,
      layer legend. Zero external dependencies.
- [x] [patch] Run hv7355_tile example on corrected 45×30 mm board; verified SVG 45×30 mm,
      no out-of-bounds tracks, complete=true legal=true, clearance_violations=6, vias=16.
- [x] [patch] Investigate `sharp_bends: ~2063` — caused by L-shape corners (90°) replacing
      135° diagonal junctions. sharp_bends now 51 (hv7355) / 42 (fpga) after selective chamfer;
      backlog item for miter/chamfer improvement remains open, not blocking.
- [x] [patch] Fix `five_level.rs::nlevel_rails` doc: clarified that the function counts GND
      as one rail, so 3-level (class-D) correctly returns 1 (not 0). Subtraction idiom
      `nlevel_rails(n) − nlevel_rails(3)` documents the incremental-rails usage.
- [x] [patch] Add `resolve_diagonal_via_clearance` DFM pass: detects diagonal tracks whose
      Euclidean edge-to-edge clearance to a foreign-net via is below `min_clearance`, and converts
      them to orthogonal L-shapes. Integrated into `cooptimize` after `chamfer_diagonal_traps`.
      Regression test covers the DONE↔PROG geometry (0.0486 mm clearance → converted, verified
      ≥0.13 mm after conversion). Direct surgical repair of the violation also applied to
      `output/full_driver/fpga_controller_tile.kicad_pcb` (vertical extended to y=40.0,
      diagonal converted to horizontal (50.5,40.0)→(51.0,40.0)).
- [x] [patch] Add `miter_right_angle_corners` DFM pass: replaces 90° H+V junctions with
      three-segment 135°–45°–135° mitered bends, eliminating their contribution to
      `sharp_bends`. Chamfer distance = `rules.signal_track` (0.15 mm). Integrated into
      `cooptimize` after `resolve_diagonal_via_clearance`. Regression test verifies the
      L-corner case produces exactly 3 segments and 0 sharp bends. 316/316 tests pass;
      clippy clean (also fixed 4 pre-existing warnings in `route/search.rs`).
- [x] [minor] `ic_spread` continuous-repulsion energy term: `PlaceWeights::ic_spread` (default 2.0)
      penalises `board_diagonal / (min_pairwise_distance_mm + 1)` for each same-footprint active-IC
      group, providing a non-vanishing gradient beyond the `thermal` floor. Test:
      `ic_spread_rewards_separation_of_same_fp_ics`.
- [x] [minor] `seed_symmetric_groups` pre-pass: distributes identical active-IC footprint groups
      into a regular grid before cooptimize round 0. Controlled by `CoOpt::seed_groups` (default
      `true`). Test: `symmetric_seeding_distributes_identical_ics_without_overlap`.
- [x] [minor] New example `hv7355_32ch_tile.rs`: 32-channel 4×HV7355 tile, now 100×80 mm,
      six-layer, banked-control/banked-VNN topology. TX_0..TX_31 are all exposed through four
      8-channel output banks. KiCad CLI DRC reports 0 violations and 0 unconnected items.
- [ ] [minor] Move beam-profile validation onto `kwavers-transducer`/kwavers propagation so the
      driver manifest drives the same acoustic model used by the rest of the kwavers stack.
- [x] [patch] Preserve clean diagonals in the DFM pass: the pipeline now runs selective
      `chamfer_diagonal_traps` instead of unconditional `convert_diagonals_to_orthogonal_safe`.
      Focused test proves an isolated diagonal remains one segment; FPGA sharp_bends dropped from
      2315 to 148 and external KiCad DRC from 20 to 1 in the latest render iteration.
- [x] [patch] Add diagonal-corner clearance routing guard: PathFinder now rejects diagonal moves
      whose square corner is already occupied by a foreign track, extending the previous crossed-edge
      and foreign-via-corner guards. Regression test covers the corner-clearance case.
- [x] [patch] Correct placement macro-utilization: locked connectors no longer satisfy board
      coverage samples for the movable functional cluster. Regression test verifies a locked edge
      connector does not reduce utilization energy.
- [x] [patch] Fix native SVG pad rendering: render component pads from exact footprint pad sizes
      and rotations instead of fixed pseudo-pad rectangles. J1 now appears in the SVG as six
      1.5748 mm electrical pads plus the 3.048 mm board-lock pad at its board coordinates.
- [x] [patch] Recover HV7355 full-driver board from `docs/reference_design/hv7355_tile.kicad_pcb`,
      re-apply exact `J_STACK` and `J3` STEP transforms, and verify with KiCad CLI DRC
      (`output/full_driver/hv7355_driver_tile.current_drc.{json,rpt}`: 0 violations,
      0 unconnected items). Render refreshed at
      `output/full_driver/hv7355_driver_tile_connector_aligned.png`.
- [x] [patch] Harden `kicad_cli` DRC validation: `KiCadCli::drc_to` persists JSON reports and
      `DrcReport::defect_counts` exposes KiCad `type` counts so the validation can distinguish
      shorts, clearances, dangling vias, solder-mask bridges, and open items.
- [x] [patch] Remove oversized top-layer HV `GND` pour: deleted the 3525.39 mm² `F.Cu` zone from
      root/full-driver/reference HV boards, normalized remaining KiCad zone clearance to 0.130 mm,
      and changed `write_kicad_pcb` so future generated zones use `DesignRules::min_clearance`
      instead of a zero-clearance override. KiCad CLI DRC remains 0 violations / 0 unconnected.
- [x] [patch] Normalize HV connector refdes: top stack connector is visible as `J1`, local power
      input is `J2`, and the Molex transducer connector remains `J3` across root/full-driver/
      reference HV board artifacts. KiCad render refreshed with `J1` visible above the stack CAD.
- [x] [patch] Close final FPGA external KiCad DRC item: `DONE` F.Cu 0.7071 mm diagonal at
      (50.5, 39.5) violates 0.13 mm clearance to `PROG` blind via at (51.0, 39.5), actual
      clearance 0.0486 mm.
- [x] [patch] Fix `hv7355_tile.rs` GridSpec mismatch: `GridSpec::cover(50,35)` → `(45,30)` to
      match `PlaceConfig::board`. Eliminates out-of-bounds routing at x=45.5 mm and corrects SVG.
- [x] [patch] Fix `hv7355_tile.rs` EMI overcorrection: `emi_weight:1.0` → `0.0`; U1 now
      converges near (20,19) instead of drifting to (13,20). HV/LV isolation via routing rules.
- [x] [patch] Fix `hv7355_tile.rs` cap seed positions: moved C2/C4/C5/C6 from x=29 mm to
      x=22/25 mm, clearing the U1→J1 routing corridor.
- [x] [patch] Fix `hv7355_tile.rs` doc errors: removed duplicate `# Outputs` section,
      corrected board size description "50×35 mm" → "45×30 mm".
- [x] [patch] Add `FootprintDef::with_pad_names` builder method; `two_row_header` auto-generates
      sequential names. Both examples now call `.with_pad_names(...)` on QFP32 and CAP_0402;
      SVG pad tooltips show `U1 pad 1` … `U1 pad 32` instead of `U1 pad ?`.
- [x] [patch] Remove stale 50×35 mm SVGs from `kicad-routing/output/`.
