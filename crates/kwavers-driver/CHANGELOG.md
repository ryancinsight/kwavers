# Changelog

## 0.1.0 - Unreleased

### Added
- **Phase 6 workspace move + kwavers-transducer wiring**: crate moved from
  `leoneuro/driver/kicad-routing/` to `crates/kwavers-driver/` (parent kwavers workspace). Standalone
  `[workspace]` table removed; parent `Cargo.toml` members updated; `kwavers-transducer = { path =
  "../kwavers-transducer", optional = true }` dep added; `kwavers = ["dep:kwavers-transducer"]`
  feature wired. `KwaversSim::simulate` filled — calls `kwavers_transducer::design_array` to get the
  exact element geometry from step parameters (1-D linear, `aperture_x = 0`, `ColumnsAsChannels`),
  then computes pressure/MI/ISPPA from the realized `design.n_channels` and `design.aperture_y_m()`.
  More accurate than `InCrateAcousticSim` for arrays where pitch quantization shifts the channel count
  away from the nominal `step.lanes` value. `grating_lobe_free` now sourced from transducer geometry
  (`ArrayDesign.grating_lobe_free`) rather than the driver-side `max_grating_free_steer_deg`
  approximation; `in_far_field` uses the realized aperture after quantization. Standalone `[profile.*]`
  sections removed (workspace member profiles are no-ops; workspace root governs). `Cargo.toml`
  `0.3.12` → `0.3.13`.
- **Phase 5 experiment framework** (4 Phase-0 placeholder files filled + 4 new files + `mod.rs` + `kwavers = []` feature marker): `stimulus.rs` (`Stimulus` DIP trait + `DefaultStimulus` manifest wrapper), `acoustic.rs` (`AcousticSimulator` DIP trait + `PressureMap` output + `InCrateAcousticSim` real in-crate physics impl using `crate::physics::acoustic` — same computation as `validate_against_budget` but operating on the already-computed `KwaversBeamStep + EnergyBudgetReport` pair with no manifest required at simulation time + `#[cfg(feature="kwavers")] KwaversSim` DipSeam stub for Phase 6), `thermal.rs` (`ThermalState` + `propagate_thermal` 0-D steady-state θ_jc model), `dispatch.rs` (`LaneBinding` + `TileDispatch` equal-partition lane→tile lookup table, 96ch/4-tile → 24ch/tile), `metrics.rs` (`ExperimentMetrics` acoustic+thermal aggregate + `build_beam_report` 4-check kwavers-beam PhysicsReport assembler sourced from simulated scalars — not the analytical pre-step estimate — so future `KwaversSim` scalars propagate correctly into the check), `recorder.rs` (`ExperimentRecord` deterministic output bundle + `artifact_key` frequency×lanes×focal deterministic string key), `runner.rs` (`run_experiment<S: AcousticSimulator>` DIP-injected orchestrator: step→simulate→thermal→report + `ExperimentReport`). 18 new value-semantic tests covering positive/negative/boundary cases across all 7 sub-modules and the full `run_experiment` pipeline. `Cargo.toml` `0.3.11` → `0.3.12`. Full suite **433/433 green**; experiment slice clippy-clean; zero new warnings from the experiment subtree.
- **Phase 4m pipeline slice migration** (flat `src/pipeline.rs`, 1827 LOC — the place↔route co-optimization orchestrator, carved into a 6-file `src/pipeline/` subtree split by **role**): `config.rs` (the `CoOpt` tunables + Default + the per-role dissipation model), `result.rs` (`CoOptResult` + its impl), `place_board.rs` (`place_to_board` + `RoutingInputs` + the placement-stage keepout/repulsion helpers the loop drives — `block_mechanical`, `block_component_bodies`, `apply_emi_pair_repulsion`, `clamp_component_inside`, `seed_symmetric_groups`, `FINE_PITCH_ESCAPE`), `cooptimize.rs` (the `cooptimize` loop + the `cooptimize_min_layers`/`cooptimize_min_area` variants + the loop's local score/clearance helpers), `tests.rs` (verbatim), and the `mod.rs` facade. Unlike `dfm`'s independent passes, this orchestrator is genuinely coupled, so the carve threaded the call graph through `pub(super)` seams: `cooptimize` drives four `place_board` helpers; the `CoOptResult` impl in `result` calls `component_clearance_clean` from `cooptimize` (a benign mutual module reference); and two test-exercised helpers (`role_dissipation_w`, `grid_occupancy_shorts`) are `pub(super)`. `crate::pipeline::*` stays **byte-identical** — lib.rs's 7-symbol crate-root re-export (`cooptimize`/`cooptimize_min_area`/`cooptimize_min_layers`/`place_to_board`/`CoOpt`/`CoOptResult`/`RoutingInputs`) and the leoneuro example's `use kwavers_driver::pipeline::CoOpt` resolve through the facade unchanged. Carved mechanically (block-extraction script; per-file imports pruned; the test module re-acquired the board/geom/route imports the flat module had supplied via `super::*`). All source files ≤ 455 LOC. `Cargo.toml` `0.3.10` → `0.3.11`. `cargo nextest run --lib pipeline::` 10/10; full suite **415/415 green**; pipeline fmt + clippy clean.
- **Phase 4l dfm slice migration** (flat `src/dfm.rs`, 2430 LOC — the largest god-file, carved into a 7-file `src/dfm/` subtree split by **pass role**): `copper.rs` (copper-area passes — `widen_for_ampacity`, `quietest_layer`, `ground_pour`), `vias.rs` (`dedup_vias`, `teardrops`, `plane_distribute_net`), `tracks.rs` (track-geometry passes — `merge_collinear`, `pad_entry_stubs`, `split_track_body_junctions`, `trim_dangling_stubs`, `remove_orphan_copper`, with the file-local `point_on_segment_interior`/`track_order_key` helpers), `diagonal.rs` (diagonal removal/repair — `convert_diagonals_to_orthogonal`/`_safe`, `chamfer_diagonal_traps`, `resolve_diagonal_via_clearance`), `miter.rs` (`miter_right_angle_corners` — the 90°→135° chamfer-insertion pass, split out as the opposite concern from the diagonal-removal passes so both files stay under the 500-line target), `tests.rs` (verbatim), and the `mod.rs` facade. Each DFM pass is a self-contained board transformation with no production cross-call between passes, and all three private helpers are used only within their own file (none needed `pub(super)`). `crate::dfm::*` stays **byte-identical** — lib.rs's 8-symbol crate-root re-export and `pipeline.rs`'s eight `crate::dfm::` pass calls (`merge_collinear`/`dedup_vias`/`chamfer_diagonal_traps`/`miter_right_angle_corners`/`pad_entry_stubs`/`remove_orphan_copper`/`resolve_diagonal_via_clearance`/`split_track_body_junctions`) all resolve through the facade with zero rewrites. The carve was mechanical (each pass moved verbatim by a block-extraction script; per-file `use` headers pruned to exactly what each file references). All source files ≤ 465 LOC (the 2430-line god-file is gone). `Cargo.toml` `0.3.9` → `0.3.10`. `cargo nextest run --lib dfm::` 21/21; full suite **415/415 green**; dfm fmt + clippy + doc clean. (This was the peer's DRC-closure lane; carved during a confirmed lull after they finished their `audit` carve.)
- **Phase 4k units slice migration** (flat `src/units.rs`, 692 LOC + 12 tests, carved into a 7-file `src/units/` subtree split by **role**): `length.rs` (the `Nm` `#[repr(transparent)]` integer-nanometre board-space newtype + its arithmetic, independent of the soft-unit machinery), `quantity.rs` (the `Unit` kind-marker trait, the `#[repr(transparent)]` `Float<U>` wrapper, and tolerance-aware `approx_eq`), `kinds.rs` (the 10 ZST unit-kind markers + the 10 concrete `Float<Kind>` aliases — `Hz`/`Ohm`/`Watt`/`Kelvin`/`Celsius`/`Volt`/`Amp`/`Henry`/`Farad`/`Coulomb`), `factories.rs` (SI-prefix constructors kHz/MHz/GHz/mΩ/kΩ/mW/µW/nF/pF/µF/nH/µH/nC/µC + the K↔°C↔°F temperature bridge), `arithmetic.rs` (the `impl_same_unit_arith` + `impl_scalar_mul_div` macros with their invocations + the hand-written cross-unit dimensional products — Ohm's law `Ω·A=V`, power `V·A=W`, charge `F·V=C`, and inverses), `tests.rs` (12 verbatim), and the `mod.rs` facade. This is the **first carve to use the extract-tests-first order** (write `tests.rs` before any source sub-file), the process fix for the Phase 4j data-loss incident. `crate::units::*` stays **byte-identical** — lib.rs's 11-type `pub use units::{…}` is unchanged and `crate::geom::Nm` (`pub use crate::units::Nm`) resolves through the facade, so the ≈230 `Nm` call-sites compile untouched; the macro/generic coupling is preserved by co-locating each `macro_rules!` with its invocations in `arithmetic.rs`. The unit system was already zero-cost (`#[repr(transparent)]` throughout), so no optimization was forced (`const fn` constructors are blocked by `f64::round` on `Nm` and would be speculative YAGNI elsewhere). All 7 files ≤ 157 LOC. `Cargo.toml` `0.3.8` → `0.3.9`. `cargo nextest run --lib units::` 12/12; full suite **415/415 green**; slice fmt + clippy + doc clean.
- **Phase 4j validate slice migration** (flat `src/validate.rs`, 1272 LOC, carved into a 5-file `src/validate/` subtree split by **role** — the driver→transducer validation seam): `check.rs` (the `Check` upper/lower primitive with signed margin + `PhysicsReport` `all_pass` gate), `board_checks.rs` (board-geometry checks — HV creepage `min_hv_spacing_mm`, ampacity `worst_ampacity_margin_mm`, the shared `core_checks`, `ViaCensus`/`via_census`, `microvia_aspect_check`, `net_length_mm`/`group_skew_mm`), `kwavers_beam.rs` (the **driver→transducer seam**: the typed `KwaversBeamStep` the `kwavers-transducer` simulator consumes, `manifest_to_kwavers_beam_step`, the `KwaversBeamValidation` prediction set, and `validate_against_budget`), `tests.rs`, and the `mod.rs` facade. `crate::validate::*` stays **byte-identical** (top-level `pub mod` → directory; external references are doc-links only in `audit::critic`/`audit::fault_report`/`manifest::energy_budget`, resolved unchanged); SSOT safety bounds + `Check` names stay sourced from `crate::ssot`. The perf surface was reviewed and deliberately left as-is (the 4-element `resistor_margin_w` clones are intentional duplicate fields for consumer ergonomics; `min_hv_spacing_mm`'s O(n²) is a one-shot sign-off proxy the audit already backstops — optimizing either would be premature).
  - **⚠ Test reconstruction (data loss):** the original 662-line `mod tests` block was **lost** — the flat `validate.rs` was removed (a tool auto-resolving the transient `validate.rs`↔`validate/mod.rs` module ambiguity in the concurrent-edit environment) *before* its test body could be extracted into the slice, and leoneuro is git-ignored so there is no VCS copy. `tests.rs` is a **from-contract reconstruction**: 12 genuine value-semantic tests, each assertion derived analytically from the function contracts (Check directions/margins, PhysicsReport gate, via census + VIPPO, micro-via aspect ratio, net length / group skew, HV creepage spacing, ampacity headroom, and the full kwavers-beam pass + reject paths). All 12 pass — coverage is restored, but it should be cross-checked against the original intent if a copy resurfaces.
  - `Cargo.toml` `0.3.7` → `0.3.8`. Full suite **415/415 green**; `validate/` fmt-clean + zero new clippy/doc warnings.
- **Phase 4i component_db slice migration + static-table optimization** (flat `src/component_db.rs`, 634 LOC + 7 tests, carved into a 6-file `src/component_db/` subtree split by **role**): `pulser_ic.rs` (the `PulserIc` datasheet record, `StockStatus`, and the per-IC property accessors — pin capacitance, supply/signal pin counts, decoupling, package/board area), `catalog.rs` (the pulser-IC table behind `available_pulsers`), `dcdc.rs` (`DcDcModule` HV-bias-rail record + its table + `available_dcdc_modules`), `compare.rs` (`PulserComparison`, `compare_pulsers`, `recommend_96ch_architecture`), `tests.rs` (7 verbatim), and the `mod.rs` facade.
  - **Memory optimization (Changed, [minor]):** `PulserIc` and `DcDcModule` are composed entirely of `&'static str` / scalar / `&'static [f64]` fields (no `String`), so both datasheet tables — previously `vec![…]` rebuilt and **heap-allocated on every call** — are now compile-time **`static` slices**. `available_pulsers` / `available_dcdc_modules` return `&'static [PulserIc]` / `&'static [DcDcModule]` instead of `Vec<_>`: **zero per-call allocation**, the const/zero-cost form for analytically-fixed datasheet data. Caller impact: `compare_pulsers` consumes via `.iter()` (unchanged); the one test `for p in &pulsers` became `for p in pulsers`; `available_dcdc_modules` has zero consumers. The leaf module has no external callers.
  - `Cargo.toml` `0.3.6` → `0.3.7`. Verified green: `cargo nextest run --lib component_db::` 7/7 pass; full suite 420/421 (the single failure is the peer's `audit::tests::dirty_fields_…` meta-test, mid-restructure during their concurrent `audit.rs` carve — not this slice). The `static` tables const-construct; slice fmt-clean + zero new clippy/doc warnings.
- **Phase 4h optim slice migration + dead-allocation cleanup** (flat `src/optim.rs`, 528 LOC + 5 tests, carved into a 6-file `src/optim/` subtree split by **role**): `context.rs` (the input contexts — `ArrayGeometry` + `new`, `ThermalContext`, `PdnConfig`, `EmiContext`, with their Defaults), `report.rs` (the `DesignReport` output aggregate), `evaluate.rs` (`evaluate_design_point`, the one-shot physics orchestrator across driver/thermal/acoustic/EMI/PDN), `kernels.rs` (standalone design-limit helpers — `max_safe_duty_thermal`, `ringing_exceeds_breakdown`, `hot_track_resistance`), `tests.rs` (5 verbatim), and the `mod.rs` facade.
  - **Cleanup ([patch], behaviour-identical):** removed a dead N-element `Vec` heap allocation in `evaluate_design_point` — it called `focused_delay_profile_s(...)` and immediately discarded the result via `let _ = delays;`. The comment claimed it "validated the delay profile is well-formed" but asserted nothing, so it was pure dead computation. No `DesignReport` field depended on it; the value-semantic tests (`design_report_is_fully_populated`, `thermal_derating_lowers_efficiency`) pass unchanged. One fewer heap allocation per design-point evaluation; its now-unused import dropped too.
  - `crate::optim::*` stays byte-identical and `optim` is a leaf consumer (no external callers), so zero caller churn. All 6 files ≤ 158 LOC. `Cargo.toml` `0.3.5` → `0.3.6`. `cargo nextest run --lib` 421/421 green, slice fmt-clean + zero new clippy/doc warnings.
- **Phase 4g driver slice migration** (flat `src/driver.rs`, 706 LOC + 13 tests, carved into a 7-file `src/driver/` subtree split by **physics role**): `pulser.rs` (the core class-D HV-pulser loss model — `PulserOp` operating point → `PulserDissipation` breakdown via `pulser_dissipation`, splitting the dynamic ½CV² loss between device and series damping resistor), `reactive.rs` (matching-network / reactive-drive / ringdown / switching-node math — `tuning_inductor_h`, `load_quality_factor`, `damping_resistor_ohm`, `ringdown_cycles`, `reactive_drive_power_w`, `driver_efficiency`, `switching_node_ringing_v`), `rating.rs` (thermal-duty + package-power-rating limits — `max_safe_duty`, `chip_power_rating_w`, `power_rating_check`, `thermally_derated_efficiency`, `PowerOverload`, `PowerRatingReport`), `sweep.rs` (the frequency-sweep driver-loss optimiser — `FreqSweepPoint`, `sweep_driver_loss`, `find_best_freq`), `compare.rs` (cross-IC comparison — `ComponentComparison`, `compare_driver_ics_at`), `tests.rs` (13 verbatim), and the `mod.rs` facade. The `DEFAULT_THETA_JC_K_PER_W` θ_jc constant is kept `pub(super)` in `mod.rs` as the single source of truth shared by the `sweep` + `compare` sub-files (off the `crate::driver` surface). Because `driver` is already a top-level `pub mod`, the directory carve keeps `crate::driver::*` **byte-identical** (lib.rs's 20-symbol `pub use driver::{…}` + the module-path `crate::driver::PowerOverload` unchanged; the four external callers `component_db`/`manifest::energy_budget`/`optim`/`pipeline` that use `crate::driver::{PulserOp, pulser_dissipation, driver_efficiency}` resolve via the facade — zero rewrites). Two stale doc references corrected in-move (`crate::thermal::junction_temperature_k` → `crate::physics::thermal::…` post-Phase-3b; a malformed `use///` doc line in `switching_node_ringing_v`). All 7 files ≤ 240 LOC. `Cargo.toml` `0.3.4` → `0.3.5`. `cargo nextest run --lib` 421/421 green, slice fmt-clean + zero new clippy/doc warnings.
- **Phase 4f manifest slice migration** (flat `src/manifest.rs`, 1422 LOC + 21 tests, carved into a 7-file `src/manifest/` subtree split by **schema role**): `stimulation.rs` (the acoustic-protocol schema — `StimulationProgram` article-class preset + `TileStimulationProfile` per-tile PRF/SHIFT/PHASE/RAMP, with their protocol-load proxies), `resistor.rs` (`ResistorPackage` — the IPC-7351-rated SMD damping-resistor footprint enum + signed rate-to-margin converter), `driver_manifest.rs` (`DriverManifest` schema + the deterministic key-value text round-trip `to_text`/`from_text`/`read` covering v1/v2 and single-stim/per-tile forms with mixed-keyset and gappy-tile parser guards + the protocol-load accessors), `energy_budget.rs` (`EnergyBudgetInputs`/`EnergyBudgetReport` + a **second `impl DriverManifest`** block hosting `validate_v2_energy_budget`, the routed-board ampacity/dissipation gate), `extract.rs` (`hv_manifest_from_board` board→manifest builder), `tests.rs` (21 verbatim), and the `mod.rs` facade. The 387-line `DriverManifest` impl is split across two files via two `impl` blocks (serialization+accessors vs energy budget; both bind the same type in the same module so cross-file `self.method()` calls resolve unchanged). Because `manifest` is already a top-level `pub mod`, the directory carve keeps `crate::manifest::*` **byte-identical** (lib.rs `pub use manifest::{hv_manifest_from_board, DriverManifest, EnergyBudgetInputs, EnergyBudgetReport, ResistorPackage, TileStimulationProfile}` unchanged, and the module-path `crate::manifest::StimulationProgram` preserved; zero outward caller rewrites). Schema-key/lane constants stay sourced from `crate::ssot`. Source files all ≤ 275 LOC. `Cargo.toml` `0.3.3` → `0.3.4`. `cargo nextest run --lib` 421/421 green, slice fmt-clean + zero new clippy/doc warnings.
- **Phase 4a kicad_cli slice migration** (flat `src/kicad_cli.rs`, 748 LOC + 6 tests, carved into a 5-file `src/kicad_cli/` subtree split by **role**): `cli.rs` (the `KiCadCli` process wrapper + `DrcOptions` — locate/spawn the external binary, drive `pcb drc`/`pcb render`/Gerber-drill-pos-BOM export; `pub(super)` `drc_args` + `locate_on_path`), `drc.rs` (the `DrcReport`/`DrcDefectCount` model + the version-tolerant permissive `parse_drc_json` that hand-scans KiCad 7/8/9/10 JSON variants, `pub(super)`), `fab.rs` (the `FabBundle` artifact set + `summary_lines` + private `count_dir_files`), `tests.rs` (6 verbatim), and the `mod.rs` facade. Because `kicad_cli` is already a top-level `pub mod`, the directory carve keeps the public path `crate::kicad_cli::*` **byte-identical** (lib.rs `pub use kicad_cli::{DrcOptions, DrcReport, FabBundle, KiCadCli}` unchanged; zero outward caller rewrites — no caller referenced a private item). All 5 files ≤ 290 LOC (the 748-line god-file is gone). `Cargo.toml` `0.3.2` → `0.3.3`. `cargo nextest run --lib` 421/421 green, slice fmt-clean + zero new clippy/doc warnings.
- **Phase 4e stack slice migration** (flat `src/stack.rs`, 893 LOC + 11 tests, carved into an 8-file `src/stack/` subtree split by **role**, not file-size symmetry): `plan.rs` (single-board thermal/height/capacity optimiser — `StackConstraints`, `StackPlan`, `board_rise_k`, `optimize_stack`), `role.rs` (`StackBoardRole` controller/driver enum), `manifest.rs` (`StackBoardManifest` text round-trip + `stack_board_manifest_from_board` extraction), `compatibility.rs` (`StackCompatibility` + `verify_stack_pair` connector-mating check), `shield.rs` (full controller-plus-HV shield stack — `ShieldStackPlan`, `ShieldStackAssembly`, `assemble_shield_stack`, `optimize_shield_stack`), `util.rs` (`pub(super)` geometry/canonicalisation helpers `canonical_stack_net`/`board_{width,height}_mm`/`check_close` shared by `manifest` + `compatibility`), `tests.rs` (11 verbatim), and the `mod.rs` facade. Because `stack` is already a top-level `pub mod`, the directory carve keeps the public path `crate::stack::*` **byte-identical** (zero outward caller rewrites — unlike the physics slices that moved `crate::<flat>` → `crate::physics::<slice>`); the two internal doc-links (`src/error.rs`, `src/error/validate.rs`) resolve unchanged through the facade. All 8 files ≤ 268 LOC (the 893-line god-file is gone). `Cargo.toml` `0.3.1` → `0.3.2`. `cargo nextest run --lib` 421/421 green (zero net test-count delta), zero new clippy/doc warnings from the slice.
- **`.bak-phase3b` cleanup**: deleted the 5 untracked pre-Phase-3b snapshots (`src/driver.rs.bak-phase3b`, `src/lib.rs.bak-phase3b`, `src/optim.rs.bak-phase3b`, `src/pdn.rs.bak-phase3b`, `src/pipeline.rs.bak-phase3b`) so future archaeologists see the `.bak-phase3b` era is closed rather than puzzle over why pre-3b imports in `git blame` don't reconcile to the current tree.
- **Phase 3f si slice migration** (carved into 5-file `src/physics/si/` subtree: 8 existing fns carried across + 3 NEW APIs — `impedance_target` (signal-line branching-match target), `return_loss_db` (single-call RL for caller-loop iteration over freq bands), and `channel_operating_margin_db` (IEEE amplitude-ratio COM) — added to fill out the frequency-band-aware impedance-budget surface). Flat `src/si.rs` retired; crate-root re-export at `crate::physics::si::{...}`. cargo build/test/doc clean.
- **Phase 3g acoustic slice migration** (carved into 8-file `src/physics/acoustic/` subtree: `mod.rs` + `wavelength.rs` + `grating.rs` + `focus.rs` + `element.rs` + `safety.rs` + `nonlinear.rs` + `tests.rs`). 18 prior pub fns carried across split by **physical role**, not file-size symmetry: `wavelength.rs` (λ + BVD series-branch motional `f_s = 1/(2π√(L_s·C_s))` + textbook BVD anti-resonance `f_p = (1/2π)·√((C_s + C_0)/(L_s·C_s·C_0))` per Kino §3.4 / IEEE Std 176 — couples motional branch with static dielectric `C_0`); `grating.rs` (steering limits + ULA array factor); `focus.rs` (relative delays + nearest-step quantisation + worst-case quantisation error); `element.rs` (per-element Fresnel-range + directivity + f-number + span→pitch + coherent focal gain); `safety.rs` (FDA MI + tissue derating + continuous-RMS intensity + ISPPA + pulse-echo round-trip attenuation); `nonlinear.rs` (Earnshaw normalised σ). 3 NEW APIs added: `bvd_anti_resonance_hz(ls_h, cs_f, c0_f)` (textbook BVD anti-resonance), `isppa_w_per_m2(p_neg, z0, duty)` (FDA Track-3 spatial-peak pulse-average intensity, SSOT-distinct from continuous-RMS `acoustic_intensity_w_per_m2`), `round_trip_attenuation_db(α, f, z)` (pulse-echo two-way loss, SSOT-distinct from one-way `tissue_attenuation_db`). The 13 prior inline tests from `src/acoustic.rs::mod tests` plus the new SSOT-distinction + range tests for the 3 NEW APIs consolidated into the slice-wide `tests.rs` (cargo test --lib reports 20/0 green for `physics::acoustic`). Flat `src/acoustic.rs` retired; crate-root re-export at `crate::physics::acoustic::{...}` is the canonical 21-symbol surface (byte-identical to prior flat exposure minus the swap from `bvd_parallel_resonance_hz` to `bvd_anti_resonance_hz`). `Cargo.toml` version bump `0.3.0` → `0.3.1`. cargo build clean (1 unrelated `Rot` unused-import warning in src/place/footprint.rs), cargo test --lib 421/0 green, cargo doc strict-clean (Bucket 1 = src/physics/acoustic/) → zero errors. Residual cargo doc strict-clean errors fall under Buckets 2-5 (Phase 3a-pre-existing `crate::physics::ampacity::track_resistance` ambiguous-link, Phase 3prior `crate::thermal`/`crate::pdn`/`crate::emi`/`crate::si` top-level retirement, Phase 3prior intra-slice `tests`-as-doc-link, etc.) and are documented in `docs/MIGRATION.md ## Phase 3 follow-ups ## Phase 3 doc-strict clean-up (placeholder)` as **Phase 3h** scope so that's a follow-up rather than blocking Phase 3g closure.
- **Phase 3e pdn remainder slice migration** (carved into 5-file `src/physics/pdn/` subtree: 7 free fns split by physical role into `target_impedance.rs` / `impedance.rs` / `cavity.rs` / `tests.rs`). `Cargo.toml` `0.2.14` → `0.2.15`. Side-effect regression fix in the same turn: `sed -i 's|capacitive_drive_currenta\b|capacitive_drive_current_a|g' src/audit.rs` (`src/audit.rs:179/3042/3093` carried a missing-underscore typo from the Phase 3d emi sed-replace that broke the build at Phase 3e startup).
- **Phase 3d emi slice migration** (carved into 8-file `src/physics/emi/` subtree):
  - `src/emi.rs` (388 LOC + 8 tests) carved into `src/physics/emi/{mod.rs, scene.rs, loop.rs,
    trace_partial.rs, losses.rs, overshoot.rs, radiated.rs, tests.rs}`. `pub mod emi;`
    declared at `src/physics/mod.rs`; flat `pub mod emi;` retired from `src/lib.rs`; flat
    `src/emi.rs` no longer exists.
  - **Cross-slice dependency preserved**: `scene::commutation_loops` walks
    `crate::place::component::Component` + `crate::place::footprint::{FootprintDef, Role}`
    — the Tier-2 dependency has been in place since `place` closed at Phase 2c.
  - **`r#loop` keyword trap**: `loop` is reserved (start of `loop {}`), so the on-disk file stays `loop.rs` but the mod decl MUST be `pub mod r#loop;` and all imports use `super::r#loop::{...}`. The slice facade documents this with an inline module-docstring paragraph (see `src/physics/emi/mod.rs`'s `r#loop` paragraph).
  - **Slice-private internals**: `pub(super) const MU0` (vacuum permeability) shared
    between `loop.rs::loop_inductance_nh` and `trace_partial.rs::trace_partial_inductance_nh`;
    `pub(super) fn polygon_area_mm2` callable from `scene.rs::commutation_loops` (the sole
    consumer). Both are NOT re-exported from the slice facade so `crate::physics::MU0` and
    the raw shoelace helper stay slice-internal.
  - **Explicit named `pub use` re-exports** in `src/physics/emi/mod.rs` (NOT glob
    `pub use scene::*;` like the ampacity slice) — locks the 10-item API surface and
    matches the 3c-dielectric discipline.
  - `src/lib.rs::pub use physics::emi::{capacitive_drive_current_a, commutation_loops,
    gate_drive_power_w, inductive_overshoot_v, loop_inductance_nh, radiated_emi_dbuv_m,
    reverse_recovery_loss_w, switching_loss_w, trace_partial_inductance_nh, CommutationLoop};`
    is the canonical crate-root surface, byte-identical to the prior flat
    `pub use emi::{...}` re-export.
  - **Source-code + doc-link re-routing** at 9 outward-caller sites across 5 files:
    `src/audit.rs` (3 doc-links + 2 source-code refs to `commutation_loops` /
    `capacitive_drive_current_a`), `src/optim.rs` (line 22 `use crate::emi::{...}` source-code
    import + 1 CommutationLoop doc-link), `src/driver.rs` (2 doc-links), `src/place/footprint.rs`
    (1 doc-link in the `capacitance_f` field-doc), `src/rules.rs` (2 doc-links on
    `ic_switching_dv_v` / `ic_switching_risetime_s`). The 5 references in `src/audit.rs`
    landed via `sed -i` rather than `str_replace` because the 317K-char audit.rs exceeds
    `str_replace`'s 100K-char patch-display limit.
  - `src/physics/mod.rs` `# Cut-over status` docstring refreshed from "Three slices" to
    "Four slices" with a Phase 3d paragraph explaining the 8-submodule split + the
    `r#loop` keyword trick + the slice-private internals (`MU0` / `polygon_area_mm2`); the
    "remaining flat modules" list dropped `emi` (now `[pdn, si, acoustic]`).
  - `Cargo.toml` version bump `0.2.13` → `0.2.14`.
- **Phase 3c dielectric slice migration** (per user-task numbering; MIGRATION.md formal table calls this `Phase 3b` — see MIGRATION.md' `## Phase 3b — dielectric slice migration` section for the dual-naming rationale):
  - `src/dielectric.rs` (134 LOC + 4 tests) carved into a 5-file `src/physics/dielectric/`
    subtree (`mod.rs`, `paschen.rs`, `ipc2221_spacing.rs`, `caf.rs`, `tests.rs`). `pub mod
    dielectric;` declared at `src/physics/mod.rs`; flat `pub mod dielectric;` retired from
    `src/lib.rs`. The flat `src/dielectric.rs` no longer exists.
  - Explicit named `pub use` re-exports in `src/physics/dielectric/mod.rs` (NOT glob
    `pub use X::*;` like the ampacity slice) — keeps slice-private air constants (`A_AIR`,
    `B_AIR`, `GAMMA`) from leaking into the slice-level API surface. The 3-line `pub use`
    rationale is documented as a `//` line comment (NOT `//!`) so it doesn't surface as part
    of the published rustdoc API contract.
  - `src/lib.rs::pub use physics::dielectric::{air_breakdown_possible, caf_ttf_relative, ipc2221_min_spacing_mm, paschen_breakdown_v, paschen_min_air};` is the canonical crate-root
    surface, byte-identical to the prior flat `pub use dielectric::{...}` re-export.
  - Single source-code doc-link in `src/rules.rs:370` updated from
    `crate::dielectric::ipc2221_min_spacing_mm` → `crate::physics::dielectric::ipc2221_min_spacing_mm`
    so `cargo doc --no-deps` resolves all intra-doc links cleanly. No other source-code callers
    reference the dielectric module by path (the SSOT link in `rules.rs`'s `CreepageRule`
    docstring is the only fan-out).
  - `src/physics/mod.rs` `# Cut-over status` docstring refreshed from "Two slices" to
    "Three slices" with a Phase 3c paragraph explaining the SSOT role (Paschen air-breakdown
    kinetics + IPC-2221B Table 6-1 B1 external uncoated spacing + Rudra/IPC-TR-476 CAF
    relative time-to-failure).
  - `Cargo.toml` version bump `0.2.12` → `0.2.13`.
- **Phase 3b thermal slice migration** (new 7-file sub-tree):
  - `src/thermal.rs` (468 LOC + 6 tests) carved into a 7-file `src/physics/thermal/` subtree
    (`mod.rs`, `joule_source.rs`, `ir_drop.rs`, `via_conductance.rs`, `electrothermal.rs`,
    `transient.rs`, `tests.rs`). `pub mod thermal;` declared at `src/physics/mod.rs` and the flat
    `pub mod thermal;` in `src/lib.rs` retired. The flat `src/thermal.rs` no longer exists.
  - `IrDrop` struct + `ir_drop(network) -> Option<IrDrop>` solver promoted out of `src/pdn.rs`
    into `src/physics/thermal/ir_drop.rs` so the electro-thermal coupling chain (`ir_drop` →
    `joule_source` → `solve_electrothermal`) sits in one crate plane. `src/pdn.rs` keepsonly the decoupling/resonance/impedance/plane-cavity half: 7 free fns
    (`target_impedance_ohm`, `holdup_capacitance_f`, `plane_resonance_hz`,
    `self_resonant_freq_hz`, `max_decoupling_distance_mm`, `pdn_impedance_at_freq`,
    `anti_resonance_hz`) + 4 decoupling tests.
  - 9 thermal/IR-drop tests consolidated into `src/physics/thermal/tests.rs` (6 thermal + 3
    IR-drop). `src/lib.rs::pub use physics::thermal::{ir_drop, junction_temperature_k,
    solve_board, solve_electrothermal, temperature_derated_resistance, thermal_time_constant_s,
    thermal_via_conductance, transient_rise_k, IrDrop, ThermalField};` is the canonical surface.
  - `crate::physics::mod.rs` docstring refreshed from "Phase 0 placeholder" to a forward-tracking
    "Cut-over status" section listing ampacity (3a) + thermal (3b) as already cut over; the
    remaining slices (dielectric / emi / pdn / si / acoustic) stay on their flat `src/<name>.rs`
    paths until their Phase-3 cuts land.
- **Phase 3a ampacity slice migration** (`src/physics/ampacity/` sub-tree): `src/ampacity.rs`
  (257 LOC + 7 tests) carved into a 7-file subtree. `pub mod ampacity;` declared at
  `src/physics/mod.rs`; flat `src/ampacity.rs` retired. The `track_resistance` function (the
  sole Tier-2 upstream enabler for both `thermal::joule_source` and `pdn::ir_drop`) lives at
  `crate::physics::ampacity::track_resistance`.
- `ic_spread` continuous-repulsion energy term in `PlaceConfig::weights`: penalises
  `board_diagonal / (min_pairwise_distance_mm + 1)` for each group of same-footprint active ICs,
  providing a non-vanishing gradient that spreads identical ICs even after they exceed the
  `thermal_spacing` floor.
- `seed_symmetric_groups` pre-pass in `cooptimize`: before round 0, identical active-IC footprint
  groups are distributed into a regular grid across the usable board area so the first routing pass
  starts from a well-spread initial placement. Controlled by the new `CoOpt::seed_groups` flag
  (default `true`; set `false` in ablation tests that deliberately start from a clustered seed).
- `CoOpt::seed_groups: bool` field (default `true`) to opt out of the symmetric pre-seeding pass.
- New example `examples/hv7355_32ch_tile.rs`: 32-channel HV7355 driver tile on a 100×80 mm
  6-layer board. Four HV7355 IC abstractions (U1–U4, each 8 TX channels = TX_0..TX_31), 16
  decoupling capacitors, one banked control connector, four 8-channel output banks, and one HV
  power connector are routed with banked control/negative-rail nets. KiCad CLI DRC reports
  0 violations and 0 unconnected items on the generated board. The generated KiCad board now
  attaches visible stock package/header CAD bodies to U1-U4, C1-C16, and J1-J6 so the render is not
  copper-only; the footprint topology remains a comparison abstraction, not exact 24-channel shield
  parity.
- DFM post-processing now splits same-net track-body junctions into explicit endpoints after
  right-angle mitering and pad-entry stub insertion, aligning the internal dangling audit with
  KiCad's `track_dangling` warning semantics.
- Generated outputs are now organized by purpose: canonical HV board variants live under
  `output/boards/hv7355_24ch_tile/` and `output/boards/hv7355_32ch_tile/`, while legacy examples,
  standalone manifests, reports, renders, archived fabrication bundles, and old root-generated
  KiCad artifacts are separated into named folders instead of the root of `output/` or the project
  root.
- `examples/fpga_tile_exact.rs` now builds TX net names and manifest/diagnostic text without
  `format!`, avoiding the MSYS2/lld `alloc::fmt::format` link failure during the all-target
  test build while keeping the exact-model example in the normal gate.
- The kwavers resistor-margin validation check label is now a single module constant, so tests and
  report generation cannot drift between `≥` and `>=` spellings.
- Regression tests: `symmetric_seeding_distributes_identical_ics_without_overlap` (pipeline) and
  `ic_spread_rewards_separation_of_same_fp_ics` (place/mod).
- Core nm-domain board, routing, placement, physics, DFM, verification, and KiCad emit modules.
- Stackable FPGA controller and HV7355 driver examples under `output/full_driver`.
- Board-backed driver manifest with transducer TX nets and FPGA programming evidence.
- Deterministic beamforming image renderer for -45, 0, and +45 degree steering from the generated
  driver manifest.
- Focal-arrival pulse-envelope beam rendering with lateral/axial 6 dB width metrics.
- Connector CAD inventory for downloaded Molex Micro-Fit, Samtec TSW, and related mating assets.
- Stack-board manifests, FPGA/HV stack compatibility checking, and shield-stack thermal/height
  planning.
- Complete shield-stack assembly manifest mapping one FPGA/controller board plus four 24-channel HV
  shields to global `TX_0..TX_95`.
- Component CAD inventory and generated `component_accuracy_*.kv` manifests.
- Extracted exact local CAD for FPGA options `XC7A100T-2FGG484C` and `XC7A35T-1FTG256C`.
- Generated article-comparison figures for the stack, board renders, and beamforming outputs.
- Exact Molex `0430452400` HV transducer connector import/routing for `J2`, including plated-pin
  access groups and NPTH board-lock keepouts.
- Exact downloaded Molex/Samtec footprint imports for FPGA power `J1`, FPGA JTAG `J3`, FPGA
  `J_STACK`, and HV `J_STACK`, with pad-name based net mapping.
- Exact Molex `0430450600` PCB-header STEP inventory for FPGA power `J1` at
  `docs/cad_models/430450600_stp/430450600.stp`.
- FPGA controller 3D render now attaches the Molex `0430450600` body to `J1` and a 23 mm
  BGA-484 package body to `U1`.
- KiCad footprint import now preserves 3D model offset/rotation transforms, fixing the Molex
  `0430450600` STEP-to-pad alignment on FPGA `J1`.
- KiCad footprint import now normalizes model offsets after courtyard recentering, applying the same
  origin shift to STEP bodies that it applies to pads. The 32-channel HV7355 comparison tile now
  derives stock two-row pin-header STEP offsets from its centered vertical 2-column pad grids,
  preventing zero-offset or rotated-grid header bodies from drifting off the rendered plated holes.
- Thermal/EMI placement feedback now keeps its own coefficient scale instead of being damped by the
  global congestion-feedback weight; thermal feedback uses the normalized solved temperature field
  rather than a binary hotspot mask.
- KiCad CLI DRC wrapper can now write persistent JSON reports and records per-defect `type` counts
  so opens, shorts, clearances, dangling vias, and solder-mask bridge regressions are visible by
  class.
- KiCad CLI DRC wrapper now supports explicit zone refill/save options for programmatically modified
  boards, and generated PCB saves reject duplicate KiCad object UUIDs before writing.
- KiCad CLI DRC JSON parsing now separates `severity = error` from `severity = warning` entries
  inside KiCad 10 `violations[]`; warning-only reports no longer count as hard DRC failures, while
  `unconnected_items[]` still fails the report.
- FPGA `J3` and FPGA/HV `J_STACK` pin-header STEP transforms are now aligned to the generated pad
  grids and verified with KiCad CLI renders.
- HV `J2` Molex `0430452400` STEP transform now moves up over the generated transducer pin rows.
- Recovered the full-driver HV7355 board from the KiCad-clean reference design and re-applied the
  connector STEP transforms; current KiCad CLI DRC reports 0 violations and 0 unconnected items.
- Removed the oversized top-layer `GND` pour from the HV7355 board render/artifacts and changed
  generated KiCad zone emission to use the active design-rule clearance instead of hardcoded
  zero-clearance zone overrides.
- Normalized HV7355 connector refdes on the board artifacts: `J1` is the visible top stack
  connector, `J2` is the Molex `0430452400` transducer connector with the local
  `docs/cad_models/430452400/430452400.step` body, and `J4` is the remaining local power/input
  connector artifact.
- Extracted the exact Molex `0430450400` / `430450400` board-header CAD from
  `docs/cad_archives/430450400.zip` and attached `docs/cad_models/430450400/430450400.step` to the
  HV `J4` output artifacts.
- Added value-semantic import coverage for the exact Molex `0430450400` KiCad-v6 footprint: four
  electrical pins, one NPTH board-lock keepout, courtyard dimensions, power-pin classification, and
  the embedded STEP model token are checked against the downloaded CAD.
- Replaced the previous temporary `J4` mating-receptacle render body with the exact board-header
  STEP. A direct exact-through-hole footprint swap without rerouting was rejected after KiCad DRC
  reported 1219 violations and 2 unconnected items on the trial board.
- Re-audited exact-J4 trial boards after UUID repair: the unrouted exact footprint candidate reports
  0 hard errors, 3 warnings, and 4 unconnected items; routed margin/local candidates remove opens
  but still produce 15-19 hard KiCad errors. These rejected trial artifacts are not retained as
  current board artifacts; the production HV board is the KiCad-clean exact-J4 output.
- Promoted the exact `MOLEX_430450400` through-hole footprint for HV `J4` into
  `output/full_driver/hv7355_driver_tile.kicad_pcb`, placed it at the right board edge in the
  power-input y-region with the exact `430450400.step` model, rerouted VPP/GND/P3V3/P5V through
  internal-layer breakouts, and verified the promoted board with KiCad CLI DRC at 0 violations and
  0 unconnected items.
- Added production-artifact regression coverage that rejects a return to the previous
  `J_PWR_Molex_0430450400_demo` footprint and asserts the exact `J4` footprint/model/orientation
  tokens are present while rejecting the old lower-quadrant J4 placement.
- Corrected the HV `J4` `430450400.step` local model rotation from `90 deg` to `0 deg`, matching
  the downloaded Molex footprint coordinate system and preventing the rendered connector body from
  drifting off the plated-hole pattern.
- Added `CoOptResult::manufacturing_clean` / `manufacturing_blockers` and updated examples to refuse
  writing KiCad outputs when routing is incomplete, illegal, hard-internal-DRC dirty, LVS dirty, or
  component-courtyard dirty.
- Removed stale generated KiCad boards and rejected `_j4_*` trial artifacts that carried current
  DRC errors; the remaining non-third-party KiCad boards sweep clean with 0 errors and
  0 unconnected items.
- Restored all example targets after the placement-isolation API change by exporting `Axis` and
  `IsolationDomain` through the crate root and updating stale example `Component`/`PlaceConfig`
  literals to use the defaulted isolation fields.
- Restored all-target clippy on `examples/fpga_tile_exact.rs` by keeping the used `VCC_1V8` net
  parameter live and removing stale unused JTAG/SPI ball-group constants.
- Radius-based EMI hotspot feedback and an EMI objective in the adversarial co-optimizer.
- Footprint rotation policies for placement optimization.
- Internal high-speed reference-plane margin checking.
- Internal high-speed adjacent-reference-plane presence checking.
- Internal high-speed inner-layer dual-ground-reference checking.
- Internal power-reference stitching-capacitor checking for high-speed tracks using power planes as
  references.
- Internal differential-pair power-reference stitching-capacitor symmetry checking.
- Internal split-plane stitching-capacitor proximity checking.
- Internal split-plane stitching-capacitor reference-bridge checking.
- Internal same-net non-ground plane-hotspot via-spacing checking.
- Internal reference-plane intrusion checking for signal tracks routed through plane zones.
- Internal ground-reference-plane fragmentation checking for same-net pour islands.
- Internal analog/digital split-domain reference-plane checking.
- Internal mixed-domain shared-ground return-current checking.
- Internal high-speed active-component edge keepout checking.
- Internal high-speed termination-resistor placement checking.
- Internal high-speed stub topology checking.
- Internal differential-pair layer and via-count symmetry checking.
- Internal same-interface differential-pair layer-set checking for explicitly indexed interfaces.
- Internal same-interface differential-pair via-count checking for explicitly indexed interfaces.
- Internal differential-pair keepout checking.
- Internal differential-pair component-courtyard intrusion checking.
- Internal high-speed layer-transition ground-via checking.
- Internal differential-pair transition ground-via symmetry checking.
- Internal high-speed source/sink terminal ground-return checking.
- Internal high-speed via-to-pad proximity checking.
- Internal high-speed via-diameter checking.
- Internal decoupling-capacitor ground-via proximity checking.
- Internal active-IC internal power-plane support checking.
- Internal high-speed via-stub checking.
- Internal unfilled non-ground via-in-pad checking.
- Internal blind/buried via drill-size checking.
- Internal differential-pair length-matching checking.
- Internal differential-pair segment length-matching checking for via-delimited local skew.
- Internal parallel-bus length-skew checking for explicitly indexed bus nets.
- Internal differential-pair pad-entry breakout symmetry checking.
- Internal differential-pair pad-entry breakout length checking.
- Internal differential-pair spacing-variation checking for constant P/N impedance spacing.
- Internal differential-pair coupling-capacitor symmetry checking.
- Internal differential-pair coupling-capacitor package-size checking.
- Internal differential-pair via-station symmetry checking.
- Internal high-speed parallel-spacing crosstalk checking.
- Internal adjacent-layer high-speed broadside-parallel checking.
- Low-weight board-utilization placement energy to reduce large unused regions after board-size
  selection.
- Low-weight similar-component alignment placement energy for identical package/role instances.
- Hot-device airflow blockage placement energy for connector courtyards intersecting active/power
  package cooling corridors.
- Functional-region placement energy for components sharing local non-global nets.
- Functional-region rail-domain placement energy for components sharing matching VCC/GND power-pin
  net sets.
- Functional-region connector-ingress placement energy for connector-local net flow into the board
  core.
- Functional-region dogleg placement energy for local signal chains that force orthogonal turns
  through an intermediate component.
- Functional-region main-chip pad-proximity placement energy for direct active-IC signal links.
- Functional-region intrusion placement energy for unrelated components placed inside another local
  functional block.
- Signal-region power-isolation placement energy for power-conversion components entering local
  signal-block halos.
- Connector-EMI regional placement energy for sensitive active ICs with connected signal pads.
- Internal clock-specific high-speed parallel-spacing checking.
- Internal virtual split-domain routing checking for analog/digital signal traces.
- **Track D v2 follow-up: per-tile stimulation programme + 96-lane binding + kwavers-conformant
  energy budget** (`src/manifest.rs`, `examples/v2_per_tile_stim.rs`,
  `output/manifests/v2_per_tile_stim.kv`).
  - New `TileStimulationProfile` struct with per-tile `prf_hz`, `shift_s`, `phase_deg`,
    `ramp_s` plus inherited `tbd_s`/`sd_s`/`isi_s`/`tt_s`/`vpp_v`/`dead_time_s` fields.
    `TileStimulationProfile::from_article_with(prf, shift, phase, ramp)` projects the
    MWSCAS 2024 article preset onto each HV tile per-tile; `protocol_load_j_s()` and
    `clamp(tbd*prf)` expose the kwavers-consumable scalars.
  - New `DriverManifest::tile_profiles: Vec<TileStimulationProfile>` field (defaults to
    empty for v1 / single-stim v2). New public constants `TX_LANES_V2 = 96` and
    `CHANNELS_PER_TILE_V2 = 24` carry the 4-tile × 24-channel shield-stack contract.
  - `to_text()` emits `stim_tile_{i}_*` keys (10 per tile) when `tile_profiles` is
    non-empty; falls back to legacy `stim_*` keys when only `stimulation` is set.
    `from_text()` autodetects tile-form by the `stim_tile_0_prf_hz` probe and rejects
    mixed-keyset (`stim_*` + `stim_tile_*`) and gappy-tile-sequence inputs so a
    truncated file cannot hand kwavers a partial beam profile.
  - New `DriverManifest::is_full_stack_v2()` (tile_profiles.len()==4 &&
    `stimulation.is_none()` && 96-lane binding), `per_tile_load_j_s()`, and
    `stack_load_j_s()` aggregation helpers.
  - New `DriverManifest::validate_v2_energy_budget(EnergyBudgetInputs) ->
    Result<EnergyBudgetReport, String>` asserts the 96-lane binding + 4-tile profile
    count and rejects degenerate inputs (`c_load_f <= 0`, `ampacity_headroom_a <= 0`)
    on the way in. Sums per-tile `protocol_load_j_s`, computes worst-case
    duty-weighted peak tile current `(c_load · v_pp · f · 24 · frame_duty)`, compares
    against the routed board's ampacity headroom, and surfaces per-tile device /
    pulser (`dynamic + gate + recovery`) / series-damping-resistor wattage vectors
    via `pulser_dissipation(...)` so kwavers callers can size the damping-resistor
    package directly. 11 new unit tests (`manifest::tests`) including the
    round-trip, 96-lane binding, ampacity-headroom pass/fail paths, mixed-keyset
    rejection, and gappy-tile-sequence rejection.
  - New `examples/v2_per_tile_stim.rs` constructs 4 distinct tile profiles, emits
    `output/manifests/v2_per_tile_stim.kv`, round-trips through `from_text()`, runs the
    validator end-to-end, and demonstrates both the success and under-routed
    failure paths (ampacity_headroom_a = 0.05 A surfaces a typed error). Stdout
    echoes kwavers-consumable scalars (`lanes`, `total_load`, `peak_i`,
    `duty_weighted_i`, `frame_duty`, `ampacity_head`, `margin`) plus per-tile
    averages (`device_total_w`, `total_pulser_w`, `damping_resistor_w`) the
    propagation pipeline reads straight off stdout if it lacks the .kv file.
- **Track D v2 follow-up: kwavers beam-propagation pre-step adapter** at the
  architectural seam (`src/validate.rs::validate_against_budget`).
  - New typed pre-step struct `KwaversBeamStep { lanes, aperture_m,
    frequency_hz, sound_speed_m_s, focal_m, timing_step_s, pitch_m,
    wavelength_m, f_number }` every kwavers-transducer consumer reads
    verbatim from the manifest scalar contract. Built by
    `manifest_to_kwavers_beam_step(&manifest, &budget) -> Result<KwaversBeamStep, String>`,
    gated on `is_full_stack_v2()`, manifest/budget `lanes` parity, and
    non-finite geometry rejection.
  - New `KwaversBeamValidation` output: `step`, `focal_pressure_pa`
    (article-anchored coherent-N model: `focal_pressure_gain(N) ×
    per_element_peak_i × 9.375e6 Pa/A`), `grating_lobe_free` (max-steer
    ≥ 89°), `in_far_field` (focus vs. Fraunhofer distance), `isppa_w_cm2`,
    `mechanical_index`, `axial_extent_mm`, `lateral_extent_mm`, and a
    `PhysicsReport` aggregating `focal_pressure_pa ≥ 1 MPa` (transduction
    floor), `MI < 10` (cavitation ceiling), `grating-lobe-free ≥ 89°`.
  - New `validate_against_budget(&manifest, &budget) -> Result<KwaversBeamValidation, String>`
    is the architectural seam entry point. Until kwavers-transducer is
    available the function reads the in-crate physics in [`crate::acoustic`];
    the integration point carries a clear
    `// TODO(kwavers-transducer): replace with crate::kwavers_transducer::simulate(&step) -> PressureMap`
    marker showing exactly where the real kwavers call would land — no struct
    migration required to swap the in-crate physics for the kwavers call.
  - 9 new unit tests under `validate::tests`: `manifest_to_kwavers_beam_step_adapts_a_full_v2_stack`,
    `manifest_to_kwavers_beam_step_rejects_a_non_v2_manifest`,
    `manifest_to_kwavers_beam_step_rejects_a_stale_budget_with_mismatched_lanes`,
    `validate_against_budget_smoke_runs_on_a_4_tile_v2_manifest`,
    `validate_against_budget_rejects_a_non_v2_manifest`,
    `validate_against_budget_predictions_are_in_sensible_ranges`,
    `validate_against_budget_grating_lobe_free_check_fires_correctly`,
    `validate_against_budget_fOV_against_TX_LANES_V2_constant`. All passing.
- **Track D v2 follow-up: per-tile damping-resistor power-rating check** at the
  kwavers-side seam (`src/manifest.rs`, `src/validate.rs`, `examples/v2_per_tile_stim.rs`).
  - New `ResistorPackage` enum (`Smd1206 = 250 mW`, `Smd2512 = 1 W`,
    `Smd4527 = 2 W`) with IPC-7351 70 °C ambient derating. `max_power_w()` and
    `name()` (`"1206"` / `"2512"` / `"4527"`) carry the rated envelope; `power_rating_check(w)`
    returns `Ok(margin_w)` (always ≥ 0) on the well-rated branch and a structured
    `ResistorRatingError { package, max_w, actual_w }` with `over_w()` accessor on
    over-rate. Three packages cover the article-class HV7355 envelope: Smd4527 holds a
    0.87 W margin on tile[3] at the article-class 50 pF / 150 V / +50 Hz stagger
    point; Smd2512 fits when the matching cap is tightened to 35 pF; Smd1206 always
    rejects so the placement annealer surfaces the package choice rather than
    silently accepting the wattage.
  - New `EnergyBudgetInputs::damping_footprint: ResistorPackage` field (fourth
    input). Inline per-tile rating check in `validate_v2_energy_budget` rejects with
    `tile[i]: pkg (rated X W at 70 °C) cannot dissipate Y W in the series damping
    resistor (over by Z W)`. Rejection ordering: stack-level ampacity gate first
    (single design-fix pass), then per-tile rating (scoped to the offending tile).
  - Per-tile `per_tile_resistor_margin_w: Vec<f64>` propagated to the kwavers
    seam: `EnergyBudgetReport` → `KwaversBeamStep::resistor_margin_w` →
    `KwaversBeamValidation::resistor_margin_w`. The kwavers consumer reads the
    per-tile margin directly to plan footprint bumps (`Smd2512 ⇒ Smd4527`) and
    matching-cap tightening without re-deriving `pulser_dissipation` — the
    `// TODO(kwavers-transducer): replace with crate::kwavers_transducer::simulate`
    marker in `validate_against_budget` calls this seam contract out explicitly.
  - New 4th safety `Check::lower("resistor margin (per-tile min) ≥ 0 W", …)` in
    the `validate_against_budget` `PhysicsReport` locks the post-rejection
    invariant at the seam: a future contributor who removes the upstream
    rejection gate is caught at this Check before the bug ships.
  - 355 lib tests green (`cargo test --lib`, 13 in `manifest::tests`, 16 in
    `validate::tests`, remaining in the rest of the suite). Example:
    `cargo run --release --example v2_per_tile_stim` (`output/manifests/v2_per_tile_stim.kv`
    sidecar + kwavers-side stdout). Lib tests include
    `v2_tile_form_validation_rejects_underrated_resistor_package` (Smd1206 ⇒
    article-class reject), `resistor_package_power_rating_check_surfaces_margin_and_rejection_paths`
    (pure API coverage of well-rated / at-edge / over-rated for Smd1206, Smd2512,
    Smd4527), and the reverted well-sized fixture (`c_load_f: 50e-12` + Smd4527)
    that gives the article-class envelope a valid IPC-7351 home. Validate tests
    include `manifest_to_kwavers_beam_step_propagates_resistor_margin_verbatim`
    (KwaversBeamStep mirrors the budget margin vector exactly) and
    `validate_against_budget_resistor_margin_check_is_the_per_tile_minimum`
    (- **Track D v2 follow-up: per-tile damping-resistor gatekeeper liftout** at the
  kwavers-side seam (`src/manifest.rs`, `src/validate.rs`).
  - `ResistorPackage::power_rating_check(...) -> Result<f64, ResistorRatingError>` is
    **rename-and-replaced** by `ResistorPackage::power_margin_w(self, dissipation_w) -> f64`
    returning a SIGNED margin (`max_w - dissipation_w`); the `ResistorRatingError` struct
    + `over_w()` accessor are deleted from the public API surface.
  - The `DriverManifest::validate_v2_energy_budget` **inline rejection gate is lifted**:
    the former `tile[i]: pkg (rated X W at 70 C) cannot dissipate Y W (over by Z W)` error path
    no longer fires. Signed per-tile margins propagate through
    `EnergyBudgetReport::per_tile_resistor_margin_w` ->
    `KwaversBeamStep::resistor_margin_w` ->
    `KwaversBeamValidation::resistor_margin_w` verbatim so the kwavers consumer sees both
    the headroom (for `Smd2512 => Smd4527` bumps) AND the under-rate magnitude (for
    matching-cap tightening) without any validator truncation at the seam.
  - New kwavers-side safety constant `KWVERS_MIN_RESISTOR_MARGIN_W = 0.0` (analogous to
    `KWVERS_MIN_FOCAL_PRESSURE_PA` and `KWVERS_MI_CAVITATION_CEILING`) -- the **sole
    gatekeeper** on the per-tile resistor rating via the 4th
    `Check::lower("resistor margin (per-tile min) >= 0 W", min, KWVERS_MIN_RESISTOR_MARGIN_W, "W")`
    against the per-tile vec's min. The Check is no longer redundant -- it can actually fail
    now when a chosen footprint under-rates the article-class envelope (e.g. `Smd1206` on
    50 pF / 150 V / +50 Hz per-tile PRF stagger => `report.all_pass == false` on the 4th
    Check, focal-pressure / MI / grating-lobe-free checks still pass). Tightening the
    constant to a positive slack floor (e.g. `0.005` W) is one edit + regression-test re-pin away.
  - 358 lib tests green (`cargo test --lib`: 13 in `manifest::tests`, 17 in `validate::tests`,
    remaining in the rest -- +1 new validate test over the prior Smd4527-extension baseline).
    New tests:
    `power_margin_w_returns_signed_under_over_and_at_the_edge` (pure API coverage of
    well-rated + at-edge + over-rated signed floats for all three packages);
    `v2_tile_form_validation_surfaces_signed_margins_for_underrated_resistor_package`
    (the old `_rejects_underrated_...` test rewritten -- validator now returns Ok with a
    negative per-tile margin vec instead of Err on a 1206 fixture); and
    `validate_against_budget_kwavers_check_shuts_underrated_resistor_package` (kwavers-side
    sole-gatekeeper regression lock: a 1206 fixture asserts `report.all_pass == false` on
    the 4th Check, the other 3 kwavers-side safety checks still pass -- focal-pressure floor
    / MI ceiling / grating-lobe-free steer).
kwavers-side Check encoding pinned).

- **Track D v2 follow-up: per-tile damping-resistor margin slack floor** at the
  kwavers-side seam (`src/validate.rs` + `examples/v2_per_tile_stim.rs`).
  - `KWVERS_MIN_RESISTOR_MARGIN_W`: `0.0` ⇒ `0.05 W` (50 mW). Real headroom BUDGET
    above the IPC-7351 70 °C AMBIENT ceiling (vs the prior "fits by exactly its rating
    with zero thermal margin" semantic). Catches "fits by its rating" choices
    before stack-temperature drift in the field crosses the IPC ceiling; further
    tightening (e.g. `0.10 W` for pessimistic drift) is one constant edit + a
    regression-test re-pin away.
  - **Test ripple.** `validate::tests::v2_budget` helper now uses
    `ResistorPackage::Smd2512He` (1.5 W; was `Smd2512` = 1 W) so every per-tile
    margin lands ≳ +0.5 W ≫ 0.05 W floor; `report.all_pass == true` and
    `>= KWVERS_MIN_RESISTOR_MARGIN_W` assertions hold over the choice of floor.
    Sole-gatekeeper test (`..._kwavers_check_shuts_underrated_...`) softened from
    `check.value < 0.0` to `check.value < KWVERS_MIN_RESISTOR_MARGIN_W`;
    `report.all_pass == false` still triggers on `Smd1206` (per-tile margins
    ≲ −0.5 W ≪ 0.05 W floor). BLOCKER re-fix at
    `(check.margin - (check.value - check.limit)).abs() < 1e-12` locks the
    `Check::lower` type-level invariant `margin = value − limit` (the
    `margin = expected_min` form broke when limit ≠ 0.0).
  - **Example CLI.** `examples/v2_per_tile_stim.rs` step 5b doc + assertions now
    reference `KWVERS_MIN_RESISTOR_MARGIN_W = 0.05 W` slack floor;
    `tight_check.value >= 0.05 − 1e-12` regression-locks the gatekeeper at the
    seam. Step 5c demonstrates the inverse narrative path: `Smd2512 = 1 W` on
    the +150 Hz PRF stagger envelope yields tile[3] margin `+0.000859375 W`
    (FP64-exact) — BARELY ABOVE the 1 W IPC ceiling, BARELY BELOW the 0.05 W
    slack floor; the kwavers-side 4th `Check` SHUTS the fixture while the other
    3 (focal-pressure / MI / grating-lobe-free) STILL pass (n_failing == 1).
    Closes the gatekeeper-narrative triad: under-rated (Smd2512 step 5c) →
    tight-but-still-fits (Smd2512He step 5b) → comfortable (Smd4527 step 4-5).


### Changed
- Hard DRC clearance is separated from adversarial near-short risk.
- Co-optimization now judges the seed floorplan before annealed feedback rounds and ranks discrete
  grid-cell/via-column shorts out of complete legal candidates.
- Example routing uses a 0.5 mm lattice with continuous pad/via keepout guards.
- Component assembly verification includes courtyard spacing and explicit 3D model envelopes.
- The status documentation now states that the current FPGA example is the executable `XC7A-QFP`
  abstraction, not an exact production FPGA footprint.
- Beamforming images now render a pulsed focal envelope rather than a continuous-wave ridge plot.
- The HV tile now uses three daisy-chained `HV7355K6-G` routed abstractions for 24 local TX outputs.
- The documentation now separates electrical 24-channel HV tile completeness from exact-footprint
  component completeness.
- Internal keep-in verification now checks final track/via copper edge clearance after ampacity
  widening, matching the KiCad board-edge DRC failure mode.
- Routed pad-entry stubs are emitted as DFM copper so KiCad's continuous-geometry connectivity
  agrees with the router's grid-node LVS for exact imported footprints.
- Generated footprint reference fields are emitted on `F.Fab`, avoiding invalid undersized
  silkscreen text and silkscreen-overlap warnings on dense generated boards.
- Placement rotation proposals now respect footprint policy: active ICs, connectors, and power parts
  retain floorplanned orientation by default; decoupling/passive parts may only flip 180 degrees.
- The adversarial audit now flags high-speed traces inside ground/power zones when the reference
  plane boundary is closer than 3 trace widths.
- The adversarial audit now flags high-speed tracks whose start, midpoint, and end lack adjacent
  ground/power reference-plane coverage.
- The adversarial audit now flags inner-layer high-speed tracks that lack ground-zone coverage on
  both adjacent layers.
- The adversarial audit now flags high-speed tracks referenced only to a power plane unless nearby
  C* power-to-ground stitching capacitors exist at both endpoints.
- The adversarial audit now flags P/N power-reference stitching capacitors on differential pairs
  whose pair-axis stations differ by more than the configured symmetry tolerance.
- The adversarial audit now uses the configured split-plane stitching-capacitor proximity budget
  instead of accepting capacitors several millimetres from the crossing.
- The adversarial audit now requires split-plane stitching capacitors to bridge the crossed
  reference zone to another ground/power reference net.
- The adversarial audit now flags same-net non-ground via clusters whose outer pad gaps are below
  the configured 15 mil plane-hotspot spacing budget.
- The adversarial audit now evaluates same-net serpentine spacing as copper edge gap instead of
  centerline spacing for the guide's 4W adjacent-copper rule.
- The adversarial audit now flags length-compensation meanders whose midpoint is more than the
  configured 15 mm guide budget from a same-net bend root.
- The adversarial audit now rejects acute same-net bend geometry while accepting 135 degree bends.
- The adversarial audit now flags non-plane signal tracks routed through ground/power reference
  zones on the same layer, preventing signal copper from carving selected plane layers.
- The adversarial audit now flags same-net ground reference pours split into multiple same-layer
  thermal-relief islands.
- The adversarial audit now flags explicit analog/digital signal nets routed over the opposite
  AGND/DGND split-ground reference zone.
- The adversarial audit now flags analog and digital signal tracks that share one adjacent ground
  reference zone while crossing or running inside the sensitive keepout distance.
- The adversarial audit now flags analog and digital signal tracks that cross the inferred virtual
  split line between analog-domain and digital-domain pad centroids.
- Analog/digital net-name classification now lives in the board domain model and is shared by split
  reference-plane and virtual split-line audits.
- Diagonal acid-trap chamfering now selects the orthogonal L-corner from the offending acute
  junction instead of always using horizontal-first replacement.
- The adversarial audit now flags active ICs carrying TX/OUT/TRIG/HV nets when their courtyards
  enter the 3 mm high-speed component edge keepout, while allowing edge connectors.
- The adversarial audit now flags resistor-like high-speed terminators when no active IC pad on the
  terminated high-speed net is within the configured 2 mm placement budget.
- The adversarial audit now flags connected high-speed T-branches so TX/OUT/TRIG/HV routing can
  reject stub topology and prefer daisy-chain routing.
- The router now grows multi-terminal Signal/HV nets as daisy chains from the current chain tip
  instead of allowing tree branches from any previously routed point.
- Each PathFinder negotiation pass now routes HV and Signal nets before Power/Ground nets while
  preserving caller-visible route order.
- PathFinder route ordering now uses guide-derived route stages, so crystal-associated nets and
  decoupling power/ground legs reserve routing resources before general controlled-signal nets.
- The adversarial audit now flags differential-pair members routed on different layer sets or with
  different via counts.
- The adversarial audit now flags explicitly indexed differential-pair interfaces whose pairs use
  different routed layer sets.
- The adversarial audit now flags explicitly indexed differential-pair interfaces whose pairs use
  different total routed via counts.
- The adversarial audit now flags unrelated signal copper inside differential-pair keepout
  corridors, including generic signal, clock, and adjacent-pair spacing rules.
- The adversarial audit now flags high-speed layer-transition vias without a nearby ground transition
  via.
- The adversarial audit now flags differential-pair layer-transition ground vias whose P/N stations
  differ by more than the configured symmetry tolerance.
- The adversarial audit now flags high-speed source/sink pads without nearby ground return copper.
- The adversarial audit now flags high-speed vias whose nearest same-net pad is outside the
  configured local-placement budget.
- The adversarial audit now flags high-speed vias whose outer diameter exceeds the selected
  design-rule via diameter.
- The adversarial audit now flags SMD decoupling capacitor ground pads whose nearest ground via is
  outside the configured 1 mm local-return budget.
- The adversarial audit now flags SMD decoupling capacitor power pads that cannot reach the
  associated IC power pin on a shared copper layer.
- The adversarial audit now flags associated decoupling capacitors whose cap-to-IC power/ground
  commutation-loop area exceeds the configured budget.
- The adversarial audit now flags active IC power/ground pads that lack same-net internal plane
  copper underneath the pad.
- The adversarial audit now flags high-speed vias whose physical barrel extends beyond the signal's
  actually used layers.
- The adversarial audit now flags unfilled vias placed directly inside non-ground SMD pads.
- The adversarial audit now flags differential-pair members whose routed lengths differ by more than
  the configured tolerance.
- The adversarial audit now flags differential-pair members whose shared-layer routed segment
  lengths differ by more than the configured tolerance, even when total pair length still matches.
- The adversarial audit now flags explicitly indexed parallel-bus net groups whose routed length
  skew exceeds the configured 2 mm budget.
- The adversarial audit now flags differential-pair pad-entry breakout distances whose P/N mismatch
  exceeds the configured tolerance, even when total routed length still matches.
- The adversarial audit now flags matched-but-long differential-pair pad-entry breakouts whose
  uncoupled length exceeds the configured local budget.
- The adversarial audit now flags differential-pair members whose parallel P/N spacing varies by
  more than the configured tolerance.
- The adversarial audit now flags one-sided or station-mismatched P/N AC-coupling capacitors on
  routed differential pairs.
- The adversarial audit now flags C* AC-coupling capacitors on routed differential pairs whose
  courtyard exceeds the configured 0603-class package budget.
- The adversarial audit now flags differential-pair vias whose P/N station placement differs by
  more than the configured tolerance, even when via counts match.
- The adversarial audit now flags unrelated long parallel high-speed traces closer than the
  configured 3W spacing while exempting true differential-pair mates.
- The adversarial audit now flags unrelated high-speed traces that overlap in parallel on adjacent
  copper layers instead of crossing orthogonally or holding lateral separation.
- The adversarial audit now flags unrelated component courtyards that intrude between parallel
  differential-pair members.
- The adversarial audit now applies the wider 50 mil clock keepout to unrelated long parallel
  high-speed runs when either net is clock-like.
- Placement energy now reports a macro-grid utilization term that rewards spread across the
  available keep-in area while preserving routing and manufacturing objectives as higher-priority
  costs.
- Placement energy now reports an axis-alignment term that rewards common package orientation for
  repeated parts without penalizing 180-degree passive flips.
- Placement energy now reports an airflow-blockage term that penalizes connectors placed in the
  nearest-edge cooling corridor to active IC and power packages.
- Placement energy now reports a regional cohesion term that keeps local schematic subsections
  compact while ignoring nets common to every component.
- Placement regional energy now groups components by matching power-pin rail domains, so common
  ground does not merge components using different VCC domains into one floorplan region.
- Placement regional energy now groups explicitly associated support parts with their main IC, so
  crystal/bias/support components without local net sharing still remain in the intended functional
  block.
- Placement regional energy now pulls X*/Y* clock-source support components toward associated IC
  pads on shared nets, matching the guide's route-crystals-first shortest-route rule.
- Placement regional energy now pulls D*/TVS* suppressor passives toward incoming connectors on
  shared nets, matching the connector-side surge-protection placement rule.
- The adversarial audit now flags same-net vias placed inside the incoming connector-to-D*/TVS*
  suppressor path, so suppressor placement cannot pass with an inductive via in the clamp segment.
- Placement regional energy now penalizes unrelated components whose centers sit inside another
  local functional block's bounding region.
- Placement regional energy now also penalizes unrelated component courtyards that intrude into a
  local functional block envelope when their centers remain outside the block.
- Placement regional energy now penalizes power-conversion package courtyards that enter a local
  signal block's isolation halo, separating high-current circuitry from sensitive signal regions.
- Placement regional energy now penalizes active ICs with connected signal pads whose courtyards
  enter a connector EMI halo, separating sensitive high-speed devices from connector noise sources.
- Placement energy now penalizes crossed two-terminal local-net flight lines so placement prefers
  smoother signal flow before detailed routing.
- Placement regional energy now penalizes folded local point-to-point chains whose intermediate
  component has both connected neighbors on the same side, preferring unidirectional functional
  block flow before routing.
- Placement regional energy now penalizes connector-local net flow that does not point inward from
  the nearest board edge toward the board core, preserving smooth I/O ingress before routing.
- Placement energy now penalizes unrelated component courtyards that block another two-terminal
  local net's direct routing channel.
- Physics-guided routing now derives high-speed edge and reference-margin budgets from
  `DesignRules`, adding a Signal/HV board-edge cost gradient before DRC.
- Physics-guided routing now rasterizes existing Signal/HV track proximity into a preferred-spacing
  field so later high-speed routes are biased toward wider separation when board space is available.
- Physics-guided routing now rasterizes adjacent-layer Signal/HV track proximity so high-speed
  routes prefer lateral separation before broadside-parallel DRC rejection.
- Physics-guided routing now applies a class-aware via cost, charging Signal/HV layer transitions
  above power/ground transitions so high-speed nets reduce impedance-disrupting vias before audit.
- Placement energy now pulls resistor-like passives toward active IC pads on their shared net,
  giving termination placement a direct objective before high-speed audit rejection.
- Placement annealing now applies a force-directed proposal bias for resistor-like terminators
  toward active IC pads on their shared net, matching the termination placement energy.
- Physics-guided routing now adds a reference-plane cost penalty for Signal/HV nodes without
  adjacent ground/power zone coverage.
- Physics-guided routing now adds an inner-layer dual-ground cost penalty for Signal/HV nodes that
  lack ground-zone coverage on both adjacent layers.
- Physics-guided routing now prefers ground-backed Signal/HV reference planes over power-only
  references while still preferring power-backed routing over unreferenced routing.
- Physics-guided routing now grades Signal/HV nodes by adjacent reference-zone boundary margin so
  search prefers stronger return-plane margin before DRC.
- Physics-guided routing now applies a soft bottom-layer penalty to Signal/HV nets, preferring
  top-side routing when competing routes are otherwise equivalent.
- Co-optimization clean-board selection now shares final verification's hard DRC predicate, so
  guide-derived routing faults cannot drift between optimizer acceptance and final verification.
- Assembly verification now flags bottom-side SMD footprints and through-hole pads that do not
  include the top assembly side, matching the guide's same-side SMD/top-side through-hole rule.
- The adversarial audit now flags blind and buried vias whose drill exceeds the configured
  blind/buried via drill limit.
- KiCad DRC parsing now handles array-valued JSON reports and numeric count fields with whitespace;
  the HV7355 example now uses a typed net-context struct for exact pad-name wiring.
- XC7A100T CAD tests now assert actual downloaded vendor symbol/footprint facts for power balls and
  MGT differential-pair names.
- The adversarial audit now flags same-layer different-net track segments that physically cross
  between grid nodes, and PathFinder rejects crossed foreign diagonal edges plus diagonal moves that
  clip a foreign via corner. On the checked FPGA example iteration this reduced external KiCad DRC
  failures from 327 to 20 and eliminated internal `track_crossing_violations`.

### Changed (routing algorithm — 2026-06-23)
- **Targeted rip-up** (`route_with_obstacles`): only nets that participate in at least one
  over-capacity grid node are ripped up and re-routed each iteration. Legal nets stay in
  place, eliminating redundant Dijkstra searches in later iterations when only a small
  fraction of nets remain congested. Uses a new `Grid::overuse_bitset()` per-iteration
  snapshot; first iteration still routes all nets unconditionally.
- **History decay** (`PathFinderParams::history_decay = 0.05`): `accumulate_history` now
  accepts a `decay: f32` argument that multiplies the history of nodes no longer over-capacity
  by `1 - decay` each iteration. Prevents stale congestion gradients from permanently biasing
  the router away from channels that have since cleared.
- **Stall-break schedule** (`route_with_obstacles`): consecutive iterations where the
  overused-node count fails to decrease increment `stall_count`; at ≥ 3 stalls the present
  factor receives a `2^(stall_count/3)` boost (capped at 8×) to break congestion equilibria
  where two nets keep swapping the same overloaded resource.

### Changed (correctness / integrity — 2026-06-23)
- **`charge_recycling_violations` detection** (`audit.rs`): wired `detect_charge_recycling_violations_board()` to `audit()`.
  Fires one violation per N-level-capable IC (MD1715, MAX14815, STHVUP32) on a board that lacks a
  charge-recycling bus net (`CHR*`/`CR_*`/`CHREC*`). Previously always returned 0.
- **`pulse_skip_violations` detection** (`audit.rs`): wired `detect_pulse_skip_violations()` to
  `audit()`. Vacuous when `DesignRules::max_skip_fraction == 0.0` (default); fires 1 violation
  when configured skip fraction exceeds the 5% RMS pressure-error tolerance on the TX channel
  count. `DesignRules` gains `max_skip_fraction` (default 0.0) and `pressure_error_tol` (default
  0.05) fields.
- **`validate.rs`**: removed debug `println!` that was printing to stdout on every call to
  `min_hv_spacing_mm` in library code.
- **`driver.rs`**: fixed cross-swapped `ComponentComparison.total_w` / `.device_w` field
  assignments: `total_w` now correctly holds `d.device_total` (full IC heat: dynamic + gate +
  recovery); `device_w` now holds `d.dynamic_device` (R_on share of switching loss only). Added
  `DEFAULT_THETA_JC_K_PER_W = 40.0` constant replacing two undocumented magic literals.
- **`acoustic.rs`**: added ±90° guard in `focused_delay_profile_s`; `tan(±π/2) = ±∞` now returns
  a zero delay vector instead of propagating NaN through callers.
- **`stack.rs`**: replaced input-dependent `assert!` in `optimize_stack` with a graceful early
  return emitting `StackPlan { feasible: false, limiter: "zero channels..." }` for zero-channel
  or zero-capacity inputs.
- **`pipeline.rs`**: `cooptimize_min_layers` and `cooptimize_min_area` now return
  `Option<CoOptResult>` (returning `None` on empty `layer_options`/`sizes` slices) instead of
  panicking. `cooptimize`'s `rounds == 0` expect message documents the programmer-error invariant.
- **`kicad_cli.rs`**: fixed three `let _ =` / `.ok()` silent discards — `create_dir_all` errors
  now propagate, render failures log with an inline comment, and temp-file pre-cleanup is
  documented.
- **`manifest.rs`**: fixed vacuous error message (`names.len()` is always 0 in the `is_empty`
  branch); added doc comments explaining the article operating-point constants and the
  `"pending-controller-manifest"` placeholder sentinel.
- **`audit.rs`**: fixed copy-paste error in `charge_reservoir_violations_fire_on_under_provisioned_buck`
  test: hotspot assertion now checks `r3` (violation path) not `r2` (vacuous path).
- `risk_score` now folds `charge_recycling_violations * 10.0` and `pulse_skip_violations * 8.0`.

### Changed (physics enhancement — 2026-06-23)
- **`acoustic.rs`**: added `focal_pressure_gain(n)` (coherent N-element pressure gain),
  `acoustic_intensity_w_per_m2(p_rms_pa, z0_rayl)` (pressure → acoustic intensity model),
  and `nonlinear_shock_parameter(p0, f, z, ρ, c, B/A)` (Earnshaw plane-wave shock distance /
  normalized σ for nonlinear regime assessment). 3 new tests.
- **`thermal.rs`**: added `junction_temperature_k(T_amb, ΔT_board, θ_jc, P)` (completes the
  thermal chain to silicon junction) and `temperature_derated_resistance(R_dc, T_ref, T_op, α)`
  (IEC 60228 copper TCR correction for hot-board IR-drop accuracy). 2 new tests.
- **`pdn.rs`**: added `pdn_impedance_at_freq(caps, f)` (frequency-domain |Z(f)| of a
  capacitor bank modelled as series ESR+ESL+1/jωC admittance) and `anti_resonance_hz(L_bulk,
  C_local)` (parallel LC anti-resonant peak frequency for bulk/local cap selection). 3 new tests.
- **`si.rs`**: added `stripline_impedance(w, t, b, er)` (Wadell centered-stripline formula for
  inner-layer controlled impedance), `differential_microstrip_impedance(w, h, s, er)`
  (odd-mode differential Z via crosstalk coupling), and `risetime_degradation_ps_per_m`
  (skin-effect + dielectric-loss edge degradation estimate). 3 new tests.
- **`emi.rs`**: added `radiated_emi_dbuv_m(f, A_loop, I_pk, r)` — small-loop far-field emission
  estimate per CISPR 22 at 3 m/10 m test distance. 2 new tests.
- **`driver.rs`**: added `switching_node_ringing_v(I_pk, L_loop_nH, C_sw)` (LC characteristic-
  impedance overshoot above the supply rail when the commutation loop rings after switch-off) and
  `thermally_derated_efficiency(op, T_j, α_Rds, P_acoustic)` (efficiency with hot Rds_on via
  temperature coefficient from the datasheet). 2 new tests.
- **New `src/optim.rs`**: `evaluate_design_point(op, array, thermal, pdn, emi, p_acoustic, B/A, Z0)`
  combines all physics modules into a single `DesignReport` covering electrical, thermal, acoustic,
  PDN, and EMI outputs in one structured call. Auxiliary helpers: `max_safe_duty_thermal`,
  `ringing_exceeds_breakdown`, `hot_track_resistance`. 5 new tests.
- **`io.rs`**: replaced all 42 `let _ = writeln!(...)` / `let _ = write!(...)` silent discards
  with file-local `wln!` / `w!` macros that call `.expect("invariant: String write_fmt never
  fails")`, making the infallibility invariant explicit at every call site.

### Verified (2026-06-23)
- `cargo clippy --all-targets --all-features -- -D warnings` (clean)
- `cargo nextest run` (308 tests)
- `cargo test --doc` (clean)
- `cargo doc --no-deps`
- KiCad CLI DRC on both full-driver boards: 0 violations, 0 unconnected items.
- Generated stack manifests: FPGA/HV compatibility PASS; current 96-channel shield stack is
  assembly-complete at 5 boards, 60 mm height, and `TX_0..TX_95` global channel coverage.
- KiCad CLI DRC on both full-driver boards after the copper-edge keep-in fix: 0 violations,
  0 unconnected items.
- KiCad CLI DRC on both full-driver boards after exact FPGA/HV stack connector replacement:
  0 violations, 0 unconnected items.

### Fixed (DFM / physics / render — 2026-06-23)
- **`convert_diagonals_to_orthogonal_safe` (`dfm.rs`)**: the safe-corner check now seeds
  `cell_net` from all axial track cells (full run, not just endpoints), all via barrels (all
  layers in the via span), and all pad cells (through-hole = all layers, SMD = declared layer).
  Root cause of the DOUT/TMS LVS short: a 45° diagonal’s L-corner was placed on a cell
  occupied mid-track or at a via barrel of a foreign net; endpoint-only seeding passed the
  foreign-net check; the LVS barrel-union then connected the corner to the foreign net. The fix
  mirrors the LVS union rules exactly.
- **`detect_serpentine_spacing_violations` (`audit.rs`)**: added minimum parallel-run overlap
  guard (10× track width = 1.5 mm for 0.15 mm trace). Short L-shape DFM stub legs (≤0.5 mm)
  are now excluded from the serpentine-spacing check; `serpentine_spacing_violations` drops
  from 61 to 0 on the FPGA tile.
- **`detect_diff_pair_violations` (`audit.rs`)**: added minimum segment-length guard (1 mm)
  before building a diff-pair corridor rectangle. L-shape DFM stub legs (0.5 mm) are
  excluded; `diff_pair_violations` drops from 3 to 0 on the FPGA tile.
- **Example compilation (`examples/fpga_tile.rs`, `examples/hv7355_tile.rs`)**: `fn place`
  changed from `impl Into<String>` to a generic `<S: Into<String>>` parameter, fixing E0562
  on MSYS2 Rust 1.95.0. Call sites with double-comma syntax and missing `&mut comps` argument
  repaired. `DEBUG DCO` diagnostic println loop removed.

### Added (render — 2026-06-23)
- **`src/render.rs`**: new `render_board_svg(board, comps, lib) -> String` and
  `save_board_svg(path, board, comps, lib)` functions. Renders copper layers bottom-to-top
  (F.Cu topmost), via annular rings + drill holes, SMD and through-hole pads, component
  courtyard rectangles, reference designator labels, and a layer legend. Exported via
  `lib.rs` as `kicad_routing::{render_board_svg, save_board_svg}`. Zero external dependencies
  (std only). Both tile examples now emit `<board>.svg` alongside the `.kicad_pcb`.

### Fixed (FPGA optimizer iteration — 2026-06-23)
- **DFM diagonal handling (`pipeline.rs`, `dfm.rs`)**: the routing pipeline now preserves clean
  45-degree segments and applies `chamfer_diagonal_traps` only where a diagonal forms an acute-angle
  DFM trap. This replaces the previous unconditional diagonal-to-L conversion that produced visible
  staircase geometry on the FPGA tile. Regression test: clean isolated diagonals remain one segment.
- **Diagonal corner clearance (`route/search.rs`)**: PathFinder now rejects a diagonal step when
  either square corner is already owned by a foreign track. This extends the earlier crossed-diagonal
  and foreign-via-corner guards to adjacent diagonal/track clearance failures. Regression test:
  a net routes around a foreign occupied corner instead of using the direct diagonal.
- **Placement utilization (`place/energy.rs`)**: macro-utilization samples now consider movable
  non-connector components first; locked connectors no longer mask an off-centre functional cluster.
  Regression test: adding a locked connector does not reduce the board-coverage penalty.
- **Native SVG pad rendering (`render.rs`)**: component pads are now drawn from the placed footprint
  definitions with real pad width/height and rotation, instead of fixed 0.28 mm pseudo-pads from the
  routing-terminal list. Connector pins now render as inspectable footprint copper; pad tooltips
  identify refdes, pad name, and board coordinate.

### Verified (2026-06-23 — FPGA optimizer iteration)
- `cargo nextest run` focused DFM/routing/placement/render regressions: 5/5 passing.
- `cargo run --release --example fpga_tile`:
  6-layer route, LVS ok (opens 0, shorts 0), `track_crossing_violations=0`, `sharp_bends=148`
  (down from 2315), external KiCad DRC = 1 violation (down from 20), unconnected items = 0.

### Verified (2026-06-23 — DFM/physics/render)
- `cargo nextest run --lib` (294/294 — all passing)
- `cargo clippy --all-targets --all-features -- -D warnings` (clean)
- `cargo run --release --example fpga_tile`: LVS ok (opens 0, shorts 0); acid_traps=33;
  clearance_violations=301; serpentine_spacing_violations=0; diff_pair_violations=0;
  SVG render emitted to `output/examples/fpga_controller_tile/fpga_controller_tile.svg`.

### Fixed (visualization / connection / example correctness — 2026-06-23)
- **`examples/hv7355_tile.rs` — GridSpec mismatch (root cause of out-of-bounds routing)**:
  `GridSpec::cover` was `(50.0, 35.0)` while `PlaceConfig::board` was `(45.0, 30.0)`. The
  router used 5 mm of phantom routing space past the board edge, placing tracks at x = 45.5 mm
  on a 45 mm board. Fixed by aligning GridSpec to `(45.0, 30.0)` so the routing grid and
  the physical board agree. SVG viewBox now correctly reads `57.0 × 42.0` (was `62 × 47`);
  board outline rect `width=45 height=30` (was `width=50 height=35`).
- **`examples/hv7355_tile.rs` — EMI repulsion overcorrection**: `emi_weight: 1.0` caused
  `apply_emi_pair_repulsion` to fire each round with a fixed ~3 mm/round step, pushing U1
  24 mm away from J1 over 8 rounds. U1 converged to (13.85, 20.84) leaving only 0.7 mm
  routing gap to J1; routing degraded to clearance_violations=14, crossings=34, vias=26.
  Root cause: J1 carries BOTH HV (TX_0/TX_1) and LV (MOSI/MISO/SCK/CSN) pads, so U1-J1
  EMI repulsion is always active; HV/LV isolation must be enforced by routing clearance rules
  (`holohv()` creepage/clearance), not by physical separation. Fixed by `emi_weight: 0.0`.
  U1 now converges to (20.30, 18.92), clearance_violations=6, crossings=26, vias=16.
- **`examples/hv7355_tile.rs` — cap seeds in routing corridor**: moved C2/C4/C5/C6 seed
  positions from x=29 mm (adjacent to J1 left edge at 30.3 mm) to x=22/25 mm, preventing
  caps from blocking the U1→J1 routing corridor between rounds.
- **`examples/hv7355_tile.rs` — doc errors**: removed duplicate `# Outputs` section
  (lines 33–42 were an exact copy of lines 23–31); corrected module-level board size
  description from "50×35 mm" to "45×30 mm"; updated example comment from `50×35` to `45×30`.
- **`src/place/footprint.rs` — `FootprintDef::with_pad_names`**: added builder method to
  set pad identifiers for SVG tooltip rendering and `pad_index` name-based wiring.
  `two_row_header` now auto-generates sequential pad names ("1" … `rows*per_row`).
- **`examples/fpga_tile.rs`, `examples/hv7355_tile.rs`**: both `build_qfp32` and
  `build_cap_0402` now call `.with_pad_names(...)` so SVG tooltips read
  `U1 pad 1 at …` instead of `U1 pad ? at …`.
- **`kicad-routing/output/` — stale SVGs removed**: `fpga_controller_tile.svg` and
  `hv7355_driver_tile.svg` in `kicad-routing/output/` reflected the 50×35 mm GridSpec and
  are now deleted; current outputs write to `driver/output/` (process cwd).

### Verified (2026-06-23 — visualization/connection fix)
- `cargo nextest run` (308/308 — all passing, 14 new tests since last baseline)
- `cargo clippy --all-targets --all-features -- -D warnings` (clean)
- `cargo run --example fpga_tile`: complete=true legal=true; acid_traps=23;
  clearance_violations=9; crossings=24; dangling=0; vias=15; SVG: viewBox=-6 -6 62 47,
  board 50×35 mm, max_x=24 mm, max_y=21 mm (all copper in bounds).
- `cargo run --example hv7355_tile`: complete=true legal=true; acid_traps=31;
  clearance_violations=6; crossings=26; dangling=0; vias=16; risk_score=3669.8;
  SVG: viewBox=-6 -6 57 42, board 45×30 mm, max_x=44 mm, max_y=25 mm (all copper in bounds);
  no tracks past x=45 mm boundary (old: x=45.5 mm tracks present).
