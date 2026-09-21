## KW-PY-FLOOR — kwavers-python's published floor matched only itself 2026-09-21

`kwavers-python` built `abi3-py38` while `aequitas-python` moved to
`abi3-py310`. The two are installed together -- `pykwavers` depends on
`aequitas_python` -- so the floors have to agree or a Python 3.8 or 3.9 user can
install one distribution and not the other. Both versions are past end of life.
`kwavers-python` moved to `abi3-py310`, keeping the stable ABI and so one wheel
per platform.

Verified at the new floor: `tools/generate_surface.py --check` reports the
committed stubs current; `cargo check -p kwavers-python --lib` is clean; and
`pytest tests/test_generated_surface.py tests/test_typed_consumer.py` is 8
passed on conda CPython 3.13.12. Not verified here: the wheel build and the full
Python suite, which need a maturin wheel of a large stack, and the hosted runs.

`uv.lock` was deliberately left alone. `uv lock --check` fails against the
**committed** pyproject as well -- uv resolves 129 packages for the declared
floor, against 77 for the 3.10 floor -- so the lockfile was already stale and
this change is not its cause; a plain `uv lock` rewrites 2904 lines and upgrades
unrelated packages. Filed as `KW-PY-UVLOCK-STALE-2026-09-21`.

## KW-EXAMPLES-115 — Seismic example partition closure 2026-08-21

**Closed 2026-09-21 on its own acceptance criterion.**

The acceptance was structural: every seismic example leaf below the 500-line
target, with the example test inventory intact. Measured now, the largest file
in the tree is `examples/seismic_imaging/planar_artifacts.rs` at **488** lines,
then `examples/seismic_imaging_demo.rs` at **471** and
`examples/seismic_imaging_3d_demo.rs` at **352**; the remaining 24 leaves run
from 6 to 283 lines. Nothing in the tree exceeds the target.

The 281-line increment ledger this entry used to carry -- "X now has its own
`seismic_imaging/<leaf>.rs` leaf ... Nextest passes N/N, the 2-D entry point is M
lines ... remains in progress" -- was a per-slice progress log whose numbers
later slices superseded; its last recorded 2-D figure of 1,024 lines is now 471,
so it described a state that no longer existed. That is why a delivered item
still read as in progress. Recover any increment with
`git log --oneline -- crates/kwavers/examples/seismic_imaging/`.

Not re-run for this closure: the example Nextest and mdBook gates the ledger
recorded green at each slice (116/116 and 69/69 in its last entries). The
closure rests on the structural acceptance, which is the part the item asked
for; the example test inventory is unchanged, since this change touches no
example source.

## ATLAS-KWAVERS-HEPHAESTUS-FDTD-107 — Collocated FDTD provider cutover — Apollo co-evolution blocker 2026-08-18

The old consumer-owned collocated FDTD implementation in
`kwavers-gpu/src/gpu/fdtd.rs` and `gpu/shaders/fdtd.wgsl` is deleted. Kwavers'
GPU/CPU equivalence runner now constructs validated Hephaestus `Fdtd3dParams`,
`FdtdMedium`, and `FdtdVelocity` buffers, dispatches the provider-owned WGPU
velocity/pressure kernels, and compares the result with an independent native
f32 CPU stencil. Provider acquisition and dispatch failures are explicit; no
CPU fallback is used. The validator applies the derived f32 absolute-or-
relative error rule, including near-zero values.

Evidence: Kwavers feature-enabled `cargo check --all-targets`, strict Clippy,
focused Nextest (22/22), the affected top-level allocation test (2/2), and
GPU-enabled doctests pass locally with the local Hephaestus provider. The
upstream Hephaestus contract test passes two sequential steps at exact head
`7bc9944852a6ba92d4ff265b9fff9bc8c81e3567`. Kwavers benchmark-regression run
`32095365142` passes at exact head `5155f32e8`. The final workflow repair is
on `2295bfff7`; its exact-head hosted matrix is blocked by the Apollo
co-evolution state.

The prior CI benchmark lane was cancelled at its 30-minute limit while blocked
in `apt-get update`; no Cargo benchmark step had started. The subsequent Test
Suite Coverage lane was cancelled at its 45-minute job limit in the same
install step. Commit `4e11cf555` applies bounded retries and HTTP(S) timeouts;
`0a3446dac` additionally bounds every Ubuntu package-manager process with an
8-minute deadline and a 30-second termination grace period across the CI,
architecture, CUDA-container, and benchmark workflows. The workflow repairs do
not change benchmark inputs or production code. Every tracked `apt-get`
invocation is now deadline-wrapped.

The exact-head merge ref inherits Kwavers main's `apollo-fft ^0.27.0`, while
Apollo default still exposes `0.26.0`. CI run `32099296963` fails in beta
build job `95596582400`, and architecture run `32099297012` fails in clean
architecture job `95596582553`, both during Cargo resolution before source
checks. Apollo PR #104 source `38192bed` is itself blocked by a stale locked
workspace and benchmark measurement manifest: Rust run `32096086258` and
benchmark run `32096086273` fail on those exact requirements. Lowering Kwavers
back to `0.26.0` would contradict its merged API migration, so no consumer
fallback or compatibility path is added. Re-open this item when Apollo's
`0.27.0` default lands and rerun the exact-head matrix.

The separate pressure-only `gpu::compute::fdtd_gpu` path and the disconnected
f64 `FdtdGpuAccelerator` solver seam remain residuals. They are not treated as
the collocated provider contract and require a separate ownership decision.

## ATLAS-KWAVERS-HEPHAESTUS-VIS-104 — GPU visualization initialization boundary — verification pending 2026-08-17

The feature-enabled `VisualizationEngine::render_multi_field` path validates
field dimensions and then silently returns `Ok(())` when either the renderer or
the data pipeline is absent. `VisualizationEngine::create` intentionally
leaves both resources uninitialized; only `initialize_gpu` establishes the
GPU rendering precondition. This is a correctness defect because a valid input
can report success without processing any field.

The repair is consumer-local: return the existing typed
`SystemError::FeatureNotAvailable` when the GPU resource pair is absent, keep
the initialized GPU path responsible for all fields, and keep CPU fallback
behind the non-GPU feature only. The FDTD/provider implementation gap remains
separate and is not changed here. Source head `b275b7115` passes the required
feature-enabled hosted matrix; the PM-only follow-up head is pending the
same exact-head rerun. Local compilation is blocked before package
diagnostics by the shared Atlas overlay's stale Asclepius checkout requiring
`aequitas ^0.1.0` versus `0.2.0`.

## Kwavers → mnemosyne allocation-locality axis closure — 2026-08-16 (Atlas gitlink scope)

The *execution* half of the placement seam is folded onto mnemosyne-heap,
closing the kwavers → mnemosyne allocation-locality axis
(`first_touch_memory`/`bind_memory_to_node` → mnemosyne-heap). Kwavers commit
`152c4a7d1` (branch `codex/kwavers-mnemosyne-numa`, head `08df5730f`) deletes
the hand-rolled `bind_memory_to_node` / `allocate_interleaved_memory` /
`first_touch_memory` primitives in `crates/kwavers-core/src/arena/numa/memory.rs`
(net −235 lines) and re-points `NumaAwareAllocator` (`layout/numa_aware.rs`),
`SoAFieldBuffer` (`batch/soa_buffer.rs`), and the parallel first-touch fan-out
at `mnemosyne_heap::numa::{bind_to_node, first_touch}`. `first_touch_memory_parallel`
stays consumer-local because mnemosyne sits below moirai and cannot depend on
an executor; `MAX_NUMA_NODES` (kwavers 256) is deleted — the nodemask bound is
mnemosyne's (1024). Mnemosyne `5ca0461` adds `mnemosyne-heap::numa` and routes
`PlacementHint::Numa(node)` through `bind_to_node` inside `TieredHeap::alloc`,
so the axis splits cleanly: Themis owns the placement vocabulary, mnemosyne
owns the kernel memory-policy execution, Moirai owns the parallel fan-out.

The fold branch was merged to kwavers main via PR #382 (merge `b74aa7ab3`);
PR #383 normalizes the ADR statuses. Atlas records the merged default
`1d7c6899` (gitlink-only advance via `update-index --cacheinfo`); the kwavers
working tree is peer-dirty on `codex/kwavers-floatelement-roots` and left
untouched per the concurrent-agents disjoint-scope rule. mnemosyne already
records `5ca0461`. Atlas root tracks the same closure as
`ATLAS-KWAVERS-MNEMOSYNE-LOCALITY-001`; this kwavers-side record is the
consumer-owned mirror. See `backlog.md` → `KWAVERS-MNEMOSYNE-LOCALITY-1`.

## FloatElement root-emulation closure — 2026-08-13
## KWAVERS-PLACEMENT-REDUNDANCY-001 — hand-rolled NUMA stack duplicates themis/mnemosyne SSOT (themis half closed 2026-08-15; mnemosyne half open)

kwavers-core `arena/numa/` reimplements placement vocabulary that themis owns
(`NumaTopology::detect()` vs `themis::CpuTopology::detect()`, `current_numa_node`
vs `themis::query::current_numa_node`) and allocation locality that mnemosyne
owns (`first_touch_memory(_parallel)`/`bind_memory_to_node` vs
`mnemosyne-heap::alloc(PlacementHint)`). The `set_thread_affinity` setter has
no provider home (themis = placement vocabulary; moirai sets worker affinity
internally without a public setter). Closure slice: direct `themis-topology`
dep in kwavers-core + mnemosyne allocation routing + affinity-setter
resolution. Assessment recorded at KWAVERS-PLACEMENT-AXIS-ASSESS-001.

- **themis half CLOSED** (PR #371, merged `62df796`): `arena/numa/` now uses
  `themis::CpuTopology`/`NumaNodeId`/`PlacementHint`/`current_numa_node`;
  `NumaTopology`, `NumaAllocPolicy`, and the hand-rolled `current_numa_node`
  are deleted; `set_thread_affinity` stays a thin kwavers-local execution
  wrapper (assessment option A).
- **mnemosyne half still open**: `first_touch_memory(_parallel)` /
  `allocate_interleaved_memory` / `bind_memory_to_node` still hand-roll the
  mbind/VirtualAllocExNuma execution that mnemosyne-heap's
  `alloc(PlacementHint)` owns. Separate closure slice (kwavers → mnemosyne
  allocation locality), not this one.

### KWAVERS-FLOATELEMENT-ROOTS-001 — powf emulation → provider roots (closed 2026-08-13)

Kwavers emulated scalar roots through `powf` fractional powers instead of the
provider-owned sign-preserving Eunomia `FloatElement` root surface — a
redundancy gap: `powf(1/3)`, `powf(-0.5)`, `powf(-1/3)`, `powf(0.25)`, and
`powf(1/20)` sites re-derived roots that Eunomia owns natively (`cbrt`,
`rsqrt`, `nth_root`), with `powf(-1/3)` additionally losing sign preservation
on negative inputs.

The gap is closed: all 10 sites across 12 files now route through
`FloatElement::{cbrt, rsqrt, nth_root}` (sign-preserving, libm-backed
defaults, native f64 overrides), and the `eunomia` dependency edges are
declared for `kwavers-driver` and `kwavers-core`. Evidence: `cargo check
--all-targets --offline` rc=0 and the full suite passes 6130/6130 (15
skipped). Merged as kwavers PR #364 (`1cb63974`), resolving eunomia at
`1a52590`.

## Capability surveys by keyword produce false positives on name collisions

Recorded 2026-08-12 from the Fullwave 2.5 parity comparison.

**Pattern.** Grepping for a capability by name finds the word, not the
capability. Two collisions in one survey:

- `domain decomposition` matched 20 files and read as "multi-GPU partitioning
  present". It is PSTD-vs-FDTD **method** selection - an unrelated feature that
  happens to share the term of art.
- `multi.?gpu` matched 29 files and read the same way. Those are device
  contexts, P2P queries and transfer queues - real infrastructure, but no solver
  splits a grid across devices.

A third was self-inflicted: `rg -li "convex\|curvilinear"` reported zero hits
because ripgrep read the escaped pipe as a literal, and both features exist. A
zero result deserves the same suspicion as a positive one.

**Rule.** A capability is present when something *calls* it for the purpose in
question, not when the phrase appears. Confirm a survey hit by finding the
consumer - and confirm a survey miss by re-running the query a second way before
reporting a gap.

## Tapered analysis gates bias spectral-ratio attenuation measurements

Recorded 2026-08-12 from KW-SOL-072.

**Pattern.** A windowed-DFT attenuation measurement (`alpha = -ln(P_far/P_near)/d`)
applies a taper to the analysis gate. In a *dispersive* medium the far-sensor
pulse is broadened relative to the near-sensor pulse, so the taper weights the
two differently. The resulting bias is multiplicative in alpha and independent
of sensor separation -- indistinguishable, by inspection, from the medium
genuinely absorbing less than prescribed.

**Why it is hard to catch.** Every instinct points at the physics: the fit, the
time step, the boundary treatment, the scheme. Here all four were eliminated
before the instrument was suspected, and the analytic exoneration of the scheme
(von Neumann analysis) was what finally redirected the search.

**The discriminating test.** Vary the sensor separation. A genuine attenuation
gives a separation-independent alpha; an additive contaminant (reflection,
leakage, offset) scales as 1/d; a multiplicative instrument bias stays a fixed
*fraction* at every separation. That one measurement classifies the error
before any hypothesis about its mechanism.

**Rule.** Do not taper a gate whose signal already decays to zero inside it --
there is nothing to truncate, so a taper only adds a position-dependent weight.
Centre the gate on the true emission time, not on step zero plus transit.

## Review 2026-07-29 — PR #325 unresolved blocker closeout

Three non-outdated review threads remained valid at `fc3ac0308`. The repository
manifest overrode Aequitas with a required `../aequitas` checkout; dense
non-empty vessel masks could produce no centerline and divide by
`f64::MIN_POSITIVE`; and both historical-baseline metadata checks omitted
`--locked` after manifest alignment.

The closeout removes the Aequitas path patch and records one canonical Git
source at `ce3ef7a6` in `Cargo.lock`, rejects a non-empty mask with no usable
centerline through an exact typed error regression, and adds `--locked` to both
metadata checks. Both benchmark jobs copy all 26 tracked package manifests plus
`Cargo.lock`. No source fallback, compatibility adapter, or fabricated diameter
path is present. Exact-head hosted gates remain the final closure evidence
because the mutable local Atlas overlay has an unrelated Apollo dual-path
collision and does not reproduce the pinned CI provider graph.

## Review 2026-07-31 — KW-AEQ-MET-04 source-level closure

Vessel metrics closed at the source level. The only review finding on the slice
was redundant centerline extraction in `VesselSegmentation::segment`: the
classification already derived the validated centerline used for the physical
diameter, so the classifier now returns that centerline with the classification
and segmentation reuses it for physical total length. That removes one mask
traversal and one medial-axis pass without changing the Aequitas
`Length`/`Velocity` contracts or the invalid-input behavior.

The audit itself is closed. `KWAVERS-AEQ-MET-04` is implemented on PR #325,
merged as Kwavers main `cc5c9c4dd`; exact-head Code Coverage passed in 39m33s and
Test Suite Coverage in 32m45s, with the remaining required matrix checks green.
No missing Aequitas metric dimension remains in the named CFDrs, Helios, or
Kwavers consumer boundaries, and every audited public complex value stays
Eunomia-backed with no separate imaginary unit required.

The ~5,100 lines that stood here were a chronological ledger: several hundred
`Review <date>: ...` and `RESOLVED` entries, per-package lint and Nextest
counts, and increment-by-increment closure prose. That is what git, the PR
threads and CI already hold, and it is the ledger shape the board rule forbids,
so it deletes rather than archives. Recover any increment with `git log --grep`
on the PR or commit it names, or from this file's own history.

**Open residuals the ledger was the only home of, kept so they are not
re-derived:**

- **CLD-1 stays open** for k-wave/experimental erosion validation and the
  nonlinear frontier extensions named in its row of the clinical/domain table
  below -- nonlinear RT/RM interface *evolution* (not just growth rates), fully
  implicit `dp/dt`, nonlinear large-amplitude cloud scattering, multi-directional
  screening. Branch reconciliation is *not* the gap: the four cloud branches are
  ancestors of `main` and their content already flowed in.
- **CUDA FDTD execution stays open** until real CUDA kernels and value-semantic
  WGPU/CUDA differential tests exist, and **FDTD GPU equivalence stays open**
  until the FDTD solver has a provider-generic Leto/Hephaestus implementation.
  Both are owned by `backlog.md` → `KW-GPU-060`.
- **The `kwavers-solver` manifest-level Rayon dependency stays open** until the
  remaining solver direct-Rayon and ndarray-parallel holdouts are migrated.
- **Tyche does not yet provide genuine Morris or Saltelli/Sobol estimators**, so
  those sensitivity contracts stay absent rather than approximated downstream.
- **The linked-example compile gate stays open.** `mdbook test docs/book` and
  `mdbook build docs/book` pass (the 286 mis-parsed fences now declare `text` or
  `rust,ignore`), but `cargo check -p kwavers --examples --locked` stops before
  compilation on the shared Atlas overlay's stale lock, so the ignored fences
  are not evidence that the linked examples compile.
- **A follow-up performance item is owed** for the abdominal FWI test path: the
  package run takes about 141 s and the paired abdominal filter about 110 s,
  against a 30 s slow-test budget.
- **Hosted verification of the Hephaestus aggregate `DeviceLimits` propagation**
  (WGPU, CUDA, baseline and beamforming builders) remained open at the time of
  the ledger; hosted and focused feature lanes own it.
- **The 128-cubed-by-100 GPU parity performance closure gate** was graph-blocked
  before compilation by provider drift (the shared RITK checkout requiring
  `apollo-fft ^0.24.0` while the local Apollo checkout declared `0.25.0`);
  re-check before treating it as open.

## clinical/ + domain/ (therapy planning · imaging recon · transducers · grid/medium/source)

| ID | Sev | file:line | Gap | Revision |
|----|-----|-----------|-----|----------|
| CLD-1 | C → **PARTIALLY ADDRESSED (2026-06-19)** | `kwavers-therapy/.../lithotripsy/cavitation_cloud.rs` | **Single-bubble dynamics now real:** the cloud erosion is driven by the actual **Gilmore (1952) compressible single-bubble collapse** (`representative_max_radius`/`inertial_collapse_energy`), capturing inertial growth `R_max ≫ R0` under rarefaction — replacing the static-R0 linear proxy. Tests: `R_max(12 MPa) > 3·R0`, deeper rarefaction erodes more. This implements the "Gilmore + Mach corrections" the code comment listed as absent. **Still open (collective / research-frontier):** multi-bubble acoustic coupling + emission back-reaction, cloud-scale energy focusing (Maeda & Colonius 2018), shock-bubble Richtmyer-Meshkov / Rayleigh-Taylor cloud instabilities, inter-phase mass transfer. Erosion carries an empirical `erosion_efficiency` (Sapozhnikov 2002) — collective cloud erosion is not a closed, "100%-accurate" problem in any library. **UPDATE (ADR 027): snapshot→time-resolved coupling DONE** — each cell now carries a real `(R,Ṙ)` state integrated by the canonical adaptive Keller-Miksis solver under the local instantaneous pressure across calls; keystone test proves a cloud cell == the standalone integrator bit-for-bit. Remaining open = the *collective* effects above. **UPDATE (ADR 028): inter-bubble acoustic coupling DONE** — `bubble_radiated_pressure = (ρ/d)(R²R̈+2RṘ²)` couples each cell to its neighbours (two-pass explicit scheme), opt-in (`coupling_enabled`, default off for cost). Tests: closed-form radiated pressure, 1/d scaling, coupling alters a two-bubble trajectory, lone bubble unaffected. **UPDATE (ADR 029): cloud-scale shielding DONE** — the incident field is screened by the cloud's void fraction (`commander_prosperetti_attenuation`, reused) via Beer-Lambert along the incident axis (`shielded_pressure`), opt-in (`shielding_enabled`, default off). Tests: closed-form exponential decay, no-nuclei pass-through, denser-screens-more. **UPDATE (ADR 030): self-consistent (implicit) coupling DONE** — fixed-point iteration of the coupling field (`coupling_pressure_field`), reusing the KM acceleration each iterate; opt-in (`implicit_coupling`, default off). Tests: returned field satisfies its own fixed-point equation, implicit differs from explicit under close coupling. **UPDATE (ADR 031): strong-regime solver DONE** — `CouplingScheme::ImplicitDirect` exactly solves the affine coupling system `(I−D·G)S=e` (robust where fixed-point diverges; self-consistent to ~1e-9 at 20 µm coupling), plus `ImplicitFixedPoint{under_relaxation}`. **UPDATE (ADR 032): four frontier refinements DONE** — (1) `dp/dt` coupling (`couple_pressure_rate`: lagged FD rate `(driving−prev_total)/dt` fed into the affine source acceleration; system stays exact since R̈ is affine in dp/dt); (2) `R(t)`-dependent shielding (`shielding_radius_dependent`: instantaneous per-cell R in the CP resonance, quasi-static); (3) cloud-interface RT/RM linear growth-rate **diagnostic** (`interface_instability`: σ_RT=√(A·k·a), ȧ_RM=k·Δv·a₀·A, A=β/(2−β)); (4) sparse/matrix-free solver (`CouplingScheme::ImplicitIterative`: `solve_lsqr_matfree` + on-the-fly `G_ab`, O(active) memory, matches dense to 1e-6). All opt-in; defaults reduce to ADR 027-031. **Now remaining (deepest frontier):** nonlinear RT/RM interface *evolution* (not just growth rates), fully implicit `dp/dt`, nonlinear large-amplitude cloud scattering, multi-directional screening, and a k-wave/experimental erosion comparison. | open: k-wave/experimental validation |
| CLD-2 | **RESOLVED (2026-07-12 → superseded 2026-08-17)** | `orchestrator/{execution.rs,methods.rs}`, `config.rs`, ~~`kzk_solver_plugin/solver.rs`~~ `kzk/plugin.rs` | Original wiring through `kzk_solver_plugin` had three live physics defects (ATLAS-KWAVERS-KZK-LINEAR-080). **Superseded 2026-08-17 (`5c553d36b`):** deleted buggy plugin; rewired via `KzkPlugin` adapter onto correct `kzk/` module (`KZKSolver`+`KZKConfig`). Collimated + focused paths now correct. | resolved |
| CLD-3 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/hifu_planning/types.rs:60` | Rewrote "Theorem"→"closed-form approximation" w/ validity regime (linear/paraxial F#≳1/homogeneous) + refs (O'Neil 1949, Cobbold 2007); named the magic 0.7 `MINUS6DB_ELLIPSOID_FILL_FACTOR` + flagged unvalidated (value preserved). | done |
| CLD-4 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/physics/mod.rs:47,50` | Category mismatch: `TISSUE_IMPEDANCE` is the nominal *matching-layer design load* (fixed manufactured hardware, `Z_match=√(Z_pzt·Z_load)`, Szabo/Cobbold), NOT a per-voxel sim medium — CT-derivation does not apply; documented to prevent re-flag. `BACKING_IMPEDANCE` was DEAD (no refs) — removed. | done |
| CLD-5 | ~~H~~ RESOLVED (2026-06-01) | `domain/source/transducers/phased_array/config.rs:34` | "Ignores user freq" is false — `Default` is correctly nominal; no constructor drops a passed freq; `satisfies_nyquist` already takes `sound_speed`. Real defect was SSOT dup of `2.5` (geometry + freq field) → single `DEFAULT_CENTER_FREQUENCY_HZ` const. | done |
| CLD-6 | ~~H~~ DOC'D (2026-06-01) | `clinical/therapy/lithotripsy/bioeffects.rs:191` | Documented Pennes-perfusion omission + its CONSERVATIVE (over-estimating) direction for a safety index; cited Pennes 1948; pointer to bioheat solver for quantitative dose. | done |
| CLD-7 | H | `clinical/therapy/therapy_integration/orchestrator/microbubble.rs:197` | uniform microbubble conc; no advection/cluster dynamics | document/extend |
| CLD-8 | M | `domain/boundary/bem/manager/assembly.rs:85` | `.unwrap()` on `last()` w/o bounds | safe `.last().copied()` |
| CLD-9 | M | `clinical/.../hifu_planning/tests.rs:115,156` | focal-spot tested only vs itself, not k-wave/analytic | add reference baseline |
| CLD-10 | M | `domain/source/transducers/focused/bowl/tests.rs:20` | bowl geometry tested, pressure field NOT vs k-wave | add field test |
| CLD-11 | M → **DONE (2026-06-20)** | `domain/boundary/cpml/config/cpml_config.rs:214` + `kwavers/tests/cpml_absorption_quality.rs` | Added `theoretical_reflection_decays_monotonically_with_thickness` (Collino&Tsogka 2001): strict-decrease + bounded-(0,target] property test, params analytically chosen to avoid FP underflow. **Courant sub-item DONE:** `test_cpml_stable_across_thicknesses` (Komatitsch&Martin 2007) sweeps PML thickness {6,8,10,12} at a fixed CFL `dt`, asserting for each that the post-propagation energy is finite (no blow-up), decays below initial (stably absorbing), and absorption is monotone non-decreasing in thickness — empirical proof the CFS-CPML preserves CFL stability regardless of thickness. Refactored the single-thickness test onto a shared `run_cpml_absorption(thickness)` helper (SSOT). | done |
| CLD-12 | ~~M~~ RESOLVED (2026-06-01) | `clinical/imaging/reconstruction/transcranial_ust/medium.rs:14` | `AIR_REJECTION_HU=-300` was a verbatim SSOT DUP of canonical `ct_acoustics::HU_BRAIN_BODY_THRESHOLD=-300` (Aubry 2003 ref). Deleted local const, switched 8 call sites (medium.rs+volume.rs) to canonical. Value drives a *qualitative* slice-selection count (robust to ±100 HU), not a calibrated mapping — no scanner-validated tolerance test warranted. | done |
| CLD-13 | ~~M~~ DONE (2026-06-01) | `domain/imaging/photoacoustic/types.rs:21,127` | Added `PressureFieldSeries` newtype (own leaf `pressure_series.rs`) wrapping `Vec<Array3<f64>>` with a validating constructor (non-empty + dimensionally uniform) and `Deref<[Array3<f64>]>` (zero consumer churn — all slice/`iter`/index callers unchanged). Both struct fields + 3 construction sites wrapped. 4 value-semantic ctor tests (accept/empty/ragged/round-trip). NB: `Array3<f64>` isn't a primitive — the captured invariant is intra-series dimension consistency, not a unit marker; cross-field time-alignment stays test-covered. | done |
| CLD-14 | ~~L~~ DONE (2026-06-01) | various | Audit framing ("uncited magic numbers") was largely false: `LENS_CURVATURE_FACTOR=0.7` already named; `crosstalk 0.1` already `// 10% (typical)`-commented; both erf impls already cited A&S 7.1.26. Real finding = DUPLICATION: two identical A&S 7.1.26 erf copies (`histotripsy.rs`, `clinical_scenarios/scenario/mod.rs`). Hoisted to canonical `math::statistics::erf` (named const + cite + error bound + 3 value-semantic tests); both sites delegate. SSOT. | done |

## kwavers-driver: missing gitignored KiCad fixtures [test-infra]
- `component_accuracy::tests::{hv_driver_artifact_uses_exact_j4_power_header,
  hv7355_32ch_artifact_has_renderable_component_models}` fail reading
  `crates/kwavers-driver/tests/fixtures/boards/hv7355_{24,32}ch_tile/*.kicad_pcb`
  (os error 3 — path not found). The fixtures directory is **gitignored** and
  absent, so these tests can only pass where a developer generated the boards
  locally. Pre-existing test-design defect (tests depend on absent gitignored
  artifacts), independent of the Atlas migration.
- DoR: either commit small deterministic `.kicad_pcb` fixtures (build-size
  budget permitting), generate them in a build/test setup step, or gate the
  tests behind a fixture-present guard. Owner decision needed on fixture policy.

## kwavers-therapy: abdominal FWI preprocessing exceeds test-time budget [perf]
- `theranostic_guidance::tests::abdominal::{abdominal_preprocessing_keeps_external_skin_between_target_and_aperture,
  abdominal_preprocessing_selects_one_connected_treatment_component}` terminate
  at the therapy profile timeout (90 s). Both call `run_theranostic_inverse` on
  64×64×3 / 72×72×3 grids; the passing `abdominal_theranostic_inverse_recovers_lesion_support`
  (42×42×3) already runs 16–19 s (near the 30 s slow threshold), so the FWI
  inverse scales super-linearly past the budget at the larger grids.
- DoR (profile-first per performance_engineering): flamegraph
  `run_theranostic_inverse` at 64×64×3, identify the hot path (forward/adjoint
  PSTD loop, per-iteration cost, leto array-op constants vs pre-migration),
  and optimize the real component — never raise the timeout or shrink the grid
  (test-gaming). Classify migration-regression vs inherent FWI cost by comparing
  the 42×42×3 wall-time against the pre-migration commit.
- **Partial characterization (2026-07-12):** the O(N) setup (elastic-medium,
  orchestrator) is negligible; cost is dominated by the per-timestep 3-D FFT loop
  in the spectral forward/adjoint solver (`config.iterations=12` ×
  `elastic_fwi_iterations=3` × n_time-steps × several FFTs each). NOT a plan-caching
  issue — apollo already caches via `f64::get_3d_plan` (`PlanCacheProvider`).
  The optimizable overhead is the kwavers FFT FACADE
  (`kwavers-math/src/fft/mod.rs`): `fft_3d_array` allocates a fresh leto `Array3` +
  element-wise `from_apollo_complex` conversion every call, and `fft_3d_array_into`
  double-copies (`out.assign(&fft_3d_array(field))`) instead of routing to apollo's
  zero-alloc `fft_3d_array_into`/`_typed_into`. NEXT (needs a CONFIRMED profile — a
  prior attempt instrumented `prof_fft_take()` but ran out of budget before running
  it): confirm facade-conversion vs inherent-FFT split, then eliminate the double
  allocation/conversion (route facade `_into` through apollo `_into`, reuse scratch
  buffers across timesteps) and verify no accuracy regression on the elastic_fwi
  convergence tests. Dedicated effort; do not rush.

## State refresh (2026-07-17) — elastic-FWI objective history defect

- **Finding:** hosted Test Suite Coverage job `87949355634` ran with
  `--test-threads=1`, reached 5,339/5,630 tests, then terminated
  `inverse::elastography::elastic_fwi::tests::fwi_outperforms_linear_inversion`
  at 90.010 seconds. Serial test scheduling therefore does not explain the
  timeout.
- **Root cause:** observed-data synthesis and each objective-only forward
  misfit cloned six `Array3<f64>` field components at every time step before
  sampling a small receiver set. The adjoint path needs displacement histories;
  the objective path does not.
- **Correction:** `ElasticWaveSolver::propagate_point_forces_recording`
  executes the same propagation loop and records receiver displacement directly.
  FWI synthesis and forward misfit now use it, while gradient evaluation retains
  full histories. A focused regression proves the recorded traces are exactly
  equal to traces sampled from the full history.
- **Evidence tier:** value-semantic differential regression plus empirical
  timing. The focused clinical FWI contract passed 1/1 in 29.123 seconds under
  the unchanged solver inputs and assertions.
- **Residual:** a fresh hosted matrix must confirm the result before Kwavers
  merges and its Atlas parent gitlink advances.

## KWAVERS-COUPLING-CONTRACT-001 — Medium-aware field-coupling inputs

`MultiphysicsFieldCoupler` currently receives only the pressure, thermal, and
optical field volumes. Its photoelastic, optical-absorption, and acoustic-
absorption coefficients therefore remain nominal constants selected by the
current API contract. This is distinct from `AcousticOpticalSolver`, which
already accepts a caller-supplied photoelastic coefficient. The field-coupler
API needs a typed medium-property provider before per-voxel and
frequency-dependent values can be implemented without hidden global state.

The 2026-08-17 cleanup removes the misleading TODO/placeholder wording and
records this boundary explicitly; no coefficient or field-update semantics
change in that cleanup.
