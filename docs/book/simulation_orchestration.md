# Chapter 21 — Simulation Orchestration: The Capability Catalog

> **Prerequisite:** Chapter 2 (FDTD/PSTD time loop), Chapter 8 (sensor recording),
> Chapter 5 (cavitation/bubble dynamics). Familiarity with directed acyclic graphs
> and topological sort is assumed.

---

## 21.1 Scope

A multi-physics simulation is a *graph computation*. Each enabled physics
domain (acoustic propagation, thermal diffusion, elastic stress, cavitation
inception, …) reads some fields, writes others, and must execute in an order
consistent with its data dependencies. Building this graph by hand for every
study is tedious, error-prone, and hides the scientific intent inside
imperative wiring code.

This chapter formalises kwavers' alternative: a **capability catalog** that
takes a declarative description of which physics is on (a `PhysicsConfig`)
and emits a runnable, dependency-ordered `PluginManager`. We prove the two
load-bearing theorems — the *catalog's coverage theorem* (every enabled
capability resolves deterministically to a concrete plugin or a structured
error) and the *manager's scheduling theorem* (topological sort of the
field-dependency graph yields a correct, cycle-detecting execution order) —
then walk through a Passive Acoustic Mapping (PAM) workflow for cerebral
cavitation detection as the worked example.

---

## 21.2 The plugin contract

A *plugin* is the minimal abstraction over one physics domain's per-step
update. The contract lives in `kwavers_solver::plugin::Plugin` and reduces to a
few pieces of metadata plus an update method:

```rust
pub trait Plugin: Debug + Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn required_fields(&self) -> Vec<UnifiedFieldType>;   // reads
    fn provided_fields(&self) -> Vec<UnifiedFieldType>;   // writes
    fn update(&mut self, fields: &mut Array4<f64>, grid: &Grid,
              medium: &dyn Medium, dt: f64, t: f64,
              context: &mut PluginContext<'_>) -> KwaversResult<()>;
    // … initialize / finalize / priority / state …
}
```

The `required_fields` / `provided_fields` declarations are the **only** I/O
contract the manager understands. They are the edges of the dependency graph
on the field-vertex set $V_F = \{ \text{Pressure}, \text{Density},
\text{Temperature}, \dots \}$ (the full enumeration is in
`UnifiedFieldType`).

> **Definition 21.1 (Plugin field signature).**
> *For a plugin $P$, define $\text{Req}(P) \subseteq V_F$ and
> $\text{Prov}(P) \subseteq V_F$ as the sets returned by `required_fields`
> and `provided_fields`. The signature $(\text{Req}(P), \text{Prov}(P))$
> determines $P$'s placement in the execution order.*

---

## 21.3 The capability lattice

A *capability* is a strongly-typed enum variant naming one physics
behaviour. The capability set is `PhysicsModelType`:

| Capability | Sub-mode parameters |
|------------|---------------------|
| `LinearAcoustics`     | `solver_type: AcousticSolver ∈ {FDTD{order}, PSTD{spectral_accuracy}, DG{polynomial_order}}`, `boundary_conditions: PhysicsBoundaryCondition` |
| `NonlinearAcoustics`  | `equation_type: NonlinearEquation ∈ {Westervelt, Kuznetsov, KZK}`, `harmonics` |
| `BubbleDynamics`      | `model: BubbleModel ∈ {RayleighPlesset, KellerMiksis, KellerHerring, Gilmore}`, `nucleation` |
| `ThermalDiffusion`    | `bioheat`, `perfusion` |
| `OpticalPropagation`  | `scattering`, `anisotropy` |
| `MechanicalStress`    | `wave_kind: ElasticWaveKind ∈ {Isotropic}` |

(`PhysicsModelType` has these **six** variants. `MechanicalStress` (ADR 021) is wired into
`PhysicsCatalog` as `MechanicalStressPlugin`, a `Plugin` owning a real
`ElasticPstdOrchestrator` (leapfrog λ/μ stress–velocity PSTD); it provides isotropic pressure
`p = -⅓ tr(σ)` to the unified field and — unlike the acoustic-fluid path — supports shear waves
since `μ > 0`. It is *not* the previously-deleted `μ ≡ 0` `ElasticWavePlugin` duplicate. The
`ElasticWaveKind` enum is additive-by-design, leaving room for anisotropic/nonlinear modes.)

All three `NonlinearAcoustics` equation types now resolve to real plugins: `KzkSolverPlugin`,
`WesterveltSolverPlugin`, and `KuznetsovSolverPlugin`. The latter two are thin `Plugin`
adapters that forward `update` to the validated `WesterveltWave` / `KuznetsovWave` full-wave
solvers of Chapter 3 (`forward::nonlinear::{westervelt_spectral, kuznetsov}`) via
`AcousticWaveModel::update_wave` — no physics is duplicated in the adapter.

A `PhysicsConfig` is just a list of these (each tagged `enabled: bool`) plus
a string-keyed parameter bag for global state. Two key properties:

1. **Closed under composition.** Any subset of capabilities is a valid
   `PhysicsConfig`. The lattice is the power set of the variants above; the
   join operation is set union.
2. **Serializable.** The enum is `serde`-derived end to end, so a config can
   be persisted, version-controlled, and exchanged between Rust and Python
   call sites without losing type information.

---

## 21.4 The catalog dispatch theorem

The catalog is one function:

```rust
PhysicsCatalog::build(config, grid, medium, dt) -> KwaversResult<PluginManager>
```

It walks `config.models`, skips disabled entries, and dispatches each
enabled `PhysicsModelType` variant to its concrete plugin constructor.
Source: `kwavers_solver::plugin::catalog`.

> **Theorem 21.1 (Catalog determinism and exhaustiveness).**
> *For every variant $v$ of `PhysicsModelType`, `PhysicsCatalog::build_plugin(v, …)`
> returns either (i) a `Box<dyn Plugin>` whose construction has fully
> succeeded, or (ii) a structured `ConfigError::InvalidValue` naming both
> the variant and the originating index `models[i]`. There is no silent
> fallback, no default plugin, and no panic path.*

**Proof sketch.**
The `match` over `PhysicsModelType` is *exhaustive* — Rust's type system
rejects the program if any variant is unhandled (`#[non_exhaustive]` is not
applied; adding a variant forces a catalog update). Each arm is one of:

- A `Ok(Box::new(ConcretePlugin::new(…)?))` call. If the inner constructor
  returns `Err`, the `?` propagates it unchanged. No silent recovery.
- An `Err(unsupported(idx, name, hint))` call returning a `ConfigError`
  whose message contains both the originating index and the variant name.

By exhaustiveness and structural induction over the match arms, every
variant produces exactly one of these two outcomes.  $\square$

The practical consequence: capability gaps are *visible at runtime through
the type system's exhaustiveness guarantee*, not hidden behind silent
no-ops. Adding a new physics domain is one capability variant + one match
arm; failing to add the second is a compile error, not a runtime mystery.

---

## 21.5 The scheduling theorem

`PluginManager::add_plugin` calls `resolve_dependencies` after every
insertion. That routine builds the field-dependency graph and topologically
sorts it. The relevant theorem governs *when* an execution order exists.

Build the directed graph $G = (V_P, E_P)$ over the registered plugins:

- $V_P$ = the set of registered plugins.
- $(P_i, P_j) \in E_P$ iff $\text{Prov}(P_i) \cap \text{Req}(P_j) \ne
  \emptyset$ (some field $P_j$ reads is written by $P_i$).

> **Theorem 21.2 (Sound scheduling).**
> *(a) If no field is provided by two distinct plugins
> ($\forall i \ne j,\ \text{Prov}(P_i) \cap \text{Prov}(P_j) = \emptyset$)
> and (b) $G$ is acyclic, then `resolve_dependencies` returns an execution
> order $\pi: V_P \to \{0, 1, \ldots, n-1\}$ such that for every edge
> $(P_i, P_j) \in E_P$, $\pi(P_i) < \pi(P_j)$.*
> *Otherwise it returns `Err(ValidationError::FieldValidation { … })`
> naming the offending plugins.*

**Proof.** The implementation
(`kwavers_solver::plugin::manager` — DFS topo-sort in the `dependency` submodule)
is a standard DFS-based topological sort with three-state node colouring
(unvisited, visiting, visited). Provider conflicts are detected eagerly
when the `provides: HashMap<UnifiedFieldType, usize>` insertion finds an
existing key (lines 287–302). Cycle detection triggers when DFS re-enters
a `visiting` node (lines 327–329). Both produce structured errors that
identify the offending plugins by metadata id. The standard DFS topo-sort
correctness argument (Cormen et al. 2009, Thm. 22.7) applies to the
remaining case.  $\square$

This is the property that lets the catalog stay declarative: the user
expresses *which physics is on*; the manager works out the *order* from the
field signatures.

---

## 21.6 Worked example — PAM for cerebral cavitation detection

Passive Acoustic Mapping (PAM, Coviello et al. 2015; Salgaonkar et al. 2009;
O'Reilly & Hynynen 2013) reconstructs the spatial distribution of acoustic
emissions inside tissue from a passive multi-element receive aperture. In
focused-ultrasound therapy of the brain it is the primary tool for
distinguishing *stable* (sustained, harmonic) cavitation — a marker of
controlled BBB opening — from *inertial* (broadband, collapse) cavitation —
a marker of incipient haemorrhage.

The forward simulation needed to develop a PAM reconstruction algorithm
must propagate broadband emissions through the skull and capture them at
a virtual aperture. With the catalog, that simulation is one config:

```rust
use kwavers_solver::plugin::{PhysicsCatalog, PluginManager};
use kwavers_physics::factory::{
    PhysicsConfig, PhysicsModelConfig,
    models::{PhysicsModelType, AcousticSolver, PhysicsBoundaryCondition},
};
use kwavers_grid::Grid;

let mut config = PhysicsConfig::new();
config.models.clear();
config.models.push(PhysicsModelConfig {
    model_type: PhysicsModelType::LinearAcoustics {
        solver_type: AcousticSolver::PSTD { spectral_accuracy: true },
        boundary_conditions: PhysicsBoundaryCondition::Absorbing { pml_layers: 12 },
    },
    enabled: true,
    parameters: Default::default(),
});

let manager = PhysicsCatalog::build(&config, &grid, &skull_medium, dt)?;
```

`skull_medium` is a `HeterogeneousMedium` whose density and sound-speed
volumes are derived from a CT scan (see Chapter 15, *Transcranial
Ultrasound*). The PSTD solver was chosen because its zero numerical
dispersion at any frequency below Nyquist preserves the broadband emission
spectrum on which PAM depends.

The post-processing path — synthesising the receive aperture on a
hemispherical sensor array, time-reversing or back-projecting to the
target plane — lives in `kwavers_analysis::signal_processing::pam`
and is the subject of Chapter 13 (*Theranostics*, §13.4).

### 21.6.1 Why the catalog matters here

A direct `PluginManager::add_plugin` workflow would have required the
researcher to (a) construct `PSTDPlugin` with the right `PSTDConfig`,
(b) understand that the absorbing boundary lives inside the plugin not the
config, (c) know which UnifiedFieldType variants the plugin reads and
writes, (d) hope no later refactor breaks the implicit ordering. The
catalog erases all four — the researcher writes only the *capability
intent*, and the dispatcher enforces the rest.

When the same study needs a **second** capability (e.g. adding
`ThermalDiffusion` to model skull heating during long PAM sessions),
the change is one entry in `config.models`. The dependency graph
re-resolves automatically; the execution order may change, but the
correctness guarantee is unchanged.

---

## 21.7 Visualisation: the dependency DAG of an enabled config

The companion script `crates/kwavers-python/examples/book/ch22_simulation_orchestration.py`
renders the capability fan-out and the field-dependency DAG for a
representative multi-capability config (PSTD acoustics + thermal diffusion +
elastic stress). It produces two figures committed to
`docs/book/figures/ch22/`:

- `capability_fanout.svg` — `PhysicsConfig → PhysicsCatalog → plugins`
  with the structured-error fork drawn explicitly.
- `field_dependency_dag.svg` — the field-vertex graph $G$ with the
  topological execution order overlaid.

![Capability fan-out: PhysicsConfig → PhysicsCatalog → plugins, with the structured-error fork.](figures/ch22/capability_fanout.svg)

*Figure 21.1. The catalog dispatch (§21.4): each enabled `PhysicsModelType` resolves to a concrete plugin or a structured error — no silent fallback.*

![Field-dependency DAG with the topological execution order overlaid.](figures/ch22/field_dependency_dag.svg)

*Figure 21.2. The field-dependency graph $G$ (§21.5) for a PSTD-acoustics + thermal-diffusion config, with the topological execution order the manager derives from the plugin field signatures.*

Run it with

```bash
python crates/kwavers-python/examples/book/ch22_simulation_orchestration.py
```

The script depends only on `matplotlib` and `networkx`; no compiled crate
is needed (the figure represents the *logical* graph, not a live solver
state).

---

## 21.8 The BubbleDynamics sub-graph: four model variants, one plugin

### 21.8.1 Model selection and the SRP boundary

The `BubbleDynamicsPlugin` exposes a single `Plugin` interface over four
`BubbleModel` variants:

| `BubbleModel` variant | ODE | Integrator | Valid regime |
|---|---|---|---|
| `KellerMiksis` | KM (Keller & Miksis 1980) | Adaptive multi-step via `BubbleField` | Linear–moderate nonlinear |
| `KellerHerring` | KH (typed wrapper over the same KM-family kernel) | Adaptive multi-step via `BubbleField` | Same regime envelope as KM, contractually selected KH |
| `RayleighPlesset` | RP (Rayleigh 1917) | Same, `use_compressibility = false` | Incompressible, small-amplitude |
| `Gilmore` | Gilmore-Tait (Gilmore 1952) | Classical RK4 via `GilmoreSolver::step_rk4` | Violent collapse, wall Mach > 0.1 |

**Architectural invariant (SRP):** the RK4 loop for the Gilmore ODE lives
inside `GilmoreSolver::step_rk4`, not in the plugin layer.  The plugin's
`update()` method calls `solver.step_rk4(state, p_acoustic, t, dt)` and is
not responsible for integration strategy.  This means swapping from classical
RK4 to an adaptive Dormand-Prince integrator requires editing only
`gilmore.rs`, not the plugin.

### 21.8.2 The dp/dt coupling in the KM path

The Keller-Miksis radiation-damping term requires `dp/dt`:

```text
KM equation (Keller & Miksis 1980, Eq. 3):
(1 - Ṙ/c) R R̈ + (3/2)(1 - Ṙ/(3c)) Ṙ² =
    (1 + Ṙ/c) (p_B - p₀)/ρ  +  R/ρc · dp_B/dt
```

The plugin stores the previous-step pressure array and computes a
backward-difference estimate at each voxel:

```text
dp/dt[i,j,k] ≈ (p_n[i,j,k] − p_{n-1}[i,j,k]) / dt    (O(dt) accuracy)
```

This is consistent with the O(Mach) accuracy level of the KM equation itself.
The Gilmore path requires no dp/dt because the Tait enthalpy difference
already encodes the compressible liquid response exactly to second order in
the wall Mach number.

### 21.8.3 Visualization: bubble radius dynamics under four variants

The companion example
`crates/kwavers-python/examples/book/ch21_simulation_orchestration.py` drives all
four bubble model variants through the PyO3 binding layer — the same
Rayleigh–Plesset, Keller–Miksis (plus the KH wrapper alias), and Gilmore paths
the plugin catalog dispatches to (§21.4) —
and produces Figure 21.3. No ODE is reimplemented in Python; only the drive
set-up and plotting live there:

```python
import numpy as np
import pykwavers as kw

# 5-µm air bubble in water, 1 MHz, 200 kPa, 4 acoustic periods.
R0, RHO, C0, P0 = 5e-6, 1000.0, 1500.0, 101325.0
SIGMA, MU, GAMMA, PV = 0.0725, 1e-3, 1.4, 0.0
F0, PA = 1.0e6, 200_000.0
T_END, N = 4.0 / F0, 8000

# All physics in the Rust core (the plugin catalog's four bubble variants).
t,  r_rp,  _    = kw.solve_rayleigh_plesset(R0, 0.0, P0, PA, F0, T_END, N,
                                            RHO, SIGMA, GAMMA, MU, PV)
_,  r_km,  _    = kw.solve_keller_miksis(R0, 0.0, P0, PA, F0, T_END, N,
                                         RHO, SIGMA, GAMMA, MU, PV, C0)
_,  r_kh, _     = kw.solve_keller_herring(R0, 0.0, P0, PA, F0, T_END, N,
                                         RHO, SIGMA, GAMMA, MU, PV, C0)
_,  r_gil, rdot = kw.solve_gilmore(R0, 0.0, P0, PA, F0, T_END, N,
                                   RHO, SIGMA, GAMMA, MU, PV, C0)
# plot r_rp/R0, r_km/R0, r_kh/R0, r_gil/R0 vs t and rdot (see the example script)
```

![Bubble radius dynamics under the four kwavers variants.](figures/ch21sim/fig01_bubble_ode_comparison.png)

*Figure 21.3. Normalised radius R(t)/R₀ (left) and Gilmore wall velocity (right)
for a 5-µm air bubble in water under a 1-MHz, 200-kPa drive, integrated by
`kw.solve_rayleigh_plesset` / `kw.solve_keller_miksis` / `kw.solve_keller_herring` /
`kw.solve_gilmore`
(§21.8.3). The bubble expands to ≈1.85 R₀ and collapses to ≈0.54 R₀.*

**Interpretation.** At this modest 200-kPa amplitude the four variants agree
closely. Rayleigh–Plesset slightly overshoots the expansion peaks because it
carries no acoustic-radiation loss, while Keller–Miksis and Gilmore — which add
the O(Mach) liquid-compressibility correction — nearly coincide and damp the peaks
a few percent lower. The Gilmore wall velocity shows the sharp inertial
collapse/rebound (≈ ±19 m/s near t ≈ 3.7 µs). The plugin selection rule
`GilmoreSolver::should_use_gilmore` switches to the Tait-enthalpy formulation once
the wall Mach number exceeds 0.1, which at higher drive amplitudes than shown
here produces the strongest (smallest-minimum-radius) collapse.

### 21.8.4 Theorem 21.3 — Gilmore RK4 positivity, equilibrium, and forced-contraction invariants

**Theorem 21.3** (Gilmore RK4 well-posedness): the classical RK4 step
`GilmoreSolver::step_rk4(state, p_acoustic, t, dt)` satisfies:

1. **Positivity.** `R_{n+1} > 0` for any input — the radius is bounded away from
   zero by the per-stage floor `R_i ≥ f64::MIN_POSITIVE`.
2. **Equilibrium stability.** Starting from the mechanical equilibrium
   (`BubbleState::new` sets `p_gas = p₀ + 2σ/R₀`, so the wall pressure
   `p_wall = p_gas − 2σ/R₀ = p₀` matches `p_∞ = p₀`, giving `H = 0`), under zero
   acoustic forcing the radius stays within 0.1 % of `R₀` over one step.
3. **Forced contraction.** Under a compressive instantaneous field
   (`p_∞ = p₀ + p_a > p₀` at the compression phase), the wall velocity is driven
   non-positive: `Ṙ_{n+1} ≤ 0`.

**Proof sketch.** (1) The RK4 weight vector `[1/6, 1/3, 1/3, 1/6]` is strictly
positive and each stage clamps `R_i ≥ f64::MIN_POSITIVE`, so the convex update
cannot produce a non-positive radius. (2) At `BubbleState::new` the Tait enthalpy
difference `H = H_wall − H_∞ = 0` because `p_wall = p_∞ = p₀`, so the Gilmore RHS
acceleration vanishes to leading order and the step is a fixed point up to
`O(dt²)`. (3) Raising `p_∞` above `p_wall` makes `H < 0`, a net inward force, so
the integrated `Ṙ` is non-positive. ∎

These properties are verified by `step_rk4_bubble_stable_at_equilibrium` (2) and
`step_rk4_compressive_forcing_contracts_bubble` (1, 3) in
`kwavers_physics::acoustics::bubble_dynamics::gilmore` (`tests.rs`). Note: the
default `BubbleState::new` is at mechanical equilibrium (not underpressured), so
contraction under *zero* forcing does not occur — contraction requires either a
compressive field or an explicitly sub-equilibrium gas pressure.

---

## 21.9 References

- Coviello C., Kozick R., Choi J., Gyöngy M., Jensen C., Smith P.P.,
  Coussios C.C. *Passive Acoustic Mapping utilizing optimal beamforming
  in ultrasound therapy monitoring.* J. Acoust. Soc. Am. 137(5),
  pp. 2573–2585, 2015. doi:10.1121/1.4916694
- Salgaonkar V.A., Datta S., Holland C.K., Mast T.D. *Passive cavitation
  imaging with ultrasound arrays.* J. Acoust. Soc. Am. 126(6),
  pp. 3071–3083, 2009. doi:10.1121/1.3238260
- O'Reilly M.A., Hynynen K. *A super-resolution ultrasound method for
  brain vascular mapping.* Med. Phys. 40(11), 110701, 2013.
  doi:10.1118/1.4823762
- Cormen T.H., Leiserson C.E., Rivest R.L., Stein C. *Introduction to
  Algorithms*, 3rd ed., MIT Press, 2009. Section 21.4 (Topological sort,
  Theorem 21.7).
- Treeby B.E., Cox B.T. *k-Wave: MATLAB toolbox for the simulation and
  reconstruction of photoacoustic wave fields.* J. Biomed. Opt. 15(2),
  021314, 2010. doi:10.1117/1.3360308 (background on plugin-style
  simulation orchestration in a comparable open-source toolbox).
