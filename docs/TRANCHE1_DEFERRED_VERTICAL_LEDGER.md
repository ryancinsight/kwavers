# Tranche 1 Deferred Vertical Ledger

This ledger records repo areas explicitly deferred from tranche 1 while the
canonical acoustic/GPU/Python slice and its core infrastructure dependencies are
 hardened. It is not a backlog dump. Each entry exists because it was observed
in the current tree and is outside the tranche-1 implementation boundary.

## Acoustic Outside Canonical Slice

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/solver/forward/hybrid/**` | Partial implementation / broad coupling | Hybrid coupling is not required for the canonical PSTD/FDTD/GPU parity slice | Stable canonical PSTD/FDTD contracts | Tranche 3 |
| `kwavers/src/solver/forward/bem/**` | Incomplete closure / different numerical family | BEM validation and singular quadrature are outside tranche-1 acoustic time-domain scope | Dedicated exterior-domain validation plan | Tranche 4 |
| `kwavers/src/solver/forward/sem/**` | Simplified search / mesh handling | SEM closure depends on mesh and high-order basis work not in tranche 1 | Gaia-backed meshing decisions | Tranche 4 |

## Non-Acoustic Physics

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/physics/optics/**` | Placeholder / approximation debt | Optical diffusion and sonoluminescence are not part of the canonical acoustic parity target | Separate optics validation program | Tranche 5 |
| `kwavers/src/physics/chemistry/**` | Domain-specific placeholder debt | Sonochemistry does not affect tranche-1 acoustic solver acceptance | Physics-specific benchmark set | Tranche 6 |
| `kwavers/src/physics/thermal/**` | Broad coupled-physics surface | Thermal models are out of scope for canonical acoustic propagation closure | Bioheat/therapy coupling architecture | Tranche 5 |

## Clinical / Imaging

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/clinical/**` | Placeholder / workflow breadth | Clinical workflows are downstream consumers, not prerequisites for solver closure | Stable solver/runtime acceptance contract | Tranche 6 |
| `kwavers/src/domain/imaging/**` | Broad API churn / partial migration | Imaging integration is active but not required for tranche-1 solver parity | Settled medium/sensor/source interfaces | Tranche 5 |
| `kwavers/src/analysis/visualization/**` | Visualization-specific warnings and GPU surface | Not part of canonical acoustic execution or parity acceptance | Renderer/streaming refactor plan | Tranche 5 |

## Registration / Meshing

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `ritk/**` | Separate toolkit vertical | Registration is explicitly outside tranche-1 bounded scope | Clinical/imaging registration product decision | Tranche 7 |
| `gaia/**` | Separate toolkit vertical | Meshing is not required for Cartesian PSTD/FDTD tranche completion | SEM/BEM/FEM mesh requirements | Tranche 7 |

## Neural / PINN / ML

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/solver/inverse/pinn/**` | Large experimental surface | PINN work is not required for canonical forward-solver acceptance | Dedicated ML acceptance contract | Tranche 8 |
| `kwavers/src/analysis/ml/**` | Simplification / placeholder debt | ML utilities are not part of solver/runtime correctness gates | Product prioritization and dataset policy | Tranche 8 |
| `kwavers/src/analysis/signal_processing/beamforming/neural/**` | Explicit placeholder debt | Not required for tranche-1 acoustic propagation parity | Neural beamforming design decision | Tranche 8 |

## Hybrid / BEM / FEM / SEM

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/solver/forward/helmholtz/**` | Separate numerical regime | Frequency-domain Helmholtz closure is not needed for time-domain acoustic tranche | Dedicated frequency-domain validation matrix | Tranche 4 |
| `kwavers/src/solver/forward/hybrid/**` | Broad coupling debt | Depends on closure of multiple external-domain methods | Completed BEM/FEM/SEM sub-verticals | Tranche 5 |
| `kwavers/src/solver/inverse/reconstruction/seismic/**` | Approximation-heavy inverse workflows | Not required for tranche-1 forward solver correctness | Inverse-problem acceptance plan | Tranche 6 |

## Remaining Placeholder / Simplification Debt Observed Outside Scope

| Path | Issue Type | Why Deferred | Blocking Dependency | Suggested Tranche |
|---|---|---|---|---|
| `kwavers/src/analysis/signal_processing/localization/**` | Simplified estimators/tests | Outside canonical acoustic propagation slice | Signal-processing tranche definition | Tranche 5 |
| `kwavers/src/simulation/**` | Mock/test backend surfaces | Legacy simulation orchestration is not the tranche-1 canonical path | Decide canonical runtime façade | Tranche 5 |
| `kwavers/src/infrastructure/api/**` | Placeholder operational debt | API/security concerns are not gating solver correctness | Product deployment scope | Tranche 6 |

## Notes

- This ledger is intentionally conservative: it records deferred scope, not an
  implementation promise.
- Tranche 1 acceptance is limited to:
  - canonical acoustic/GPU/Python closure
  - core error/recovery/telemetry/runtime infrastructure hardening
- New deferred items discovered during implementation should be appended here
  only if they are outside tranche-1 scope and would otherwise be mistaken for
  silent omissions.
