# First-Wave Inventory and Gate Map

This document is the executable inventory for the first remediation wave:

- `kwavers` acoustic forward solvers
- `pykwavers` parity and binding surfaces

It uses the scientific acceptance contract in [SCIENTIFIC_ACCEPTANCE_CONTRACT.md](/d:/kwavers/docs/SCIENTIFIC_ACCEPTANCE_CONTRACT.md) as the source of truth.

## Gate Legend

- `Gate 0`: inventory and source-of-truth ownership identified
- `Gate 1`: public interface and file ownership corrected
- `Gate 2`: implementation complete, no silent placeholders
- `Gate 3`: validated against literature, analytical solutions, or `k-wave-python`
- `Gate 4`: performance and memory closure complete

## First-Wave Modules

| Subsystem | Primary Path | Contract Surface | Current Gate | Notes |
|---|---|---|---|---|
| Grid | `kwavers/src/domain/grid` | domain config/types | Gate 0 | Domain SSOT; retain as ownership root |
| Medium | `kwavers/src/domain/medium` | domain config/types | Gate 0 | Feeds both FDTD and PSTD parity |
| Source | `kwavers/src/domain/source` | domain config/types | Gate 0 | Delay/apodization remains parity-critical |
| Sensor | `kwavers/src/domain/sensor` | domain config/types | Gate 0 | Recorder and ordering are parity-critical |
| CPML | `kwavers/src/domain/boundary/cpml` | boundary config/types | Gate 0 | Boundary validation required for Gate 3 |
| FFT | `kwavers/src/math/fft` | shared numerical backend | Gate 0 | Must converge to one execution SSOT |
| FDTD | `kwavers/src/solver/forward/fdtd` | `FdtdConfig`, `FdtdWorkspace` | Gate 1 | Scientific metadata exists; workspace surface added |
| PSTD | `kwavers/src/solver/forward/pstd` | `PSTDConfig`, `PstdWorkspace` | Gate 1 | Scientific metadata exists; workspace surface added |
| Validation | `kwavers/src/solver/validation` | validation metadata/types | Gate 1 | Contract surface already present |
| PyO3 bindings | `pykwavers/src/lib.rs` | stable solver/domain wrappers | Gate 1 | Remove placeholder result metadata |
| Parity examples | `pykwavers/examples/*transducer*` | deterministic scenario harnesses | Gate 0 | `us_bmode_phased_array` still failing |
| Parity tests | `pykwavers/tests/*parity*` | parity thresholds and reports | Gate 0 | Use `k-wave-python` as authority |

## First-Wave Execution Rules

1. Run Rust validation with:
   `cargo nextest run --profile review30 --workspace --lib --tests`
2. Any test exceeding the 30-second per-test policy must be decomposed into smaller validations.
3. No production-facing result API may return placeholder metadata.
4. `Workspace` APIs must report steady-state memory expectations explicitly.
