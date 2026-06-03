# Documentation Index — Kwavers

`docs/` was pruned during the workspace-crate-split refactor (ADR-011) to remove
~465 historical sprint/phase/audit reports and stale duplicate PM artifacts that
were a merge-conflict surface and no longer reflected the codebase. The pruned
files remain recoverable from git history.

## What lives here

| Path | Contents |
|------|----------|
| [`book/`](book/) | The Kwavers book — chapters, figures, validation narratives. |
| [`ADR/`](ADR/) | Architecture Decision Records (the architecture SSOT). |
| [`gpu/pstd_shader_abi.md`](gpu/pstd_shader_abi.md) | WGSL storage-buffer ABI for `pstd.wgsl` (referenced by the shader). |

## Architecture Decision Records

Numbered chronologically by decision date; rewritten to current reality and
audited 2026-06-03.

| ADR | Topic | Status |
|-----|-------|--------|
| [001](ADR/001-adaptive-beamforming-consolidation.md) | Adaptive-beamforming consolidation | Implemented |
| [002](ADR/002-sensor-array-processing-consolidation.md) | Sensor array-processing consolidation | Implemented |
| [003](ADR/003-signal-processing-analysis-layer.md) | Signal processing → analysis layer | Implemented |
| [004](ADR/004-domain-material-property-ssot.md) | Domain material-property SSOT | Implemented |
| [005](ADR/005-pinn-training-stabilization.md) | PINN training stabilization | Implemented (1 limitation) |
| [006](ADR/006-workspace-pyo3-bindings-architecture.md) | Workspace + PyO3 bindings architecture | Implemented (partly superseded by 011) |
| [007](ADR/007-solver-forward-domain-grouping.md) | `solver::forward` domain grouping | Phase 1 done; Phase 2 open |
| [008](ADR/008-compute-backend-trait-wiring.md) | `ComputeBackend` trait wiring | Implemented |
| [009](ADR/009-pykwavers-elastic-bindings.md) | pykwavers elastic-wave bindings | A.1–A.4 done; A.3.5 open |
| [010](ADR/010-fwi-finite-window-pstd-born.md) | Finite-window PSTD Born forward | Implemented |
| [011](ADR/011-workspace-crate-split.md) | **Workspace crate split** | Implemented; facade is pure re-export |

## Project-management SSOT (repo root, not under `docs/`)

| File | Purpose |
|------|---------|
| [`../README.md`](../README.md) | Project overview, install, status. |
| [`../backlog.md`](../backlog.md) | Strategy / prioritized work. |
| [`../CHECKLIST.md`](../CHECKLIST.md) | Tactical task tracking. |
| [`../gap_audit.md`](../gap_audit.md) | Physics/numerics gap audit. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Version history. |
