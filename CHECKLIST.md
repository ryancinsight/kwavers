# Project Checklist

## Phase 1: Foundation (0-10%)
- [ ] 100% Audit/Planning/Gap Analysis
- [x] Detect Root/VCS & initialize formal artifacts (`checklist.md`, `backlog.md`, `gap_audit.md`)
- [x] Verify `k-wave` and `k-wave-python` present in `external/`
- [x] Check existing modules for circular dependencies and cross-contamination (Solvers, Domains, Simulation, Clinical, Analysis, Physics, Math, Core)

## Phase 2: Execution & Restructuring (10-50%)
- [x] Eliminate circular dependencies & cross-contaminations
- [x] Clean codebase: Remove dead/deprecated code, resolve all warnings, avoid build logs
- [x] Establish deep nested file structure (3-5+ levels) with parent/child hierarchies
- [x] Enforce Single Source of Truth via shared accessors

## Phase 3: Component Validation vs k-Wave (50%+)
- [x] Grids: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Sources: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Signals: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Sensors: Implement in `kwavers`, wrap in `pykwavers`, validate vs `k-wave`
- [x] Solvers: Implement in `kwavers` (using BURN for GPU/Autodiff for PINN), wrap in `pykwavers`, validate vs `k-wave`
