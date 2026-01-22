# Axisymmetric Solver Removal Migration Guide

## Overview
The deprecated `solver::forward::axisymmetric` module has been removed from kwavers as of 2026-01-21.

## Reason for Removal
The axisymmetric solver was marked as deprecated with the note: "Use domain-level projections instead". The solver module contained:
- Axisymmetric k-space pseudospectral solver (`kspaceFirstOrderAS` equivalent)
- Discrete Hankel Transform implementation
- Cylindrical coordinate handling

However, this functionality has been superseded by the unified `CylindricalTopology` in `domain::grid::topology` which provides better integration with the rest of the codebase.

## Migration Path

### Before (deprecated):
```rust
use kwavers::solver::forward::axisymmetric::{AxisymmetricSolver, AxisymmetricConfig};

let config = AxisymmetricConfig::new(grid_size, spacing);
let solver = AxisymmetricSolver::new(config, medium)?;
let result = solver.solve()?;
```

### After (recommended):
```rust
use kwavers::domain::grid::topology::CylindricalTopology;
use kwavers::solver::forward::pstd::PseudospectralSolver;

// Use cylindrical topology with standard PSTD solver
let topology = CylindricalTopology::new(grid, symmetry_axis)?;
let solver = PseudospectralSolver::with_topology(topology, medium)?;
let result = solver.solve()?;
```

## Alternative Approaches

For axially-symmetric problems, you have two options:

### Option 1: Use Cylindrical Topology (Recommended)
The `CylindricalTopology` in `domain::grid::topology` provides proper handling of cylindrical coordinates and can be used with any solver that supports custom topologies.

### Option 2: Use 3D Solver with Symmetry
For simple cases, you can also model the full 3D problem if computational resources allow, or use symmetry boundary conditions with the standard `PSTDSolver`.

## Affected Components

**Removed Files:**
- `src/solver/forward/axisymmetric/mod.rs`
- `src/solver/forward/axisymmetric/config.rs`
- `src/solver/forward/axisymmetric/solver.rs`
- `src/solver/forward/axisymmetric/transforms.rs`

**Modified Files:**
- `src/solver/forward/mod.rs` - Removed axisymmetric exports

## Impact Analysis

✅ **No Breaking Changes for Users:**
- No tests depend on axisymmetric solver
- No examples use axisymmetric solver
- No benchmarks use axisymmetric solver
- Module was already marked as deprecated

## Questions?

If you were using the axisymmetric solver and need migration assistance, please refer to:
- `domain/grid/topology.rs` - Cylindrical topology implementation
- `solver/forward/pstd/mod.rs` - PSTD solver with topology support
- File an issue on GitHub for specific migration questions

## Timeline

- **Deprecated:** Sprint 208 (previous development cycle)
- **Removed:** 2026-01-21 (this audit)
- **Migration Window:** N/A (no active users found)
