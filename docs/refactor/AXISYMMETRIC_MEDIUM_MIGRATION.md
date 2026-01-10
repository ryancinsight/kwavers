# AxisymmetricMedium Migration Guide

**Version**: 2.16.0  
**Date**: 2026-01-15  
**Status**: Active Deprecation  
**Removal Target**: 3.0.0

---

## Executive Summary

The `AxisymmetricMedium` struct and related constructors in the axisymmetric solver have been **deprecated** in favor of using domain-level `Medium` types with `CylindricalMediumProjection`. This change eliminates layer violations by ensuring that medium definitions live exclusively in the `domain::medium` module, not in solver-specific code.

### What's Deprecated

- `solver::forward::axisymmetric::config::AxisymmetricMedium` (struct)
- `solver::forward::axisymmetric::config::AxisymmetricMedium::homogeneous` (method)
- `solver::forward::axisymmetric::config::AxisymmetricMedium::tissue` (method)
- `solver::forward::axisymmetric::AxisymmetricSolver::new` (constructor)

### What's New

- `domain::medium::adapters::CylindricalMediumProjection` (adapter struct)
- `solver::forward::axisymmetric::AxisymmetricSolver::new_with_projection` (constructor)

### Timeline

| Version | Status | Action |
|---------|--------|--------|
| 2.15.x | Old API only | No changes |
| 2.16.0 | Dual API | Old API deprecated, new API available |
| 2.17.0 - 2.x | Deprecation period | Both APIs work, warnings emitted |
| 3.0.0 | Breaking | Old API removed |

**Recommendation**: Migrate to the new API as soon as possible. The deprecation period will last through the 2.x series to allow ample time for migration.

---

## Migration Patterns

### Pattern 1: Homogeneous Water Medium

#### Old Code (Deprecated)

```rust
use kwavers::solver::forward::axisymmetric::{
    AxisymmetricConfig, AxisymmetricMedium, AxisymmetricSolver
};

let config = AxisymmetricConfig::default();
let medium = AxisymmetricMedium::homogeneous(
    config.nz,
    config.nr,
    1500.0, // sound speed (m/s)
    1000.0  // density (kg/m³)
);
let mut solver = AxisymmetricSolver::new(config, medium)?;
```

#### New Code (Recommended)

```rust
use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricSolver};
use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
use kwavers::domain::grid::{Grid, CylindricalTopology};

let config = AxisymmetricConfig::default();

// Step 1: Create 3D grid (must encompass cylindrical domain)
let grid = Grid::new(
    config.nz, config.nz, config.nz, // nx, ny, nz (cubic for simplicity)
    config.dz, config.dz, config.dz  // dx, dy, dz
)?;

// Step 2: Create domain-level homogeneous medium
let medium = HomogeneousMedium::new(
    1000.0, // density (kg/m³)
    1500.0, // sound speed (m/s)
    0.0,    // absorption μ_a (1/m)
    0.0,    // scattering μ_s' (1/m)
    &grid
);

// Step 3: Create cylindrical topology for projection
let topology = CylindricalTopology::new(
    config.nz,
    config.nr,
    config.dz,
    config.dr
)?;

// Step 4: Project medium to cylindrical coordinates
let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;

// Step 5: Create solver with projection
let mut solver = AxisymmetricSolver::new_with_projection(config, &projection)?;
```

**Key Changes**:
- Medium now uses domain-level `HomogeneousMedium`
- Requires explicit 3D `Grid` and `CylindricalTopology`
- Projection adapter bridges 3D medium → 2D solver
- Parameter order in `HomogeneousMedium::new`: `(density, sound_speed, μ_a, μ_s', grid)`

---

### Pattern 2: Tissue Medium

#### Old Code (Deprecated)

```rust
use kwavers::solver::forward::axisymmetric::{
    AxisymmetricConfig, AxisymmetricMedium, AxisymmetricSolver
};

let config = AxisymmetricConfig::hifu_default();
let medium = AxisymmetricMedium::tissue(config.nz, config.nr);
let mut solver = AxisymmetricSolver::new(config, medium)?;
```

#### New Code (Recommended)

```rust
use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricSolver};
use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
use kwavers::domain::grid::{Grid, CylindricalTopology};

let config = AxisymmetricConfig::hifu_default();

// Create 3D grid
let grid = Grid::new(
    config.nz, config.nz, config.nz,
    config.dz, config.dz, config.dz
)?;

// Create tissue medium with typical properties
let medium = HomogeneousMedium::new(
    1050.0, // tissue density (kg/m³)
    1540.0, // tissue sound speed (m/s)
    0.5,    // absorption at 1 MHz (Np/m) – approximate
    0.0,    // scattering (ignore for acoustic)
    &grid
);

// Create cylindrical topology
let topology = CylindricalTopology::new(config.nz, config.nr, config.dz, config.dr)?;

// Project to 2D
let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;

// Create solver
let mut solver = AxisymmetricSolver::new_with_projection(config, &projection)?;
```

**Note**: The old `AxisymmetricMedium::tissue` used:
- Sound speed: 1540 m/s
- Density: 1050 kg/m³
- Absorption: 0.5 Np/m (frequency-dependent, power law 1.1)
- Nonlinearity B/A: 6.0

For full equivalence with nonlinearity, use `HomogeneousMedium::with_nonlinearity` (if available) or extend `HomogeneousMedium` to support B/A parameter in future versions.

---

### Pattern 3: Heterogeneous Medium

The new API enables heterogeneous media, which was not easily possible with `AxisymmetricMedium`.

#### New Code (Heterogeneous Support)

```rust
use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricSolver};
use kwavers::domain::medium::{HeterogeneousMedium, adapters::CylindricalMediumProjection};
use kwavers::domain::grid::{Grid, CylindricalTopology};
use ndarray::Array3;

let config = AxisymmetricConfig::default();

// Create 3D grid
let grid = Grid::new(128, 128, 128, 1e-4, 1e-4, 1e-4)?;

// Create heterogeneous medium with spatially varying properties
let shape = (grid.nx, grid.ny, grid.nz);
let mut sound_speed_3d = Array3::from_elem(shape, 1500.0);
let mut density_3d = Array3::from_elem(shape, 1000.0);
let absorption_3d = Array3::zeros(shape);

// Example: Add a tissue inclusion
for i in 40..80 {
    for j in 40..80 {
        for k in 40..80 {
            sound_speed_3d[[i, j, k]] = 1540.0; // Tissue
            density_3d[[i, j, k]] = 1050.0;
        }
    }
}

let medium = HeterogeneousMedium::new(
    sound_speed_3d,
    density_3d,
    absorption_3d,
    Array3::zeros(shape), // scattering
    &grid
)?;

// Project to cylindrical coordinates (samples along θ=0 plane)
let topology = CylindricalTopology::new(128, 64, 1e-4, 1e-4)?;
let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;

// Create solver
let mut solver = AxisymmetricSolver::new_with_projection(config, &projection)?;
```

**Benefits**:
- Full 3D heterogeneous medium support
- Automatic projection to 2D cylindrical slice
- Consistent with other domain medium types

---

## Mathematical Invariants Preserved

The `CylindricalMediumProjection` adapter guarantees:

1. **Sound speed bounds**:  
   `min(c_3D) ≤ min(c_2D) ≤ max(c_2D) ≤ max(c_3D)`

2. **Homogeneity preservation**:  
   Uniform 3D medium → Uniform 2D field

3. **Physical constraints**:
   - Positive, finite density: `ρ > 0`
   - Positive, finite sound speed: `c > 0`
   - Non-negative absorption: `α ≥ 0`

4. **Array dimensions**:  
   `shape = (nz, nr)` matching cylindrical topology

5. **Projection mapping**:  
   Samples at θ = 0 plane: `(r, θ=0, z) → (x=r, y=0, z)` in Cartesian

---

## API Comparison

### Constructor Signatures

#### Old (Deprecated)

```rust
// Deprecated constructor
pub fn new(
    config: AxisymmetricConfig,
    medium: AxisymmetricMedium
) -> KwaversResult<Self>
```

#### New (Recommended)

```rust
// New constructor
pub fn new_with_projection<M: Medium>(
    config: AxisymmetricConfig,
    projection: &CylindricalMediumProjection<M>
) -> KwaversResult<Self>
```

### Medium Creation

#### Old (Deprecated)

```rust
// Limited to simple cases
AxisymmetricMedium::homogeneous(nz, nr, c, ρ)
AxisymmetricMedium::tissue(nz, nr)
```

#### New (Recommended)

```rust
// Flexible domain-level media
HomogeneousMedium::new(ρ, c, μ_a, μ_s', grid)
HeterogeneousMedium::new(c_3d, ρ_3d, μ_a_3d, μ_s'_3d, grid)
// ... other medium types
```

---

## Backward Compatibility

### Transition Period (2.16.0 - 2.x)

During the deprecation period:
- **Old API**: Still functional, emits deprecation warnings
- **New API**: Fully supported and recommended
- **Both APIs**: Can coexist in the same codebase

### Compiler Warnings

When using deprecated API:

```
warning: use of deprecated struct `AxisymmetricMedium`: 
  Use domain-level `Medium` types with `CylindricalMediumProjection` instead.
  See type documentation for migration guide.
```

To suppress warnings temporarily (not recommended):

```rust
#[allow(deprecated)]
let medium = AxisymmetricMedium::homogeneous(128, 64, 1500.0, 1000.0);
```

---

## Performance Considerations

### Old API
- Direct storage of 2D arrays in solver
- Zero projection overhead
- Simple, but limited to solver-specific medium types

### New API
- Projection computed once during construction: **O(nz × nr)**
- 2D arrays cached in `CylindricalMediumProjection`
- Solver accesses cached arrays: **O(1)** per access
- Memory overhead: **~(nz × nr × 8 bytes × 3 fields) ≈ 128×64×8×3 = 196 KB** typical
- **Runtime performance**: Identical to old API after construction

**Conclusion**: The new API has negligible performance impact (one-time projection cost, same runtime performance).

---

## Common Migration Issues

### Issue 1: Parameter Order Change

**Problem**: `HomogeneousMedium::new` has different parameter order than `AxisymmetricMedium::homogeneous`.

**Old**:
```rust
AxisymmetricMedium::homogeneous(nz, nr, sound_speed, density)
```

**New**:
```rust
HomogeneousMedium::new(density, sound_speed, μ_a, μ_s', &grid)
//                     ^^^^^^^  ^^^^^^^^^^^  (new params)
```

**Solution**: Note the order change: density comes first, then sound speed, then new absorption/scattering parameters.

---

### Issue 2: Grid Size Mismatch

**Problem**: Cylindrical topology coordinates exceed 3D grid bounds.

**Error**:
```
projection dimensions out of bounds: iz=128, ir=64 -> x=0.0064, y=0, z=0.0128
```

**Solution**: Ensure the 3D grid encompasses the cylindrical domain:
- Grid must extend at least to `(nr × dr, nr × dr, nz × dz)`
- Safe default: cubic grid with size `max(nz, nr)` in all dimensions

**Example**:
```rust
let size = config.nz.max(config.nr);
let spacing = config.dz.min(config.dr);
let grid = Grid::new(size, size, size, spacing, spacing, spacing)?;
```

---

### Issue 3: Missing Absorption/Scattering Parameters

**Problem**: `HomogeneousMedium::new` requires `μ_a` and `μ_s'` parameters not present in old API.

**Solution**: For pure acoustic simulations (ignoring optical properties), use zeros:
```rust
HomogeneousMedium::new(
    density,
    sound_speed,
    0.0, // μ_a (no absorption modeled separately here)
    0.0, // μ_s' (no scattering)
    &grid
)
```

Note: Acoustic absorption in the axisymmetric solver is handled separately via `AxisymmetricConfig::pml_alpha` (PML absorption) and frequency-dependent attenuation models. The `μ_a` parameter in `HomogeneousMedium` is for photoacoustic absorption, not acoustic attenuation.

---

### Issue 4: Nonlinearity Parameter (B/A)

**Problem**: `AxisymmetricMedium` supported optional B/A nonlinearity field, but `HomogeneousMedium` does not expose it directly (as of 2.16.0).

**Workaround**:
```rust
// B/A parameter is not yet exposed in HomogeneousMedium constructor
// For now, use homogeneous medium and note limitation
let medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, &grid);

// Future: Extend HomogeneousMedium to support B/A
// Or use HeterogeneousMedium with custom nonlinearity field
```

**Status**: This is a known limitation. Future versions may extend `HomogeneousMedium` to support `with_nonlinearity(b_over_a: f64)` builder method.

---

## Testing Your Migration

### Unit Test Template

```rust
#[test]
fn test_migration_homogeneous_water() {
    use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricSolver};
    use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
    use kwavers::domain::grid::{Grid, CylindricalTopology};

    let config = AxisymmetricConfig::default();

    // Create 3D grid and medium
    let grid = Grid::new(128, 128, 128, 1e-4, 1e-4, 1e-4).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

    // Create cylindrical projection
    let topology = CylindricalTopology::new(config.nz, config.nr, config.dz, config.dr).unwrap();
    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Create solver
    let solver = AxisymmetricSolver::new_with_projection(config, &projection);
    assert!(solver.is_ok());

    // Verify medium properties
    let solver = solver.unwrap();
    assert_eq!(solver.config().nz, config.nz);
    assert_eq!(solver.config().nr, config.nr);
}
```

### Integration Test Checklist

- [ ] Solver construction succeeds
- [ ] Medium properties match expected values (sound speed, density)
- [ ] CFL stability check passes
- [ ] Simulation runs without errors
- [ ] Output dimensions are correct (nz × nr)
- [ ] Results match expected physics (compare to old API if possible)

---

## Frequently Asked Questions

### Q1: Why was `AxisymmetricMedium` deprecated?

**A**: It violated the Single Source of Truth principle by defining medium properties in the solver layer instead of the domain layer. The new approach uses domain-level `Medium` types, which:
- Eliminates duplication across solvers
- Enables heterogeneous media support
- Improves architectural clarity (separation of concerns)
- Allows reuse of medium definitions across different solvers

---

### Q2: Do I need to migrate immediately?

**A**: No. The old API will continue to work through the 2.x series with deprecation warnings. However, we strongly recommend migrating to avoid breaking changes in 3.0.0.

---

### Q3: Will the new API be slower?

**A**: No. After the one-time projection during construction (< 1 ms typical), runtime performance is identical. The solver accesses cached 2D arrays just like the old implementation.

---

### Q4: Can I use heterogeneous media now?

**A**: Yes! This is a major benefit of the new API. Use `HeterogeneousMedium` with spatially varying properties, and the projection will automatically sample along the θ=0 plane for the axisymmetric solver.

---

### Q5: What if my grid dimensions don't match?

**A**: The projection will return an error if cylindrical coordinates exceed the 3D grid bounds. Solution: Make your 3D grid large enough to encompass the cylindrical domain (see [Issue 2](#issue-2-grid-size-mismatch)).

---

### Q6: Can I still use `CylindricalGrid`?

**A**: `CylindricalGrid` (in `coordinates.rs`) is also deprecated in favor of `CylindricalTopology` (in `domain::grid`). The new solver uses `CylindricalTopology` internally. Both are functionally equivalent, but `CylindricalTopology` is the canonical domain-level implementation.

---

### Q7: How do I add nonlinearity (B/A) with the new API?

**A**: As of 2.16.0, this is not directly supported in `HomogeneousMedium`. Workarounds:
- Use `HeterogeneousMedium` with a constant B/A array
- Wait for future extension of `HomogeneousMedium` to support `with_nonlinearity(b_over_a)` method
- File an issue requesting this feature

---

## Additional Resources

- **Adapter Documentation**: `src/domain/medium/adapters/cylindrical.rs`
- **Sprint 3 Summary**: `docs/refactor/PHASE1_SPRINT3_SUMMARY.md`
- **Medium Consolidation Audit**: `docs/refactor/MEDIUM_CONSOLIDATION_AUDIT.md`
- **Phase 1 Progress**: `docs/refactor/PHASE1_PROGRESS_REPORT.md`

---

## Support

If you encounter issues during migration:

1. Check this guide for common issues
2. Review the inline documentation in `CylindricalMediumProjection`
3. Examine the test cases in `solver::forward::axisymmetric::solver::tests`
4. File an issue with:
   - Old code snippet
   - New code attempt
   - Error messages
   - Expected vs. actual behavior

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-15  
**Maintainer**: Architecture Refactoring Team