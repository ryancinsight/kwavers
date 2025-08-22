# Pragmatic Action Plan for Kwavers

## Immediate Actions (Do Now)

### 1. Stop Adding Features
- No more physics models
- No more examples
- No more abstractions

### 2. Set Up CI/CD
```yaml
# .github/workflows/rust.yml
name: Rust
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - run: cargo build --verbose
    - run: cargo test --verbose
```

### 3. Delete Unnecessary Code
Remove these directories entirely:
- `src/factory/` - Over-engineered, use direct construction
- `examples/` - Keep only 5 basic examples
- `src/ml/` - Not core functionality
- `src/visualization/` - Separate concern

## Week 1: Simplification

### Core Module Structure (Target)
```
src/
├── physics/        # Core physics (keep)
│   ├── wave.rs     # Basic wave equation
│   ├── fdtd.rs     # FDTD solver
│   └── pstd.rs     # PSTD solver
├── grid.rs         # Grid management (keep)
├── medium.rs       # Medium properties (simplify)
├── boundary.rs     # Boundary conditions (keep)
├── source.rs       # Wave sources (simplify)
└── lib.rs          # Public API
```

### Examples to Keep
1. `basic_wave_propagation.rs`
2. `fdtd_example.rs`
3. `pstd_example.rs`
4. `heterogeneous_medium.rs`
5. `benchmark.rs`

Delete the other 25 examples.

## Week 2: Core Functionality

### Simplify API
```rust
// Instead of factory pattern:
let grid = Grid::new(nx, ny, nz, dx, dy, dz);
let medium = HomogeneousMedium::water();
let solver = FDTDSolver::new(grid, medium);
solver.run(steps);

// Not this:
let config = SimulationConfigBuilder::new()
    .with_grid_config(GridConfig::new())
    .with_medium_factory(MediumFactory::new())
    .build();
let simulation = SimulationFactory::create(config);
```

### Remove Abstractions
- Delete plugin system (use traits directly)
- Remove managers (unnecessary indirection)
- Eliminate builders (use constructors)

## Week 3: Testing & Documentation

### Minimal Test Suite
```rust
#[test]
fn test_wave_propagation() {
    // Simple test that wave propagates
}

#[test]
fn test_fdtd_stability() {
    // CFL condition test
}

#[test]
fn test_boundary_conditions() {
    // PML absorption test
}
```

### Honest Documentation
```markdown
# Kwavers

A research-grade acoustic wave simulation library.

## Status
- Alpha quality
- Not production ready
- Use at your own risk

## Features
- FDTD solver
- PSTD solver
- Basic wave propagation

## Installation
```

## Success Metrics

### Before
- 369 files
- 50,000+ lines
- 30 examples
- Unknown test coverage
- No CI/CD

### Target (1 Month)
- 50 files
- 10,000 lines
- 5 examples
- 80% test coverage
- Working CI/CD

## What NOT to Do

1. **Don't add features** until core is stable
2. **Don't optimize** prematurely
3. **Don't abstract** without proven need
4. **Don't claim** production readiness
5. **Don't hide** problems

## Long Term (3-6 Months)

Only after simplification:
1. Performance benchmarks
2. GPU support (if needed)
3. Python bindings (if requested)
4. Additional physics (if validated)

## The Hard Truth

This project needs to:
1. **Admit** it's research code
2. **Delete** 70% of current code
3. **Focus** on core value
4. **Prove** it works with CI/CD
5. **Simplify** ruthlessly

## Final Recommendation

Consider starting fresh with lessons learned:
- Keep physics algorithms
- Discard architecture
- Build minimal version first
- Add complexity only when needed
- Test everything continuously

**Remember**: Perfect is the enemy of good. Ship something simple that works.