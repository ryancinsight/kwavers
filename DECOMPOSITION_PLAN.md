# PSTD GOD OBJECT DECOMPOSITION PLAN

## Current State: 1126 lines - MASSIVE SRP VIOLATION

## Decomposition Structure:

```
src/solver/pstd/
├── mod.rs              # Module exports only (~50 lines)
├── config.rs           # Configuration structures (~100 lines)
├── solver.rs           # Core PSTD solver logic (~250 lines)
├── spectral.rs         # FFT/spectral operations (~200 lines)
├── kspace.rs           # k-space corrections (~150 lines)
├── boundaries.rs       # Boundary condition handling (~150 lines)
├── sources.rs          # Source injection (~100 lines)
├── absorption.rs       # Absorption models (~100 lines)
└── validation.rs       # Configuration validation (~50 lines)
```

## Responsibilities to Extract:

1. **Configuration** - PstdConfig, validation rules
2. **Spectral Operations** - FFT, wavenumber computation, derivatives
3. **k-space Corrections** - Correction factors, methods
4. **Boundary Handling** - PML, periodic, Dirichlet
5. **Source Management** - Source injection, scaling
6. **Absorption** - Power law, stokes, fractional
7. **Core Solver** - Time stepping, field updates

## Implementation Order:
1. Create config.rs - Move configuration
2. Create spectral.rs - Extract FFT operations
3. Create kspace.rs - Extract corrections
4. Create boundaries.rs - Extract boundary logic
5. Create solver.rs - Core solver only
6. Update mod.rs - Clean exports