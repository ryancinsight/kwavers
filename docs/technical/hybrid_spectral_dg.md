# Hybrid Spectral-DG Methods Documentation

## Overview

The Hybrid Spectral-Discontinuous Galerkin (Spectral-DG) methods module provides a sophisticated numerical framework that combines the high accuracy of spectral methods in smooth regions with the robustness of discontinuous Galerkin methods near shocks and discontinuities.

## Key Features

- **Automatic Method Switching**: Detects discontinuities and automatically switches between spectral and DG methods
- **High-Order Accuracy**: Spectral accuracy in smooth regions, robust shock handling in discontinuous regions
- **Conservation Guarantees**: Ensures conservation properties are maintained at method interfaces
- **Flexible Configuration**: Customizable detection thresholds, polynomial orders, and transition widths

## Architecture

The module follows SOLID design principles with clear separation of concerns:

```
solver/forward/pstd/dg/
├── mod.rs                     # Main module interface
├── traits.rs                  # Common traits and interfaces
├── discontinuity_detector.rs  # Discontinuity detection algorithms
├── spectral_solver.rs         # FFT-based spectral solver
├── dg_solver/                 # Discontinuous Galerkin solver
├── coupling.rs                # Solution coupling and conservation
└── shock_capturing/           # Stabilization utilities
```

## Usage Example

```rust
use kwavers::{HybridSpectralDGConfig, HybridSpectralDGSolver};
use kwavers::domain::grid::Grid;
use std::sync::Arc;

// Create computational grid
let grid = Arc::new(Grid::new(128, 128, 128, 1.0, 1.0, 1.0).expect("grid creation failed"));

// Configure the hybrid solver
let config = HybridSpectralDGConfig {
    discontinuity_threshold: 0.1,      // Sensitivity for discontinuity detection
    spectral_order: 8,                 // Order of spectral method
    dg_polynomial_order: 3,            // Polynomial order for DG
    adaptive_switching: true,          // Enable automatic switching
    conservation_tolerance: 1e-10,     // Conservation error tolerance
};

// Create the solver
let _solver = HybridSpectralDGSolver::new(config, grid);
```

## Components

### Discontinuity Detection

The discontinuity detector supports multiple detection strategies:

- **Gradient-based**: Detects large gradients and curvatures
- **Wavelet-based**: Uses Haar wavelets to identify high-frequency content
- **Combined**: Uses both methods for robust detection

```rust
use kwavers::solver::pstd::dg::DiscontinuityDetector;
use kwavers::domain::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

let grid = Arc::new(Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed"));
let field: Array3<f64> = Array3::zeros((64, 64, 64));

// Detect discontinuities
let detector = DiscontinuityDetector::new(0.1);
let mask = detector.detect(&field, &grid)?;
let _ = mask;
```

### Spectral Solver

The spectral solver uses FFT for high-order accuracy:

- **De-aliasing**: 2/3 rule filtering to prevent aliasing errors
- **Spectral derivatives**: Exact derivatives in Fourier space
- **Stability**: Automatic CFL condition based on order

```rust
// Spectral solver is created internally by HybridSpectralDGSolver
// But can be used standalone:
use kwavers::solver::pstd::dg::spectral_solver::RegionPSTDSolver;
let _spectral = RegionPSTDSolver::new(8, grid.clone());
```

### DG Solver

The DG solver provides robust shock handling:

- **Legendre basis**: Uses Legendre polynomials for element representation
- **Upwind flux**: Godunov flux for stability
- **Variable order**: Configurable polynomial order

```rust
// DG solver is created internally by HybridSpectralDGSolver
// But can be used standalone:
use kwavers::solver::pstd::dg::{BasisType, DGConfig, DGSolver, FluxType, LimiterType};

let dg_config = DGConfig {
    polynomial_order: 3,
    basis_type: BasisType::Legendre,
    flux_type: FluxType::LaxFriedrichs,
    use_limiter: true,
    limiter_type: LimiterType::Minmod,
    shock_threshold: 0.1,
};
let _dg = DGSolver::new(dg_config, grid.clone()).expect("DG solver creation failed");
```

### Solution Coupling

The coupling module ensures smooth transitions:

- **Smooth blending**: Gradual transition between methods
- **Conservation correction**: Maintains integral conservation
- **Interface smoothing**: Reduces spurious oscillations

## Configuration Options

### HybridSpectralDGConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `discontinuity_threshold` | `f64` | 0.1 | Sensitivity for discontinuity detection (lower = more sensitive) |
| `spectral_order` | `usize` | 8 | Order of spectral method (higher = more accurate but stricter CFL) |
| `dg_polynomial_order` | `usize` | 3 | Polynomial order for DG basis functions |
| `adaptive_switching` | `bool` | true | Enable automatic method switching |
| `conservation_tolerance` | `f64` | 1e-10 | Maximum allowed conservation error |

## Performance Considerations

1. **Grid Size**: Spectral methods require FFT, which scales as O(N log N)
2. **Detection Overhead**: Discontinuity detection adds ~10-20% overhead
3. **Memory Usage**: Requires storage for masks and transition regions
4. **Parallelization**: FFT operations can be parallelized with appropriate FFT library

## Best Practices

1. **Threshold Tuning**: Start with default threshold (0.1) and adjust based on problem
2. **Order Selection**: Higher spectral order for smooth problems, lower for mixed problems
3. **Conservation Monitoring**: Check conservation errors, especially for long simulations
4. **Testing**: Always validate against known solutions or simpler methods

## Limitations

1. **Periodic Boundaries**: Current spectral implementation assumes periodic boundaries
2. **Structured Grids**: Designed for regular Cartesian grids
3. **Single Field**: Currently handles one field at a time (extend for systems)

## Future Enhancements

1. **Non-periodic boundaries**: Chebyshev polynomials for non-periodic domains
2. **Adaptive order**: Dynamically adjust polynomial order based on solution
3. **Multi-field systems**: Coupled systems of PDEs
4. **GPU acceleration**: CUDA/OpenCL kernels for FFT and DG operations

## References

1. Hesthaven, J.S. and Warburton, T., "Nodal Discontinuous Galerkin Methods", Springer, 2008
2. Boyd, J.P., "Chebyshev and Fourier Spectral Methods", Dover, 2001
3. Cockburn, B. and Shu, C.W., "The Runge-Kutta Discontinuous Galerkin Method", J. Comp. Phys., 1998
