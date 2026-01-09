# Advanced Features Tutorial

This tutorial covers the advanced features of Kwavers, including performance profiling, validation, and benchmark suites.

## Table of Contents

1. [Performance Profiling](#performance-profiling)
2. [Validation](#validation)
3. [Benchmark Suite](#benchmark-suite)
4. [Advanced Numerical Methods](#advanced-numerical-methods)
5. [Multi-Physics Simulations](#multi-physics-simulations)

## Performance Profiling

The performance profiling infrastructure provides comprehensive insights into your simulation's performance characteristics.

### Basic Usage

```rust
use kwavers::analysis::performance::profiling::PerformanceProfiler;
use kwavers::domain::grid::Grid;

// Create profiler
let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3).expect("grid creation failed");
let profiler = PerformanceProfiler::new(grid);

// Profile a computation
{
    let _scope = profiler.scope("pressure_update");
    let _ = 1 + 1;
}

// Record allocations/deallocations
profiler.allocate(1024 * 1024);
profiler.deallocate(1024 * 1024);

// Generate report
let report = profiler.report();
println!("{}", report.report());
```

### Roofline Analysis

The profiler includes roofline analysis to identify performance bottlenecks:

```rust
println!("{}", report.performance_analysis);
```

### Cache Profiling

Monitor cache behavior for optimization:

```rust
profiler.cache.record_hit(1);
profiler.cache.record_miss(1);

let cache_profile = profiler.cache.profile();
println!("L1 hit rate: {:.3}", cache_profile.statistics.l1_hit_rate());
```

## Validation

Validate your simulations against analytical solutions and literature benchmarks.

### Running Validation Tests

```rust
use kwavers::solver::utilities::validation::NumericalValidator;

let validator = NumericalValidator::new();
let results = validator
    .validate_all()
    .expect("numerical validation failed");

println!("PSTD stable: {}", results.stability_tests.pstd_stable);
println!("FDTD stable: {}", results.stability_tests.fdtd_stable);
println!("PSTD phase error: {}", results.dispersion_tests.pstd_phase_error);
```

### Available Test Cases

1. **Homogeneous Propagation**: Validates basic wave propagation
2. **PML Absorption**: Tests boundary condition effectiveness
3. **Heterogeneous Medium**: Validates layered media simulation
4. **Nonlinear Propagation**: Tests harmonic generation
5. **Focused Transducer**: Validates phased array focusing
6. **Time Reversal**: Tests time reversal reconstruction

### Custom Validation Tests

For reference parity comparisons, configure the PSTD solver in reference mode:

```rust
use kwavers::solver::pstd::config::CompatibilityMode;
use kwavers::solver::pstd::PSTDConfig;

let mut config = PSTDConfig::default();
config.compatibility_mode = CompatibilityMode::Reference;
```

## Benchmark Suite

The automated benchmark suite provides comprehensive performance testing.

### Running Benchmarks

```rust
use kwavers::analysis::performance::ProductionBenchmarks;

let benchmarks = ProductionBenchmarks::new(100, 1000);
let results = benchmarks.run_all();

for result in results {
    println!("{}", result.report());
}
```

### Benchmark Components

- **PSTD Solver**: Pseudo-spectral time domain performance
- **FDTD Solver**: Finite-difference time domain performance
- **Kuznetsov Solver**: Nonlinear acoustics performance
- **AMR**: Adaptive mesh refinement efficiency
- **GPU**: GPU acceleration benchmarks

## Advanced Numerical Methods

### Adaptive Mesh Refinement (AMR)

```rust
use kwavers::domain::grid::Grid;
use kwavers::solver::amr::AMRSolver;
use ndarray::Array3;

let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");
let mut amr = AMRSolver::new(&grid, 3).expect("amr init failed");

let pressure_field: Array3<f64> = Array3::zeros((64, 64, 64));

amr.adapt_mesh(&pressure_field, 0.1).expect("amr mesh adaptation failed");

// Get statistics
let stats = amr.memory_stats();
println!("Octree nodes: {}", stats.nodes);
println!("Octree leaves: {}", stats.leaves);
```

### IMEX Time Integration

For problems with multiple time scales:

```rust
use kwavers::solver::forward::imex::{IMEXRKConfig, IMEXRKType, IMEXRK};

let config = IMEXRKConfig {
    scheme_type: IMEXRKType::ARK3,
    embedded_error: false,
};

let _scheme = IMEXRK::new(config);
```

### Spectral-DG Methods

For robust shock handling:

```rust
use kwavers::{HybridSpectralDGConfig, HybridSpectralDGSolver};
use kwavers::domain::grid::Grid;
use std::sync::Arc;

let config = HybridSpectralDGConfig {
    discontinuity_threshold: 0.01,
    spectral_order: 8,
    dg_polynomial_order: 4,
    adaptive_switching: true,
    conservation_tolerance: 1e-10,
};

let grid = Arc::new(Grid::new(128, 128, 128, 1.0, 1.0, 1.0).expect("grid creation failed"));
let _solver = HybridSpectralDGSolver::new(config, grid);
```

## Multi-Physics Simulations

### Fractional Derivative Absorption

Model realistic tissue absorption:

```rust
use kwavers::domain::medium::absorption::{AbsorptionCalculator, AbsorptionModel, TissueAbsorption, TissueType};
use ndarray::Array3;

let model = AbsorptionModel::Tissue(TissueAbsorption::new(TissueType::Liver));
let absorption = AbsorptionCalculator::new(model);

let mut pressure: Array3<f64> = Array3::zeros((64, 64, 64));
let frequency_hz = 1e6;
let dt = 1e-7;
absorption.apply_absorption(&mut pressure, frequency_hz, dt)?;
```

### Anisotropic Materials

Model directionally-dependent properties:

```rust
use kwavers::domain::medium::anisotropic::{AnisotropyType, StiffnessTensor};

// Transversely isotropic (muscle fiber)
let stiffness = StiffnessTensor::transversely_isotropic(
    11e9,   // C11
    5.5e9,  // C12
    2.5e9,  // C13
    2e9,    // C33
    3e9,    // C44
)
.expect("invalid stiffness tensor");

assert_eq!(stiffness.anisotropy_type, AnisotropyType::TransverselyIsotropic);
```

### Multi-Rate Integration

Efficiently handle multiple time scales:

```rust
use kwavers::solver::time_integration::TimeScaleSeparator;
use ndarray::Array4;

let mut separator = TimeScaleSeparator::new(&grid);
let fields = Array4::<f64>::zeros((kwavers::solver::TOTAL_FIELDS, grid.nx, grid.ny, grid.nz));
let time_scales = separator
    .analyze(&fields, 1e-12)
    .expect("time-scale analysis failed");

// Automatic sub-cycling
for scale in &time_scales {
    println!("Time scale: {:.2e}s", scale);
}
``` 

## Best Practices

### Memory Efficiency

1. Use `grid.zeros_array()` for array allocation (DRY principle)
2. Prefer iterators over index-based loops
3. Use slices and views instead of copying data

```rust
// Good: Iterator-based approach
field.indexed_iter_mut()
    .for_each(|((i, j, k), value)| {
        *value = compute_value(i, j, k);
    });

// Good: Slice operations
field.slice_mut(s![.., .., 0]).fill(0.0);
```

### Performance Optimization

1. Profile before optimizing
2. Use the benchmark suite to track improvements
3. Consider AMR for problems with localized features

```rust
// Profile critical sections
{
    let _scope = profiler.scope("critical_computation");
    // Your computation
}

println!("{}", report.performance_analysis);
```

### Validation

Always validate against known solutions:

```rust
use kwavers::solver::utilities::validation::NumericalValidator;

let validator = NumericalValidator::new();
let results = validator
    .validate_all()
    .expect("numerical validation failed");

println!("PSTD stable: {}", results.stability_tests.pstd_stable);
println!("FDTD stable: {}", results.stability_tests.fdtd_stable);
```

## Example: Complete Simulation Pipeline

Here's a complete example combining multiple advanced features:

```rust
use kwavers::analysis::performance::profiling::PerformanceProfiler;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::physics::plugin::PluginManager;
use kwavers::solver::fdtd::{FdtdConfig, FdtdPlugin};
use ndarray::Array4;

fn simulation_step_example() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let profiler = PerformanceProfiler::new(grid.clone());

    let mut plugin_manager = PluginManager::new();
    let fdtd_plugin = FdtdPlugin::new(FdtdConfig::default(), &grid).expect("plugin init failed");
    plugin_manager.add_plugin(Box::new(fdtd_plugin)).expect("add plugin failed");
    plugin_manager.initialize(&grid, &medium).expect("plugin init failed");

    let mut fields = Array4::zeros((kwavers::solver::TOTAL_FIELDS, grid.nx, grid.ny, grid.nz));
    let dt = 1e-7;
    let t = 0.0;

    {
        let _scope = profiler.scope("plugin_step");
        let _ = _scope;
    }

    let report = profiler.report();
    let _ = report;
}
```

## Troubleshooting

### Performance Issues

1. **Low grid updates/second**: Check roofline analysis for bottleneck
2. **High memory usage**: Enable AMR for sparse problems
3. **Poor cache performance**: Consider data layout optimization

### Validation Failures

1. **Numerical dispersion**: Increase grid resolution or use PSTD
2. **Boundary reflections**: Verify PML configuration
3. **Energy drift**: Check time step stability criteria

### Common Errors

```rust
// Error: Grid too coarse for wavelength
if grid.dx > wavelength / 10.0 {
    return Err(ValidationError::GridResolution(
        "Grid spacing must be < λ/10".to_string()
    ).into());
}

// Error: Time step too large
let cfl = c_max * dt / grid.dx;
if cfl > 0.5 {
    return Err(ValidationError::Stability(
        format!("CFL = {} > 0.5", cfl)
    ).into());
}
```

## Further Reading

- [Performance Optimization Guide](performance_optimization.md)
- [Numerical Methods Reference](numerical_methods.md)
- [Physics Models Documentation](physics_models.md)
- [API Reference](../api/index.html)
