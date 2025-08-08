# Advanced Features Tutorial

This tutorial covers the advanced features of Kwavers, including performance profiling, k-Wave validation, and benchmark suites.

## Table of Contents

1. [Performance Profiling](#performance-profiling)
2. [k-Wave Validation](#k-wave-validation)
3. [Benchmark Suite](#benchmark-suite)
4. [Advanced Numerical Methods](#advanced-numerical-methods)
5. [Multi-Physics Simulations](#multi-physics-simulations)

## Performance Profiling

The performance profiling infrastructure provides comprehensive insights into your simulation's performance characteristics.

### Basic Usage

```rust
use kwavers::performance::profiling::{PerformanceProfiler, ProfileReport};
use kwavers::{Grid, HomogeneousMedium};

// Create profiler
let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
let profiler = PerformanceProfiler::new(&grid);

// Profile a computation
{
    let _scope = profiler.time_scope("pressure_update");
    // Your computation here
}

// Record memory events
profiler.record_memory_event(
    1024 * 1024, 
    MemoryEventType::Allocation,
    Some("field allocation".to_string())
);

// Generate report
let report = profiler.generate_report()?;
report.print_summary();
```

### Roofline Analysis

The profiler includes roofline analysis to identify performance bottlenecks:

```rust
// The report includes roofline analysis
println!("Performance bound: {:?}", report.roofline.bound_type);
println!("Arithmetic intensity: {:.2} FLOP/byte", report.roofline.arithmetic_intensity);
println!("Achieved: {:.1} GFLOP/s", report.roofline.achieved_gflops);
```

### Cache Profiling

Monitor cache behavior for optimization:

```rust
profiler.update_cache_stats(|stats| {
    stats.l1_hits += 1000;
    stats.l1_misses += 10;
});
```

## k-Wave Validation

Validate your simulations against the industry-standard k-Wave toolbox.

### Running Validation Tests

```rust
use kwavers::solver::validation::KWaveValidator;

let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
let validator = KWaveValidator::new(grid);

// Run all tests
let report = validator.run_all_tests()?;
report.print_summary();

// Check if all tests passed
if report.all_passed() {
    println!("All k-Wave validation tests passed!");
}
```

### Available Test Cases

1. **Homogeneous Propagation**: Validates basic wave propagation
2. **PML Absorption**: Tests boundary condition effectiveness
3. **Heterogeneous Medium**: Validates layered media simulation
4. **Nonlinear Propagation**: Tests harmonic generation
5. **Focused Transducer**: Validates phased array focusing
6. **Time Reversal**: Tests time reversal reconstruction

### Custom Validation Tests

Create your own validation tests:

```rust
use kwavers::solver::validation::{KWaveTestCase, ReferenceSource};

let custom_test = KWaveTestCase {
    name: "my_test".to_string(),
    description: "Custom validation test".to_string(),
    tolerance: 1e-3,
    reference: ReferenceSource::Literature("Paper et al. 2024".to_string()),
};
```

## Benchmark Suite

The automated benchmark suite provides comprehensive performance testing.

### Running Benchmarks

```rust
use kwavers::benchmarks::{BenchmarkSuite, BenchmarkConfig, OutputFormat};

// Configure benchmarks
let config = BenchmarkConfig {
    grid_sizes: vec![64, 128, 256],
    time_steps: 100,
    iterations: 5,
    enable_gpu: true,
    enable_amr: true,
    output_format: OutputFormat::Markdown,
};

// Run benchmarks
let mut suite = BenchmarkSuite::new(config);
let report = suite.run_all()?;
report.print_summary();

// Export results
report.export_csv("benchmark_results.csv")?;
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
use kwavers::solver::amr::{AMRManager, AMRConfig, WaveletType};

let config = AMRConfig {
    max_level: 3,
    min_level: 0,
    refine_threshold: 0.1,
    coarsen_threshold: 0.01,
    refinement_ratio: 2,
    buffer_cells: 2,
    wavelet_type: WaveletType::Daubechies4,
    interpolation_scheme: InterpolationScheme::Conservative,
};

let mut amr = AMRManager::new(config, &grid)?;

// Adapt mesh based on field
amr.adapt_mesh(&pressure_field)?;

// Get statistics
let stats = amr.get_statistics();
println!("Compression ratio: {:.2}", stats.compression_ratio);
```

### IMEX Time Integration

For problems with multiple time scales:

```rust
use kwavers::solver::imex::{IMEXScheme, IMEXRKConfig, IMEXRKType};

let config = IMEXRKConfig {
    scheme_type: IMEXRKType::ARK3,
    adaptive_timestepping: true,
    tolerance: 1e-6,
    safety_factor: 0.9,
    max_timestep_ratio: 2.0,
};

let imex_scheme = IMEXScheme::IMEXRK(config);
```

### Spectral-DG Methods

For robust shock handling:

```rust
use kwavers::solver::spectral_dg::{HybridSpectralDGConfig, HybridSpectralDG};

let config = HybridSpectralDGConfig {
    discontinuity_threshold: 0.01,
    spectral_order: 8,
    dg_polynomial_order: 4,
    adaptive_switching: true,
    conservation_tolerance: 1e-10,
};

let solver = HybridSpectralDG::new(config, &grid)?;
```

## Multi-Physics Simulations

### Fractional Derivative Absorption

Model realistic tissue absorption:

```rust
use kwavers::medium::absorption::FractionalDerivativeAbsorption;

let absorption = FractionalDerivativeAbsorption::new(
    1.1,    // Power law exponent for liver
    0.5,    // α₀ at 1 MHz (Np/m)
    1e6,    // Reference frequency (Hz)
);

// Apply to simulation
absorption.apply(&mut pressure, &grid, dt)?;
```

### Anisotropic Materials

Model directionally-dependent properties:

```rust
use kwavers::medium::anisotropic::{StiffnessTensor, AnisotropyType};

// Transversely isotropic (muscle fiber)
let stiffness = StiffnessTensor::transversely_isotropic(
    11e9,   // C11
    5.5e9,  // C13
    2.5e9,  // C33
    2e9,    // C44
    3e9,    // C66
);

let propagator = AnisotropicWavePropagator::new(stiffness, density, &grid)?;
```

### Multi-Rate Integration

Efficiently handle multiple time scales:

```rust
use kwavers::solver::time_integration::TimeScaleSeparator;

let separator = TimeScaleSeparator::new(&grid);
let time_scales = separator.analyze_system(&fields, &medium)?;

// Automatic sub-cycling
for scale in &time_scales {
    if scale.is_stiff {
        println!("Stiff component: {} (τ = {:.2e}s)", scale.component, scale.time_scale);
    }
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
    let _scope = profiler.time_scope("critical_computation");
    // Your computation
}

// Check performance bounds
if matches!(report.roofline.bound_type, PerformanceBound::MemoryBound) {
    println!("Consider cache blocking or data layout optimization");
}
```

### Validation

Always validate against known solutions:

```rust
// Run physics validation tests
use kwavers::physics::validation_tests;

// Validate wave equation
validation_tests::test_1d_wave_equation_analytical()?;

// Validate nonlinear acoustics
validation_tests::test_kuznetsov_second_harmonic()?;
```

## Example: Complete Simulation Pipeline

Here's a complete example combining multiple advanced features:

```rust
use kwavers::*;
use kwavers::performance::profiling::PerformanceProfiler;
use kwavers::solver::amr::{AMRManager, AMRConfig};
use kwavers::physics::mechanics::acoustic_wave::KuznetsovWave;

fn advanced_simulation() -> KwaversResult<()> {
    // Setup
    let grid = Grid::new(256, 256, 256, 1e-3, 1e-3, 1e-3);
    let profiler = PerformanceProfiler::new(&grid);
    
    // Configure AMR
    let amr_config = AMRConfig::default();
    let mut amr = AMRManager::new(amr_config, &grid)?;
    
    // Configure nonlinear solver
    let kuznetsov_config = KuznetsovConfig {
        enable_nonlinearity: true,
        enable_diffusivity: true,
        ..Default::default()
    };
    let mut solver = KuznetsovWave::new(&grid, kuznetsov_config);
    
    // Initialize
    let mut pressure = grid.zeros_array();
    initialize_focused_source(&mut pressure, &grid);
    
    // Time stepping
    let dt = 1e-6;
    let n_steps = 1000;
    
    for step in 0..n_steps {
        // Profile time step
        let _scope = profiler.time_scope("time_step");
        
        // Adaptive mesh refinement
        if step % 10 == 0 {
            amr.adapt_mesh(&pressure)?;
        }
        
        // Solve
        solver.update(&pressure, &medium, dt)?;
        
        // Record statistics
        if step % 100 == 0 {
            let stats = amr.get_statistics();
            println!("Step {}: compression ratio = {:.2}", 
                    step, stats.compression_ratio);
        }
    }
    
    // Generate performance report
    let report = profiler.generate_report()?;
    report.print_summary();
    
    Ok(())
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