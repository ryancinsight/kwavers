# Adaptive Mesh Refinement (AMR) Usage Guide

## Overview

The Adaptive Mesh Refinement (AMR) module in Kwavers provides dynamic grid refinement for efficient simulation of multi-scale phenomena. It automatically refines the computational mesh in regions requiring higher resolution while maintaining coarser grids elsewhere, achieving 60-80% memory reduction and 2-5x performance improvements.

## Key Features

- **Wavelet-based error estimation**: Detects regions requiring refinement using multi-resolution analysis
- **Octree-based 3D refinement**: Efficient spatial hierarchy for managing refined regions
- **Conservative interpolation**: Preserves physical quantities during refinement/coarsening
- **Multiple interpolation schemes**: Linear, cubic, and conservative methods
- **Flexible refinement criteria**: Customizable thresholds and buffer zones

## Basic Usage

### 1. Creating an AMR Manager

```rust
use kwavers::domain::grid::Grid;
use kwavers::solver::amr::AMRSolver;

// Create base grid
let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");

// Initialize AMR solver with a maximum refinement level
let max_level = 5;
let mut amr = AMRSolver::new(&grid, max_level)?;
```

### 2. Adapting the Mesh

```rust
use ndarray::Array3;

// Your solution field (e.g., pressure, velocity)
let solution: Array3<f64> = compute_solution();

// Adapt mesh based on solution features
let threshold = 1e-3;
amr.adapt_mesh(&solution, threshold)?;
```

### 3. Monitoring Memory Usage

```rust
let stats = amr.memory_stats();
println!("Octree nodes: {}", stats.nodes);
println!("Octree leaves: {}", stats.leaves);
println!("Octree memory: {:.2} MB", stats.memory_bytes as f64 / (1024.0 * 1024.0));
```

## Advanced Configuration

### Wavelet Types

Choose the appropriate wavelet based on your solution characteristics:

- **Haar**: Simplest, good for discontinuities and sharp fronts
- **Daubechies(N)**: Compact support; select order `N` via `WaveletBasis::Daubechies(N)`
- **CDF(p, q)**: Biorthogonal CDF wavelets via `WaveletBasis::CDF(p, q)`

### Interpolation Schemes

The AMR interpolator supports the following schemes internally:

- **Linear**
- **Cubic**
- **Conservative**

`AMRSolver` constructs a `ConservativeInterpolator` using the conservative scheme by default.

```rust
use kwavers::solver::amr::ConservativeInterpolator;
use ndarray::Array3;

let interpolator = ConservativeInterpolator::new();
let coarse = Array3::from_elem((4, 4, 4), 1.0);

let fine = interpolator.prolongate(&coarse);
let coarse_again = interpolator.restrict(&fine);

assert_eq!(fine.dim(), (8, 8, 8));
assert_eq!(coarse_again.dim(), (4, 4, 4));
```

### Custom Error Estimators

```rust
use kwavers::solver::amr::ErrorEstimator;

let estimator = ErrorEstimator::new();
let error = estimator.estimate_error(&solution)?;
let _ = error;
```

## Integration with Solver

### Time-Stepping with AMR

```rust
// Main simulation loop
for step in 0..n_steps {
    // Adapt mesh every N steps or based on error
    if step % adapt_interval == 0 {
        amr.adapt_mesh(&fields, threshold)?;
    }
    
    // Advance solution
    advance_timestep(&mut fields, dt);
}
```

### Multi-Physics with AMR

```rust
// Different refinement criteria for different physics
let acoustic_error = estimate_acoustic_error(&pressure);
let thermal_error = estimate_thermal_error(&temperature);
let chemical_error = estimate_chemical_error(&concentration);

// Combined refinement criterion (field-valued)
let combined_error = acoustic_error * 0.5 + thermal_error * 0.3 + chemical_error * 0.2;

amr.adapt_mesh(&combined_error, 0.1)?;
```

## Performance Optimization

### 1. Refinement Frequency

Balance accuracy and performance by adjusting adaptation frequency:

```rust
// Adaptive refinement interval based on solution dynamics
let adapt_interval = if max_error > 0.1 {
    1  // Refine every step for rapidly changing solutions
} else if max_error > 0.01 {
    5  // Refine every 5 steps for moderate changes
} else {
    10 // Refine every 10 steps for slowly varying solutions
};
```

### 2. Buffer Zone Optimization

Adjust buffer cells based on wave propagation:

```rust
// Larger buffer for high-speed phenomena
config.buffer_cells = (wave_speed * dt / dx).ceil() as usize + 1;
```

### 3. Level Limiting

Prevent over-refinement in specific regions:

```rust
// Limit refinement near boundaries
if near_boundary(i, j, k) {
    max_local_level = config.max_level - 1;
}
```

## Example: Focused Ultrasound Simulation

```rust
use kwavers::domain::grid::Grid;
use kwavers::solver::amr::AMRSolver;
use ndarray::Array3;

fn simulate_focused_ultrasound() -> KwaversResult<()> {
    let grid = Grid::new(256, 256, 256, 0.5e-3, 0.5e-3, 0.5e-3).expect("grid creation failed");
    let mut amr = AMRSolver::new(&grid, 6)?;
    let threshold = 1e-4;
    
    // Initialize fields
    let mut pressure = Array3::zeros((256, 256, 256));
    let mut velocity = Array3::zeros((256, 256, 256));
    
    // Time stepping
    for step in 0..n_steps {
        // Adapt based on pressure gradients
        if step % 5 == 0 {
            amr.adapt_mesh(&pressure, threshold)?;
        }
        
        // Update fields...
        update_pressure(&mut pressure, &velocity, dt);
        update_velocity(&mut velocity, &pressure, dt);
    }
    
    // Final statistics
    let stats = amr.memory_stats();
    println!("Simulation complete:");
    println!("  Octree nodes: {}", stats.nodes);
    println!("  Octree leaves: {}", stats.leaves);
    
    Ok(())
}
```

## Troubleshooting

### Common Issues

1. **Excessive refinement**: Increase `threshold` or reduce `max_level`
2. **Insufficient resolution**: Decrease `threshold` or increase `max_level`
3. **Memory usage**: Reduce `max_level`
4. **Oscillations**: Prefer gradient-based criteria and smoother thresholds

### Performance Profiling

```rust
// Enable AMR profiling
let start = Instant::now();
amr.adapt_mesh(&field, threshold)?;
let adapt_time = start.elapsed();

println!("Adaptation took: {:?}", adapt_time);
println!("Updates/second: {:.0}", 1.0 / adapt_time.as_secs_f64());
```

## Best Practices

1. **Start Conservative**: Begin with lower `max_level` and tighter thresholds
2. **Monitor Convergence**: Track how refinement affects solution accuracy
3. **Balance Resources**: Trade-off between accuracy and computational cost
4. **Validate Conservation**: Use conservative interpolation for physical quantities
5. **Profile Performance**: Measure speedup compared to uniform fine grid

## References

- Berger, M. J., & Oliger, J. (1984). "Adaptive mesh refinement for hyperbolic partial differential equations"
- Popinet, S. (2003). "Gerris: a tree-based adaptive solver for the incompressible Euler equations"
- Teyssier, R. (2002). "Cosmological hydrodynamics with adaptive mesh refinement"
