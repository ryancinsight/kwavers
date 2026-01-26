# Source Localization Algorithms

This document provides an overview of the source localization algorithms implemented in the kwavers library.

## Overview

Source localization determines the spatial coordinates of acoustic sources from sensor array measurements. The library provides three complementary algorithms:

1. **Trilateration** - Basic geometric solution using time-of-arrival (TOA)
2. **Multilateration** - Advanced TDOA with least-squares optimization
3. **MUSIC** - Super-resolution subspace-based localization

## Algorithm Comparison

| Algorithm | Type | Min Sensors | Resolution | Computational Cost | Best Use Case |
|-----------|------|-------------|------------|-------------------|---------------|
| Trilateration | Geometric | 4 | λ/2 | Low (O(N)) | Real-time, single source |
| Multilateration | Optimization | 4+ | λ/2 | Medium (O(N²)) | Overdetermined systems, robust |
| MUSIC | Subspace | K+1 | Sub-wavelength | High (O(N³)) | High SNR, multi-source |

Where:
- N = number of sensors
- K = number of sources
- λ = acoustic wavelength

## 1. Trilateration

**File**: `src/analysis/signal_processing/localization/trilateration.rs`

### Mathematical Basis

Solves the geometric intersection of spheres centered at each sensor:

```
||r_s - r_i|| = c·(t_i - t_0)
```

Using time-difference-of-arrival (TDOA) to eliminate unknown emission time t_0.

### Key Features

- Gauss-Newton iterative solver
- Automatic initial guess (sensor centroid)
- Convergence monitoring
- Uncertainty estimation

### Usage Example

```rust
use kwavers::analysis::signal_processing::localization::{Trilateration, TrilaterationConfig};

let sensors = vec![
    [0.0, 0.0, 0.0],
    [0.01, 0.0, 0.0],
    [0.0, 0.01, 0.0],
    [0.0, 0.0, 0.01],
];

let config = TrilaterationConfig {
    sound_speed: 1540.0,  // m/s (soft tissue)
    ..Default::default()
};

let localizer = Trilateration::new(sensors, config)?;
let result = localizer.localize(&arrival_times)?;

println!("Source: {:?}", result.position);
println!("Uncertainty: ±{:.3} mm", result.uncertainty * 1000.0);
```

### Output

```rust
LocalizationResult {
    position: [x, y, z],        // m
    uncertainty: f64,            // m (std dev)
    residual: f64,               // m (RMS error)
    iterations: usize,
    converged: bool,
}
```

## 2. Multilateration

**File**: `src/analysis/signal_processing/localization/multilateration.rs`

### Mathematical Basis

Overdetermined TDOA system solved via weighted least squares:

```
(J^T W J + λI)·Δx = -J^T W r
```

where:
- J = Jacobian matrix of residuals
- W = weight matrix (inverse variance)
- λ = Levenberg-Marquardt damping parameter
- r = residual vector

### Key Features

- **Weighted Least Squares**: Accounts for heterogeneous sensor uncertainties
- **Levenberg-Marquardt**: Adaptive damping for robust convergence
- **GDOP Calculation**: Geometric Dilution of Precision metric
- **Iterative Refinement**: Corrects linearization errors

### Usage Example

```rust
use kwavers::analysis::signal_processing::localization::{Multilateration, MultilaterationConfig};

let sensors = vec![
    [0.0, 0.0, 0.0],
    [0.02, 0.0, 0.0],
    [0.0, 0.02, 0.0],
    [0.0, 0.0, 0.02],
    [0.01, 0.01, 0.0],
    [0.01, 0.0, 0.01],
];

let config = MultilaterationConfig {
    sound_speed: 1540.0,
    use_weighted_ls: true,
    max_iterations: 50,
    ..Default::default()
};

let mut multi = Multilateration::new(sensors, config)?;

// Optional: Set sensor uncertainties for weighted LS
let uncertainties = vec![1e-9, 1e-9, 5e-9, 5e-9, 5e-9, 5e-9];  // seconds
multi.set_sensor_uncertainties(uncertainties)?;

let result = multi.localize(&arrival_times)?;

// Calculate geometry quality
let gdop = multi.calculate_gdop(&result.position)?;
println!("GDOP: {:.2} (lower is better)", gdop);
```

### GDOP Interpretation

- **GDOP < 2**: Excellent geometry
- **GDOP 2-5**: Good geometry
- **GDOP 5-10**: Moderate geometry
- **GDOP > 10**: Poor geometry (consider repositioning sensors)

## 3. MUSIC Localization

**File**: `src/analysis/signal_processing/localization/music.rs`

### Mathematical Basis

MUltiple SIgnal Classification via eigendecomposition of sensor covariance:

```
R = U_s Λ_s U_s^H + U_n Λ_n U_n^H
```

MUSIC pseudospectrum:

```
P_MUSIC(θ) = 1 / ||U_n^H · a(θ)||²
```

where:
- R = sensor covariance matrix
- U_s = signal subspace (K largest eigenvectors)
- U_n = noise subspace (N-K smallest eigenvectors)
- a(θ) = steering vector at location θ

### Key Features

- **Super-Resolution**: Resolves sources closer than λ/2
- **Subspace Methods**: Exploits eigenstructure
- **Grid Search**: 3D spatial spectrum computation
- **Peak Detection**: Automatic multi-source identification
- **Covariance Estimation**: From snapshot data

### Usage Example

```rust
use kwavers::analysis::signal_processing::localization::{MusicLocalizer, MusicConfig};
use ndarray::Array2;
use num_complex::Complex;

let sensors = vec![
    [0.0, 0.0, 0.0],
    [wavelength, 0.0, 0.0],
    [0.0, wavelength, 0.0],
    [0.0, 0.0, wavelength],
];

let config = MusicConfig {
    frequency: 1e6,              // Hz
    sound_speed: 1500.0,         // m/s
    num_sources: 2,              // number of sources
    x_bounds: [0.0, 0.02],       // m
    y_bounds: [0.0, 0.02],       // m
    z_bounds: [0.0, 0.02],       // m
    grid_resolution: 0.0002,     // m (0.2 mm)
    peak_threshold: 0.5,         // relative to max
    ..Default::default()
};

let music = MusicLocalizer::new(sensors, config)?;

// Method 1: Directly from covariance matrix
let result = music.localize(&covariance)?;

// Method 2: Estimate covariance from snapshots
let covariance = music.estimate_covariance(&snapshots)?;
let result = music.localize(&covariance)?;

// Results
println!("Detected {} sources", result.sources.len());
for (i, pos) in result.sources.iter().enumerate() {
    println!("Source {}: {:?}", i+1, pos);
    println!("Peak value: {:.2}", result.peak_values[i]);
}

// Eigenvalues for diagnostics
println!("Eigenvalues: {:?}", result.eigenvalues);
```

### Output

```rust
MusicResult {
    sources: Vec<[f64; 3]>,          // Detected positions (m)
    peak_values: Vec<f64>,           // Pseudospectrum peaks
    spectrum: Vec<f64>,              // Full 3D spectrum (flattened)
    grid_shape: [usize; 3],          // [nx, ny, nz]
    signal_subspace_dim: usize,      // K
    eigenvalues: Vec<f64>,           // Sorted descending
}
```

## Performance Characteristics

### Computational Complexity

- **Trilateration**: O(N × I) where I = iterations (~10-50)
- **Multilateration**: O(N² × I) for Levenberg-Marquardt
- **MUSIC**: O(N³) for eigendecomposition + O(M) for grid search (M = grid points)

### Memory Requirements

- **Trilateration**: O(N)
- **Multilateration**: O(N²) for Jacobian storage
- **MUSIC**: O(N² + M) for covariance + spectrum

### Accuracy vs SNR

Localization error scales approximately as:

```
σ_loc ≈ λ / (2π √(SNR × N))
```

For high SNR and many sensors:
- Trilateration: ~λ/10
- Multilateration: ~λ/20 (with good geometry)
- MUSIC: ~λ/100 (super-resolution)

## Algorithm Selection Guide

### Use Trilateration when:
- Real-time processing is critical
- Single source localization
- Minimal computational resources
- Straightforward implementation needed

### Use Multilateration when:
- Overdetermined system (N > 4)
- Heterogeneous sensor quality
- Robust solution required
- GDOP analysis is important

### Use MUSIC when:
- Multiple sources need resolution
- High SNR available
- Sub-wavelength precision required
- Frequency is well-defined (narrowband)
- Computational resources available

## Common Pitfalls

### 1. Poor Sensor Geometry
**Problem**: Sensors in a line or plane
**Solution**: Use GDOP calculation to verify geometry
**Fix**: Add sensors to form 3D tetrahedral configuration

### 2. Time Synchronization Errors
**Problem**: Sensor clocks drift
**Solution**: Use weighted least squares with uncertainty weights
**Fix**: Periodic clock synchronization

### 3. Sound Speed Variations
**Problem**: Inhomogeneous medium
**Solution**: Use adaptive sound speed estimation
**Fix**: Measure local sound speed or use ray tracing

### 4. Multipath Interference
**Problem**: Reflections create ghost sources
**Solution**: Use coherence analysis or matched filtering
**Fix**: Apply spatial filtering or time-gating

## Testing

### Unit Tests

```bash
# Test individual algorithms
cargo test --lib analysis::signal_processing::localization::trilateration
cargo test --lib analysis::signal_processing::localization::multilateration
cargo test --lib analysis::signal_processing::localization::music
```

### Integration Tests

```bash
# Test all localization algorithms together
cargo test --test localization_integration
```

### Test Coverage

- ✅ Basic functionality (creation, validation)
- ✅ Single source localization
- ✅ Symmetric sensor arrays
- ✅ Off-axis sources
- ✅ Overdetermined systems
- ✅ Weighted least squares
- ✅ Noisy measurements
- ✅ GDOP calculation
- ✅ Covariance estimation
- ✅ Cross-algorithm validation

## References

### Trilateration
- Foy, W. H. (1976). "Position-Location Solutions by Taylor-Series Estimation"
  *IEEE Trans. Aerospace and Electronic Systems*, AES-12(2), 187-194

### Multilateration
- Smith, J. O., & Abel, J. S. (1987). "Closed-Form Least-Squares Source Location from Range Differences"
  *IEEE Trans. ASSP*, 35(12), 1661-1669
- Chan, Y. T., & Ho, K. C. (1994). "A Simple and Efficient Estimator for Hyperbolic Location"
  *IEEE Trans. Signal Processing*, 42(8), 1905-1915

### MUSIC
- Schmidt, R. O. (1986). "Multiple Emitter Location and Signal Parameter Estimation"
  *IEEE Trans. Antennas and Propagation*, 34(3), 276-280
- Stoica, P., & Nehorai, A. (1989). "MUSIC, Maximum Likelihood, and Cramér-Rao Bound"
  *IEEE Trans. ASSP*, 37(5), 720-741

## Future Enhancements

Planned additions (see TODO_AUDIT.md):

- [ ] Bayesian filtering for trajectory tracking
- [ ] Kalman filtering for continuous localization
- [ ] Particle filters for non-Gaussian tracking
- [ ] GPU-accelerated MUSIC grid search
- [ ] Adaptive grid refinement (coarse-to-fine)
- [ ] Multi-path rejection filters
- [ ] Wavefront curvature analysis

## License

Part of the kwavers library - see main LICENSE file.
