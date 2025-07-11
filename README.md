# KWAVERS - Acoustic Wave Simulation Framework

KWAVERS is a high-performance computational physics framework focused on simulating ultrasound wave propagation in heterogeneous biological media. It specializes in modeling nonlinear acoustic waves, thermal effects, and cavitation for applications in medical ultrasound, High-Intensity Focused Ultrasound (HIFU), and sonoluminescence.

## Key Features

- Nonlinear acoustic wave propagation using k-space and time-domain methods
- Linear elastic wave propagation in isotropic media (new!)
- Focused ultrasound simulation with various transducer geometries
- Heterogeneous tissue modeling with realistic acoustic properties (now including basic elastic properties)
- Cavitation dynamics and bubble-field interactions
- Thermal effects including heat diffusion and tissue absorption
- Light transport for sonoluminescence phenomena
- Parallelized computation using Rayon and ndarray for high performance

## Project Structure

```
kwavers/
├── src/
│   ├── boundary/             # Boundary conditions (PML, etc.)
│   ├── grid/                 # Grid representation and operations
│   ├── medium/               # Material properties and heterogeneous media
│   │   ├── heterogeneous/    # Spatially-varying media models
│   │   └── tissue/           # Tissue-specific acoustic properties
│   ├── physics/              # Physical models
│   │   ├── mechanics/        # Mechanical wave phenomena
│   │   │   ├── acoustic_wave/# Acoustic wave propagation models
│   │   │   └── cavitation/   # Bubble dynamics and cavitation
│   │   ├── thermal/          # Heat transfer and diffusion
│   │   ├── optics/           # Light propagation models
│   │   ├── chemical/         # Chemical reactions and transport
│   │   └── scattering/       # Scattering models
│   ├── solver/               # Main solver infrastructure
│   ├── source/               # Acoustic sources (transducers)
│   ├── time/                 # Time stepping and simulation control
│   └── utils/                # Utility functions (FFT, etc.)
├── examples/                 # Example simulations
└── benches/                  # Performance benchmarks
```

## Usage

KWAVERS simulations typically follow this workflow:

1. Define a computational grid
2. Create a medium with appropriate acoustic properties
3. Set up acoustic sources (transducers)
4. Configure boundary conditions
5. Set up a solver with physics components
6. Run the simulation
7. Analyze and visualize results

### Basic Example

```rust
use kwavers::{
    boundary::PMLBoundary,
    grid::Grid,
    medium::HomogeneousMedium,
    physics::mechanics::acoustic_wave::NonlinearWave,
    source::LinearArray,
    solver::Solver,
    time::Time,
};
use std::sync::Arc;

fn main() {
    // Create grid
    let grid = Grid::new(128, 128, 128, 0.0005, 0.0005, 0.0005);
    
    // Create medium
    let mut medium = HomogeneousMedium::new();
    medium.set_sound_speed(1500.0);
    let medium_arc = Arc::new(medium);
    
    // Set up source
    let frequency = 1.0e6;
    let source = LinearArray::new(/* parameters */);
    
    // Set up boundary
    let boundary = PMLBoundary::new(/* parameters */);
    
    // Configure time stepping
    let dt = 0.1e-6;
    let num_steps = 1000;
    let time = Time::new(dt, num_steps);
    
    // Create solver
    let mut solver = Solver::new(
        grid,
        time,
        medium_arc,
        Box::new(source),
        Box::new(boundary)
    );
    
    // Run simulation
    solver.run();
    
    // Process results
    // ...
}
```

## Performance

The framework is optimized for computational efficiency and uses parallel processing to accelerate simulations. Key performance features include:

- Parallelized array operations using Rayon
- Optimized memory access patterns
- Precomputation of frequently used values
- Efficient k-space FFT-based solvers
- Selective physics component activation to reduce computational load

### Performance Optimizations

The codebase has been extensively optimized to achieve high performance:

1. **Memory Management**
   - Pre-allocated arrays to avoid repeated allocations
   - Thread-local storage for temporary buffers
   - Lazy initialization of large data structures
   - Optimized memory access patterns for better cache utilization

2. **Parallel Processing**
   - Efficient parallelization using Rayon's parallel iterators
   - Chunked processing for better load balancing
   - Parallel FFT and array operations
   - Minimized thread synchronization points

3. **Computational Optimizations**
   - Precomputed coefficients for expensive calculations
   - Optimized mathematical operations (multiplication by inverse instead of division)
   - Reduced complex number operations
   - Branchless operations for better CPU pipeline utilization

4. **FFT Optimizations**
   - Cached FFT/IFFT instances for repeated grid sizes
   - Optimized buffer management to reduce allocations
   - Parallel complex number conversions
   - Enhanced Laplacian calculations

These optimizations result in significant performance improvements, with the most compute-intensive components (cavitation modeling, FFT operations, and nonlinear wave propagation) showing the largest speedups.

## License

This code is provided under the MIT License. See LICENSE file for details.
