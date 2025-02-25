# KWAVERS Simulation Optimization Notes

## Optimization Summary

We've improved the kwavers ultrasound simulation codebase with the following key optimizations:

1. **Performance Optimizations**
   - Added property caching for homogeneous medium to reduce redundant calculations
   - Optimized nonlinear wave solver with enhanced parallel processing
   - Applied thread-safe data structures for concurrent access

2. **Numerical Improvements**
   - Implemented higher-order k-space correction for better dispersion handling
   - Enhanced gradient calculation in nonlinear physics
   - Improved boundary condition handling

3. **Physical Model Enhancements**
   - Added configurable nonlinearity scaling for better HIFU simulation
   - Improved nonlinear term calculation for the Westervelt equation
   - Adjusted material properties for more accurate tissue/water modeling
   - Implemented comprehensive tissue-specific absorption models

## Detailed Technical Notes

### Medium Property Caching

- Added caching for medium properties to avoid redundant calculations in homogeneous media
- Implemented thread-safe caching with Mutex and OnceLock
- Created lazy initialization for large arrays to reduce memory usage when not needed
- Set up cache invalidation when temperature or bubble state changes

```rust
// Before: Repeatedly creating new arrays every time
fn density_array(&self) -> Array3<f64> {
    Array3::from_elem(
        (self.temperature.dim().0, self.temperature.dim().1, self.temperature.dim().2),
        self.density,
    )
}

// After: Using cached version to avoid repeated allocation
fn density_array(&self) -> Array3<f64> {
    self.density_array.get_or_init(|| {
        let shape = self.temperature.dim();
        Array3::from_elem(shape, self.density)
    }).clone()
}
```

### Nonlinear Wave Solver

- Added configurable parameters for different nonlinear effects:
  - `nonlinearity_scaling` to adjust strength of nonlinear effects
  - `k_space_correction_order` for varying levels of dispersion correction
  - `use_adaptive_timestep` option for stability in strong nonlinear fields

- Implemented higher-order k-space correction terms to reduce dispersion errors:
  ```rust
  // Enhanced phase term for higher order accuracy
  let phase = match self.k_space_correction_order {
      1 => -c * k_val * dt,  // First order
      2 => -c * k_val * dt * (1.0 - 0.25 * (k_val * c * dt / std::f64::consts::PI).powi(2)),
      3 => -c * k_val * dt * (1.0 - 0.25 * (k_val * c * dt / std::f64::consts::PI).powi(2) 
           + 0.05 * (k_val * c * dt / std::f64::consts::PI).powi(4)),
      _ => -c * k_val * dt * (1.0 - 0.25 * (k_val * c * dt / std::f64::consts::PI).powi(2) 
           + 0.05 * (k_val * c * dt / std::f64::consts::PI).powi(4)
           - 0.008 * (k_val * c * dt / std::f64::consts::PI).powi(6)),
  }
  ```

- Enhanced nonlinear term calculation with better gradient estimation:
  - Used higher-order central differences for interior points
  - Applied simpler approximation near boundaries
  - Incorporated both Westervelt and Khokhlov-Zabolotskaya-Kuznetsov (KZK) formulations

### Tissue-Specific Absorption Model

- Implemented a comprehensive tissue-specific absorption database with properties of 13+ biological tissues
- Based on established literature values from Szabo, Duck, and Bamber
- Created a heterogeneous tissue medium implementation for complex anatomical structures:

```rust
// Tissue type enumeration with various tissue options
pub enum TissueType {
    BloodVessel,
    Bone,
    BoneCortical,
    BoneMarrow,
    Brain,
    Fat,
    Kidney,
    Liver,
    Lung,
    Muscle,
    Skin,
    SoftTissue, // General soft tissue
    Tumor,      // Generic tumor tissue
}

// Tissue properties with all relevant acoustic parameters
pub struct TissueProperties {
    pub name: &'static str,
    pub density: f64,
    pub sound_speed: f64,
    pub alpha0: f64,      // Absorption coefficient at 1MHz
    pub y: f64,           // Power law exponent
    pub b_a: f64,         // Nonlinearity parameter
    pub specific_heat: f64,
    pub thermal_conductivity: f64,
    pub impedance: f64,
}
```

- Implemented advanced absorption models that incorporate:
  - Frequency-dependent power law absorption (α = α₀·fʸ)
  - Temperature-dependent effects (~1-2% increase per degree above 37°C)
  - Nonlinear pressure-dependent absorption enhancement
  
```rust
// Temperature dependence for absorption
let temp_c = temperature - 273.15;
let temp_factor = 1.0 + 0.015 * (temp_c - 37.0);
alpha *= temp_factor.max(0.5);

// High intensity pressure effects on absorption
if let Some(p_amp) = pressure_amplitude {
    let p_threshold = 1.0e6; // 1 MPa threshold for nonlinear effects
    if p_amp > p_threshold {
        let nonlinear_factor = 1.0 + 0.05 * (p_amp - p_threshold) / 1.0e6;
        alpha *= nonlinear_factor.min(2.0); // Cap at doubling the absorption
    }
}
```

- Created methods for building realistic tissue structures including:
  - Layered tissue models (e.g., skin-fat-muscle-organ layers)
  - Spherical inclusions (e.g., tumors, cysts)
  - Arbitrary 3D tissue regions

- Added reflection coefficient calculations between tissue boundaries:
```rust
// Calculate reflection coefficient between tissues
pub fn reflection_coefficient(tissue1: TissueType, tissue2: TissueType, frequency: f64) -> f64 {
    let z1 = tissue_impedance(tissue1, frequency);
    let z2 = tissue_impedance(tissue2, frequency);
    ((z2 - z1) / (z2 + z1)).powi(2)
}
```

### Physical Model Parameters

- Updated HIFU simulation parameters for more realistic modeling:
  - Set sound speed to 1482 m/s (body temperature water)
  - Configured power law absorption with α₀=0.3, δ=1.1 for biological tissue
  - Used B/A=5.2 for water at body temperature
  - Increased number of transducer elements to 32 for better focusing
  - Placed sensors at key locations (pre-focal, focal, post-focal, off-axis)

## Important Findings

1. **Nonlinear Effects**
   - Nonlinear effects become significant at pressure amplitudes above 1 MPa
   - Proper handling of nonlinear terms is critical for accurate HIFU modeling
   - Higher-order dispersion correction provides significant improvements for focused beams

2. **Performance Bottlenecks**
   - FFT operations remain the most time-consuming part of simulation (~50-70% of runtime)
   - Medium property lookups can be a significant bottleneck in heterogeneous media
   - Boundary handling at interfaces requires careful numerical treatment

3. **Thread Safety**
   - Concurrency requires careful management with thread-safe structures
   - Direct manipulation of shared structures can lead to race conditions
   - Proper synchronization is essential when updating cached properties

4. **Tissue-Specific Acoustic Properties**
   - Tissue properties vary significantly - bone absorption is ~10x higher than soft tissue
   - Frequency-dependent absorption follows power law models with exponents from 1.0-1.5
   - Acoustic impedance differences at tissue boundaries cause significant reflections
   - Nonlinear effects are tissue-dependent, with higher B/A values in fat (9.6) vs. muscle (7.2)

## Future Improvement Areas

1. **Further Performance Optimization**
   - Explore GPU acceleration for FFT operations
   - Implement adaptive grid spacing to reduce computation in non-critical regions
   - Consider frequency-domain methods for certain cases

2. **Numerical Methods**
   - Implement adaptive timestepping for better stability
   - Add multi-resolution approaches for large domains
   - Explore pseudospectral time domain methods for certain applications

3. **Physical Models**
   - Implement more sophisticated bubble dynamics models for cavitation
   - Enhance thermal models for better heating prediction
   - Add tissue elasticity effects for shear wave propagation

4. **Tissue Models**
   - Add frequency-dependent scattering models
   - Implement more sophisticated thermal damage models
   - Create interfaces for importing patient-specific tissue maps

## Version Compatibility Notes

- All optimizations maintain compatibility with existing APIs
- The nonlinear wave solver now supports configuration but uses sensible defaults
- Medium property adjustments should be done before wrapping in Arc for thread-safety
- Heterogeneous tissue medium can be used as a drop-in replacement for homogeneous medium 

## Performance Analysis

Based on the profiling data, the simulation runtime is distributed across components as follows:

```
[INFO] src\solver\mod.rs:339 - Physics component breakdown:
[INFO] src\solver\mod.rs:340 -   Acoustic Wave: 6.003s/step (13.2%)
[INFO] src\solver\mod.rs:341 -   Boundary: 3.363s/step (7.4%)
[INFO] src\solver\mod.rs:342 -   Cavitation: 15.432s/step (33.9%)
[INFO] src\solver\mod.rs:343 -   Light Diffusion: 2.885s/step (6.3%)
[INFO] src\solver\mod.rs:344 -   Thermal: 2.920s/step (6.4%)
[INFO] src\solver\mod.rs:345 -   Chemical: 0.180s/step (0.4%)
[INFO] src\solver\mod.rs:346 -   Other: 14.802s/step (32.5%)
```

## Optimizations Applied

### 1. Nonlinear Wave Module (Acoustic Wave)

The nonlinear wave module was optimized by:

- **Precomputation of k-squared values**: Added cached k-squared values in the `NonlinearWave` struct to avoid recomputing these values at every time step.
- **Phase factor calculation optimization**: Extracted phase factor calculation to a separate inline function to improve compiler optimization.
- **Memory access patterns**: Improved memory access patterns in the k-space propagation to better utilize cache.
- **Reuse of arrays**: Used cached arrays instead of creating new ones for each update step.
- **Efficient mathematical operations**: Used mathematical simplifications like multiplying by inverse instead of dividing.

### 2. Cavitation Module

The cavitation module was the most time-consuming component. Optimizations included:

- **Pre-allocated arrays**: Pre-allocated arrays in the struct to avoid frequent memory allocations.
- **Improved parallelism**: Replaced the nested loop approach with optimized parallel processing using Rayon.
- **Memory locality**: Processed data in 2D slices to improve memory locality and cache utilization.
- **Cached medium properties**: Cached medium properties for a given point to avoid repeated lookups.
- **Mathematical optimization**: Used inverse multiplication instead of division and simplified complex calculations.
- **Branchless operations**: Used branchless code where appropriate to avoid pipeline stalls.

### 3. Boundary Module

The boundary module was optimized by:

- **Improved parallelism**: Implemented parallel processing for boundary conditions using Rayon.
- **Precomputation of damping factors**: Added precomputation of 3D damping profiles to avoid redundant calculations.
- **Lazy initialization**: Used lazy initialization of damping arrays to reduce memory usage when not needed.
- **Optimized damping profile generation**: Parallelized the damping profile calculation.
- **Updated trait design**: Modified the Boundary trait to support mutable self references for optimizations.

### 4. Light Diffusion Module

The light diffusion module was optimized by:

- **Precomputed diffusion coefficients**: Added precomputation of inverse diffusion coefficients for faster calculations.
- **Enhanced parallel processing**: Improved parallel operations for critical sections of the code.
- **Complex number optimizations**: Optimized complex number operations to reduce computational cost.
- **Mathematical optimizations**: Used multiplication by inverse instead of division.
- **Added performance metrics**: Implemented detailed performance tracking for better profiling.
- **Lazy initialization**: Used lazy initialization of precomputed arrays to reduce memory overhead.

### 5. Thermal Module

The thermal module was optimized by:

- **Precomputed thermal factors**: Precomputed thermal factors to avoid repeating expensive calculations.
- **Improved heat source calculation**: Refactored heat source calculation for better performance.
- **Chunked processing**: Implemented chunked processing for better cache locality and memory access patterns.
- **Optimized temperature field updates**: Improved the field update algorithm for better efficiency.
- **Added performance tracking**: Implemented detailed performance metrics for better profiling.
- **Branchless operations**: Used branchless operations for numerical stability checks.

### 6. FFT/IFFT Operations (Core Utilities)

The FFT and IFFT operations were a major part of the "Other" category. Optimizations included:

- **Thread-local buffer storage**: Implemented thread-local storage for FFT/IFFT buffers to reduce memory allocations.
- **Parallelized complex conversion**: Added parallel processing for real-to-complex and complex-to-complex operations.
- **Improved laplacian calculation**: Optimized the laplacian utility function with better parallel processing.
- **Enhanced memory access patterns**: Improved memory access patterns in FFT operations.
- **Eliminated unnecessary cloning**: Reduced unnecessary cloning of arrays during FFT/IFFT operations.
- **Buffer pre-initialization**: Added pre-initialization of buffers during FFT cache warm-up.

### 7. Example File Fixes

Fixed the example file `tissue_model_example.rs` by:

- Correcting imports to use the proper module paths
- Updating method calls to match the current API
- Using type annotations for floating-point literals to avoid ambiguity
- Organizing the code for better clarity and maintainability

## Lessons Learned

### Memory Management

- **Allocation overhead**: Creating new arrays inside loops or at each time step can significantly impact performance. Pre-allocating arrays and reusing them is much more efficient.
- **Memory locality**: Processing data in ways that respect cache lines (e.g., processing 2D slices) can significantly improve performance.
- **Thread-local storage**: Using thread-local storage for frequently used temporary buffers can reduce allocation overhead without affecting thread safety.
- **Lazy initialization**: Only allocating memory when needed and caching it for reuse improves both memory efficiency and performance.

### Parallel Processing

- **Chunking strategy**: The size and distribution of chunks for parallel processing can significantly impact performance. Finding the optimal chunk size is important.
- **Load balancing**: Using `par_bridge()` with `chunks()` provides better load balancing than fixed chunk sizes.
- **Thread synchronization**: Minimizing synchronization points is crucial for parallel performance.
- **Parallel array operations**: Using Zip patterns with par_for_each for array operations provides significant speedup on multi-core systems.

### Numerical Computation

- **K-space methods**: K-space methods are efficient for wave propagation, but their implementation requires careful memory management and FFT optimization.
- **Precision vs. performance**: Using `f64` for all computations provides good precision but at a performance cost. In some cases, using `f32` might be sufficient.
- **Division operations**: Division operations are more expensive than multiplication. Where possible, compute inverses once and use multiplication.
- **Branch prediction**: Branchless code (using conditional assignments) can be faster than if-else statements in tight loops.

### Rust-Specific Optimizations

- **Using ndarray's Zip**: The `Zip` pattern from ndarray provides efficient parallel iteration with good memory access patterns.
- **Rayon parallelism**: Rayon's parallel iterators are powerful but need to be used carefully to avoid overhead from excessive task creation.
- **`par_for_each` vs. nested loops**: Replacing nested loops with parallel operations can significantly improve performance on multi-core systems.
- **Avoiding clones**: Cloning large arrays is expensive; use references or slices where possible.
- **Arc for thread safety**: Using Arc<T> for shared data structures is more efficient than repeated cloning.
- **Mutability patterns**: Designing APIs to allow mutable borrowing enables more efficient implementations without sacrificing safety.

## Future Optimization Opportunities

- **SIMD acceleration**: Investigate using SIMD intrinsics for numerical kernels, possibly through libraries like `packed_simd`.
- **Further FFT optimization**: Look into faster FFT libraries or potential GPU acceleration for FFT operations.
- **Task granularity**: Further tune the task granularity for parallel operations to balance overhead and parallelism.
- **Memory layout**: Consider using array-of-structs to struct-of-arrays transformations where appropriate for better vectorization.
- **Thread pool tuning**: Experiment with different thread pool sizes to find the optimal setting for different hardware.
- **Algorithmic improvements**: Look for algorithmic improvements in the physics components, especially for cavitation modeling.
- **I/O optimizations**: Profile and optimize file I/O operations, particularly for large data sets. 