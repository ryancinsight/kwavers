# Hybrid Methods in Kwavers - Coupling Strategies Guide

## Overview

Kwavers implements several hybrid numerical methods for coupling different discretization techniques. Each method is optimized for specific problem characteristics and offers unique advantages for acoustic wave simulation.

## Available Hybrid Methods

### 1. FDTD-FEM Coupling
**File**: `src/solver/forward/hybrid/fdtd_fem_coupling.rs`

#### When to Use
- **Multi-scale problems**: Fine-scale features in localized regions
- **Complex geometries**: Sharp edges, corners, or small features
- **Adaptive resolution**: Different grid densities in different regions
- **Time-domain problems**: When temporal evolution is important

#### Mathematical Foundation
```text
FDTD Domain: ∂²u/∂t² = c²∇²u + f    (structured grid)
FEM Domain:  ∇²u + k²u = f          (unstructured mesh)
Interface:   u₁ = u₂, ∂u₁/∂n = ∂u₂/∂n
```

#### Advantages
- ✅ Handles complex geometries in FEM domain
- ✅ Efficient for large uniform regions (FDTD)
- ✅ Natural time-domain evolution
- ✅ Conservative field transfer via Schwarz method

#### Disadvantages
- ❌ Higher computational cost than single-method approaches
- ❌ Requires careful interface treatment for stability

#### Example Use Cases
- Ultrasound transducer modeling with complex piezoelectric elements
- Scattering from objects with fine geometric features
- Waveguides with varying cross-sections

---

### 2. PSTD-SEM Coupling
**File**: `src/solver/forward/hybrid/pstd_sem_coupling.rs`

#### When to Use
- **High-accuracy requirements**: Spectral convergence needed
- **Smooth solutions**: Solutions without sharp discontinuities
- **Geometric flexibility**: Complex domains requiring unstructured meshes
- **Frequency-domain problems**: Steady-state harmonic analysis

#### Mathematical Foundation
```text
PSTD: Global spectral (FFT-based) - O(N log N) complexity
SEM:  Local spectral elements - exponential convergence
Interface: Modal basis transformation for field continuity
```

#### Advantages
- ✅ Exponential convergence (spectral accuracy)
- ✅ Both methods leverage FFT for efficiency
- ✅ Natural coupling between spectral representations
- ✅ Superior accuracy for smooth wave fields

#### Disadvantages
- ❌ Less robust for discontinuous solutions (shocks, sharp edges)
- ❌ Higher memory requirements than low-order methods
- ❌ Complex implementation of modal transformations

#### Example Use Cases
- High-precision acoustic scattering calculations
- Ultrasound imaging with phase-sensitive applications
- Room acoustics and reverberation analysis

---

### 3. BEM-FEM Coupling
**File**: `src/solver/forward/hybrid/bem_fem_coupling.rs`

#### When to Use
- **Unbounded domains**: Radiation to infinity (exterior problems)
- **Complex interior geometry**: Detailed modeling of internal structures
- **Scattering problems**: Objects in infinite media
- **Natural boundary conditions**: Automatic radiation conditions

#### Mathematical Foundation
```text
FEM Domain:  ∇²u + k²u = f    in Ω (bounded, complex geometry)
BEM Domain:  ∫_Γ G(x,y) ∂u/∂n ds = u(x)    on Γ (unbounded exterior)
Interface:   u₁ = u₂, ∂u₁/∂n = ∂u₂/∂n on Γ
```

#### Advantages
- ✅ Automatic radiation boundary conditions (no artificial boundaries)
- ✅ Exact far-field radiation patterns
- ✅ Efficient for exterior scattering problems
- ✅ Handles unbounded domains naturally

#### Disadvantages
- ❌ More complex implementation than single-domain methods
- ❌ Requires careful treatment of hypersingular integrals
- ❌ Limited to linear problems (BEM formulation)

#### Example Use Cases
- Underwater acoustics (submarines, sonar systems)
- Medical ultrasound in tissue (unbounded propagation)
- Electromagnetic scattering from antennas
- Seismic wave propagation in layered earth models

---

## Method Comparison Matrix

| Method | Accuracy | Efficiency | Geometry | Boundaries | Time/Freq |
|--------|----------|------------|----------|------------|-----------|
| FDTD-FEM | 2nd-4th order | High | Complex | Artificial | Time |
| PSTD-SEM | Spectral | High | Complex | Artificial | Frequency |
| BEM-FEM | High | Medium | Complex | Natural | Frequency |

## Choosing the Right Hybrid Method

### Decision Flowchart

```
Start: Acoustic Wave Problem
├── Time-domain evolution needed?
│   ├── Yes → FDTD-FEM coupling
│   └── No  → Frequency-domain analysis
│       ├── Unbounded domain?
│       │   ├── Yes → BEM-FEM coupling
│       │   └── No  → Continue
│       └── Spectral accuracy needed?
│           ├── Yes → PSTD-SEM coupling
│           └── No  → Single-method approach
```

### Problem Characteristics Guide

#### For Complex Geometry + Unbounded Domain
- **Primary Choice**: BEM-FEM coupling
- **Why**: BEM handles exterior perfectly, FEM handles interior complexity
- **Example**: Ultrasound probe in infinite tissue medium

#### For High Accuracy + Complex Geometry
- **Primary Choice**: PSTD-SEM coupling
- **Why**: Spectral convergence with geometric flexibility
- **Example**: Precise scattering cross-section calculations

#### For Multi-Scale Problems + Time Evolution
- **Primary Choice**: FDTD-FEM coupling
- **Why**: Time-domain evolution with adaptive resolution
- **Example**: HIFU treatment planning with detailed transducer modeling

## Implementation Details

### Common Coupling Strategies

#### Schwarz Alternating Method
```rust
// Iterative coupling between domains
for iteration in 0..max_iterations {
    // 1. Solve Domain 1 with Domain 2 boundary conditions
    solve_domain_1(boundary_from_domain_2);

    // 2. Extract interface solution from Domain 1
    interface_solution = extract_interface(domain_1);

    // 3. Apply to Domain 2 with relaxation
    apply_to_domain_2(interface_solution * omega + old_solution * (1-omega));

    // 4. Solve Domain 2
    solve_domain_2(boundary_from_domain_1);

    // 5. Check convergence
    residual = compute_interface_residual();
    if residual < tolerance { break; }
}
```

#### Conservative Field Transfer
```rust
// Ensure integral quantities are preserved across interfaces
conservative_interpolate(source_field, source_grid, target_grid) {
    // Use high-order interpolation that preserves moments
    // For spectral methods: modal basis transformation
    // For finite methods: conservative remapping
}
```

### Stability Considerations

#### Relaxation Parameters
- **FDTD-FEM**: ω = 0.8 (conservative for explicit schemes)
- **PSTD-SEM**: ω = 0.9 (higher for implicit spectral coupling)
- **BEM-FEM**: ω = 0.7 (lower for boundary integral stability)

#### Convergence Acceleration
- ** Aitken acceleration**: For slowly converging iterations
- ** Interface smoothing**: Laplacian smoothing near boundaries
- ** Overlap methods**: Larger overlap regions for better convergence

## Performance Optimization

### Memory Management
- **Domain decomposition**: Minimize interface data transfer
- **Sparse representations**: Use sparse matrices for interface operators
- **Buffer management**: Reuse allocated memory between iterations

### Computational Efficiency
- **Load balancing**: Distribute computational work evenly
- **Parallel coupling**: Concurrent solution of different domains
- **Adaptive coupling**: Reduce iterations for well-conditioned problems

## Validation and Testing

### Convergence Tests
```rust
// Test convergence with grid refinement
for refinement_level in 1..4 {
    grid_size *= 2;
    solve_hybrid_problem(grid_size);
    check_convergence_rate(); // Should approach expected order
}
```

### Analytical Benchmarks
- **Plane waves**: Test interface continuity
- **Spherical waves**: Validate radiation conditions
- **Gaussian beams**: Check field conservation

### Stability Analysis
- **Eigenvalue analysis**: Check coupling operator eigenvalues
- **Time-step restrictions**: Verify CFL-like conditions
- **Interface condition number**: Assess conditioning of coupling

## Future Extensions

### Additional Hybrid Methods
- **BEM-FDTD**: Time-domain exterior problems
- **FEM-SEM**: Multi-scale spectral elements
- **PSTD-BEM**: Spectral boundary methods

### Advanced Features
- **Adaptive coupling**: Dynamic domain decomposition
- **Multi-domain coupling**: More than two coupled domains
- **Nonlinear coupling**: For nonlinear wave equations

## References

1. **Wu, T. (2000)**. "Pre-asymptotic error analysis of BEM and FEM coupling"
2. **Kopriva, D. A. (2009)**. "Implementing spectral methods for PDEs"
3. **Farhat & Lesoinne (2000)**. "Two-level FETI methods for Stokes problems"
4. **Hesthaven & Warburton (2008)**. "Nodal DG methods"

## Usage Examples

### Basic BEM-FEM Coupling
```rust
use kwavers::solver::forward::hybrid::{BemFemSolver, BemFemCouplingConfig};

// Setup FEM mesh for interior complex geometry
let fem_mesh = create_fem_mesh();
// Define BEM boundary elements for exterior domain
let bem_boundary = vec![0, 1, 2, 3]; // Boundary element indices

let config = BemFemCouplingConfig::default();
let wavenumber = 2.0 * PI * frequency / speed_of_sound;

let mut solver = BemFemSolver::new(config, fem_mesh, bem_boundary, wavenumber)?;

// Solve coupled system
let fem_initial_guess = vec![0.0; fem_mesh.nodes.len()];
let bem_boundary_guess = vec![0.0; bem_boundary.len()];

solver.solve(fem_initial_guess, bem_boundary_guess)?;

// Check convergence
let (converged, iterations, history) = solver.convergence_info();
println!("Converged: {}, Iterations: {}", converged, iterations);
```

This hybrid methods framework provides kwavers with the flexibility to tackle complex acoustic wave problems that would be difficult or impossible to solve with single discretization techniques.