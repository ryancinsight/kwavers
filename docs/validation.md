# Physics Validation Documentation

This document describes the validation tests and benchmarks used to verify the correctness of the Kwavers physics simulations.

## Overview

Kwavers includes comprehensive validation against known analytical solutions from physics. These tests ensure that our numerical implementations are accurate and reliable.

## Validation Categories

### 1. Finite Difference Convergence

**Test**: `benchmark_fd_convergence()`

Tests the convergence of finite difference approximations for the Laplacian operator.

- **Analytical Solution**: For `u(x) = sin(2πx/L)`, the Laplacian is `∇²u = -(2π/L)² sin(2πx/L)`
- **Expected Behavior**: Second-order convergence (error ∝ dx²)
- **Results**: Demonstrates proper O(dx²) convergence

### 2. Heat Diffusion

**Test**: `test_heat_diffusion_analytical()`

Validates heat diffusion against the analytical solution for a Gaussian initial condition.

- **Equation**: ∂T/∂t = α∇²T
- **Initial Condition**: T(x,y,0) = T₀ exp(-r²/2σ₀²)
- **Analytical Solution**: T(x,y,t) = T₀(σ₀/σₜ)² exp(-r²/2σₜ²)
  - Where: σₜ = √(σ₀² + 4αt)
- **Physical Parameters**:
  - Thermal diffusivity α = 1.4×10⁻⁷ m²/s (water)
  - Initial width σ₀ = 5 mm

### 3. Acoustic Wave Propagation

**Test**: `test_1d_wave_equation_analytical()`

Tests the 1D wave equation with sinusoidal initial conditions.

- **Equation**: ∂²p/∂t² = c²∇²p
- **Initial Condition**: p(x,0) = A sin(kx)
- **Analytical Solution**: p(x,t) = A sin(kx - ωt)
  - Where: ω = ck, k = 2π/λ
- **Physical Parameters**:
  - Sound speed c = 1500 m/s (water)
  - Frequency f = 1 MHz

### 4. Acoustic Absorption

**Test**: `test_acoustic_absorption_beer_lambert()`

Validates acoustic absorption against the Beer-Lambert law.

- **Law**: I(x) = I₀ exp(-αx)
- **Power Law Model**: α(f) = α₀(f/f₀)^y
- **Test Cases**:
  - Frequency-independent absorption (y = 0)
  - Linear frequency dependence (y = 1)
  - Tissue-like absorption (α₀ = 0.5 dB/cm/MHz)

### 5. Standing Waves

**Test**: `test_standing_wave_rigid_boundaries()`

Tests standing wave formation between rigid boundaries.

- **Boundary Conditions**: v = 0 at x = 0, L
- **Analytical Solution**: p(x,t) = A cos(kx) cos(ωt)
- **Modes**: k = nπ/L for integer n

### 6. Spherical Wave Spreading

**Test**: `test_spherical_spreading_3d()`

Validates geometric spreading in 3D.

- **Law**: p ∝ 1/r for spherical waves
- **Test**: Measures pressure ratio at different radii
- **Expected**: p₁/p₂ = r₂/r₁

### 7. Numerical Dispersion

**Test**: `test_numerical_dispersion()`

Analyzes phase velocity errors for different wavelengths.

- **Method**: Propagate sinusoidal waves for one period
- **Measure**: Phase shift due to numerical dispersion
- **Parameters**: Points per wavelength (4, 8, 16, 32, 64)

## Benchmark Results

### Accuracy Benchmarks

Run with: `cargo run --example accuracy_benchmarks`

| Test | Typical Error | Status |
|------|--------------|--------|
| FD Laplacian (nx=256) | ~5×10⁻⁵ | ✓ PASS |
| Heat Diffusion | < 5% | ✓ PASS |
| Power Law Absorption | ~10⁻¹⁴ | ✓ PASS |
| Time Integration | Variable | Needs fix |

### Performance Considerations

1. **Grid Resolution**: Error scales as O(dx²) for second-order schemes
2. **CFL Condition**: Stability requires CFL ≤ 0.5 for explicit schemes
3. **Wavelength Resolution**: Need >8 points per wavelength for <10% dispersion error

## Usage in Development

These validation tests serve multiple purposes:

1. **Regression Testing**: Ensure changes don't break existing functionality
2. **Accuracy Verification**: Confirm numerical methods are correctly implemented
3. **Parameter Selection**: Guide choice of grid resolution and time steps
4. **Documentation**: Provide examples of proper usage

## Adding New Validation Tests

To add a new validation test:

1. Identify an analytical solution relevant to your physics
2. Implement the test in `src/physics/validation_tests.rs`
3. Add benchmark in `src/benchmarks/accuracy.rs` if needed
4. Document the test in this file
5. Ensure the test passes with reasonable tolerances

## References

1. **Heat Equation**: Carslaw & Jaeger, "Conduction of Heat in Solids"
2. **Wave Equation**: Morse & Ingard, "Theoretical Acoustics"
3. **Absorption**: Duck, "Physical Properties of Tissue"
4. **Numerical Methods**: LeVeque, "Finite Difference Methods for PDEs"