# Autodiff Stress Gradients — Quick Reference

**Module**: `src/solver/inverse/pinn/elastic_2d/loss.rs`  
**Feature**: `pinn`  
**Status**: Production-ready (Phase 4 Task 4 complete)

---

## TL;DR

```rust
use kwavers::solver::inverse::pinn::elastic_2d::loss::compute_elastic_wave_pde_residual;

// Forward pass: network predicts displacement
let displacement = model.forward(coords);
let u = displacement.slice([.., 0..1]);
let v = displacement.slice([.., 1..2]);

// Compute PDE residual (all gradients via autodiff)
let (residual_x, residual_y) = compute_elastic_wave_pde_residual(
    u, v, x, y, t,
    rho,    // Density (kg/m³)
    lambda, // Lamé first parameter (Pa)
    mu      // Shear modulus (Pa)
);

// Compute loss
let pde_loss = loss_computer.pde_loss(residual_x, residual_y);
```

---

## Available Functions

### 1. Full PDE Residual (Most Common)

```rust
pub fn compute_elastic_wave_pde_residual<B: AutodiffBackend>(
    u: Tensor<B, 2>,      // x-displacement [N, 1]
    v: Tensor<B, 2>,      // y-displacement [N, 1]
    x: Tensor<B, 2>,      // x-coordinates [N, 1] (autodiff enabled)
    y: Tensor<B, 2>,      // y-coordinates [N, 1] (autodiff enabled)
    t: Tensor<B, 2>,      // time [N, 1] (autodiff enabled)
    rho: f64,             // Density
    lambda: f64,          // Lamé first parameter
    mu: f64,              // Shear modulus
) -> (Tensor<B, 2>, Tensor<B, 2>)
```

**Returns**: `(R_x, R_y)` where `R = ρ ∂²u/∂t² - ∇·σ`

**Use Case**: Computing PDE residual loss in training loop

---

### 2. Spatial Terms Only

```rust
pub fn displacement_to_stress_divergence<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>)
```

**Returns**: `(div_x, div_y)` = stress divergence components

**Use Case**: Steady-state problems, validation of spatial derivatives

---

### 3. Individual Stages (Advanced)

#### Stage 1: Displacement Gradients
```rust
pub fn compute_displacement_gradients<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
```
**Returns**: `(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)`

#### Stage 2: Strain from Gradients
```rust
pub fn compute_strain_from_gradients<B: AutodiffBackend>(
    dudx: Tensor<B, 2>,
    dudy: Tensor<B, 2>,
    dvdx: Tensor<B, 2>,
    dvdy: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
```
**Returns**: `(ε_xx, ε_yy, ε_xy)`

#### Stage 3: Stress from Strain
```rust
pub fn compute_stress_from_strain<B: AutodiffBackend>(
    epsilon_xx: Tensor<B, 2>,
    epsilon_yy: Tensor<B, 2>,
    epsilon_xy: Tensor<B, 2>,
    lambda: f64,
    mu: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
```
**Returns**: `(σ_xx, σ_yy, σ_xy)`

#### Stage 4: Stress Divergence
```rust
pub fn compute_stress_divergence<B: AutodiffBackend>(
    sigma_xx: Tensor<B, 2>,
    sigma_xy: Tensor<B, 2>,
    sigma_yy: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>)
```
**Returns**: `(∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y)`

#### Stage 5: Time Derivatives
```rust
pub fn compute_time_derivatives<B: AutodiffBackend>(
    u: Tensor<B, 2>,
    t: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>)
```
**Returns**: `(∂u/∂t, ∂²u/∂t²)`

---

## Complete Training Loop Example

```rust
use burn::tensor::{backend::AutodiffBackend, Tensor};
use kwavers::solver::inverse::pinn::elastic_2d::{
    loss::{compute_elastic_wave_pde_residual, LossComputer},
    config::LossWeights,
};

fn training_step<B: AutodiffBackend>(
    model: &ElasticPINN2DModel<B>,
    loss_computer: &LossComputer,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    t: Tensor<B, 2>,
    rho: f64,
    lambda: f64,
    mu: f64,
) -> Tensor<B, 1> {
    // Forward pass
    let coords = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);
    let displacement = model.forward(coords);
    let u = displacement.clone().slice([0..displacement.dims()[0], 0..1]);
    let v = displacement.slice([0..displacement.dims()[0], 1..2]);

    // Compute PDE residual via autodiff
    let (residual_x, residual_y) = compute_elastic_wave_pde_residual(
        u, v, x, y, t, rho, lambda, mu
    );

    // Compute loss
    let pde_loss = loss_computer.pde_loss(residual_x, residual_y);

    // Optionally add boundary/initial/data losses
    // let total_loss = loss_computer.total_loss(pde_loss, bc_loss, ic_loss, data_loss);

    pde_loss
}
```

---

## Validation Test Example

```rust
use kwavers::solver::inverse::pinn::elastic_2d::loss::compute_elastic_wave_pde_residual;

#[test]
fn test_plane_wave_pde_residual() {
    // Material properties
    let rho = 2000.0;
    let lambda = 1e9;
    let mu = 0.5e9;

    // Generate plane wave test points
    let nx = 50;
    let nt = 20;
    let x_vals: Vec<f32> = (0..nx).map(|i| i as f32 * 0.1).collect();
    let t_vals: Vec<f32> = (0..nt).map(|i| i as f32 * 0.01).collect();

    // Create tensors
    let x = Tensor::<Backend, 2>::from_floats(/* ... */, &device);
    let y = Tensor::zeros_like(&x);
    let t = Tensor::<Backend, 2>::from_floats(/* ... */, &device);

    // Analytical plane wave solution
    let u_analytical = compute_plane_wave_displacement(x, t, /* ... */);
    let v_analytical = Tensor::zeros_like(&u_analytical);

    // Compute PDE residual
    let (residual_x, residual_y) = compute_elastic_wave_pde_residual(
        u_analytical, v_analytical, x, y, t, rho, lambda, mu
    );

    // Check residual is small
    let residual_norm = (residual_x.powf_scalar(2.0) + residual_y.powf_scalar(2.0))
        .mean()
        .sqrt();

    assert!(residual_norm.into_scalar() < 1e-3, "PDE residual too large");
}
```

---

## Material Parameters

### Lamé Parameters
- **λ** (lambda): First Lamé parameter
  - Units: Pa (N/m²)
  - Typical range: 1e8 to 1e11 Pa
  - Relation: `λ = ν E / ((1+ν)(1-2ν))` for Young's modulus E, Poisson's ratio ν

- **μ** (mu): Shear modulus (second Lamé parameter)
  - Units: Pa (N/m²)
  - Typical range: 1e8 to 1e11 Pa
  - Relation: `μ = E / (2(1+ν))`

- **ρ** (rho): Density
  - Units: kg/m³
  - Typical range: 1000 to 8000 kg/m³

### Common Materials

| Material | λ (GPa) | μ (GPa) | ρ (kg/m³) |
|----------|---------|---------|-----------|
| Aluminum | 52.0    | 26.0    | 2700      |
| Steel    | 110.0   | 80.0    | 7850      |
| Tissue   | 2.0     | 0.5     | 1060      |

---

## Mathematical Background

### Elastic Wave Equation
```
ρ ∂²u/∂t² = ∇·σ
```

### Strain-Displacement
```
ε_xx = ∂u/∂x
ε_yy = ∂v/∂y
ε_xy = 0.5(∂u/∂y + ∂v/∂x)
```

### Hooke's Law (Isotropic)
```
σ_xx = (λ + 2μ) ε_xx + λ ε_yy
σ_yy = λ ε_xx + (λ + 2μ) ε_yy
σ_xy = 2μ ε_xy
```

### Stress Divergence
```
(∇·σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y
(∇·σ)_y = ∂σ_xy/∂x + ∂σ_yy/∂y
```

---

## Common Pitfalls

### ❌ Forgetting to Enable Autodiff
```rust
// Wrong: x, y, t are not autodiff-enabled
let x = Tensor::from_floats(coords_x, &device);
```

### ✅ Correct: Ensure Gradient Tracking
```rust
// Correct: x, y, t from network input or explicitly require_grad()
let coords = Tensor::from_floats(raw_coords, &device).require_grad();
let x = coords.slice([.., 0..1]);
```

### ❌ Shape Mismatches
```rust
// Wrong: u and v must be [N, 1], not [N]
let u = displacement.slice([.., 0]);  // Returns [N]
```

### ✅ Correct Shapes
```rust
// Correct: explicit 2D slicing
let u = displacement.slice([.., 0..1]);  // Returns [N, 1]
```

### ❌ Wrong Parameter Units
```rust
// Wrong: lambda in kPa, mu in GPa (inconsistent)
let lambda = 1e6;  // kPa
let mu = 1e9;      // Pa (GPa)
```

### ✅ Consistent Units
```rust
// Correct: all in Pa
let lambda = 1e9;  // Pa
let mu = 0.5e9;    // Pa
let rho = 2000.0;  // kg/m³
```

---

## Performance Notes

- **Autodiff Overhead**: ~2-3x slower than finite differences per evaluation
- **Training Benefit**: Exact gradients → faster convergence → fewer epochs needed
- **Memory**: Gradient tracking increases memory by ~2x (stores computational graph)
- **Optimization**: Consider caching intermediate tensors (strain, stress) if used multiple times

---

## Troubleshooting

### Error: "Gradient not available"
**Cause**: Input tensor doesn't support autodiff  
**Fix**: Ensure x, y, t have `.require_grad()` or come from autodiff-enabled source

### Error: "Shape mismatch"
**Cause**: Displacement components are 1D instead of 2D  
**Fix**: Use slice syntax `[.., 0..1]` instead of `[.., 0]`

### Error: "Device mismatch"
**Cause**: Tensors on different devices (CPU vs GPU)  
**Fix**: Ensure all tensors created on same device via `.device()` query

### High residual norm
**Cause**: Network not trained, or wrong material parameters  
**Fix**: Check training convergence, verify λ, μ, ρ values are physically reasonable

---

## References

- Implementation: `src/solver/inverse/pinn/elastic_2d/loss.rs`
- Tests: `tests/pinn_elastic_validation.rs`
- Documentation: `docs/phase4_task4_complete.md`
- Theory: Raissi et al., "Physics-informed neural networks" (2019)