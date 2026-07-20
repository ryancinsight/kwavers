# Example: Validate 2D PINN

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example validate_2d_pinn --features pinn`  
**Source**: [`crates/kwavers/examples/validate_2d_pinn.rs`](../../../crates/kwavers/examples/validate_2d_pinn.rs)

## What This Example Demonstrates

This example validates the 2D wave equation Physics-Informed Neural Network (PINN) implementation against analytical solutions. It demonstrates convergence behavior and accuracy metrics.

## Validation Approach

| Aspect | Description |
|--------|-------------|
| Analytical Solution | Exact 2D wave equation solution |
| PINN Solution | Neural network approximation |
| L2 Error | Global accuracy metric |
| Convergence | Error vs. training iterations |

## Key Code Snippet

```rust
// Create PINN for 2D wave equation
let pinn = create_2d_wave_pinn()?;

// Train and evaluate
for epoch in 0..max_epochs {
    pinn.train_step()?;
    let error = pinn.compute_l2_error()?;
    
    println!("Epoch {}: L2 error = {:.6e}", epoch, error);
}

// Validate against analytical solution
let analytical = create_analytical_solution()?;
let validation_error = compute_validation_error(&pinn, &analytical)?;
```

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
