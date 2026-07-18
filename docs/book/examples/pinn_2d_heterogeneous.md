# Example: PINN 2D Heterogeneous

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_2d_heterogeneous --features pinn`  
**Source**: [`crates/kwavers/examples/pinn_2d_heterogeneous.rs`](../../../../crates/kwavers/examples/pinn_2d_heterogeneous.rs)

## What This Example Demonstrates

This example demonstrates Physics-Informed Neural Networks (PINN) for solving the 2D wave equation in heterogeneous media. It shows how the network handles spatially varying material properties.

## Features

| Feature | Description |
|---------|-------------|
| Heterogeneous Medium | Spatially varying sound speed |
| 2D Wave Equation | Pressure field in 2D |
| PINN Training | Physics-informed loss function |
| Convergence Analysis | Error vs. training iterations |

## Key Code Snippet

```rust
// Create heterogeneous medium
let medium = create_heterogeneous_medium()?;

// Create PINN solver
let pinn = UniversalPINNSolver::new(
    domain,
    geometry,
    medium,
    config,
)?;

// Train PINN
for epoch in 0..max_epochs {
    let loss = pinn.train_step()?;
    
    if epoch % 100 == 0 {
        println!("Epoch {}: loss = {:.6e}", epoch, loss);
    }
}

// Evaluate solution
let solution = pinn.evaluate()?;
```

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
