# Example: PINN Advanced Physics

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_advanced_physics --features pinn -- --all`  
**Source**: [`crates/kwavers/examples/pinn_advanced_physics.rs`](../../../crates/kwavers/examples/pinn_advanced_physics.rs)

## What This Example Demonstrates

This example expands the PINN story beyond acoustics. It is a command-line showcase for Navier–Stokes flow, heat transfer, structural mechanics, and additional multi-physics or industrial demonstration modes.

| Component | API | Value |
|---|---|---|
| Mode selector | `--navier-stokes`, `--heat-transfer`, `--structural`, `--all` | Chooses which advanced physics domain to demonstrate |
| Console formulations | `physics_demo::*` | Prints governing equations and scenario descriptions for each selected domain |
| Feature gate | `--features pinn` | Required to enable the PINN-backed advanced-physics demo code |

## Key Code Snippet

```rust
let demo_mode = if args.len() > 1 {
    args[1].as_str()
} else {
    "--all"
};

match demo_mode {
    "--navier-stokes" => {
        physics_demo::demonstrate_navier_stokes();
    }
```

## Expected Output (if applicable)

The output depends on the chosen flag, but every path prints the selected physics domain, its equations, and the corresponding demo narrative.

## Book Chapter

[← Inverse Problems and Physics-Informed Neural Networks](../inverse_problems_and_pinns.md)
