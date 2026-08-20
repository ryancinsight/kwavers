# Example: Multiphysics Sonoluminescence

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example multiphysics_sonoluminescence --features pinn`  
**Source**: [`crates/kwavers/examples/multiphysics_sonoluminescence.rs`](../../../crates/kwavers/examples/multiphysics_sonoluminescence.rs)

This example runs the real universal PINN solver over the registered
cavitation, sonoluminescence, and electromagnetic domains. The demo workload
uses two epochs and 32 collocation points so it remains bounded for local and
CI execution. It prints the returned total loss and per-domain final losses and
convergence flags. It does not fabricate luminosity, conservation, or
literature-agreement values.

```bash
cargo run -p kwavers --example multiphysics_sonoluminescence --features pinn
```

The feature gate is required because the example depends on the PINN provider.

[← Nonlinear Acoustics](../nonlinear_acoustics.md)
