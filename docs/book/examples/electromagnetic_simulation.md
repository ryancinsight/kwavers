# Example: Electromagnetic Simulation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example electromagnetic_simulation --features pinn`  
**Source**: [`crates/kwavers/examples/electromagnetic_simulation.rs`](../../../../crates/kwavers/examples/electromagnetic_simulation.rs)

## What This Example Demonstrates

This example demonstrates electromagnetic wave simulation using Physics-Informed Neural Networks (PINN). It covers three problem types:

| Problem Type | Description |
|---|---|
| Electrostatics | Electric potential in parallel plate capacitor |
| Magnetostatics | Magnetic vector potential around current-carrying wire |
| Wave Propagation | EM wave propagation in free space |

## Key Code Snippet

```rust
use kwavers_solver::inverse::pinn::ml::electromagnetic::{EMProblemType, ElectromagneticDomain};
use kwavers_solver::inverse::pinn::ml::universal_solver::UniversalSolverGeometry2D;
use kwavers_solver::inverse::pinn::ml::{PinnEMSource, UniversalPINNSolver};

// Electrostatic parallel plate capacitor
let domain: ElectromagneticDomain<Backend> = ElectromagneticDomain::new(
    EMProblemType::Electrostatic,
    8.854e-12,                   // Vacuum permittivity
    4e-7 * std::f64::consts::PI, // Vacuum permeability
    0.0,                         // No conductivity
    vec![0.01, 0.01],            // 1cm x 1cm domain
)
.add_pec_boundary(BoundaryPosition::Top)    // +V plate
.add_pec_boundary(BoundaryPosition::Bottom) // Ground plate
.add_pec_boundary(BoundaryPosition::Left)   // Side wall
.add_pec_boundary(BoundaryPosition::Right); // Side wall

let geometry = UniversalSolverGeometry2D::rectangle(0.0, 0.01, 0.0, 0.01);
```

## Features Demonstrated

- Domain configuration for different EM problem types
- Boundary condition setup (PEC, PMC, impedance)
- Material property specification
- PINN training with physics-informed loss
- GPU acceleration (when available)
- Result visualization and validation

## Book Chapter

[← Therapy and Theranostics](../theranostics.md)
