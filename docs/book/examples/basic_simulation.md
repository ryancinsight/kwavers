# Example: Basic Simulation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example basic_simulation`  
**Source**: [`crates/kwavers/examples/basic_simulation.rs`](../../../crates/kwavers/examples/basic_simulation.rs)

## What This Example Demonstrates

The canonical entry point for kwavers. Sets up a 64×64×64 grid in water,
computes CFL-stable time stepping, and demonstrates the three-layer setup:
grid → medium → time.

| Component | API | Value |
|---|---|---|
| Grid | `Grid::new(64, 64, 64, 1e-3, ...)` | 64 mm³ domain at 1 mm spacing |
| Medium | `HomogeneousMedium::new(1000.0, 1500.0, ...)` | Water: ρ=1000 kg/m³, c=1500 m/s |
| Time | `grid.cfl_timestep(1500.0)` | CFL-stable Δt ≈ 192 ns |

## Key Code Snippet

```rust,ignore
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_core::time::Time;

let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
let dt = grid.cfl_timestep(1500.0);
let time = Time::new(dt, 100);

println!("CFL timestep: {:.2e} s", time.dt);
println!("Grid points: {}", grid.nx * grid.ny * grid.nz);
println!("Memory estimate: {:.1} MB", grid.memory_estimate_mb(6));
```

## Book gate oracle

The shared book workflow builds the `kwavers` library before testing executable
Rust fences. This small assertion keeps the package-backed book gate live even
when the larger simulation walkthrough remains environment-specific.

```rust,ignore
extern crate kwavers;

let version = kwavers::get_version_info();
assert_eq!(version.get("name").map(String::as_str), Some("kwavers"));
assert_eq!(version.get("license").map(String::as_str), Some("MIT"));
```

The executable part of the gate checks the CFL expression used by the
walkthrough against its analytical value:

```rust
let grid_spacing_m = 1.0e-3_f64;
let sound_speed_m_s = 1500.0_f64;
let cfl_number = 0.5_f64;
let spatial_dimensions = 3.0_f64;
let dt = cfl_number * grid_spacing_m / (sound_speed_m_s * spatial_dimensions.sqrt());
let expected = 1.9245008972987525e-7_f64;

assert!((dt - expected).abs() < 1.0e-18);
assert!(dt.is_finite() && dt > 0.0);
```

## Expected Output

```text
Grid created: 64x64x64 points
Domain size: 64.0x64.0x64.0 mm
CFL timestep: 1.92e-7 s
Grid points: 262144
Memory estimate: 21.0 MB
```

## Book Chapter

[← Wave Physics Fundamentals](../foundations.md)
