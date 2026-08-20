# kwavers-medium

Material and tissue models for [kwavers](https://github.com/ryancinsight/kwavers):
homogeneous and heterogeneous media, acoustic/elastic/optical/thermal/viscous properties,
absorption models, and anisotropy.

A medium answers "what are the material properties at this grid point?" for every physics
module in the stack. This crate owns that question: the property traits solvers bind to,
the concrete media that implement them, and the literature-sourced tissue parameters
behind them.

## Trait seams

Properties are segregated by physics domain so a solver depends only on what it uses:

| Trait | Provides |
|---|---|
| `CoreMedium` | Sound speed, density, absorption, nonlinearity at a point |
| `ArrayAccess` | Whole-field views of the same quantities for vectorized kernels |
| `AcousticProperties` | Acoustic-specific parameters |
| `ElasticProperties` / `ElasticArrayAccess` | Lamé parameters and shear behavior |
| `MediumOpticalProperties` | Absorption and reduced scattering coefficients |
| `ThermalProperties` | Conductivity, specific heat, perfusion |
| `ViscousProperties` | Shear and bulk viscosity |
| `Medium` | The composed super-trait for consumers that need everything |

## What it provides

| Module | Responsibility |
|---|---|
| `homogeneous` | `HomogeneousMedium` — uniform properties over the grid |
| `heterogeneous` | Voxel-wise media, including `CtMediumBuilder` for CT-derived maps |
| `anisotropic` / `viscoelastic` | Direction-dependent and relaxation-based models |
| `absorption` | `PowerLawAbsorption` and tissue absorption classes |
| `frequency_dependent` | Dispersion and frequency-dependent tissue models |
| `optical_map` | Layered and region-based optical property maps |
| `properties` | Literature tissue and fluid property tables |
| `interface` | Interface detection between dissimilar materials |
| `builder` / `validation_simulation` | `MediumBuilder`, `MediumValidator` |

## Example

```rust
use kwavers_grid::Grid;
use kwavers_medium::{CoreMedium, HomogeneousMedium};

let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
let water = HomogeneousMedium::water(&grid);

// Properties are queried per grid point; a homogeneous medium answers uniformly.
let c = water.sound_speed(8, 8, 8);
assert!((c - 1500.0).abs() < 20.0);
assert!((water.density(0, 0, 0) - water.density(8, 8, 8)).abs() < f64::EPSILON);
```

## Documentation

- API reference: <https://docs.rs/kwavers-medium>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
