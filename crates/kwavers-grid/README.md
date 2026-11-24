# kwavers-grid

Spatial discretization for [kwavers](https://github.com/ryancinsight/kwavers): Cartesian
and cylindrical grids, coordinates, topology, difference operators, k-space FFT helpers,
and geometric domains.

Every field in a kwavers simulation is defined over a `Grid`. This crate owns that
definition — the extents, the spacing, the stability limits derived from them, and the
geometric primitives that classify a point as inside or outside a region.

## What it provides

| Module | Responsibility |
|---|---|
| `structure` | `Grid`, `Bounds`, `GridDimension` — the discretization SSOT |
| `config` / `simple_config` | `GridConfig`, `GridType`, domain parameters |
| `coordinates` | `CoordinateSystem` conversions |
| `topology` | Cartesian and cylindrical topologies, neighbor relations |
| `operators` | Finite-difference and spectral operators over a grid |
| `stability` | CFL and spacing constraints derived from the discretization |
| `geometry` | `RectangularDomain`, `SphericalDomain`, `PointLocation` classification |
| `validation` | `GridValidator` — grid invariant checks |

k-space helpers (`KSpaceCalculator`, FFT shifts, optimal FFT sizes) are re-exported from
`kwavers-math` so grid consumers need one import path.

## Cartesian finite differences

Centered stencil weights are ordered from the nearest pair of neighbors outward.
The gradient has units of field per metre and the Laplacian has units of field per
square metre when the grid spacing is in metres. Each axis uses its own spacing.

`LaplacianOperator::apply_mut` writes directly into caller-owned output, including
strided views. Second-, fourth-, sixth-, and eighth-order stencils use one to four
neighbors on each side. Periodic grids wrap each axis. Homogeneous Neumann grids
reflect about the endpoint nodes. Dirichlet mode writes zero to the halo in which
the full centered stencil does not fit, including when the output buffer is reused.

## Example

Construction is fallible: zero dimensions and non-positive spacing are rejected at the
boundary rather than producing an invalid grid.

```rust
use kwavers_grid::Grid;

// 64 x 64 x 64 domain at 0.1 mm isotropic spacing.
let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4).unwrap();
assert_eq!(grid.nx, 64);

// Invalid parameters fail with a typed error instead of panicking.
assert!(Grid::new(0, 64, 64, 1e-4, 1e-4, 1e-4).is_err());
assert!(Grid::new(64, 64, 64, 0.0, 1e-4, 1e-4).is_err());
```

## Documentation

- API reference: <https://docs.rs/kwavers-grid>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
