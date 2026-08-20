# kwavers-source

Low-level acoustic, optical, and electromagnetic excitation primitives for
[kwavers](https://github.com/ryancinsight/kwavers): the `Source` trait, grid/mask-driven
sources, analytic wavefronts, custom arbitrary-signal sources, and apodization windows.

A source answers two questions for a solver: *where* on the grid does energy enter (a
spatial mask), and *what* is injected there over time (a `kwavers-signal` `Signal`). This
crate owns those primitives.

Physical transducer *devices* — bowls, phased and linear arrays, calibration, factories —
live in [`kwavers-transducer`](https://docs.rs/kwavers-transducer), which builds on this
crate and on [`kwavers-receiver`](https://docs.rs/kwavers-receiver).

## What it provides

| Module | Responsibility |
|---|---|
| `types` | The `Source` trait, `SourceField`, `SourceType`, polarization and EM wave types |
| `structs` | `PointSource`, `CompositeSource`, `TimeVaryingSource`, `NullSource` |
| `grid_source` | `GridSource`, `SourceMode` — mask-driven excitation over grid regions |
| `wavefront` | Analytic plane, spherical, Bessel, and Gaussian wavefronts |
| `custom` | Arbitrary user-supplied spatial/temporal profiles |
| `apodization` | Aperture window math shared with the device layer |
| `injection` | `SourceInjectionMode` — additive vs. hard-source injection semantics |
| `optical` / `electromagnetic` | Non-acoustic excitation primitives |

The spatial mask is computed once at initialization (`create_mask`) and reused every step;
`add_mask_into` composes several sources into one caller-owned buffer without extra
allocation.

## Example

```rust
use std::sync::Arc;
use kwavers_grid::Grid;
use kwavers_signal::SineWave;
use kwavers_source::{PointSource, Source};

let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
let signal = Arc::new(SineWave::new(1.0e6, 1.0e6, 0.0));

// Position is in metres, at the centre of the domain.
let source = PointSource::new((8e-4, 8e-4, 8e-4), signal);

let mask = source.create_mask(&grid);
// Exactly one cell is excited by a point source.
let excited: f64 = mask.iter().sum();
assert!((excited - 1.0).abs() < f64::EPSILON);
```

## Documentation

- API reference: <https://docs.rs/kwavers-source>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
