# kwavers

Ultrasound Simulation Toolbox with Cavitation and Light Physics.

`kwavers` is the thin top-level application / integration crate of the
[kwavers](https://github.com/ryancinsight/kwavers) workspace. It hosts the
binary plus the cross-cutting tests, examples, and benchmarks. **It re-exports
nothing**: the library surface lives in the per-layer crates (`kwavers-core`,
`kwavers-grid`, `kwavers-medium`, `kwavers-solver`, …), and consumers depend on
those directly. See the
[workspace README](https://github.com/ryancinsight/kwavers#readme) for the full
crate map.

## Quick start

Every snippet below is a doctest of this crate, so it cannot drift from the API.

### 1. Build a computational grid

```toml
[dependencies]
kwavers-grid = "3.0.0"
```

```rust
use kwavers_grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100³ points at 1 mm isotropic spacing.
    let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3)?;

    assert_eq!((grid.nx, grid.ny, grid.nz), (100, 100, 100));
    assert_eq!(grid.dx, 1e-3);
    Ok(())
}
```

### 2. Describe the propagation medium

```toml
[dependencies]
kwavers-core = "3.0.0"
kwavers-grid = "3.0.0"
kwavers-medium = "4.0.0"
```

`HomogeneousMedium` carries acoustic, optical, thermal, and cavitation
properties. The `water`, `tissue`, `blood`, and `air` constructors fill them
from the reference constants in `kwavers-core`; `HomogeneousMedium::new` takes
`(density, sound_speed, mu_a, mu_s_prime, grid)` where the last two are the
*optical* absorption and reduced-scattering coefficients.

```rust
use kwavers_core::constants::fundamental::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers_grid::Grid;
use kwavers_medium::{CoreMedium, HomogeneousMedium};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
    let water = HomogeneousMedium::water(&grid);

    // Point accessors come from the `CoreMedium` trait.
    let density = water.density(0, 0, 0); // kg/m³
    let sound_speed = water.sound_speed(0, 0, 0); // m/s
    let impedance = density * sound_speed; // Pa·s/m

    assert_eq!(density, DENSITY_WATER);
    assert_eq!(sound_speed, SOUND_SPEED_WATER);
    assert_eq!(impedance, DENSITY_WATER * SOUND_SPEED_WATER);
    Ok(())
}
```

### 3. Acoustic impedance and reflection

Plain arithmetic on the same quantities, no kwavers types required.

```rust
fn main() {
    // Acoustic impedance Z = ρc.
    let impedance_water = 1000.0 * 1500.0_f64; // Pa·s/m
    let impedance_air = 1.2 * 343.0_f64;

    // Pressure reflection coefficient R = (Z₂ − Z₁) / (Z₂ + Z₁).
    let reflection = (impedance_air - impedance_water) / (impedance_air + impedance_water);

    // The water/air interface is very nearly a perfect reflector.
    assert!(reflection < -0.999);
}
```

## Runnable examples

Sixty-plus end-to-end programs live in
[`crates/kwavers/examples/`](https://github.com/ryancinsight/kwavers/tree/main/crates/kwavers/examples):

```bash
cargo run -p kwavers --example basic_simulation
```

## Documentation

- API reference: <https://docs.rs/kwavers>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Architecture decision records:
  <https://github.com/ryancinsight/kwavers/tree/main/docs/ADR>

## License

MIT — see
[LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
