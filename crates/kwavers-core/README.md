# kwavers-core

Foundation layer for [kwavers](https://github.com/ryancinsight/kwavers): constants, error
types, arena allocation, and time/logging utilities.

This is the root of the workspace dependency graph. It depends on no other kwavers crate,
and every other layer depends on it — so it carries only what the whole stack shares: the
error taxonomy that failures converge on, the physical and numerical constants that must
have one value, the arena allocators that keep field-sized buffers off the hot path, and
the unit conversions between literature and SI conventions.

## What it provides

| Module | Responsibility |
|---|---|
| `error` | `KwaversError` — the workspace error taxonomy — and `KwaversResult<T>` |
| `constants` | Physical, tissue, cavitation, thermodynamic, optical, and numerical constants (SSOT) |
| `arena` | Field arenas, bump/scoped allocators, NUMA-aware pools, thread-local scratch |
| `units` | Conversions between literature units (dB/MHz/cm) and SI (Np/(rad·s·m)) |
| `time` | Simulation time bookkeeping |
| `log` | Tracing/logging initialization |
| `utils` | Shared low-level helpers |

## Example

Constants are the single source of truth for material values used across the stack:

```rust
use kwavers_core::constants::{C_WATER, DENSITY_WATER};

// Acoustic impedance of water, Z = rho * c.
let impedance = DENSITY_WATER * C_WATER;
assert!((impedance - 1.48e6).abs() < 0.1e6);
```

Absorption coefficients are published in dB/(MHz^y · cm) but integrated in SI:

```rust
use kwavers_core::units::db_per_mhz_cm_to_neper_per_rad_s_m;

// Water: alpha_0 = 0.0022 dB/(MHz^2 cm), power law exponent y = 2.
let alpha_si = db_per_mhz_cm_to_neper_per_rad_s_m(0.0022, 2.0);
assert!(alpha_si > 0.0);
```

## Features

Optional integrations are off by default so the foundation stays a clean leaf. Each one
only adds a `From` conversion into `KwaversError` (orphan-rule-bound to this crate):

- `channels` — `From<flume::RecvError>`
- `registration` — `From<ritk_registration::RegistrationError>`

## Documentation

- API reference: <https://docs.rs/kwavers-core>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
