# kwavers-physics

Physics for [kwavers](https://github.com/ryancinsight/kwavers): nonlinear acoustics, bubble
dynamics, thermal transport, optics, chemistry, and elastic waves.

This crate holds the governing equations and their discretized right-hand sides. It does
not own the time-stepping loop — that belongs to
[`kwavers-solver`](https://docs.rs/kwavers-solver) — and it does not own material data,
which belongs to [`kwavers-medium`](https://docs.rs/kwavers-medium). What it owns is the
physics itself, stated once and shared by every solver that integrates it.

## Domains

| Module | Physics |
|---|---|
| `foundations` | Wave-equation specifications and coupling traits — the SSOT the other modules implement |
| `acoustics` | Linear and nonlinear propagation, cavitation, bubble dynamics, conservation validation |
| `thermal` | Heat transfer, bioheat/Pennes diffusion, thermal dose |
| `optics` | Light transport and sonoluminescence |
| `photoacoustics` | Thermoelastic optical-to-acoustic coupling |
| `electromagnetic` | Electromagnetic wave equations |
| `chemistry` | Sonochemical kinetics |
| `therapy` | Therapeutic-ultrasound domain models (microbubble dynamics, modality types) |
| `analytical` | Closed-form kernels used as oracles and for fast planner queries |
| `field_surrogate` | Cached focal-pressure kernels for interactive planning |
| `factory` | Capability-driven plugin catalog (`PhysicsConfig` → `PluginManager`) |

## Design

Physics specifications live in `foundations` as traits; each domain module implements them.
A solver binds to the specification, not to a concrete physics implementation, so adding a
constitutive model is a new implementation rather than a change to the integration loop.

The `analytical` module is not a convenience layer — its closed-form solutions are the
independent oracles the numerical paths are verified against (conservation identities,
manufactured solutions, published reference cases).

## Related crates

- Time integration and inverse problems — [`kwavers-solver`](https://docs.rs/kwavers-solver)
- Material properties — [`kwavers-medium`](https://docs.rs/kwavers-medium)
- Field component layout — [`kwavers-field`](https://docs.rs/kwavers-field)

## Documentation

- API reference: <https://docs.rs/kwavers-physics>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
