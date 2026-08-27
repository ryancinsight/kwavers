# kwavers-simulation

Simulation orchestration for [kwavers](https://github.com/ryancinsight/kwavers): builders,
runners, multi-physics coupling, modality pipelines, backends, and result I/O.

This is where the pieces become a run. The layers below supply a grid, a medium, sources,
receivers, and solvers; this crate wires them into a configured simulation, steps it,
couples the physics domains that interact, and writes the results out.

## What it provides

| Module | Responsibility |
|---|---|
| `builder` / `configuration` / `configs` | `ConfigurationBuilder`, `Configuration`, per-physics config records |
| `core` | `CoreSimulation`, `SimulationBuilder`, `SimulationResult`, run statistics |
| `runner` | `SimulationRunner` — the time loop and its progress reporting |
| `setup` | `SimulationSetup`, `SimulationComponents` — assembled run inputs |
| `multi_physics` | Coupled solvers and field-coupling strategies across domains |
| `manager` | `PhysicsManager` — physics plugin lifecycle within a run |
| `modalities` | End-to-end modality pipelines (e.g. `PhotoacousticSimulator`) |
| `backends` / `dispatch` / `solver_factory` | Backend selection and solver construction |
| `parameters` | Output field types, formats, and performance parameters |
| `io` | Result serialization |
| `therapy` / `imaging` | Modality-specific orchestration entry points |

## Backend selection

PSTD callers select `FftBackend::Leto` or `FftBackend::Hephaestus` independently
of `SolverType::PSTD`. Leto runs Apollo's CPU transform over Leto storage;
Hephaestus prepares device-resident transforms and keeps the PSTD fields on the
GPU. A requested Hephaestus acquisition or execution failure is returned and
never silently falls back to Leto.

## Related crates

- Solvers and the backend trait seam — [`kwavers-solver`](https://docs.rs/kwavers-solver)
- Concrete GPU backends — [`kwavers-gpu`](https://docs.rs/kwavers-gpu)
- Post-run analysis — [`kwavers-analysis`](https://docs.rs/kwavers-analysis)

## Documentation

- API reference: <https://docs.rs/kwavers-simulation>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
