# kwavers-gpu

Hephaestus-backed, provider-generic GPU compute backend for
[kwavers](https://github.com/ryancinsight/kwavers).

This crate is the single home for GPU concretions: device acquisition, provider-specific
buffers and kernels, and the concrete implementations of the compute-backend trait surface
declared in [`kwavers-solver`](https://docs.rs/kwavers-solver).

## Why a separate leaf crate

The compute-backend and FDTD-accelerator traits live in `kwavers-solver`. Per the
dependency-inversion rule, the algorithm crates depend only on those abstractions; concrete
device code lives here, downstream of the solver, and is injected at the application
boundary.

Adding a vendor means adding another Hephaestus provider implementation in this crate — not
changing an algorithm layer, and not forking a kernel per vendor. The currently implemented
provider is WGPU, because the production kernels are WGSL.

## What it provides

| Module | Responsibility |
|---|---|
| `profiling` | GPU allocation bookkeeping — available unconditionally, no device dependency |
| `gpu` | Devices, buffers, and kernels (feature `gpu`) |
| `backend` | The concrete `ComputeBackend` implementation (feature `gpu`) |
| `pstd_gpu` | GPU PSTD FFT lattice (feature `gpu`) |
| `beamforming` | Provider implementations of the beamforming operations declared in `kwavers-analysis` (feature `gpu`) |
| `validation` | GPU/CPU differential equivalence checks (feature `gpu`) |

## Features

- `gpu` — enables the device-backed modules above. Everything behind it is a complete
  implementation; the feature exists for dependency and build-size management, not as a
  stub toggle.

Device selection itself is a runtime decision: the simulation layer probes for a usable
device and reports which backend it took and why. A device that is present but fails is
surfaced as an error, never a silent downgrade to CPU.

## Verification

GPU kernels are verified against unoptimized CPU reference paths under epsilon bounds
derived from reduction depth and width — reordered floating-point reductions are not
bitwise-comparable to a sequential reference, so equality is asserted only where evaluation
order provably matches.

## Documentation

- API reference: <https://docs.rs/kwavers-gpu>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
