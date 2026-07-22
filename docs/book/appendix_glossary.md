# Appendix C — Glossary

## Atlas Stack Terms

**ADR** — Architecture Decision Record. Numbered documents in `docs/ADR/` that
record significant design choices and their rationale.

**Atlas** — The umbrella workspace and monorepo for the physics simulation
suite. Individual crates are developed in sibling repos and linked via
git submodules.

**coeus** — The deep learning framework replacing `burn`. Backends: `leto`
(CPU) and `hephaestus` (GPU). The `coeus-autograd` crate provides the
differentiable programming layer for PINNs and learned physics.

**COW** — Copy-On-Write. The `std::borrow::Cow<'_, T>` pattern used throughout
to avoid unnecessary allocations; `Cow::Borrowed` paths are zero-copy.

**DIP** — Dependency Inversion Principle. High-level physics modules depend on
abstractions (traits), not on concrete array types.

**DRY** — Don't Repeat Yourself. One implementation per algorithm; no duplicated
sparse, FFT, or geometry code across crates.

**eunomia** — The numeric traits crate replacing `num-traits` and `num-complex`.
Provides `Scalar`, `RealField`, `Complex64`, `Float`, and related abstractions.

**GAT** — Generic Associated Types. Used in `leto` for lending iterators
(`for<'a> fn(&'a Self) -> Self::Item<'a>`) enabling zero-copy windowed views.

**hephaestus** — The GPU array library aligned with the `leto` API. Provides
`CsrArray`, dense `Array*`, and `BackendOps` for GPU kernel dispatch.

**hermes** — The SIMD abstraction replacing direct intrinsics. Runtime dispatch
to AVX2, NEON, or scalar fallback with a unified API.

**leto** — The CPU array library replacing `ndarray` and `nalgebra`. Provides
`Array1/2/3/D`, `ArrayView*`, `VecStorage`, sparse formats (CSR/CSC/COO),
and the geometry types from `nalgebra`.

**melinoe** — Branded types (internal halo). Provides type-safe wrappers via
`Halo<T, Brand>` for domain quantities (pressure, frequency, etc.) without
runtime overhead.

**mnemosyne** — The memory allocator crate. Provides thread-local slab pools,
arena strategies, and integration hooks for `leto` storage.

**moirai** — The concurrency runtime replacing `tokio` and `rayon`. Provides
data-parallel `Adaptive` scheduling, async execution, and cooperative tasks.

**phantom type** — A zero-sized type parameter used to carry information at
compile time without storing data. Used extensively in `melinoe` brands and
`themis` strategy types.

**PINN** — Physics-Informed Neural Network. A neural network whose loss function
includes PDE residuals, boundary, and initial conditions. Implemented in
`kwavers-solver::inverse::pinn` using `coeus-autograd`.

**PSTD** — Pseudo-Spectral Time Domain. The primary kwavers acoustic wave solver,
implemented using `apollo-fft` for k-space updates.

**ritk** — Radiation and Imaging Toolkit. Image I/O, processing, registration,
and segmentation. Replaces ITK for the Atlas physics suite.

**SoC** — Separation of Concerns. Each module owns exactly one responsibility.

**SRP** — Single Responsibility Principle. Each module has one reason to change.

**SSOT** — Single Source of Truth. Each algorithm or data type lives in exactly
one place; all consumers reference that location.

**themis** — The allocation strategy crate. Provides arena, region, and pool
strategies consumed by `mnemosyne`.

**ZST** — Zero-Sized Type. A type that occupies no memory, used for strategy
markers, phantom brands, and const-generic state machines.
