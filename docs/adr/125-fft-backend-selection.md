# Select FFT execution at the solver boundary

- Status: Accepted
- Date: 2026-08-26
- Item: `backlog.md#kw-fft-hephaestus-backend-selector`
- Depends on: ADR 039 and Hephaestus provider PR 222
- Change class: [major] [arch]

## Context

Kwavers currently exposes two independent choices for pseudo-spectral
execution: `SolverType::PSTD` constructs the host solver, while
`SolverType::PstdGpu` constructs a separate GPU adapter. The distinction is an
execution backend, not a different numerical method. Encoding it as a solver
type duplicates the selection dimension and prevents one configuration from
stating which FFT provider executes the same pseudo-spectral contract.

The host path already uses Apollo's CPU FFT arithmetic directly over Leto
arrays for one-, two-, and three-dimensional transforms. The GPU PSTD path
instead owns another FFT implementation: a power-of-two-only WGSL kernel,
twiddle storage, dispatch orchestration, and a 1,024-element axis limit. That
implementation duplicates the prepared one-, two-, and three-dimensional FFT
provider in Hephaestus and prevents provider-level resource reuse.

ADR 039 establishes the ownership boundary: Leto owns dense host storage and
layout; Hephaestus owns device buffers, transfers, synchronization, and GPU
kernel dispatch. Apollo owns CPU FFT arithmetic. Kwavers owns selection and
composition, not another FFT kernel.

## Decision

Add a closed `FftBackend` configuration with `Leto` and `Hephaestus` variants.
`SolverConfiguration::fft_backend` selects once when the pseudo-spectral solver
is assembled:

- `Leto` constructs the CPU PSTD solver, whose FFT arithmetic is provided by
  Apollo over Leto storage.
- `Hephaestus` constructs the GPU-resident PSTD adapter and prepares
  Hephaestus FFT plans against its typed device buffers.

Remove `SolverType::PstdGpu`; `SolverType::PSTD` continues to identify the
numerical method, while `fft_backend` identifies its execution provider. The
default remains `FftBackend::Leto`. A requested Hephaestus backend returns the
existing typed feature or device-acquisition error when unavailable and never
falls back to Leto.

The GPU PSTD command provider uses Hephaestus's provenance-carrying grouped
sequence as its compute-pass type. Kwavers WGSL operations borrow the raw pass
from that sequence; prepared Hephaestus FFT operations encode into the same
sequence. Backend selection and capability validation therefore remain outside
the transform loops without adding command-buffer boundaries between PSTD
operations.

Delete Kwavers's FFT shader entry point, twiddle buffer, FFT pipeline, axis
dispatch functions, FFT-only push constants, and the Apollo GPU re-export
facade after the Hephaestus cutover. The Hephaestus provider accepts arbitrary
positive dimensions, so the obsolete power-of-two and 1,024-axis restrictions
are also removed. Remaining limits derive from addressable buffer and WGPU
dispatch contracts rather than from the deleted kernel.

## Failure semantics

- `FftBackend::Hephaestus` without the GPU feature is a configuration error.
- Device acquisition, preparation, shape, and encoding failures propagate as
  errors from the operation boundary; none select another backend.
- Zero dimensions and shapes whose element count cannot be represented safely
  are rejected before allocation.
- A prepared FFT cannot encode into a command sequence from another device;
  Hephaestus validates that provenance before emitting WGPU commands.

## Verification

- Shared analytical and differential tests cover forward and inverse 1-D,
  2-D, and 3-D transforms, including non-power-of-two dimensions and inverse
  normalization.
- GPU PSTD tests exercise direct and Bluestein-backed dimensions through the
  prepared provider and compare against the CPU analytical contract within a
  bound derived from single-precision FFT error growth.
- Structural checks require no `fft_1d_smem`, local FFT dispatch routine,
  twiddle buffer, Apollo GPU facade, or `SolverType::PstdGpu` reference to
  remain.
- Prepared-plan tests establish caller-buffer reuse and same-pass composition;
  allocation evidence is reported separately from timing evidence.
- Performance measurements use bounded benchmark binaries and report medians
  with confidence intervals. A slow gate is profiled or structurally
  consolidated; its timeout or workload is not weakened.

## Alternatives rejected

### Retain separate `PSTD` and `PstdGpu` solver types

This treats one backend dimension as two numerical methods, duplicates factory
branches, and leaves no single selector for the same FFT contract.

### Put GPU FFT arithmetic in Leto or Apollo

Leto is the host storage/layout provider and Apollo is the CPU FFT engine.
Either placement would contradict ADR 039 and duplicate Hephaestus's device
lifetime, typed-buffer, dispatch, and prepared-resource ownership.

### Keep the Kwavers GPU FFT as a fallback

A fallback would preserve two implementations and silently change execution
after a requested provider fails. Provider failure remains visible instead.

## Consequences

Configuration callers replace `SolverType::PstdGpu` with
`SolverType::PSTD` plus `FftBackend::Hephaestus`. Python callers make the same
backend selection independently of the solver-method enum. Host callers retain
the existing default and require no backend annotation.

Kwavers no longer owns FFT arithmetic or GPU FFT resource construction. The
remaining provider work is measurable at its owning layer: Apollo CPU kernels,
Leto host layout/transfer boundaries, and Hephaestus prepared GPU execution can
be profiled and improved without cloning the algorithm in Kwavers.
