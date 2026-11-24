<a id="kw-sol-001"></a>

## KW-SOL-001 — PSTD exponential instability at NY=768 observed via the Python bindings (reproduction in the Rust solver required) [patch] — todo

priority: research; needs: none; scope: `kwavers-solver` PSTD (and the pykwavers build of it); milestone: none

- **Outcome:** the PSTD solver's stability boundary in grid height is known and pinned by a Rust-side test, instead of an anecdote from the bindings.
- **Driver:** observed during Build 15 (Nov 22, 2026) in the **pykwavers build** of the solver: in homogeneous water, a grid with **NY = 768 blows up exponentially (~1e29 Pa by step 300)**, while NY = 512 / 704 / 720 / 736 are stable under the same conditions. Nothing here yet distinguishes a solver defect (e.g. an FFT-size-dependent path) from a bindings/build artefact — that distinction is the work.
- **Status of evidence:** observed-via-bindings only. No Rust-side reproduction exists at filing time; no mechanism is claimed.
- **Acceptance:** a Rust solver test (or documented run) at NY = 768 in homogeneous water that either reproduces the blow-up — filed then as a solver defect with the FFT size / plan implicated — or demonstrates stability, closing this entry as a bindings-side artefact with the evidence recorded.
