<a id="kw-fdtd-avx512-doc"></a>

## KW-FDTD-AVX512-DOC — The FDTD solver doc advertises AVX-512 kernels that were deleted [patch] — todo

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `crates/kwavers-solver/src/forward/fdtd/solver/mod.rs`, `backlog/kw-sol-054.md`

- **Finding.** `solver/mod.rs:107-108` documents "optional AVX-512 SIMD pressure update kernels". a5e6ad554 (2026-09-16, `refactor(solver)!: Remove the unused FDTD stencil processors`, an ancestor of `origin/main`) deleted `forward/fdtd/avx512_stencil/`, `simd_stencil/` and `dispatch/`. `git grep -n -i avx origin/main -- crates/kwavers-solver/src/forward/fdtd` returns only that line.
- **Same cause, second record.** `KW-SOL-054` (`Repair AVX-512 FDTD layout contract`) scopes "AVX-512 FDTD kernels and their `avx512` tests", which no longer exist; its acceptance cannot be run.
- **Acceptance:** the grep above returns no hit; the doc names only the staggered-grid, CPML and Westervelt paths the solver has; `KW-SOL-054` is deleted with a5e6ad554 cited in the commit.
- **Next step:** edit the two doc lines, then delete `KW-SOL-054` and its index line.
