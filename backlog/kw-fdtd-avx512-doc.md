<a id="kw-fdtd-avx512-doc"></a>

## KW-FDTD-AVX512-DOC — The FDTD solver doc and a standing fact advertise AVX-512 kernels that were deleted [patch] — todo

priority: verification; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `crates/kwavers-solver/src/forward/fdtd/solver/mod.rs`, `gap_audit.md`

- **Finding.** `solver/mod.rs:107-108` documents "optional AVX-512 SIMD pressure update kernels". a5e6ad554 (2026-09-16, `refactor(solver)!: Remove the unused FDTD stencil processors`, an ancestor of `origin/main`) deleted `forward/fdtd/avx512_stencil/`, `simd_stencil/` and `dispatch/`. `git grep -n -i avx origin/main -- crates/kwavers-solver/src/forward/fdtd` returns only that line.
- **Same cause, second place.** The standing fact at `gap_audit.md:4582-4583` says FDTD `simd_stencil` and `avx512_stencil` "are NOT duplication"; both modules are gone, so the fact guards nothing.
- **Acceptance:** the grep above returns no hit; the doc names only the staggered-grid, CPML and Westervelt paths the solver has; `gap_audit.md` holds no `avx512_stencil` or `simd_stencil` text.
- **Next step:** edit the two doc lines, then delete the standing fact's two lines.
