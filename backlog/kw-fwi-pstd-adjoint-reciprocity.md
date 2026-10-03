<a id="kw-fwi-pstd-adjoint-reciprocity"></a>

## KW-FWI-PSTD-ADJOINT-RECIPROCITY — The PSTD adjoint-reciprocity check never ran [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers-solver/src/inverse/fwi/time_domain/forward.rs`; FWI adjoint tests

- **Verified:** `FwiParameters::build_solver_for_forward` dispatches `SolverType::{FDTD, PSTD}` to `build_fdtd_boxed`/`build_pstd_boxed` and returns `Box<dyn Solver>`; the FDTD forward smoke test and the unsupported-type rejection test pass.
- **Not verified:** the PSTD adjoint-reciprocity check, the one live residual of T15b.
- **Acceptance:** a reciprocity test over the PSTD adjoint, or a recorded decision that the FDTD check covers the contract.
- **Next step:** write the PSTD reciprocity test against the FDTD one as the structural template.
