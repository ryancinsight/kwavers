<a id="kw-gpu-048"></a>

## KW-GPU-048 — GPU PSTD output and dispatch honesty [major] — review

priority: correctness; needs: none; scope: `kwavers-gpu` PSTD output contract, `kwavers-simulation` GPU adapter and runner dispatch, `kwavers-solver` selection docs, ADR-037

- **Driver:** a downstream consumer must distinguish a real final-state GPU result from its CPU peak-envelope planner and must receive an explicit unsupported error for the CT-scale GPU constraint.
- **Acceptance:** a GPU batch returns only requested real outputs; final pressure and staggered velocity fields transfer from provider buffers when requested; `SolverType::PstdGpu` never executes CPU PSTD as a substitute.
- **Decision:** [ADR-037](docs/adr/037-gpu-pstd-output-contract.md).
- **Evidence:** GPU-feature Nextest 144/144 (one skipped) under the serialized WGPU test group; default scoped suite 1036/1036 (four skipped); warning-denied Clippy and all-feature Rustdoc clean; Hephaestus owns the aggregate buffer-limit mapping (Kwavers keeps its ordinary provider limit at 8 and requests 24/32 only for PSTD layouts).
- **Remaining capability gap:** whole-grid provider capacity stays a per-plan constraint; the peak output is KW-GPU-062.
- **Next step:** re-validate at the default branch; close if the adapter and dispatch tests are present.
