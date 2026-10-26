<a id="kw-origin-sampled-medium"></a>

## KW-ORIGIN-SAMPLED-MEDIUM — Non-PSTD paths read a heterogeneous medium at the origin [patch] — todo

priority: correctness; needs: none; scope: crates/kwavers-solver (kzk), crates/kwavers-therapy, crates/kwavers-medium/src/heterogeneous

- **Finding.** These sites read medium properties at `(0, 0, 0)` and use the result for the whole run, so a heterogeneous medium is flattened to its origin voxel:
  - `kzk/plugin.rs:123-124`
  - `kwavers-therapy/.../orchestrator/execution.rs:239-240` and `341-342`
  - `kwavers-medium/src/heterogeneous/ct.rs:146`
  - `heterogeneous/core/structure.rs:265`

  #966 removed the PSTD instances (ADR 120).
- **Next step.** For each site, decide whether a scalar is valid (the medium is homogeneous by contract) or whether per-voxel data must flow. Fix each, or record why the scalar is valid. Add a conformance regex for `(0.0, 0.0, 0.0, grid)` property reads.
- **Acceptance.** No unexplained origin sample remains. A heterogeneous test medium that is zero at the origin behaves correctly at each fixed site.
