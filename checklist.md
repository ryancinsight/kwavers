# Active correctness pass — physics, solver and facade 3.0.1

- [x] [patch] Reproduce the composed conductive sign defect and correct the shared energy balance with both callers covered by analytical regressions.
- [x] [patch] Preserve outward-positive heat-transfer power, pressure-work signs and the thermal-disable path; run the full native physics suite.
- [ ] [patch] Complete final-head hosted verification; retain the unchanged native core lint findings and hold merging for release/deployment decisions.

- [x] [patch] Verify both wheel consumers retain matching compiled build scripts before Maturin cleanup; run five workflow regressions, including exact contents/modes/links and missing-artifact rejection.
- [x] [patch] At source `2eac1317`, installed base/k-Wave wheels, 26 boundary tests, three k-Wave comparisons, 13 export-inventory tests (three existing skips), all eight benchmarks and full doctests pass under existing bounds.
- [ ] [patch] Complete KW-BUBBLE-CONDUCTION-SIGN final-head CI: the required CI gate remains blocked by three of 41 Miri cases reaching the existing 15-minute job limit; recheck after the wheel regression preflight is added.
- [x] [patch] Resolve the separately observed security advisories with Rustls 0.23.46; retain all 94 first-party Git package identities and unrelated dependency edges. The refreshed application audit passes without advisory exemptions; remove the obsolete rkyv 0.7 exemption.

- [x] [patch] Preserve exact PAM energy for contiguous and strided rows; retain checked FD gradient borrows and pass 58 focused FD regressions; all 41 facade cases and 1,104 solver tests pass (six pre-existing ignores).
- [x] [patch] Checked PAM chunks pass the unchanged full-size Miri fixture locally in 321.37 wall seconds; exact integer chunk/tail and strided regressions pass.
- [x] [patch] Per-call FD receiver reuse preserves source-scaling and objective/gradient order; 58 focused, 41 facade and 1,104 full solver tests pass. Homogeneous full-size Miri passes locally in 358.44 wall seconds.
- [x] [patch] Replace Born adjoint coordinate projection with one checked dense gradient borrow. The exact dyadic coordinate oracle and all 58 existing focused FD regressions pass; preserve arithmetic, accumulation order, stateless operators and O(voxels) scratch.
- [x] [patch] Recheck the complete native solver and facade suites: 1,105 solver tests pass with six existing ignores, and all 41 facade cases pass.
- [ ] [patch] Complete the authorized Miri performance increment and final-head verification under unchanged 15-minute limits, full scenarios, numerical tolerances and safety checks. Merge, publication and deployment remain excluded.
