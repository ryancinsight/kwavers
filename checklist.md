# Active correctness pass — physics, solver and facade 3.0.1

- [x] [patch] Reproduce the composed conductive sign defect and correct the shared energy balance with both callers covered by analytical regressions.
- [x] [patch] Preserve outward-positive heat-transfer power, pressure-work signs and the thermal-disable path; run the full native physics suite.
- [ ] [patch] Complete final-head hosted verification; retain the unchanged native core lint findings and hold merging for release/deployment decisions.

- [ ] KW-BUBBLE-CONDUCTION-SIGN CI repair: execute bounded full doctests, all benchmarks, installed-wheel contracts and real nonempty Miri harnesses on the final revision.

- [x] [patch] Preserve exact PAM energy for contiguous and strided rows; retain checked FD gradient borrows and pass 58 focused FD regressions; all 41 facade cases and 1,104 solver tests pass (six pre-existing ignores).
- [x] [patch] Checked PAM chunks pass the unchanged full-size Miri fixture locally in 321.37 wall seconds; exact integer chunk/tail and strided regressions pass.
- [x] [patch] Per-call FD receiver reuse preserves source-scaling and objective/gradient order; 58 focused, 41 facade and 1,104 full solver tests pass. Homogeneous full-size Miri passes locally in 358.44 wall seconds.
- [ ] [patch] Decide bounded full-doctest/wheel/Miri resources; the three-line 10/30-minute proposal is reviewable and remains unapplied pending approval.
