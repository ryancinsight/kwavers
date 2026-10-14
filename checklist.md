# Active correctness pass — kwavers-physics 3.0.1

- [x] [patch] Reproduce the composed conductive sign defect and correct the shared energy balance with both callers covered by analytical regressions.
- [x] [patch] Preserve outward-positive heat-transfer power, pressure-work signs and the thermal-disable path; run the full native physics suite.
- [ ] [patch] Complete final-head hosted verification; retain the unchanged native core lint findings and hold merging for release/deployment decisions.

- [ ] KW-BUBBLE-CONDUCTION-SIGN CI repair: execute bounded full doctests, all benchmarks, installed-wheel contracts and real nonempty Miri harnesses on the final revision.
