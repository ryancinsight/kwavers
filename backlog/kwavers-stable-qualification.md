<a id="kwavers-stable-qualification"></a>
## KWAVERS-STABLE-QUALIFICATION — kwavers stable-release qualification [patch] — todo
- priority: verification; milestone: M-solvers-core; basis: `cb2c7ecdb714`.
- Outcome: Restore the release baseline and qualify the current simulation cohort. Follow the [release scope](../docs/adr/138-release-scope-4-0.md) and [milestone exits](../backlog/milestones.md). [Failing security job](https://github.com/ryancinsight/kwavers/actions/runs/45037646079/job/137255042083).
- Representative manifest `kwavers 5.0.0`. Target existing stable majors and Driver 1.0.0.
- [ ] [patch] Resolve RUSTSEC-2026-0592 on the locked graph and rerun Security Audit and CI gate from run 45037646079. Record the repaired dependency path and final-head jobs.
- [ ] [patch] Replay the published Physics and Solver 5.1.0 conservation and reference-trajectory cases from an independent consumer. Check the retained state across pulse and quiet intervals and report numerical error.
- [ ] [major] Keep the facade and layer majors monotonic. Qualify the 0.5.0 Driver public surface under [proposed ADR 146](../docs/adr/146-stable-driver-surface.md), with a documented support boundary, compatibility check and downstream consumer before its freeze.
- [ ] [patch] Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).
