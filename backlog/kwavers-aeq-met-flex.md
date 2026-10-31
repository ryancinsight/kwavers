<a id="kwavers-aeq-met-flex"></a>

## KWAVERS-AEQ-MET-FLEX — Type the flexible-array dynamic metrics [arch] [major] — blocked

priority: architecture; needs: none; scope: `crates/kwavers-transducer/src/flexible/{array,beamforming,geometry,config}.rs`, direct callers and tests, ADR 070

- **Outcome:** audit and type the flexible-array dynamic metrics (timestamps, focus/speed/delay contracts, calibration confidence, deformation outputs) with Aequitas quantities; preserve raw dense mesh and signal storage boundaries.
- **Blocker (discharged 2026-10-28):** `crates/kwavers-transducer/src/flexible/array.rs` was dirty in a peer's tree (recorded 2026-08-02). Peer claims expire after an hour without a commit, so the blocker is stale; re-validate against the fetched default branch before claiming.
- **Resolution:** no live lease held the scope; taken over on takeover/kwavers-met-flex. Typed flat_array spacing (Length), tracking-noise const (Length), poisson_ratio and detection threshold (Dimensionless) + default-contract regression test; 258+2 transducer tests green, clippy -D warnings clean. Raw mesh/signal arrays and unit direction vectors stay raw per ADR 070 mesh-boundary rule. Re-open trigger (no longer blocking): the peer's work integrated or its scope released (already satisfied if no live claim holds the file).
- **Acceptance:** raw public signatures in the scoped files carry typed quantities; value-semantic regressions, package gates and the raw-public-signature scan pass.
