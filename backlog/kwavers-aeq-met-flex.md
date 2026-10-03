<a id="kwavers-aeq-met-flex"></a>

## KWAVERS-AEQ-MET-FLEX — Type the flexible-array dynamic metrics [arch] [major] — blocked

priority: architecture; needs: none; scope: `crates/kwavers-transducer/src/flexible/{array,beamforming,geometry,config}.rs`, direct callers and tests, ADR 070

- **Outcome:** audit and type the flexible-array dynamic metrics (timestamps, focus/speed/delay contracts, calibration confidence, deformation outputs) with Aequitas quantities; preserve raw dense mesh and signal storage boundaries.
- **Blocker:** `crates/kwavers-transducer/src/flexible/array.rs` was dirty in a peer's tree (recorded 2026-08-02). Peer claims expire after an hour without a commit, so the blocker is stale; re-validate against the fetched default branch before claiming.
- **Re-open trigger:** the peer's work integrated or its scope released (already satisfied if no live claim holds the file).
- **Acceptance:** raw public signatures in the scoped files carry typed quantities; value-semantic regressions, package gates and the raw-public-signature scan pass.
