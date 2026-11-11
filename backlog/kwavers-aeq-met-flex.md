<a id="kwavers-aeq-met-flex"></a>

## KWAVERS-AEQ-MET-FLEX — Type the flexible-array dynamic metrics [arch] [major] — todo

priority: architecture; needs: none; basis: a3a72689be0fbfed3f64231c52186eac79dedc15; scope: `crates/kwavers-transducer/src/flexible/{array,beamforming,geometry,config}.rs`, direct callers and tests, ADR 070

- **Outcome:** audit and type the flexible-array dynamic metrics (timestamps, focus/speed/delay contracts, calibration confidence, deformation outputs) with Aequitas quantities; preserve raw dense mesh and signal storage boundaries.
- **Acceptance:** raw public signatures in the scoped files carry typed quantities; value-semantic regressions, package gates and the raw-public-signature scan pass.
- **Open clause.** "raw public signatures in the scoped files carry typed quantities": `pub fn centroid(&self) -> [f64; 3]` (`geometry.rs:121`) returns a position as raw metres, while ADR 070 types focus coordinates as `Length` (its line 46) and `focusing_delays` in `array.rs:286` takes a position as `[Length<f64>; 3]`. The other raw public signatures a one-line `git grep` finds in the four files are position or normal arrays (`update_positions`, `geometry.rs:62`; `per_element_curvature`, `beamforming.rs:102`), which ADR 070 keeps as storage (its lines 50 to 52), and a unit direction (`steering_delays`, `array.rs:293`), which ADR 070 does not name; signatures that span lines were not scanned. An earlier resolution note recorded typed spacing, noise and threshold parameters and green package gates on a takeover branch; it was not re-run here.
- **Next step:** return `[Length<f64>; 3]` from `centroid` and update its callers; scan the multi-line signatures of the four files; re-run the package gates; then close.
