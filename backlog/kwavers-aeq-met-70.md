<a id="kwavers-aeq-met-70"></a>

## KWAVERS-AEQ-MET-70 — Inventory raw public metrics in the remaining transducer families [arch] — todo

priority: architecture; needs: none; basis: 4a5c8db8889e807011518f4aaf8013f18729e698; scope: `crates/kwavers-transducer/src/{transducers/phased_array,transducers/physics,bulk_piezo.rs,curvilinear,kwave_array,basic,factory,parallel.rs}`, their direct callers and tests

- **Outcome:** an inventory of every public raw physical field and signature in the transducer families that no `KWAVERS-AEQ-MET-*` record covers, each classified as typed with Aequitas quantities or as an explicit storage or serialization boundary, then split into one typed item per family. Closed families (focused, hemispherical, array_2d, MEMS, beamforming, ultrafast, acquisition geometry, design) are out of scope; `flexible/` stays with `KWAVERS-AEQ-MET-FLEX`.
- **Evidence:** at 4a5c8db88, `transducers/phased_array/config.rs:19,21,23,27` (`element_spacing`, `element_width`, `element_height`, `frequency`: `f64`), `:25` (`center_position: (f64, f64, f64)`), `transducers/physics/geometry.rs:24` (`pitch: f64`) and `bulk_piezo.rs:203` (`sound_speed: f64`) are public raw metrics. The closed gap records (basis `2603e307c`, `AEQ-MET-56..63`) said other families stay audit candidates until inspected.
- **Acceptance:** the inventory lists each family's public raw lines with file:line at a stated revision, and every line either maps to a filed typed item or to a named boundary with its reason; no family named in scope is left unclassified.
- **Next step:** run the raw-public-signature scan over the scoped paths, record the result in this item, and file the per-family items.
