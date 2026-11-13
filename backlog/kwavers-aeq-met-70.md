<a id="kwavers-aeq-met-70"></a>

## KWAVERS-AEQ-MET-70 — Inventory raw public metrics in the remaining transducer families [arch] — todo

priority: architecture; needs: none; basis: 4a5c8db8889e807011518f4aaf8013f18729e698; scope: every public transducer, sensor-array and array-geometry type or function in `crates/` that no closed `KWAVERS-AEQ-MET-*` record typed, including `crates/kwavers-transducer/src/{transducers/phased_array,transducers/physics,bulk_piezo.rs,passive_acoustic_mapping,curvilinear,kwave_array,basic,factory,parallel.rs}`, and their tests

- **Outcome:** an inventory of every public raw physical field and signature in that set, each classified as typed with Aequitas quantities or as an explicit storage or serialization boundary, then split into one typed item per family. Families a closed `AEQ-MET` record typed (focused, hemispherical, array_2d, rectangular, MEMS, beamforming, ultrafast, acquisition geometry, design, the PAM analysis in `kwavers-analysis`) are out; `flexible/` stays with `KWAVERS-AEQ-MET-FLEX`.
- **Evidence:** at 4a5c8db88, `kwavers-transducer` `transducers/phased_array/config.rs:19,21,23,25,27`, `transducers/physics/geometry.rs:24`, `bulk_piezo.rs:203` and `passive_acoustic_mapping/geometry.rs:10,18,19,26,34` (`PamArrayGeometry`), `kwavers-physics` `.../ultrasound/frequency_domain_fwi.rs:440` and `kwavers-receiver/src/array.rs:82` take raw `f64` metrics. The deleted gap records (basis `2603e307c`, `AEQ-MET-56..63`) said other array families stay audit candidates.
- **Acceptance:** a workspace scan for public raw-`f64` metric fields and parameters in that set, its pattern and revision recorded, lists every hit with file:line; each hit maps to a filed typed item or to a named boundary with its reason; a rerun of the scan at that revision reproduces the list.
- **Next step:** write and run the scan, record its result here, and file the per-family items.
