# Driver 0.5 external qualification

Reviewed main `a949d8d429fa3965d3ae500958c58d512bdce41f` and published 0.5.0 (`bde4385ea3a7d18ddecbcbd0e00f7075d3958c3ab437a489768b62de6b5f9023`, SHA-256). All 251 package Rust sources match main.

| Configuration | Registry 0.5.0 passed/run | Proposed 0.5.1 passed/run |
|---|---:|---:|
| Neither | 4/7 | 7/7 |
| Default I/O | 5/11 | 11/11 |
| Transducer only | 5/7 | 7/7 |
| Both | 6/11 | 11/11 |

The [external consumer](../../scripts/tests/driver_consumer/README.md) exercises the 4 mm and 0.3 mm length cases, 1 nm neighbors of the 0.5 mm limit, per-layer skew, catalog selection, per-channel power, package thermal response, the 96-lane experiment and optional KiCad exports. Rust 1.97.0 on Apple Silicon resolves Aequitas 0.5.0, Eunomia 0.12.1 and optional Transducer 6.0.0. The published input comes entirely from crates.io. The repair rehearsal changes only the Driver dependency to the unpacked candidate.

Six repairs cover invalid grid pitch, legacy manifest round trips, analytical pulse-average intensity, exported segment precision, through-drill rules and JSON filename encoding. In the burst-power fixture the device dissipates 35 mW and the resistor dissipates 70 mW. In the fallback experiment the 3.375 MPa sinusoidal peak gives 384.818412 W/cm². The original report doubled this value.

The candidate passed all 223 executed SemVer checks in each feature set (30 tool skips per set), default and all-feature internal suites, and strict all-feature Clippy. Two internal component-artifact tests return early without their unshipped board fixtures. The native KiCad file checks cover the emitted PCB electrical subset, schematic labels, rule text and project JSON. External KiCad DRC/ERC, full-format interoperability, board measurements and fabrication qualification remain open.

Under [proposed ADR 146](../adr/146-stable-driver-surface.md), the candidate stable boundary covers documented units, valid input ranges, audit counter semantics, specified electrical and thermal outputs, checked manifest schema forms and named I/O subsets. Document or narrow other reachable public items before beta. The legacy intensity helper mixes pulse and train averages through its duty argument. A separately scoped additive API could name both explicitly. No 0.6.0 addition is required by this electronics consumer. Repeat on the final revision and published package before closing the release gates.
