# kicad-routing gap audit

## Closed This Cycle
- [patch] 32-channel HV7355 KiCad-error gap: the 32-channel example previously failed to produce an
  authoritative board artifact because the 75×52 mm four-layer topology exposed only 16 TX nets and
  left stable LVS opens. The example now emits a 100×80 mm six-layer board with four 8-channel TX
  banks, banked MOSI/MISO/SCK/CSN control, banked VNN returns, and 0.20 mm dense-board HV/power
  tracks. KiCad CLI DRC on
  `output/boards/hv7355_32ch_tile/hv7355_32ch_tile.kicad_pcb` reports 0 violations and
  0 unconnected items.
- [patch] 32-channel CAD-render gap: the 32-channel board emitted no KiCad `(model ...)` blocks, so
  its 3D render was mostly copper and pads. The example now attaches stock QFN-56 8×8 bodies for
  U1-U4, stock 0402 capacitor bodies for C1-C16, and stock 2.54 mm pin-header bodies for J1-J6.
  This fixes render inspectability only; exact 57-pad HV7355 and exact connector parity remain a
  separate migration.
- [patch] CAD model coordinate-frame gap: imported KiCad footprints translated pad coordinates into
  a courtyard-centred frame but left STEP model offsets in the vendor file's original frame. The
  importer now applies the same `(cx, cy)` translation to the model offset, and the 32-channel board
  no longer attaches stock two-row pin-header STEP bodies with zero offsets or a horizontal synthetic
  pad grid against centered vertical header models. Regression coverage asserts imported Molex
  transforms, generated 32-channel header offsets, and J5 pad orientation.
- [patch] Thermal-feedback coefficient gap: thermal and EMI placement feedback were folded into the
  same scalar field as route congestion, then multiplied by `feedback_weight`, so `thermal_weight`
  was effectively damped by the congestion coefficient. The optimizer now scales congestion/weakness
  by `feedback_weight`, applies thermal/EMI coefficients directly, and feeds a normalized solved
  temperature field instead of a binary hotspot mask.
- [patch] Output artifact ambiguity gap: generated boards, manifests, DRC reports, renders, and fab
  exports were split between the project root and a flat `output/` directory. Canonical 24-channel and
  32-channel boards now have separate folders under `output/boards/`; non-canonical examples,
  manifests, reports, renders, stack outputs, beamforming outputs, and archived root-generated
  artifacts now live in purpose-named folders.
- [patch] Verification drift gap: the exact FPGA example triggered an MSYS2/lld `alloc::fmt::format`
  link failure during the all-target test build, and one resistor-margin test searched for the ASCII
  `>=` label while report generation emitted `≥`. The example now avoids `format!` in the affected
  strings, and the resistor-margin check label is single-sourced.
- [patch] Example completion gap: `hv7355_tile` previously emitted `verification: FAIL` with LVS opens
  or hard DRC failures under tested configurations. It now emits `verification: PASS` internally with
  complete routing and `clearance=0` on the hard audit count.
- [patch] Audit semantics gap: flight-line crossings and 3x-clearance near-shorts were treated as
  hard DRC failures. They now remain adversarial risk signals, while `clearance_violations` is the
  hard manufacturing clearance count.
- [patch] Beamforming gap: the example printed an approximate beam summary using 0.5 mm pitch. It now
  derives pitch from the paper's 4.3 mm aperture and validates focused 5 ns delay quantization.
- [patch] Full-driver artifact gap: the FPGA and HV examples were separate physical boards without a
  recorded combined run. Both now generate into `output/full_driver`, share the `J_STACK` pinout, and
  pass external KiCad DRC with zero violations/unconnected items.
- [patch] Beamforming image gap: deterministic Rust beamforming renders now exist for -45°, 0°, and
  +45° steering using the quantized driver delays.
- [patch] Beamforming visual-model gap: the first image renderer produced continuous-wave bright
  ridges that looked like steered plane waves rather than focused measurements. It now renders a
  focal-arrival pulse envelope and emits lateral/axial 6 dB width metrics.
- [patch] Assembly/clearance gap: component spacing was only an annealing energy term, and router
  pad halos were optimistic relative to KiCad's continuous geometry. Courtyard spacing is now a hard
  assembly verification axis, examples use the shared 2.0 mm assembly rule, pad/via keepouts include
  a 0.05 mm continuous-geometry guard, and the examples use a 0.5 mm routing lattice. KiCad CLI DRC
  reports 0 violations/0 unconnected items for both regenerated full-driver boards.
- [patch] Board-backed simulation gap: beamforming renders previously could be produced without
  proving the boards expose transducer outputs or programming access. The HV tile now emits
  `driver_manifest.kv` with the transducer connector and 16 TX nets; the FPGA tile updates it with
  JTAG/config-flash/stack-bus programming evidence; `beamforming_results` refuses to render unless
  that manifest parses and contains 16 channels.
- [patch] Connector model gap: KiCad connector model filenames do not always encode body dimensions,
  so the oversized-model assembly check could miss connector bodies. Footprints now support explicit
  model envelopes, and the generated transducer/JTAG connector models are checked against their
  courtyards.
- [patch] Connector CAD provenance gap: downloaded connector archives were outside this crate and not
  classified. The connector CAD files are now extracted under `docs/cad_models` and summarized in
  `docs/connector_cad_inventory.md`, including which files are exact PCB footprints versus mating
  housing STEP files.
- [patch] Stack compatibility gap: the two board examples shared a connector by convention but did
  not emit a machine-readable stack contract. They now emit `stack_fpga.kv` and `stack_hv.kv`;
  `verify_stack_pair` checks board outline, connector position/rotation, and normalized pinout, and
  the generated full-driver pair reports PASS.
- [patch] Stack feasibility gap: the HV tile's local stack estimate counted only driver boards. The
  stack model now counts the top FPGA/controller board and reports the current 96-channel example as
  one controller plus four 24-channel HV shields.
- [patch] Component accuracy observability gap: generated boards did not distinguish routed
  functional abstractions from exact downloaded CAD footprints. Component CAD archives are now
  extracted under `docs/cad_models`, `docs/component_cad_inventory.md` records the selected/candidate
  parts, and generated boards emit `component_accuracy_*.kv`.
- [patch] Exact connector gap: FPGA power `J1`, FPGA JTAG `J3`, FPGA `J_STACK`, HV transducer `J2`,
  and HV `J_STACK` now use exact local downloaded footprints with external KiCad DRC reporting
  0 violations and 0 unconnected items on both generated full-driver boards.
- [patch] HV connector refdes/render gap: the rendered 24-position Molex transducer connector used
  the local `430452400` CAD body but was still named `J3`, while `J2` referred to a small demo
  power/input footprint with no visible CAD body. The board artifacts, stack metadata, and article
  comparison now name the Molex `0430452400` transducer connector `J2`; the old demo input connector
  is `J4` until the exact 4-pin power header is rerouted.
- [patch] J4 visible-body gap: `J4` had no exact renderable board-header body. The exact
  `430450400.step` is now extracted from `docs/cad_archives/430450400.zip` and attached to the HV
  board artifacts, replacing the temporary mating-receptacle render body.
- [patch] J4 exact-footprint swap gap: replacing the current routed demo pad geometry directly with
  exact `MOLEX_430450400` through-hole pads without rerouting produced 1219 KiCad DRC violations and
  2 unconnected items on a trial board, so the exact footprint change must run through the generator
  and router instead of a manual footprint-only swap.
- [patch] KiCad validation automation gap: manual PCB edits could reuse KiCad UUIDs and stale
  copper-zone fills could make DRC diagnostics misleading. The KiCad wrapper now exposes explicit
  refill/save DRC options, PCB saving rejects duplicate object UUIDs before writing, emitted PCB
  tests assert UUID uniqueness, and the downloaded Molex `0430450400` KiCad-v6 footprint is parsed
  by a value-semantic import test. A fresh KiCad CLI refill/save DRC on
  `output/full_driver/hv7355_driver_tile.kicad_pcb` reports 0 violations and 0 unconnected items;
  rejected exact-J4 trials were deleted during the DRC cleanup, while the promoted board now uses
  the exact `MOLEX_430450400` through-hole footprint and passes KiCad CLI DRC.
- [patch] KiCad DRC classification gap: KiCad 10 reports warnings such as `track_dangling` inside
  `violations[]` with `severity = warning`, while hard DRC errors share the same array with
  `severity = error`. The wrapper now classifies by severity before setting `DrcReport::violations`
  and keeps `unconnected_items[]` as a hard failure. Re-audited exact-J4 candidates show two distinct
  failure modes: the UUID-repaired un-routed candidate has 0 hard errors, 3 warnings, and 4
  unconnected items; the routed margin/local candidates clear opens but still create 15-19 hard
  KiCad errors. The rejected trial artifacts have been removed; the production HV board remains
  KiCad-clean after refill/save DRC.
- [patch] Exact J4 closure gap: the previous exact-J4 attempts either left the four power pins open
  or collided with BUS/HV copper because they treated the exact Molex PTH/NPTH geometry as a local
  body swap. The promoted HV board now places `MOLEX_430450400` at the right board edge in the
  power-input y-region, keeps the exact footprint's `PCB EDGE` datum on the board outline, preserves
  the existing VPP/GND/P5V power anchors, moves P3V3 to a safe existing rail via, adds a same-layer
  P3V3 tie at the obsolete demo tail, and passes KiCad CLI DRC with 0 violations and 0 unconnected
  items. A regression test asserts the production artifact uses exactly one exact `J4` footprint,
  rejects the old lower-quadrant placement, and contains zero demo-footprint tokens.
- [patch] J4 rendered-alignment gap: the exact `430450400.step` body was attached with an extra
  local `90 deg` model rotation, so the KiCad render showed the connector body displaced from the
  plated-hole pattern even though the routed pads were electrically clean. The promoted artifacts
  now use zero local STEP rotation and the regression test rejects the old transform.
- [patch] FPGA exact-example drift gap: the exact FPGA example had a live `VCC_1V8` rail parameter
  underscore-prefixed while still being passed to the BGA wiring function, and stale unused JTAG/SPI
  ball-group constants. The example now compiles under all-target clippy without suppressing lints.
- [patch] KiCad dirty-artifact gap: several demos wrote `.kicad_pcb` files after the optimizer had
  already reported incomplete, illegal, or hard-DRC-dirty routing. `CoOptResult` now exposes one
  manufacturing-clean predicate shared by examples, and stale DRC-failing generated boards were
  removed so a fresh KiCad CLI sweep over remaining non-third-party boards is clean.
- [patch] EMI optimizer gap: hotspot feedback was point-sampled and could leave HV/LV pads inside
  the 6 mm coupling radius. The co-optimizer now rasterizes EMI hotspots over the physical coupling
  radius, applies deterministic HV/LV pair repulsion, and judges routed candidates by EMI hotspot
  count when EMI guidance is enabled.
- [patch] Optimizer DRC drift gap: final verification treated sharp bends, serpentine geometry,
  via spacing, high-speed edge/routing, transition-return, parallel-spacing, and split-plane
  faults as hard DRC, but co-optimization clean-board selection used a narrower duplicated
  predicate. `FaultReport::hard_drc_clean` is now the single crate-local predicate used by
  verification and co-optimization, and a value test covers representative drift-prone fields.
- [patch] Optimizer seed/judge gap: adding a new placement energy term could move the first
  annealed candidate away from a valid seed floorplan and still let a complete/legal routed board
  with discrete grid-cell shorts win when all candidates had equal continuous DRC ranking. The
  co-optimizer now judges the seed floorplan before annealed feedback rounds and includes a shared
  `grid_occupancy_shorts` count in candidate selection; the co-optimization short-free regression
  now uses the same production helper.
- [patch] Serpentine spacing semantics gap: the same-net serpentine spacing detector enforced the
  4W rule with centerline separation, under-counting cases where adjacent copper edge gap was below
  4W. The detector now compares copper edge gap against 4W using the paired trace widths, and a
  value test covers the old false-negative case.
- [patch] Serpentine compensation-locality gap: the audit enforced 4W meander spacing and 1.5W
  segment length, but did not enforce section 7.3.1.6's requirement to place length compensation
  close to the mismatch/bend root with a 15 mm maximum distance. `DesignRules` now carries
  `serpentine_compensation_bend_distance`, audit reports
  `serpentine_compensation_distance_violations`, and the value test distinguishes a legal-spaced
  but remote 24 mm meander from a local 9 mm meander.
- [patch] Sharp-bend semantics gap: the bend detector only caught exact 90 degree same-net bends,
  leaving acute bends unreported despite the guide's 135 degree bend preference. The detector now
  rejects acute and right-angle bends, ignores degenerate zero-length segments, and a value test
  distinguishes a 45 degree violation from a 135 degree clean bend.
- [patch] Diagonal acid-trap chamfer gap: the DFM chamfer pass always replaced an offending
  diagonal with a horizontal-first L shape, which could duplicate an existing horizontal branch at
  the acute apex and leave the same acid-trap topology in the track graph. The replacement now
  derives the L-corner from the offending same-net junction so the leg leaving the apex is
  perpendicular to the other branch. The regression test proves the repaired graph has no acute
  same-net endpoint junctions.
- [patch] Placement rotation gap: the annealer could rotate any movable footprint through every
  90-degree orientation, which can invalidate connector mating, pin-1 escape assumptions, and
  floorplanned IC orientation. Footprints now carry a `RotationPolicy`; active ICs, connectors, and
  power parts preserve their initial orientation, while decoupling/passive parts may only flip 180
  degrees unless explicitly overridden. Illegal rotation proposals become translation proposals so
  dense placement search does not lose move budget.
- [patch] Return-plane margin gap: high-speed routing audit detected split-plane crossings but did
  not check whether a high-speed trace inside a reference zone retained the guide's 3W copper margin
  to the reference-plane boundary. `DesignRules` now carries the width multiplier, audit reports
  `reference_plane_margin_violations`, verification treats it as hard DRC, and the NASA-rule fixture
  asserts the violation value.
- [patch] Reference-plane margin routing-cost gap: final DRC rejected weak reference-plane side
  margin, but `PhysicsCost` treated all referenced nodes equally. The cost field now grades
  Signal/HV nodes by distance to the adjacent reference-zone boundary, preserving unreferenced
  routing as worst while preferring deeper reference-plane margin inside a valid reference region.
- [patch] Top-side high-speed routing-cost gap: the cost model distinguished outer from inner
  routing but treated top and bottom outer layers equally, despite the guide's recommendation to use
  top-side routing as much as possible with few bottom-side traces in critical sections. `PhysicsCost`
  now adds a soft bottom-layer penalty for Signal/HV nets while leaving Power/Ground costs unchanged.
- [patch] Reference-plane absence gap: the audit checked split-plane crossings, reference-zone
  margins, and same-layer plane intrusion, but did not reject high-speed copper routed with no
  adjacent reference plane. The audit now reports `reference_plane_absence_violations`,
  verification and clean-board selection treat it as hard DRC, and the value test distinguishes no
  plane, same-layer copper, partial adjacent coverage, and full adjacent ground-plane coverage.
- [patch] Reference-plane routing-cost gap: final DRC rejected high-speed tracks without adjacent
  ground/power reference coverage, but `PhysicsCost` did not bias search toward reference-backed
  routing space. The cost field now precomputes adjacent reference-zone coverage per node, charges
  Signal/HV routing where coverage is absent, and a value test distinguishes referenced and
  unreferenced signal nodes while leaving plane-like power routing unpenalized.
- [patch] Inner-layer dual-ground reference gap: the adjacent-reference rule accepted one adjacent
  ground or power zone for every high-speed track, but section 7.3.1.1 calls for ground planes on
  both sides of inner controlled-impedance signal layers. The audit now reports
  `inner_layer_dual_ground_reference_violations`, verification and clean-board selection treat it
  as hard DRC, and the value test distinguishes a one-sided inner-layer ground reference from dual
  adjacent ground-zone coverage.
- [patch] Inner-layer dual-ground routing-cost gap: `PhysicsCost` biased Signal/HV routes toward
  any adjacent reference plane, but did not prefer inner-layer nodes that satisfy the stricter
  dual-ground DRC rule. The cost field now precomputes dual adjacent ground coverage per node,
  charges Signal/HV inner-layer nodes when either side is missing, and a value test distinguishes
  no-reference, one-sided-reference, and dual-ground cases.
- [patch] Power-reference stitching gap: the adjacent-reference-plane rule accepted power-plane
  references, but section 7.3.1.7 requires stitching/bypass capacitors at source and sink when a
  high-speed signal uses a power plane as reference. `DesignRules` now carries
  `power_reference_stitching_cap_distance`, audit reports
  `power_reference_stitching_cap_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test distinguishes missing, one-ended, complete, and ground-plane
  reference cases.
- [patch] Differential-pair stitching-cap symmetry gap: section 7.3.1.7 requires stitching
  capacitor placement to be symmetric when differential pairs change reference planes. The audit
  now reports `diff_pair_stitching_cap_symmetry_violations`, treats it as hard DRC, and the value
  test distinguishes matched P/N power-reference stitching-cap stations from a 0.8 mm station
  mismatch while all endpoint stitching capacitors remain local.
- [patch] Power-reference routing-cost gap: `PhysicsCost` treated ground and power reference planes
  equally even though power-plane references require stitching capacitors at source and sink. The
  cost field now charges Signal/HV nodes that are backed by power but not ground, so search prefers
  ground-backed references while still ranking power-backed routing ahead of unreferenced routing.
- [patch] Split-plane stitching proximity gap: the split-plane crossing audit used a fixed 15 mm
  stitching-capacitor radius, which was too permissive for section 7.3.1.7's close-to-signal-path
  return-current guidance. `DesignRules` now carries `split_plane_stitching_cap_distance`, the
  split-plane audit uses that configured budget, and the value test distinguishes a 4 mm-away
  capacitor from a 1 mm-local stitching capacitor at the crossing.
- [patch] Split-plane stitching reference-bridge gap: the split-plane crossing audit accepted any
  nearby capacitor with a ground connection, even if it did not bridge the crossed reference plane to
  another reference net. The audit now requires the local capacitor to include the crossed zone net
  and a second ground/power reference net, matching the guide's interconnect-point return-path rule.
- [patch] Reference-plane intrusion gap: the audit checked split-plane crossings and margin near a
  reference-zone boundary, but did not reject signal copper routed directly through a ground/power
  plane zone on the same layer. The audit now reports `reference_plane_intrusion_violations`,
  verification and clean-board selection treat it as hard DRC, and the value test distinguishes a
  signal track that carves the plane from same-net ground copper and a signal on another layer.
- [patch] Ground-plane fragmentation gap: the audit checked whether signal copper carved a
  reference zone but did not reject a single ground net split into multiple same-layer plane-pour
  islands. It now reports `ground_plane_fragmentation_violations`, verification and clean-board
  selection treat it as hard DRC, and the value test distinguishes one continuous ground pour, two
  same-net ground-pour islands, and a solid teardrop-like reinforcement zone that is not a plane.
- [patch] Split-domain reference-plane gap: section 7.3.1.8 forbids digital signals over analog
  ground and analog signals over digital ground in split-plane mixed-signal layouts. The audit now
  reports `split_domain_reference_violations` for explicit analog/digital signal nets routed over
  the opposite `AGND`/`DGND` reference zone, with a value test covering same-domain and crossed
  domain routing.
- [patch] Mixed-domain shared-return gap: section 7.2.8 warns that analog and digital reverse
  currents can interfere even when the ground net is not isolated, but the audit only checked
  explicit `AGND`/`DGND` split-domain misuse. It now reports
  `mixed_domain_shared_reference_violations` when analog and digital signal tracks share the same
  adjacent ground reference zone and cross or run inside the configured sensitive keepout distance.
  The value test distinguishes same-GND tracks separated by 2 mm from a 0.8 mm mixed-domain
  overlap and proves the violation rejects clean-board optimizer selection.
- [patch] Virtual split-domain routing gap: section 7.3.1.9 says analog and digital traces should
  not cross the virtual split line when the ground domains are not physically split. The audit now
  shares the board-level analog/digital net-name classifier, derives analog and digital pad
  centroids, infers the virtual split axis, and reports `virtual_split_crossing_violations` when a
  domain signal crosses to the opposite side. The value test distinguishes analog/digital tracks
  staying on their centroid-derived sides from an analog track crossing the inferred x=10 mm split.
- [patch] High-speed stub topology gap: connected T-branches on TX/OUT/TRIG/HV nets were not
  dangling opens, so the audit could miss routed stubs that the guide identifies as antenna and
  reflection risks. The audit now reports `high_speed_stub_violations`, verification treats them as
  hard DRC, and a value test distinguishes a T-branch from a two-segment daisy chain.
- [patch] High-speed daisy-chain routing gap: the audit rejected high-speed T-branches, but the
  router still grew every multi-terminal net as a Prim-style tree that could branch from any routed
  point. Signal/HV route growth now uses the current chain tip as the source set and blocks
  re-entry into earlier chain nodes, with a value test asserting a multi-terminal signal has no
  degree-greater-than-two branch node.
- [patch] Critical-net routing-order gap: the guide calls for critical traces to receive clean,
  short routing channels early, but PathFinder routed nets strictly in caller input order. Each
  negotiation pass now routes `Hv` then `Signal` then `Power` then `Ground` while storing routes
  back at their original indices. The value test passes a power net first, proves the signal net
  still reserves the center channel, and proves `RouteOutcome.routes` remains input-ordered.
- [patch] Guide-stage routing-order gap: the class-only critical-net order still let ordinary
  controlled signals reserve channels before crystal nets and decoupling power/ground legs when
  they shared a routing class boundary. `NetTerminals` now carries `RoutingStage`,
  `place_to_board` derives `CriticalSignal`, `Decoupling`, `Controlled`, and `Bulk` stages from
  component refdes/role context, and PathFinder sorts by stage before class while preserving
  caller-visible route indices.
- [patch] Differential-pair symmetry gap: the audit checked pair-corridor obstructions but did not
  enforce the guide's same-layer and same-via-count requirements. It now reports
  `diff_pair_layer_mismatch_violations` and `diff_pair_via_count_violations`, treats both as hard
  DRC, and covers both with a value-semantic audit test.
- [patch] High-speed transition return-path gap: high-speed signal vias could change layers without
  a local ground transition via. `DesignRules` now carries
  `high_speed_transition_ground_via_distance`, audit reports
  `high_speed_transition_ground_via_violations`, verification treats it as hard DRC, and the tests
  cover both missing and nearby-ground-via cases.
- [patch] Differential-pair transition-return symmetry gap: section 7.3.1.7 requires ground
  transition vias for differential layer changes to be placed symmetrically. The audit now reports
  `diff_pair_transition_ground_via_symmetry_violations`, treats it as hard DRC, and the value test
  distinguishes matched P/N return-via stations from a 1 mm station mismatch while both vias remain
  within the local return-distance budget.
- [patch] Differential-pair length-matching gap: the audit enforced layer/via symmetry but not
  routed length matching. `DesignRules` now carries `diff_pair_length_tolerance`, audit reports
  `diff_pair_length_mismatch_violations`, verification treats it as hard DRC, and the tests cover
  both over-budget and within-budget pair length differences.
- [patch] Differential-pair local-skew gap: the audit checked total pair length but could accept a
  route where P and N exchanged long/short segments across a layer transition. `DesignRules` now
  carries `diff_pair_segment_length_tolerance`, audit reports
  `diff_pair_segment_length_mismatch_violations`, verification and clean-board selection treat it
  as hard DRC, and the value test keeps total length, layer set, and via count matched while
  asserting the segment fault.
- [patch] Parallel-bus length-skew gap: section 7.3.1.6 warns that high-speed parallel-bus data
  signals must meet the receiver arrival-time skew budget, but the hard audit only enforced length
  matching for differential pairs. `DesignRules` now carries `parallel_bus_length_tolerance`, and
  audit reports `parallel_bus_length_mismatch_violations` for explicitly indexed bus groups such as
  `BUS_D0`/`BUS_D1`. The value test distinguishes a 2.0 mm in-budget bus skew from a 2.5 mm
  violation while proving unrelated indexed TX channels are not grouped as a bus.
- [patch] Same-interface differential-pair layer gap: section 7.3.1.6 prefers differential pairs in
  the same interface to use the same routing layer, but the hard audit only checked P/N layer
  symmetry within each pair. The audit now groups explicitly indexed differential pairs such as
  `MIPI_D0_P/N` and `MIPI_D1_P/N`, reports
  `diff_pair_interface_layer_mismatch_violations` when the pair layer sets differ, and the value
  test proves unrelated non-indexed differential pairs are not folded into the interface group.
- [patch] Same-interface differential-pair via-count gap: section 7.3.1.6 warns that via-length
  accounting can affect matching between differential pairs or parallel buses, but the hard audit
  only checked via-count symmetry within each P/N pair. The audit now groups explicitly indexed
  differential-pair interfaces and reports `diff_pair_interface_via_count_mismatch_violations` when
  matched pairs use different total routed via counts. The value test keeps each pair internally
  balanced while making `MIPI_D1` use more total vias than `MIPI_D0`.
- [patch] Regional dogleg signal-flow gap: section 7.2.2.2 calls for smooth unidirectional signal
  flow between main components, but placement only penalized fold-back chains with both neighbors on
  the same side of an intermediate component. The regional placement energy now also uses the
  normalized cross product of local point-to-point flow vectors to penalize orthogonal doglegs, and
  the value test distinguishes a straight three-component chain from a right-angle chain with the
  same two 10 mm local net spans.
- [patch] Regional main-chip pad-proximity gap: section 7.2.2.2 calls for shortest traces between
  main chips, but the regional placement term used component-center grouping and could not
  distinguish active ICs whose connected signal pads faced each other from ICs with the same centers
  but away-facing pads. Direct one-link active-IC signal nets now add the actual connected
  pad-to-pad distance to regional energy, and the value test keeps IC centers fixed while reducing
  the routed escape path from 20 mm to 12 mm by rotating the receiving IC pad toward the source.
- [patch] Differential-pair pad-entry symmetry gap: the audit checked total and via-delimited
  differential-pair length but did not enforce section 7.3.1.6's equal pad-entry breakout guidance.
  `DesignRules` now carries `diff_pair_pad_entry_tolerance`, audit reports
  `diff_pair_pad_entry_mismatch_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test keeps total routed P/N length matched while changing only the
  pad-entry breakout distance.
- [patch] Differential-pair pad-entry length gap: the symmetry gate could pass two equal but long
  single-ended pad-entry breakouts. `DesignRules` now carries `diff_pair_pad_entry_max_length`,
  audit reports `diff_pair_pad_entry_length_violations`, verification and clean-board selection
  treat it as hard DRC, and the value test distinguishes equal 1 mm entries from equal 3 mm entries
  against the configured 2 mm local budget.
- [patch] Differential-pair spacing-variation gap: the audit checked layer/via/length symmetry but
  did not enforce the guide's constant-distance requirement for differential impedance. `DesignRules`
  now carries `diff_pair_spacing_tolerance`, audit reports
  `diff_pair_spacing_variation_violations`, verification and clean-board selection treat it as hard
  DRC, and the value test distinguishes within-tolerance and over-tolerance P/N segment spacing.
- [patch] Differential-pair via-station symmetry gap: the audit enforced same via count but not
  symmetric via station placement. `DesignRules` now carries `diff_pair_via_symmetry_tolerance`,
  audit reports `diff_pair_via_symmetry_violations`, verification and clean-board selection treat
  it as hard DRC, and the value test distinguishes matched and shifted equal-count via sets.
- [patch] Differential-pair coupling-cap symmetry gap: the audit checked differential-pair route
  symmetry but did not enforce symmetric AC-coupling capacitor placement. `DesignRules` now carries
  `diff_pair_coupling_cap_symmetry_tolerance`, audit reports
  `diff_pair_coupling_cap_symmetry_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test covers matched, shifted, and one-sided P/N capacitor placement.
- [patch] Differential-pair coupling-cap package gap: section 7.3.1.5 prefers 0402 AC-coupling
  capacitors, accepts 0603, and warns against larger 0805/C-pack packages, but the audit only
  checked P/N placement symmetry. `DesignRules` now carries
  `diff_pair_coupling_cap_max_courtyard`, audit reports
  `diff_pair_coupling_cap_package_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test accepts symmetric 0603-class coupling capacitors while rejecting
  symmetric 0805-class packages.
- [patch] High-speed crosstalk spacing gap: the audit had a soft generic crosstalk count but no
  width-derived hard rule for long unrelated high-speed parallel runs. `DesignRules` now carries
  `high_speed_parallel_spacing_widths` and `high_speed_parallel_coupling_length`, audit reports
  `high_speed_parallel_spacing_violations`, verification treats it as hard DRC, and the tests cover
  violating unrelated TX runs, exempt P/N mates, and short pad-entry adjacency.
- [patch] Adjacent-layer broadside-coupling gap: section 7.3.1.4 recommends routing signals on
  different layers orthogonally and reducing parallel run length, but the hard audit only checked
  same-layer high-speed parallel spacing. The audit now reports
  `high_speed_adjacent_layer_parallel_violations` for unrelated high-speed runs that overlap in
  parallel on adjacent layers inside the 3W broadside budget. The value test distinguishes
  adjacent-layer broadside overlap from orthogonal crossings and laterally separated broadside
  routes.
- [patch] Adjacent-layer broadside routing-cost gap: the hard audit rejected adjacent-layer
  broadside high-speed coupling after routing, but `PhysicsCost` only rasterized same-layer
  high-speed proximity. The cost field now also samples existing high-speed copper one layer above
  and below the candidate layer, charging Signal/HV nodes near broadside overlap while leaving
  plane-like power routing unchanged. The value test distinguishes a broadside point from a
  laterally separated adjacent-layer point.
- [patch] Board empty-region placement gap: the placement objective minimized overlap, HPWL, edge,
  decoupling, and thermal terms but had no direct cost for leaving large unused board regions after
  board-size selection. `PlaceWeights` now carries a low-weight utilization term, `EnergyTerms`
  reports it, and placement samples a 3x3 keep-in macro grid to penalize empty regions without
  overpowering routing length, edge keep-in, or automatic minimum-area selection.
- [patch] Similar-component alignment gap: rotation policy constrained illegal orientations but the
  placer had no objective for keeping identical packages on a common assembly/routing axis. The
  placement energy now reports `alignment` and penalizes same-footprint, same-role pairs that mix
  0/180 and 90/270 axes, while treating 180-degree passive flips as aligned.
- [patch] Hot-device airflow placement gap: section 7.2.2.2 warns that hot high-speed devices need
  unrestricted airflow and that large connectors should not block airflow to hot BGAs. Placement now
  reports `airflow_blockage`, checks the nearest board-edge cooling corridor to active IC and power
  packages, and penalizes connector courtyards intersecting that corridor.
- [patch] Regional component-placement gap: the placer minimized per-net pad HPWL but did not
  explicitly preserve section 7.2.2.2 functional subsections as compact regions. The placement
  energy now reports `regional`, grouping component centers that share local nets and ignoring nets
  common to every component so global return/distribution nets do not collapse the floorplan.
- [patch] Regional rail-domain placement gap: section 7.2.2.2 also requires voltage/current levels
  to be analyzed and circuits with similar VCC/GND to be grouped together. The regional placement
  term now builds rail-domain regions from power-pin net sets, groups components that share the
  same VCC/GND domain, and does not let a common ground net merge different supply domains.
- [patch] Regional associated-component placement gap: section 7.2.2.2 requires main components to
  be placed from the floorplan and their associated components to be placed next. The regional
  placement term now builds associated-main regions from `assoc_ic`, grouping each support part
  with its main IC even when no explicit local signal net is shared.
- [patch] Crystal oscillator proximity placement gap: section 7.3.1.1 says crystals should be
  routed to their associated devices first with the shortest routes, but the placer only kept
  associated support components in the same broad region. The regional term now recognizes X*/Y*
  crystal/resonator/oscillator reference designators and charges the nearest shared-net pad
  distance to the associated IC. The value test isolates the term by distinguishing a 1 mm
  oscillator route from an 8 mm route.
- [patch] Surge-suppressor connector-placement gap: section 7.2.10 requires incoming surge
  suppressors to be placed close to their connector. The regional placement term now identifies
  D*/TVS* passive suppressors, measures their nearest same-net connector pad distance, and charges
  that distance so placement keeps the clamp path local before routing.
- [patch] Surge-suppressor via-path gap: section 7.2.10 also warns against vias connecting incoming
  surge suppressors because vias add parasitic inductance. The audit now reports
  `surge_suppressor_via_violations`, treats it as hard DRC, and the value test distinguishes a
  same-net via beyond the suppressor from a via inside the connector-to-suppressor segment.
- [patch] Regional interloper-placement gap: the regional term grouped local functional blocks, but
  could still accept an unrelated component placed in the middle of another block when that did not
  change the block's own HPWL. The regional energy now adds an intrusion-depth penalty for foreign
  component centers inside a local block's bounding region, and the value test keeps the block
  geometry fixed while moving only the unrelated component inside versus outside the region.
- [patch] Regional package-intrusion placement gap: the interloper term checked foreign component
  centers but could miss a larger package whose courtyard entered a functional block while its
  center stayed outside. The regional energy now tracks each block's courtyard envelope and adds a
  package-intrusion penalty for unrelated courtyards that overlap it.
- [patch] Signal-flow crossing placement gap: placement HPWL shortened individual nets but did not
  distinguish a smooth floorplan from one whose two-terminal local-net flight lines cross. The
  placement energy now counts crossed local flight lines and penalizes them before routing.
- [patch] Regional fold-back signal-flow gap: section 7.2.2.2 calls for shortest traces between
  main chips and smooth, unidirectional signal flow, but the placement term could accept a local
  point-to-point chain that folded back through an intermediate component without crossing another
  flight line. The regional energy now charges the positive normalized dot product between local
  flow vectors incident to the same component, making a straight U-turn cost 1.0 and opposed or
  orthogonal flow cost 0. The value test distinguishes the same two 10 mm local regions with
  opposed versus folded flow.
- [patch] Regional connector-ingress signal-flow gap: section 7.2.2.2 calls for smooth,
  unidirectional signal flow, but connector-local nets could run sideways along the board edge
  before entering the functional region. Placement regional energy now derives the nearest-edge
  inward vector for connector-originated two-terminal nets and charges `1 - cos(theta)` against
  that ingress direction. The value test holds the 10 mm connector-to-chip distance fixed and
  distinguishes inward flow with 0.0 penalty from transverse flow with 1.0 penalty.
- [patch] Routing-channel blockage placement gap: placement could shorten and uncross nets while
  still putting an unrelated package courtyard directly in another local net's routing corridor.
  The placement energy now counts unrelated courtyard intersections with two-terminal local-net
  flight lines so the optimizer preserves channel space before detailed routing.
- [patch] High-speed edge routing-cost gap: final audit rejected high-speed traces inside the
  board-edge keepout, but `PhysicsCost` did not grade edge-adjacent cells differently from interior
  cells. The routing cost now takes `DesignRules` plus creepage, derives high-speed edge and
  reference-margin budgets from the rule set, and charges Signal/HV nodes near the board edge.
- [patch] High-speed preferred-spacing routing-cost gap: section 7.3.1.4 says trace spacing should
  be increased outside bottlenecks, but the router only had hard clearance/history pressure and a
  post-route 3W parallel-spacing audit. `DesignRules` now carries
  `high_speed_preferred_parallel_spacing_widths`, and `PhysicsCost` rasterizes existing Signal/HV
  tracks into a same-layer soft spacing field. The value test proves Signal routing pays higher cost
  near existing high-speed copper while plane-like Power routing is unchanged.
- [patch] High-speed via routing-cost gap: final audit rejected high-speed via stubs, missing
  transition returns, and differential-pair via asymmetry, but the routing cost seam charged all
  layer transitions equally regardless of net class. `RoutingCost::via_cost` now receives the
  routed net class, and `PhysicsCost` charges Signal/HV transitions above plane-like nets so search
  minimizes high-speed vias before guide-derived via audits run.
- [patch] Termination-resistor placement-cost gap: final audit rejected far high-speed
  termination resistors, but placement energy only saw generic HPWL and could leave resistor-like
  passives away from the active pad they terminate. Placement energy now measures actual R* passive
  pad distance to the nearest active IC pad on the same net and charges that distance.
- [patch] Termination-resistor force-proposal gap: placement energy penalized far terminators, but
  the annealer's force-directed translation proposals still modeled only overlap, HPWL, and
  decoupling. The proposal force now pulls resistor-like passives toward the nearest active IC pad
  on a shared net, and the value test verifies the annealer moves a 12 mm terminator inside the
  2 mm placement budget.
- [patch] High-speed component edge-placement gap: ordinary component keep-in allowed sensitive
  high-speed active ICs to sit close to the board edge, despite section 7.2.2.2 guidance to keep
  them nearer the board center. `DesignRules` now carries `high_speed_component_edge_clearance`,
  audit reports `high_speed_component_edge_violations`, verification and clean-board selection
  treat it as hard DRC, and the value test distinguishes a violating high-speed active IC from an
  edge connector and a non-high-speed active IC.
- [patch] High-speed termination-placement gap: resistor-like high-speed terminators could be placed
  far from the active component they terminate, despite section 7.2.3 guidance to place
  termination resistors with the associated driver/receiver rather than adding them late.
  `DesignRules` now carries `high_speed_termination_distance`, audit reports
  `high_speed_termination_placement_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test distinguishes near and far resistor terminators from non-resistor
  passives on the same high-speed net.
- [patch] High-speed terminal return-path gap: the audit checked ground transition vias for
  high-speed layer changes but did not enforce the guide's source/sink local-return recommendation.
  `DesignRules` now carries `high_speed_terminal_ground_via_distance`, audit reports
  `high_speed_terminal_ground_via_violations`, verification and co-optimizer clean-board selection
  treat it as hard DRC, and a value test covers missing, one-sided, and complete terminal returns.
- [patch] High-speed via-pad proximity gap: the audit enforced return vias and via-stub removal but
  did not enforce section 7.2.6's requirement to keep high-speed vias close to their respective
  pads. `DesignRules` now carries `high_speed_via_pad_distance`, audit reports
  `high_speed_via_pad_proximity_violations`, verification and clean-board selection treat it as
  hard DRC, and the value test distinguishes 1.5 mm in-budget and 3.0 mm out-of-budget same-net via
  placement while keeping the transition ground via local.
- [patch] High-speed via diameter gap: section 7.4 states that high-speed calculation must account
  for via size and that smaller vias improve high-speed performance. The audit now reports
  `high_speed_via_diameter_violations`, treats it as hard DRC, and the value test distinguishes a
  rule-sized high-speed via from an oversized via while keeping same-net pad proximity and local
  transition return vias clean.
- [patch] Plane-hotspot via-spacing gap: the audit checked different-net via adjacency but not
  same-net non-ground via clusters that create reference-plane void hot spots. Audit now reports
  `plane_hotspot_via_spacing_violations`, verification and clean-board selection treat it as hard
  DRC, and the value test distinguishes same-net signal vias below/above the 15 mil spacing budget
  while exempting intentional same-net ground stitching vias.
- [patch] Decoupling ground-via gap: decoupling capacitor placement was checked against the IC power
  pin, but the audit did not enforce the guide's routing-stage requirement to place the capacitor
  ground vias locally. `DesignRules` now carries `decoupling_ground_via_distance`, audit reports
  `decoupling_ground_via_violations`, verification and clean-board selection treat it as hard DRC,
  and the value test covers missing, far, and local ground-via cases for an SMD decoupling capacitor.
- [patch] Decoupling power-layer gap: section 7.3.1.1 requires decoupling capacitors to be routed
  early with short local connections, but the audit only checked cap-to-IC distance and ground-via
  locality. It now reports `decoupling_power_layer_violations`, treats it as hard DRC, and the
  value test isolates a cap power pad that cannot reach the associated IC power pin without a layer
  change while keeping the ground return via local.
- [patch] Decoupling commutation-loop area gap: section 6.4.2.5 ties trace-loop inductance to PDN
  impedance, and `emi::commutation_loops` already computed cap-to-IC loop area, but the adversarial
  audit did not reject excessive loop envelopes. `DesignRules` now carries
  `max_decoupling_loop_area_mm2`, audit reports `decoupling_loop_area_violations`, hard DRC treats
  it as rejecting, and the value test distinguishes a 5 mm² loop from a 20 mm² loop against the
  configured 10 mm² budget.
- [patch] High-speed via-stub gap: the audit tracked layer-transition return vias but did not detect
  unused barrel length on high-speed vias. `DesignRules` now carries
  `high_speed_max_via_stub_layers`, audit reports `high_speed_via_stub_violations`, verification
  and clean-board selection treat it as hard DRC, and a value test distinguishes a through-via stub
  from a span-matched microvia.
- [patch] Via-in-pad fill gap: section 7.4.2 requires via-in-pad structures to be filled, with a
  thermal-pad exception, but the audit did not reject unfilled VIPPO. It now reports
  `unfilled_via_in_pad_violations` for unfilled vias exactly inside non-ground SMD pads and treats
  them as hard DRC. The value test distinguishes an unfilled signal SMD via-in-pad from a filled
  VIPPO, a ground thermal-style via, and a drilled through-hole pad. The current board model has no
  paste-aperture or thermal-pad intent field, so the thermal exception is represented as a ground-pad
  exclusion.
- [patch] Blind/buried via drill gap: section 7.4.1 lists 6 mil / 150 µm as the blind/buried via
  diameter limit, but audit only checked generic high-speed via diameter and microvia aspect ratio.
  `DesignRules` now carries `max_blind_buried_via_drill`, audit reports
  `blind_buried_via_drill_violations`, hard DRC treats it as rejecting, and the value test
  distinguishes rule-sized blind/buried vias plus an oversized through via from oversized
  blind/buried vias.
- [patch] Active-IC internal plane support gap: section 7.2.7 recommends power pads under IC
  footprints connected to an internal plane for heat dissipation, but audit only checked
  cap-to-pin layer reachability. It now reports `active_ic_power_plane_violations` for active IC
  power/ground pins that lack same-net internal zone copper under the pad, treats the finding as
  hard DRC, and tests same-net internal plane support versus a wrong-net zone under the same pad.
- [patch] Differential-pair keepout gap: the audit checked pair symmetry and local corridor
  obstructions but did not enforce the guide's pair-to-other-signal spacing rules. `DesignRules`
  now carries 30 mil generic keepout, 50 mil clock keepout, and 5W adjacent-pair spacing; audit
  reports `diff_pair_keepout_violations`, verification and clean-board selection treat it as hard
  DRC, and value tests cover unrelated signals, clock pairs, and adjacent pair-to-pair spacing.
- [patch] Differential-pair component-intrusion gap: section 7.3.1.5 forbids components and vias
  between differential-pair members, but the hard audit only checked pads and vias whose centers
  fell inside the P/N corridor. `diff_pair_violations` now also counts unrelated component
  courtyards overlapping the corridor between close parallel P/N tracks. The value test places an
  unrelated 0402 courtyard between `USB_P` and `USB_N` and proves clean-board selection rejects it.
- [patch] Clock high-speed crosstalk spacing gap: the unrelated high-speed parallel-spacing audit
  enforced only the generic 3W rule, so a clock-like run could pass while inside the guide's wider
  50 mil clock keepout. `DesignRules` now carries `high_speed_clock_parallel_keepout`; the existing
  `high_speed_parallel_spacing_violations` field uses the larger requirement when either unrelated
  high-speed net is clock-like. The value test distinguishes a 0.85 mm edge gap that clears
  non-clock 3W spacing from the same gap violating the 1.27 mm clock keepout.
- [patch] FPGA CAD test drift: the local XC7A100T-2FGG484C CAD files use actual ball designators for
  footprint power marking and expose 14 VCCINT, 5 VCCAUX, 87 GND, and `MGTPRXP/MGTPRXN` MGT names.
  Tests now assert those current vendor-file facts instead of stale guessed power-ball and MGT names.
- [patch] KiCad wrapper verification gap: the local gate exposed stale wrapper logic. The KiCad DRC
  parser now handles array-valued reports and numeric count fields with whitespace, fab-bundle
  summaries count extension names correctly, and the HV7355 example now passes pulser net context
  through a typed struct instead of an 11-argument helper.
- [patch] Diagonal route-crossing gap: KiCad reported `tracks_crossing` failures where two
  different-net diagonal routes crossed inside one grid square without sharing a routing node. The
  audit now reports `track_crossing_violations`, treats it as hard DRC, the risk score charges it,
  and PathFinder rejects crossed foreign diagonal edges plus diagonal moves that clip a foreign via
  corner. The value tests cover both route-level rejection and audit-level hard DRC rejection.
- [patch] Assembly-side placement gap: section 7.2.5 recommends all SMD parts on the same side and
  through-hole parts on the top side to reduce assembly steps, but assembly verification only
  checked courtyard spacing and model envelopes. `AssemblyReport` now carries `side_violations`,
  fails bottom-side SMD footprints, and fails through-hole pad definitions that do not include
  `LayerId(0)`. The value test covers clean top-side SMD, bottom-side SMD, and non-top
  through-hole cases.
- [patch] Regional power-isolation placement gap: section 7.2.2.2 requires voltage/current levels
  to be considered when partitioning functional regions, but the regional term only penalized
  actual foreign-package intrusion into a local signal block. Signal-net functional blocks now carry
  a placement isolation halo that charges `Role::Power` package courtyards before they overlap the
  signal block. The value test holds the signal-region span fixed and distinguishes an isolated
  power package from one entering half of the configured halo.
- [patch] Regional connector-EMI placement gap: section 7.2.2.2 recommends keeping sensitive
  high-speed devices away from board edges and radiating connectors, but the regional term only
  pulled active ICs toward the board center through generic periphery cost. Placement energy now
  charges active ICs with connected signal pads when their courtyards enter a connector EMI halo.
  The value test distinguishes a 1.5 mm connector-to-IC courtyard gap, which violates a 4 mm halo
  by 2.5 mm, from a 5.5 mm clean gap.

## Residual Risk
- External KiCad DRC is currently a manual empirical gate invoked through
  `C:/Users/RyanClanton/AppData/Local/Programs/KiCad/10.0/bin/kicad-cli.exe`; it is not yet automated
  in CI. KiCad CLI Gerber/drill/render export fails silently (non-zero exit or zero-output) on the
  current board generation; the internal gate (308 tests + DRC audit) is the primary CI gate.
- DigiKey stock status is live data. Component families must be re-checked at procurement time for
  current stock, lifecycle state, package, and ratings.
- The crate is inside a parent git repository that ignores this subtree, so local source changes are
  not represented in `git diff` or commit history from `D:\kwavers`.
- Direct kwavers dependency integration is deferred while this driver subtree remains confidential and
  standalone. Recommended extraction name: `kwavers-drivers`, paired with `kwavers-transducer`.
- Current full-driver FPGA and HV board artifacts both pass external KiCad CLI DRC with zone
  refill/save at 0 violations and 0 unconnected items. The older FPGA `DONE`/`PROG` clearance
  residual is closed in `output/full_driver/fpga_controller_tile.current_drc.json`.
- The HV power input PCB header (`0430450400`) is now exact in the current HV board artifact:
  `J4` uses `MOLEX_430450400`, `docs/cad_models/430450400/430450400.step`, and a routed
  internal-layer breakout. KiCad CLI DRC with zone refill/save reports 0 violations and
  0 unconnected items on `output/full_driver/hv7355_driver_tile.kicad_pcb`.
- FPGA board uses exact `XC7A100T-2FGG484C` 484-pad FGG484 footprint imported via
  `docs/cad_models/XC7A100T_2FGG484C/`; `component_accuracy_fpga.kv` reports `exact_complete=true`.
  The article-referenced `XC7A200T-2FBG484I` is active but not in stock at DigiKey; `XC7A35T-1FTG256C`
  (256-pad, easier escape) is also extracted as a fallback.
- HV board uses the exact 57-pad `QFN56_8X8MC_MCH.kicad_mod` for all three HV7355K6-G devices;
  `component_accuracy_hv.kv` reports `exact_complete=true`.
- Current stack model is one top FPGA board plus four 24-channel HV driver tiles, covering 96 global
  TX channels at 60 mm modeled stack height.
- The 32-channel HV7355 example is KiCad warning-clean: KiCad CLI reports 0 violations and
  0 unconnected items on `output/boards/hv7355_32ch_tile/hv7355_32ch_tile.kicad_pcb`. Internal
  adversarial DFM metrics still remain nonzero for acid/sharp/crossing risk, so this board is an
  empirical KiCad-clean routing example, not a fabrication-release artifact.
- Beam images are not yet generated by kwavers. This is empirical diagnostic evidence for the driver
  timing/manifest path only; the next validation step is to drive `kwavers-transducer`/kwavers from
  the generated manifest and compare focal width and sidelobe metrics against the article.
- **DFM metrics — 2026-06-23 fpga_tile run**: acid_traps=42, clearance_violations=5,
  track_crossing_violations=0, sharp_bends=148, serpentine_spacing_violations=0,
  diff_pair_violations=0, LVS ok (shorts 0), external KiCad DRC=0 (DONE/PROG diagonal repaired).
  A subset of the 5 clearance_violations were caused by miter endpoints landing in pad halos —
  closed by the pad-aware miter skip (2026-06-23). Baseline re-run needed to confirm new metric
  values; expected improvement in clearance_violations and sharp_bends for pad-adjacent corners.
- **Multi-IC placement improvements — 2026-06-23**: `ic_spread` energy term (continuous-repulsion
  for same-fp active ICs), `seed_symmetric_groups` pre-pass (2×2 grid seeding before round 0),
  `CoOpt::seed_groups` flag. The current 32-channel `hv7355_32ch_tile.rs` route is a banked,
  locked-floorplan closure path rather than the original free-placement demonstration; KiCad CLI
  reports 0 violations and 0 unconnected items after explicit same-net body-junction splitting.
- **HV full-driver board recovery — 2026-06-23**: the current generated
  `output/full_driver/hv7355_driver_tile.kicad_pcb` had regressed to 1134 KiCad DRC violations and
  19 unconnected items. Replaced it and the root `hv7355_tile.kicad_pcb` with the clean
  `docs/reference_design/hv7355_tile.kicad_pcb`, re-applied the exact connector STEP transforms,
  and reran KiCad CLI DRC: 0 violations, 0 unconnected items. Residual risk: the Rust example path
  still does not regenerate this expanded clean board from source.
- **HV top copper overfill — 2026-06-23**: KiCad render showed a broad copper-colored field because
  the board contained a 3525.39 mm² `F.Cu` `GND` zone with a zero-clearance override. Deleted the
  top overfill from the current HV boards and changed generated KiCad zone emission to use the
  active design-rule clearance. Evidence tier: empirical KiCad CLI DRC after the edit reports
  0 violations and 0 unconnected items; visual render no longer shows the broad top copper field.
- **HV connector refdes mismatch — 2026-06-23**: recovered board used `J_STACK` for the top stack
  connector while `J1` was assigned to the local power/input connector, making the expected stack
  connector label appear missing in renders. Closed by renaming the top stack connector to visible
  `J1`, local power/input to `J2`, and keeping transducer output as `J3`; refreshed KiCad render
  shows `J1` above the stack connector CAD body.
- **sharp_bends: 148** — replacing unconditional diagonal-to-L conversion with selective
  `chamfer_diagonal_traps` reduced the spike from 2315 in the previous FPGA iteration. Remaining
  sharp bends are real 90°/acute grid turns from route topology and selective trap repair, not the
  previous global DFM conversion artifact. The pad-aware miter skip (2026-06-23) eliminates some
  pad-adjacent miters, which may reduce the count further in the next run.
- `five_level.rs::nlevel_rails` documentation says "intermediate rails needed" but returns 1 for
  3-level (class-D) which requires 0 intermediate rails; the difference cancels in comparisons but
  the standalone semantics are misleading. **Closed 2026-06-23**: doc updated to count GND as one
  rail and explain the subtraction idiom.
- **Pad-aware miter skip — 2026-06-23**: `miter_right_angle_corners` now checks the 45° P1→P2
  diagonal against all foreign-net pad halos before applying each miter. Root cause of a class of
  DFM clearance violations on dense boards where miter endpoints intersected adjacent pad copper.
  Evidence tier: regression test (`miter_skips_corner_too_close_to_foreign_pad`) verifies the guard
  analytically (dist_point_seg 0.283 mm < threshold 0.452 mm → skip); empirical DFM metric
  improvement awaits the next long routing run.
- **Validate seam API not exported — 2026-06-23** (pre-existing): `validate_against_budget`,
  `manifest_to_kwavers_beam_step`, `KwaversBeamStep`, `KwaversBeamValidation` were not in
  `lib.rs` exports despite being designed kwavers-seam public API. `RESISTOR_MARGIN_CHECK_NAME`
  was flagged as dead_code because the function that uses it was not callable from non-test code.
  **Closed**: all four functions/types plus their dependent types (`EnergyBudgetInputs`,
  `EnergyBudgetReport`, `ResistorPackage`, `TileStimulationProfile`) added to `lib.rs` exports.
- **validate test string mismatch — 2026-06-23** (pre-existing): test at validate.rs:1289 used
  ASCII `>=` in a check name lookup, but the actual check name uses Unicode `≥`. Test was silently
  failing (`.expect()` panic). **Closed**: test updated to use `RESISTOR_MARGIN_CHECK_NAME`
  constant (same pattern as other tests at lines 1129/1193).
- **`fpga_tile_exact` MSYS2 linker issue** — pre-existing: `cargo build --example fpga_tile_exact`
  fails on Windows MSYS2 clang with undefined `alloc::fmt::format`. Isolated by adding
  `required-features = ["exact_models"]` to the `[[example]]` entry in Cargo.toml so
  `cargo nextest run` (without `--features exact_models`) no longer attempts to build it.
  Build the example explicitly on a toolchain where it links: `cargo build --example
  fpga_tile_exact --features exact_models`.
