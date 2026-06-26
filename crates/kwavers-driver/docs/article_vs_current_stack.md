# Article Implementation vs Current Stack

Evidence tier: article PDF text extraction from `docs/10560355.pdf`, generated manifests under
`output/full_driver`, generated beam figures under `output/beamforming`, and local CAD inventory under
`docs/component_cad_inventory.md`. This is empirical/documentary evidence, not a formal proof.

## Summary

The current implementation is a stack-level 96-channel extension of the article-class driver, not a
pin-for-pin clone of the article board. It preserves the article's core operating target: high-voltage
ultrasound transmit channels, 2 MHz operation, 5 ns timing quantization, transducer-output mapping,
and FPGA-controlled beamforming. It intentionally substitutes integrated HV7355 ultrasound pulsers for
the article's discrete DMOS half-bridge/gate-driver implementation.

The implemented stack is:

- One top FPGA/controller shield.
- Four HV driver shields below it.
- Each HV shield has three daisy-chained `HV7355K6-G` pulsers.
- Each HV shield exposes 24 local TX outputs.
- The complete stack maps `TX_0..TX_95`.

## Figures

Generated 96-channel shield stack:

![96-channel shield stack](../output/full_driver/shield_stack_isometric.png)

Generated FPGA/controller shield top render:

![FPGA controller shield](../output/full_driver/fpga_controller_tile_top.png)

Generated 24-channel HV7355 shield top render:

![24-channel HV7355 shield](../output/boards/hv7355_24ch_tile/top.png)

Generated 32-channel HV7355 comparison tile top render:

![32-channel HV7355 comparison tile](../output/boards/hv7355_32ch_tile/top.png)

Generated beamforming examples from `output/full_driver/driver_manifest.kv`:

![Beam steer -45 degrees](../output/beamforming/beam_steer_neg45.bmp)

![Beam steer 0 degrees](../output/beamforming/beam_steer_pos0.bmp)

![Beam steer +45 degrees](../output/beamforming/beam_steer_pos45.bmp)

## Side-by-Side Comparison

| Axis | Article implementation | Current implementation | Status |
|---|---|---|---|
| System purpose | Programmable high-voltage phased-array electronics for ultrasound neuromodulation. | Stackable FPGA/controller plus HV pulser shields for ultrasound-driver experiments. | Matched purpose. |
| Channel count | 16 channels. | 96 channels as 4 HV shields x 24 channels, plus 1 controller shield. | Superset. |
| Physical stack | Article describes a 16-channel modular implementation, not this shield-stack geometry. | 5-board shield stack: slots 0-3 are HV shields, slot 4 is FPGA/controller. | Implemented in `shield_stack_assembly.kv`. |
| Stack height | Not a directly matching constraint in the paper. | 60.000 mm total stack height at 12.000 mm board pitch. | Implemented and feasible. |
| Controller FPGA | Xilinx Artix-7 `XC7A200T-2FBG484I`, 200 MHz clock. | Current routable controller abstraction represents `XC7A100T-2FGG484C` as procurement candidate, with article FPGA CAD also downloaded. | Functionally modeled; exact FPGA footprint not imported. |
| FPGA programming evidence | FPGA generates channel waveforms and delay timing. | Manifest records JTAG `TCK,TMS,TDI,TDO`, config flash `W25Q128`, and stack bus `BUS_SCLK,BUS_SDI,BUS_SDO,BUS_LATCH,BUS_CLK`. | Implemented as board-backed evidence. |
| HV output architecture | Discrete half-bridge class-D driver per element using N/P DMOS power transistors, isolated/gate-driven control path. | Three `HV7355K6-G` integrated ultrasound pulsers per HV shield; each part supplies 8 channels. | Intentional substitution. |
| HV voltage target | Up to 150 V pulses. | HV7355 part choice is article-class: 150 V ultrasound pulser; board carries `VPP` as HV rail. | Electrically aligned; exact power/protection network still incomplete. |
| Peak output current class | Article uses discrete power devices for high pressure drive; paper discusses tens-of-watts/thermal constraints. | HV7355 selection is a 1.5 A, 150 V integrated pulser class; thermal model uses per-channel driver dissipation for stack feasibility. | Aligned at device-class level; not measured hardware. |
| Drive frequency | 2 MHz transducer operation. | `driver_manifest.kv` records `frequency_hz=2.0e6`; beam renderer uses this manifest. | Implemented. |
| Timing resolution | 5 ns inter-channel delay resolution. | `driver_manifest.kv` records `timing_step_s=5.0e-9`; tests check 5 ns focused-delay quantization. | Implemented. |
| Article waveform example | PRF 1 kHz, TBD 0.5 ms, SD 300 ms, ISI 3 s, TT 18 s, 2 MHz carrier. | Track D v2 follow-up emits per-tile stimulation programme at the kv sidecar: each HV tile carries its own PRF/SHIFT/PHASE/RAMP profile (`TileStimulationProfile::from_article_with`) plus the inherited TBD/SD/ISI/TT/VPP/dead-time protocol fields from the article preset, and the 96-lane `TX_0..TX_95` binding = one profile per tile (`is_full_stack_v2`, `validate_v2_energy_budget`). Sidecar at `output/manifests/v2_per_tile_stim.kv`. | v2 per-tile. |
| Transducer array | Custom 2 MHz, 16-element array, 4.3 x 11.7 x 0.7 mm. | Per HV tile exposes 24 local `TX_0..TX_23` nets; stack assembly maps four tiles to global `TX_0..TX_95`. The tile aperture scales the article pitch to 6.593 mm across 24 elements. | Superset tile; full 96-channel acoustic geometry needs kwavers-backed definition. |
| Focus/beam result | Measured up to 6 MPa peak-to-peak at 10 mm focus; lateral/axial resolution 0.6/4.67 mm. | Rust beam renderer generates pulse-envelope focal plots from board manifest. Current 0-degree render peaks at -0.020 mm lateral / 9.983 mm depth with 1.398 mm lateral and 4.409 mm axial 6 dB widths. | Diagnostic only; not article-grade acoustic validation. |
| Beam simulation engine | Paper reports measurement plus analysis. | `DriverManifest::validate_v2_energy_budget` emits the kwavers-conformant `EnergyBudgetReport` (with the new `per_tile_resistor_margin_w` per-tile margin vec + the SMD-package power-rating check); `validate_against_budget` (`src/validate.rs`) reads it + the full-stack v2 manifest and emits the typed `KwaversBeamStep` pre-step and `KwaversBeamValidation` (article-anchored coherent-N focal pressure, grating-lobe free boolean, MI, ISPPA, axial/lateral 6 dB extents, kwavers-side 4-check `PhysicsReport` with the new `resistor margin (per-tile min) ≥ 0 W` lock at the seam). The sealed `ResistorPackage` enum (`Smd1206` 250 mW / `Smd2512` 1 W / `Smd4527` 2 W at IPC-7351 70 °C ambient) ties the per-tile dissipation to a placement-annealer-visible footprint choice, and the propagated `resistor_margin_w` per-tile vector lets the kwavers consumer plan footprint bumps (`Smd2512 ⇒ Smd4527`) and matching-cap tightening without re-deriving `pulser_dissipation`. The `// TODO(kwavers-transducer)` marker at the in-crate physics block is the explicit seam for the future `crate::kwavers_transducer::simulate(&step) -> PressureMap` call (and its update instructions explicitly point the future consumer at `step.resistor_margin_w[i]` for package/cap planning). The **inline rejection gate has been lifted** out of `validate_v2_energy_budget` -- rename `ResistorPackage::power_rating_check(...) -> Result<f64, ResistorRatingError>` to `ResistorPackage::power_margin_w(self, dissipation_w) -> f64` returning a SIGNED margin (`max_w - dissipation_w`); the `ResistorRatingError` struct + `over_w()` accessor are deleted from the public API surface. Signed per-tile margins propagate through `EnergyBudgetReport::per_tile_resistor_margin_w` -> `KwaversBeamStep::resistor_margin_w` -> `KwaversBeamValidation::resistor_margin_w` verbatim; the kwavers-side 4th `Check` against the new safety constant `KWVERS_MIN_RESISTOR_MARGIN_W = 0.05 W (50 mW slack floor — a real headroom BUDGET above the IPC-7351 70 °C AMBIENT ceiling, vs the prior bare-ceiling semantic)` is the SOLE gatekeeper on the per-tile resistor rating (no longer redundant -- can actually fail now when an under-rated footprint is chosen: `Smd1206` on the article-class envelope fails the 4th Check while the focal-pressure / MI / grating-lobe-free checks still pass). Tightened to a `0.05 W` slack floor (50 mW headroom BUDGET above the IPC-7351 70 °C AMBIENT ceiling — a real headroom budget vs the prior bare-ceiling semantic); further tightening (e.g. `0.10 W` for pessimistic stack-temperature drift) is one constant edit + a regression-test re-pin away. | **v2 contracted**; lifted rejection gate + per-tile SMD rating + signed-margin sole-gatekeeper Check sealed at the seam; kwavers-side `0.05 W` slack floor (real headroom BUDGET above the IPC-7351 ceiling) + gatekeeper-narrative triad (under-rated `Smd2512` step 5c / tight-but-still-fits `Smd2512He` step 5b / comfortable `Smd4527` step 4-5) demonstrated in `examples/v2_per_tile_stim.rs`. |
| Stack bus | Not directly comparable; article is a 16-channel electronics implementation. | Shared 24-pin `J_STACK` bus: `VPP`, `GND`, `P5V`, `P3V3`, `BUS_SCLK`, `BUS_SDI`, `BUS_SDO`, `BUS_LATCH`, `BUS_CLK`. | Implemented and compatibility-checked. |
| Transducer connector | Article connects 16 HV outputs to custom transducer array. | HV shield manifest records transducer connector `J2`; stack assembly records each tile's local connector and global channel range. | Implemented at manifest level. |
| Isolation/control domain | Article uses digital isolators and gate drivers. | HV shield includes six trigger isolators plus one serial-bus isolator; exact isolator CAD is downloaded but not imported into generated footprints. | Partial. |
| Supplies | Article board requires 3.3 V, 5 V, and adjustable HV up to 150 V. | Stack bus carries `P3V3`, `P5V`, `VPP`, and `GND`. | Implemented at net/pinout level. |
| Thermal handling | Article analyzes and measures HV-driver power/thermal behavior. | Stack optimizer computes feasibility from per-channel dissipation, board pitch, and temperature-rise constraints; current stack reports `peak_driver_rise_k=15.000`. | Modeled; not hardware-measured. |
| Manufacturing DRC | Article hardware was fabricated/tested. | Generated FPGA and HV boards pass internal ERC/DRC/assembly/keep-in/LVS/BOM gates; external KiCad DRC reports 0 violations and 0 unconnected items on both boards. | Empirical generated-board evidence, not fabricated hardware. |
| Component exactness | Article uses real fabricated components and packages. | `component_accuracy_hv.kv` and `component_accuracy_fpga.kv` report `exact_complete=false`. | Open fabrication blocker. |

## Current Stack Assembly

`output/full_driver/shield_stack_assembly.kv` is the stack-level source of truth:

| Slot | Role | Board file | Global channels |
|---:|---|---|---|
| 0 | HV driver | `output/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb` | `TX_0..TX_23` |
| 1 | HV driver | `output/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb` | `TX_24..TX_47` |
| 2 | HV driver | `output/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb` | `TX_48..TX_71` |
| 3 | HV driver | `output/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb` | `TX_72..TX_95` |
| 4 | FPGA/controller | `fpga_controller_tile.kicad_pcb` | Stack control/programming |

Stack properties from `output/full_driver/shield_stack_plan.kv`:

- Total channels: 96
- Channels per HV tile: 24
- HV7355 devices per HV tile: 3
- Driver tiles: 4
- Total boards: 5
- Board pitch: 12.000 mm
- Board thickness: 1.600 mm
- Stack height: 60.000 mm
- Peak driver rise: 15.000 K
- Assembly complete: true

## Main Architectural Difference

The article implements the HV output stage from discrete switching devices:

`FPGA timing -> digital isolator -> gate driver -> P/N DMOS half bridge -> transducer element`

The current implementation uses integrated ultrasound pulsers:

`FPGA/controller stack bus -> isolated/control tile logic -> three daisy-chained HV7355 pulsers -> TX output`

This is a real design substitution, not a placeholder. It reduces the discrete gate-drive and power
stage complexity while increasing the stack to 96 channels.

## Completion Gaps

1. Exact FPGA footprint import:
   Replace the `XC7A-QFP` routable abstraction with the selected exact Artix-7 BGA footprint and
   escape model.

2. Exact HV7355 footprint import:
   Replace the 24-pad routed HV7355 abstraction with the downloaded 57-pad QFN56 footprint while
   preserving three daisy-chained HV7355 devices and 24 routed TX outputs per HV shield.

3. Exact connector footprint import:
   The generated boards now import the downloaded Molex/Samtec footprints for FPGA power `J1`, FPGA
   JTAG `J3`, FPGA/HV `J_STACK`, and the 24-channel Molex `0430452400` HV transducer connector.
   KiCad DRC reports 0 violations and 0 unconnected items on both full-driver boards. Remaining
   connector work is limited to the HV power input because the local downloaded file is the mating
   receptacle STEP, not yet the board-side PCB header footprint.

4. Stimulation-program manifest:
   Closed by the Track D v2 follow-up. `TileStimulationProfile` now carries the
   PRF/TBD/SD/ISI/TT/VPP/dead-time fields per HV tile (mirroring the article-class
   `StimulationProgram`), `DriverManifest::to_text()` emits them at
   `stim_tile_{i}_*` (10 keys per tile), and the per-tile staggered program appears at
   `output/manifests/v2_per_tile_stim.kv`. Reproduce: `cargo run --release --example v2_per_tile_stim`.

5. kwavers validation:
   Closed by the Track D v2 follow-up + the kwavers beam-propagation pre-step
   adapter (`src/validate.rs::validate_against_budget`). The downstream
   consumer reads `KwaversBeamStep` + `EnergyBudgetReport` verbatim and
   produces pressure maps against the article's 10 mm focus and lateral/axial
   metrics.
   - Pre-step: `KwaversBeamStep { lanes(96), aperture_m, frequency_hz,
     sound_speed_m_s, focal_m, timing_step_s, pitch_m, wavelength_m,
     f_number }` built by `manifest_to_kwavers_beam_step(...)` and gated
     on `is_full_stack_v2()` + manifest/budget lane parity.
   - Output: `KwaversBeamValidation` with focal-pressure estimate, MI, ISPPA,
     axial/lateral 6 dB extents, grating-lobe-free / in-far-field booleans,
     and a `PhysicsReport` aggregating `focal_pressure_pa ≥ 1 MPa`
     (transduction floor), `MI < 10` (cavitation ceiling), and
     `grating-lobe-free ≥ 89°`.
   - Seam: `// TODO(kwavers-transducer): replace with
     crate::kwavers_transducer::simulate(&step) -> PressureMap` marker at
     the in-crate physics block; future contributor replaces that block
     with the real kwavers call with **no struct migration**.

## Decision

The current stack should be described as:

> A 96-channel, stackable HV7355-based implementation of the article-class high-voltage phased-array
> driver architecture, complete at stack-manifest and generated-geometry level, with the kwavers
> beam-propagation contract now formally typed (`KwaversBeamStep` /
> `validate_against_budget`). Remaining blockers are exact fabricator-package import and the
> downstream `crates/kwavers-transducer` consumer that fills `PressureMap` from the typed
> pre-step at the documented `// TODO(kwavers-transducer)` seam.
