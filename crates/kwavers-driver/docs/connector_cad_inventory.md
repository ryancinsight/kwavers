# Connector CAD Inventory

Evidence tier: local file inspection plus generated-board DRC/render checks.

## Selected Connector Parts

| Function | Selected part | Local CAD status |
|---|---|---|
| HV power input | Molex `0430450400` board header | Exact vendor CAD is extracted at `docs/cad_models/430450400/`: `MOLEX_430450400.kicad_mod`, `430450400.kicad_sym`, and `430450400.step`. Current HV board artifacts use the exact `MOLEX_430450400` through-hole footprint and zero-local-rotation STEP body at `J4`; KiCad CLI DRC on the promoted board reports 0 violations and 0 unconnected items. |
| FPGA power input | Molex `0430450600` board header | Exact downloaded footprint is in `docs/cad_models/430450600/KiCADv6/footprints.pretty/CONN_SD-43045-001_06_MOL.kicad_mod`. Exact PCB-header STEP is present at `docs/cad_models/430450600_stp/430450600.stp`; the footprint carries raw vendor model offset `(3.0, 2.5643, 0)`, and the importer normalizes it to the generated courtyard-centred coordinate frame. Mating receptacle STEP is at `docs/cad_models/430250600_stp/430250600.stp`. |
| HV transducer output | Molex `0430452400` board header target | Required for the 24-channel tile. DigiKey lists `0430452400` as a 24-position right-angle Micro-Fit 3.0 header and `0430252400` as the matching 24-position receptacle housing. Exact local footprint and STEP are now imported at `docs/cad_models/430452400/`; the STEP is rendered with the verified `(0, 2.8, 0)` offset so the connector body covers the plated pin rows. Mating receptacle STEP is at `docs/cad_models/430252400_stp/430252400.stp`. The existing `430452000` folder is the older 20-position part and is intentionally not used for the 24-channel tile. |
| FPGA JTAG | Samtec `TSW-106-07-G-S` | Downloaded footprint is in `docs/cad_models/TSW_106_07_G_S/KiCADv6/footprints.pretty/CON6_1X6_TU_TSW.kicad_mod`; KiCad system STEP is used for render with the pad-grid-derived `(-6.35, 0, 0)` offset and `-90 deg` Z rotation. |
| Stack bus | Samtec `TSW-112-07-G-D` / `SSW-112-23-G-D` | Header footprint is in `docs/cad_models/TSW_112_07_G_D/KiCADv6/footprints.pretty/TSW-112-07-G-D_SAI.kicad_mod`; mated socket footprint is imported at `docs/cad_models/SSW_112_23_G_D/KiCADv6/footprints.pretty/CON24_2X12_TU_SSQ.kicad_mod`. The KiCad system 2x12 header STEP is rendered with the verified `(-13.97, -1.27, 0)` offset and `-90 deg` Z rotation after footprint recentering. |

## Additional Connector CAD — Now Extracted (2026-06-23)

The following connector archives from `driver/docs/` have been extracted into `docs/cad_models/`.
None are yet wired into generated boards.

### Samtec Headers — Wider / Taller Stack Variants

| MPN | Description | Footprint | Path |
|---|---|---|---|
| `TSW-108-08-G-D` | 2×8 2.54 mm DIP header | `CON16_2X8_TU_TSW_SAI.kicad_mod` | `docs/cad_models/TSW_108_08_G_D/KiCADv6/` |
| `TSW-112-23-G-D` | 2×12 2.54 mm header 23 mm post | `CON24_2X12_TU_TSW_SAI.kicad_mod` | `docs/cad_models/TSW_112_23_G_D/KiCADv6/` |
| `TSW-148-07-G-D` | 2×24 2.54 mm header 7 mm post | `TSW-148-07-G-D_SAI.kicad_mod` | `docs/cad_models/TSW_148_07_G_D/KiCADv6/` |
| `TSW-148-08-G-D` | 2×24 2.54 mm header 8 mm post | `SAMTEC_TSW-148-08-G-D.kicad_mod` | `docs/cad_models/TSW_148_08_G_D/` |

`TSW-148-*-G-D` (2×24 = 48 pins) supports a wider stack bus for future tiles needing more signal
pairs than the current narrow 2×12 bus.

### Other Connectors

| MPN | Vendor | Description | Footprint | Path |
|---|---|---|---|---|
| `10164359-00011LF` | Amphenol FCI | Fine-pitch board connector | `AMPHENOL_10164359-00011LF.kicad_mod` | `docs/cad_models/10164359_00011LF/` |
| `142146` | TE Connectivity (AMP) | 5-position connector | `CONN_142146_AMP.kicad_mod` | `docs/cad_models/142146/KiCADv6/` |
| `1720640002` | Molex | 2-position board connector | `CONN_SD-76825-0100_02_MOL.kicad_mod` | `docs/cad_models/1720640002/KiCADv6/` |
| `1720650002` | Molex | 2-position board connector | `CONN_SD-76829-0100_02_MOL.kicad_mod` | `docs/cad_models/1720650002/KiCADv6/` |

## Integration Constraint

The Rust footprint importer now loads the exact Molex `0430452400` `.kicad_mod` for HV `J2`, the
exact Molex `0430450600` `.kicad_mod` for FPGA `J1`, and the exact Samtec `TSW-106-07-G-S` /
`TSW-112-07-G-D` `.kicad_mod` footprints for FPGA JTAG and the inter-board stack bus. Pad-name
mapping is used for every exact connector so KiCad pad numbers, schematic pins, and routed nets stay
aligned. Imported STEP model offsets are translated by the same courtyard-centre shift applied to
imported pads, so the model body and pad grid share the generated footprint coordinate frame. The
router models plated pins as multi-layer access terminals and the DFM pass emits
explicit pad-entry copper from routed grid nodes to exact pad centers; KiCad CLI DRC on both
`output/full_driver/fpga_controller_tile.kicad_pcb` and the canonical HV copy
`output/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb` report 0 violations and
0 unconnected items.
The Molex `0430450400` PCB header abstraction is closed in the current board artifact: `J4` uses the
exact downloaded `MOLEX_430450400` through-hole footprint, the exact `430450400.step` board-header
body, and a rerouted four-net internal-layer breakout from the previous power anchors. The footprint
is a through-hole right-angle Micro-Fit header, so its four plated holes and board-lock hole are
manufacturing features of this MPN, not optional render details. A non-through-board stack interface
requires a different connector family. The rejected direct-swap trial remains useful as a regression
reference: the exact footprint cannot be treated as a cosmetic body-only replacement because its PTH
and NPTH holes require rerouting.
