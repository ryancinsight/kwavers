# Output Layout

Generated artifacts are grouped by purpose so board variants do not collide with stack or demo
outputs.

## Canonical board variants

| Folder | Contents |
|--------|----------|
| `output/boards/hv7355_24ch_tile/` | Exact 24-channel HV shield used by the 96-channel stack model. Includes KiCad board/project sidecars, DRC JSON, top render, connector render, stack manifest, and component-accuracy manifest. |
| `output/boards/hv7355_32ch_tile/` | Generated 32-channel comparison tile from `examples/hv7355_32ch_tile.rs`. Includes KiCad board/project sidecars, DRC JSON, SVG, and top render with visible stock package/header CAD bodies; it is not the exact 24-channel shield topology. |

## Supporting outputs

| Folder | Contents |
|--------|----------|
| `output/full_driver/` | Stack integration workspace for the FPGA controller, the 24-channel HV shield, stack manifests, stack renders, and fabrication exports. |
| `output/beamforming/` | kwavers-backed beamforming validation sidecar, realized tile geometry CSV, metrics CSV, and deterministic BMP visualizations generated from the driver manifest. |
| `output/examples/` | Older single-board demos and non-canonical example boards. |
| `output/manifests/` | Standalone generated manifests not tied to one board folder. |
| `output/reports/` | Standalone DRC/audit reports. |
| `output/renders/` | Standalone render experiments. |
| `output/archive/` | Previous root-level fabrication bundles and generated root artifacts retained for traceability. |
