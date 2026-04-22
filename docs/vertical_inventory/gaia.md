# `gaia` Inventory

## Overall status

Gaia is the mesh/geometry support crate and should own reusable geometry preparation for
multi-physics workflows.

## Canonical responsibilities

- reusable geometry and mesh types
- detector geometry helpers
- interface extraction
- orientation-sensitive mesh utilities

## Observed integration state

- `kwavers` depends directly on `gaia`
- some geometry extraction logic still exists in `kwavers`
- long-term ownership for detector arrays and interface extraction should live here

## Immediate tranche-one priorities

1. keep Gaia as the target home for reusable detector/interface geometry
2. defer major expansion until a canonical vertical proves shared need
3. add regression tests for orientation correctness when new geometry APIs land
