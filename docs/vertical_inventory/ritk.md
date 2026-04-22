# `ritk` Inventory

## Overall status

RITK is the registration and transform stack intended for multimodal alignment.

## Canonical responsibilities

- transforms
- similarity metrics
- rigid/affine/deformable registration
- multimodal imaging alignment

## Current integration state

- `ritk-core`, `ritk-registration`, and related crates are present and compile in the workspace
- `kwavers` depends on `ritk-core` and `ritk-registration` optionally
- multimodal workflows in `kwavers` still need a tighter canonical integration boundary

## Immediate tranche-one priorities

1. treat RITK as the only registration home
2. avoid new registration duplication in `kwavers`
3. defer substantive registration integration work until photoacoustic and acoustic-core closure
