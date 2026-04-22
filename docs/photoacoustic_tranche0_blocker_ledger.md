# Photoacoustic Tranche Zero Blocker Ledger

## Scope

This ledger captures the baseline status required before the photoacoustic hard-cutover proceeds.

## Current status

- `cargo check -p kwavers --lib`: passing as of the current workspace snapshot
- `cargo test -p kwavers --test photoacoustic_vertical`: passing
- `cargo test -p kwavers --test photoacoustic_validation`: passing
- `cargo check -p pykwavers`: passing

## Known blocker classes

### Compile blockers

- The previously observed unrelated blocker in `solver/forward/hybrid/bem_fem_coupling/coupler.rs` is no longer present in the current checked baseline.
- The workspace remains vulnerable to unrelated compile breakage because of the dirty migration state.

### Test harness blockers

- Legacy tests still targeted `simulation::modalities::photoacoustic`; these are being migrated to the canonical `PhotoacousticScenario` + `PhotoacousticRunner` surface.

### GPU / WGPU blockers

- GPU/WGPU drift remains a later tranche concern.
- Same-tranche GPU closure remains an open acceptance blocker for photoacoustics.
- Apollo already exposes the retained WGPU FFT backend surface in `apollofft-wgpu`.

### Dirty-tree collision risks

- Broad in-flight edits exist across `kwavers`, `pykwavers`, `apollo`, `gaia`, and `ritk`.
- Photoacoustic cutover must avoid reverting unrelated work and must prefer local ownership tightening over broad rewrites outside the vertical.

## Tranche-zero acceptance checklist

- `kwavers` library builds
- canonical photoacoustic test target passes
- `pykwavers` builds against canonical bindings
- legacy public ownership leaks are identified and mapped
