# ADR 003 — Signal processing migrated to the analysis layer

- **Status:** Implemented
- **Date:** 2026-01-09 · **Audited:** 2026-06-03
- **Change class:** [arch]
- **Relates:** finalizes the placement for [ADR 001](001-adaptive-beamforming-consolidation.md) and [ADR 002](002-sensor-array-processing-consolidation.md); paths later relocated by the crate split ([ADR 011](011-workspace-crate-split.md))

## Context

Signal-processing algorithms (beamforming, source localization, passive acoustic
mapping) lived in `domain::sensor`. The domain layer is meant to hold only
primitives — sensor geometry, sampling rates, field sampling — so housing array
processing there was a ~5-layer jump that risked upward (domain→physics/solver)
dependencies, conflated primitives with algorithms, and coupled reusable signal
processing to domain types. It also contradicts the literature, which treats
beamforming as array signal processing (Van Trees, Capon, Schmidt), not sensor
geometry.

## Decision

Move all signal processing out of `domain::sensor` into a dedicated analysis-layer
module. The domain keeps only sensor primitives; algorithms operate on sampled
data regardless of whether it originates from real sensors, simulation, or clinical
workflows. Remove the duplicate definitions identified in
[ADR 002](002-sensor-array-processing-consolidation.md) as part of the move.

## Current state (audited 2026-06-03)

Done — all migration phases complete and the transitional deprecation shims have
been removed. Signal processing now lives in `crates/kwavers-analysis/src/signal_processing/`:

- `beamforming/` — `time_domain/das`, `adaptive/{mvdr, subspace/{music,esmv}}`,
  `narrowband/capon`, `three_dimensional`, `neural`, `gpu`
- `localization/` — `trilateration`, `multilateration`, `tdoa`,
  `beamforming_search`, `bayesian`, `music`, `model_order`
- `pam/` — `config`, `mapper`, `processor`, `delay_and_sum`

The domain side is purified: `crates/kwavers-domain/src/sensor/beamforming/mod.rs`
retains only the hardware-geometry `SensorBeamformer` plus the
`BeamformingCoreConfig` SSOT type; `domain::sensor::localization` no longer exists,
and `domain::sensor::passive_acoustic_mapping` contains only array-geometry
primitives. No `#[deprecated] pub use crate::analysis::…` re-export shims remain.

## Consequences

- Strict downward dependency flow restored; domain is reusable and test-isolated.
- Algorithms are reusable across data sources (sensor / simulation / clinical).
- Through the `kwavers` facade these modules are still reachable as
  `kwavers::analysis::signal_processing::…`; the underlying crate is `kwavers_analysis`.
- This ADR's original target tree used pre-split `src/analysis/…` paths; the
  workspace crate split ([ADR 011](011-workspace-crate-split.md)) relocated them to
  `crates/kwavers-analysis/src/…`.
