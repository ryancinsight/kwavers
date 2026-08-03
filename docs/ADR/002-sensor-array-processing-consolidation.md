# ADR 002 — Sensor array-processing consolidation

- **Status:** Implemented (realized in the analysis layer per [ADR 003](003-signal-processing-analysis-layer.md))
- **Date:** 2025-11-12 · **Audited:** 2026-06-03 · **Revised:** 2026-08-02
- **Change class:** [arch]
- **Relates:** subsumes [ADR 001](001-adaptive-beamforming-consolidation.md); placement overridden by [ADR 003](003-signal-processing-analysis-layer.md)

## Context

The `sensor` bounded context held overlapping array-processing implementations:

- `sensor/beamforming` — conventional + adaptive + subspace + processor + steering.
- `sensor/adaptive_beamforming` — duplicate adaptive/conventional/subspace plus
  legacy files (see [ADR 001](001-adaptive-beamforming-consolidation.md)).
- `sensor/passive_acoustic_mapping/beamforming.rs` — re-implemented DAS/MVDR/
  MUSIC/ESMV with a divergent `BeamformingConfig`.
- `sensor/localization/{beamforming,algorithms}.rs` — a third DAS search rather
  than reusing the shared API.

This produced two divergent `BeamformingConfig` types and three copies of the
core algorithms — an SSOT violation that obscured the domain boundary.

## Decision

Unify all array processing behind one beamforming API with a single typed core
config and per-task config wrappers. Passive acoustic mapping (PAM) and source
localization become *consumers* of that API rather than re-implementers. Feature-
gate the experimental neural beamformer.

## Current state (audited 2026-06-03)

Done — and, per [ADR 003](003-signal-processing-analysis-layer.md), realized in
the **analysis** layer rather than under `domain::sensor`:

- **Single core config.** `BeamformingCoreConfig` is the SSOT
  (`crates/kwavers-transducer/src/beamforming/config.rs`), with Aequitas
  `Velocity` and `Frequency` fields at the public boundary.
- **PAM is a consumer.** `crates/kwavers-analysis/src/signal_processing/pam/`
  (`processor.rs`, `mapper.rs`, `delay_and_sum/`) with `PamBeamformingConfig`
  (`pam/config.rs:7`) wrapping the core config; the standalone PAM algorithm file
  is gone.
- **Localization is a consumer.** `crates/kwavers-analysis/src/signal_processing/localization/beamforming_search/`
  documents itself as a grid search over the shared SSOT; `LocalizationBeamformSearchConfig`
  wraps `BeamformingCoreConfig`. The standalone localization DAS files are gone.
- **Neural beamformer** lives under `…/beamforming/neural/` (config/features/pinn/
  distributed).

Note: `BeamformingConfig3D` (`…/beamforming/three_dimensional/config.rs:70`) is a
distinct 3-D imaging config, not a duplicate. `MUSIC` (beamformer) and
`MUSICProcessor` (DOA/source-count in `localization/music/`) are legitimately
different roles, not an SSOT violation.

## Consequences

- One beamforming API; PAM and localization no longer carry private copies.
- The former `BeamformingConfig` alias was removed on 2026-08-02. All current
  callers use the canonical `BeamformingCoreConfig` name; the branch boundary
  is the migration mechanism for the breaking public rename.
- Module placement moved from `domain::sensor` to `kwavers-analysis` — the layer
  decision in [ADR 003](003-signal-processing-analysis-layer.md) supersedes the
  domain-layer tree sketched in the original proposal.
