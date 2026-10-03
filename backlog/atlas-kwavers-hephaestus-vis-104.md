<a id="atlas-kwavers-hephaestus-vis-104"></a>

## ATLAS-KWAVERS-HEPHAESTUS-VIS-104 — Reject uninitialized GPU visualization [patch] — review 2026-08-17

priority: correctness; needs: none; scope: `crates/kwavers-analysis/src/visualization/engine/mod.rs`, `crates/kwavers-analysis/src/visualization/mod.rs`

- **Outcome:** GPU-enabled multi-field rendering returns the existing typed feature/resource error until the renderer and data pipeline are initialized; initialized GPU rendering and the CPU fallback retain every field.
- **Acceptance:** valid multi-field input never returns `Ok(())` when GPU resources are absent; initialized GPU rendering processes every field; invalid field-count input and the non-GPU fallback stay value-semantic.
- **Non-goals:** no CPU fallback behind the GPU feature, no FDTD/provider changes, no renderer ownership changes in Hephaestus or Leto.
- **State:** the source head passed the feature-enabled hosted matrix; the PM-only follow-up head had to rerun the same gate before closure.
- **Next step:** re-validate the engine at the default branch; close if the typed error is present and tested.
