# Keep visualization backend and quality selection single-source

- Status: Accepted
- Date: 2026-08-25
- Item: Atlas `backlog.md#atlas-kwavers-vis-config-2026-08-25`
- Depends on: Atlas ADR 0054 (Kwavers/Hephaestus visualization contract)

## Context

Top-level Kwavers selects `VisualizationBackend::Leto` or
`VisualizationBackend::Hephaestus`, constructs the concrete provider, and
injects it into `VisualizationEngine`. `kwavers-analysis` nevertheless retained
a public `VisualizationConfig::gpu_enabled` boolean. No production path read
the field: setting it to `false` did not prevent Hephaestus initialization, and
setting it to `true` did not acquire a device. The field therefore described a
choice it could not make and contradicted the provider typestate.

The same configuration held both `quality` and the compatibility alias
`render_quality`. Presets initialized them equally, but callers could diverge
them. Adaptive quality updated `quality`; `Renderer3D` read `render_quality` and
kept its own configuration clone. As a result, adaptive quality reported a
transition without changing subsequent rendering.

## Decision

Delete `gpu_enabled` and `render_quality`. Backend selection remains solely at
the top-level Kwavers factory, and `VisualizationConfig::quality` is the sole
render-quality value. `VisualizationEngine::auto_adjust_quality` applies a
transition to both its retained configuration and an initialized renderer.

The existing 20 percent hysteresis band and one-level transition policy remain
unchanged. Tests pin the lower, stable, and upper regions and verify that a
quality change alters the renderer's sampling density on an input whose
impulse is visible only to the denser sampling path.

## Consequences

This is a public breaking change. Struct-literal callers remove `gpu_enabled`
and rename `render_quality` to `quality`. Callers that previously expected the
boolean to select execution must instead call
`kwavers::visualization::create_visualization_provider` with the required
backend and inject the returned provider before initialization. A requested
Hephaestus backend continues to fail with its typed acquisition error and never
falls back to Leto.

`VisualizationConfig::debug` remains a low-quality, profiling-enabled render
preset; it no longer suggests that a render preset controls device selection.

## Alternatives rejected

**Make `gpu_enabled` authoritative.** A boolean cannot identify Leto versus
Hephaestus and would create a second selection path below the composition
boundary. Mapping `false` to Leto would also be a hidden fallback.

**Keep aliases synchronized.** Public mutable fields can diverge after every
write, so synchronization would be call-order dependent and retain two names
for one contract.

**Rebuild the renderer after every quality transition.** The transition only
changes operation selection. Reconstructing renderer state would discard
resources and add work unrelated to the changed value.
