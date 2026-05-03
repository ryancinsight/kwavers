# Beamforming and Image Formation

![Ultrafast diagnostic pipeline](figures/diagnostics_ultrafast_pipeline.svg)

## Scope

Beamforming covers delay-and-sum, MVDR, subspace methods, plane-wave compounding, SAFT, 3-D beamforming, neural beamforming, clutter filtering, localization, and image-quality metrics. Code ownership maps to `kwavers::analysis::signal_processing::beamforming`, `kwavers::clinical::imaging`, and `kwavers::analysis::ml`.

## Theorem: Coherent Sum Phase Alignment

If delayed channel signals `x_i(t - tau_i)` share the same phase at a target point, then their coherent sum amplitude scales linearly with aperture count in the noiseless case.

### Proof Sketch

Phase alignment makes every target contribution equal to a common complex phase. Summing `N` equal phasors gives magnitude `N` times the single-channel magnitude.

## Algorithm: Beamforming Contract

1. Store channel data, geometry, and sampling metadata separately.
2. Compute delays from geometry and local sound speed.
3. Apply aperture weighting or covariance inversion.
4. Validate point-target localization, sidelobe level, and contrast metrics.

## Implementation Targets

- Keep neural feature extraction, layer inference, physics regularization, and adaptation separated.
- Avoid covariance shortcuts that do not inspect matrix values.
- Keep beamforming benchmark inputs reproducible.

## Research Anchors

- ULTRA-SR ULM benchmark: https://doi.org/10.1109/TMI.2024.3388048
- Row-column 3-D super-resolution ultrasound: https://doi.org/10.1016/j.ultrasmedbio.2024.03.020
