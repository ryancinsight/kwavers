# Safety and Dosimetry

## Scope

Safety covers mechanical index, thermal index, cavitation dose, acoustic intensity, thermal dose, treatment limits, compliance reporting, and closed-loop safety control. Code ownership maps to `kwavers::clinical::safety`, `kwavers::clinical::therapy`, and `kwavers::physics::acoustics::therapy`.

## Theorem: Mechanical Index Scaling

For peak rarefactional pressure `p_neg` in MPa and center frequency `f` in MHz,

```text
MI = p_neg / sqrt(f).
```

### Proof Sketch

MI is a normalized cavitation-risk index. The square-root frequency scaling encodes the empirical reduction in cavitation likelihood with increasing frequency for otherwise comparable rarefactional pressure.

## Algorithm: Safety Validation

1. Compute pressure extrema from the actual simulated or measured waveform.
2. Convert to MI, TI, intensity, and dose with explicit units.
3. Compare against treatment-specific safety limits.
4. Preserve safety-controller decisions and input metrics in reports.

## Implementation Targets

- Keep MI, TI, cavitation, and thermal dose calculators separate.
- Reject nonpositive frequency, nonfinite pressure, and mismatched time axes.
- Validate controller actions with computed metric values, not status-only assertions.

## Research Anchors

- FDA diagnostic ultrasound output guidance context: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-clearance-diagnostic-ultrasound-systems-and-transducers
- Transcranial focused ultrasound safety context: https://doi.org/10.1186/s12984-025-01753-2
