# Sources and Transducers

## Scope

Sources and transducers cover tone bursts, pressure sources, velocity sources, phased arrays, focused bowls, annular arrays, flexible arrays, BLI rasterization, source ordering, and calibration. Code ownership maps to `kwavers::domain::source`, `kwavers::domain::source::kwave_array`, and `kwavers::domain::source::flexible`.

## Theorem: Delay Law for a Focused Aperture

For element position `r_i`, focus `r_f`, and sound speed `c`, the focusing delay is

```text
tau_i = (|r_i - r_f| - min_j |r_j - r_f|) / c.
```

### Proof Sketch

The wavefront from every element reaches the focus simultaneously when emission times differ by path-length excess divided by wave speed. Subtracting the minimum path delay makes all delays non-negative without changing relative phase.

## Algorithm: Source Contract

1. Define the geometric support in physical coordinates.
2. Rasterize support to grid cells with documented ordering.
3. Generate or import source signals with exact sample count and time step.
4. Validate source mask, distributed signal matrix, and field output separately.

## Implementation Targets

- Preserve Fortran-order active-cell rows for k-Wave parity.
- Keep geometric rasterization separate from signal synthesis.
- Treat flexible-array calibration as domain state, not a solver-side correction.

## Research Anchors

- k-wave-python array examples: https://k-wave-python.readthedocs.io/
- k-Wave MATLAB toolbox: http://www.k-wave.org/
