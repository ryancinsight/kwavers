# Sensors and Measurements

## Scope

Sensors cover point sensors, masks, Cartesian interpolation, directivity, detector averaging, recorder layouts, complex signals, and sonoluminescence detection. Code ownership maps to `kwavers::domain::sensor`, `kwavers::analysis::signal_processing`, and `kwavers::solver::validation`.

## Theorem: Linear Sensor Superposition

For a linear acoustic field `p` and sensor weights `w_i`, a weighted sensor output

```text
s(t) = sum_i w_i p(r_i, t)
```

is linear in `p`.

### Proof Sketch

For fields `p` and `q` and scalars `a` and `b`, substitution gives `sum_i w_i (a p_i + b q_i) = a sum_i w_i p_i + b sum_i w_i q_i`.

## Algorithm: Sensor Validation

1. Validate mask shape and sensor coordinates against the grid.
2. Preserve traversal ordering in every exported sensor matrix.
3. Compare raw detector traces before derived image metrics.
4. Record directivity and averaging kernels as inspectable parameters.

## Implementation Targets

- Keep recorder layout, interpolation, and detector physics in separate modules.
- Validate sensor directivity with angle-dependent traces.
- Keep sonoluminescence event detection separate from optical emission models.

## Research Anchors

- k-wave-python sensor examples: https://k-wave-python.readthedocs.io/
