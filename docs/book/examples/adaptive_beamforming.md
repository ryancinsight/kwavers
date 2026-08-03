# Example: Adaptive Beamforming

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example adaptive_beamforming`  
**Source**: [`crates/kwavers/examples/adaptive_beamforming.rs`](../../../crates/kwavers/examples/adaptive_beamforming.rs)

## What This Example Demonstrates

Runs Minimum Variance Distortionless Response (MVDR/Capon) adaptive beamforming
on a synthetic uniform linear array and verifies the result against an
analytical steering oracle. A far-field source is synthesized at a known angle,
the sample covariance is estimated from deterministic complex baseband
snapshots, and the Capon spectrum is scanned over candidate angles. The
executable asserts:

- the spectrum peaks at the oracle angle (within the scan step);
- the distortionless constraint `a(θ₀)ᴴ w(θ₀) = 1` holds to round-off;
- MVDR nulls a strong interferer that delay-and-sum (DAS) cannot.

| Component | API | Value |
|---|---|---|
| Steering oracle | `beamforming::narrowband::NarrowbandSteering` | phase-only plane-wave steering `a_i(θ) = exp(−j 2π f τ_i(θ))` |
| Covariance | `beamforming::covariance::CovarianceEstimator::estimate_complex` | sample covariance with forward-backward averaging |
| Adaptive weights | `beamforming::MinimumVariance::compute_weights` | `w = R⁻¹a / (aᴴR⁻¹a)` with diagonal loading |
| Verification | Hermitian quadratic `wᴴRw` | Capon spectrum peak, distortionless, and interferer-nulling assertions |

## Key Code Snippet

```rust
let w = mvdr.compute_weights(&covariance, &a)?;
let power = hermitian_quadratic(&w, &covariance);
// The Capon peak must match the steering oracle within the scan step.
assert!((theta_hat - theta0_deg).abs() <= 3.0);
```

## Expected Output

The executable prints the estimated Capon peak angle, the spectrum power at the
oracle direction, the distortionless-constraint error, and the MVDR-versus-DAS
interferer comparison. All checks must pass; a failing assertion reports the
observed value before exiting.

## Book Chapter

[← Transducer Arrays and Beamforming](../beamforming_and_image_formation.md)
