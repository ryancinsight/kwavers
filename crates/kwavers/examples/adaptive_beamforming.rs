//! Adaptive Beamforming: MVDR/Capon on a Synthetic Aperture
//!
//! Demonstrates Minimum Variance Distortionless Response (MVDR / Capon)
//! beamforming on a synthetic uniform linear array, verified against an
//! analytical steering oracle:
//!
//! 1. **Steering oracle** — a far-field source at a known angle θ₀ is
//!    synthesized with the phase-only narrowband steering model
//!    `a_i(θ) = exp(-j 2π f τ_i(θ))`, where `τ_i` is the propagation delay
//!    from the far-field direction to sensor `i`.
//! 2. **Covariance estimation** — the sample covariance
//!    `R = (1/N) Σ x_n x_nᴴ` is estimated from complex baseband snapshots
//!    with forward-backward averaging.
//! 3. **Capon spectrum** — the MVDR weight
//!    `w = R⁻¹ a / (aᴴ R⁻¹ a)` is computed at every candidate angle and the
//!    output power `P(θ) = wᴴ R w` must peak at the oracle angle θ₀.
//! 4. **Distortionless constraint** — `a(θ₀)ᴴ w(θ₀) = 1` is verified to
//!    round-off, and a second scenario shows MVDR nulling a strong
//!    interferer where delay-and-sum (DAS) cannot.
//!
//! All snapshots use deterministic pseudo-random phases, so the run is fully
//! reproducible and the assertions are input-sensitive.
//!
//! Run with: `cargo run -p kwavers --example adaptive_beamforming`

use eunomia::Complex64;
use kwavers_analysis::signal_processing::beamforming::covariance::CovarianceEstimator;
use kwavers_analysis::signal_processing::beamforming::narrowband::{
    NarrowbandSteering, NarrowbandSteeringVector,
};
use kwavers_analysis::signal_processing::beamforming::MinimumVariance;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

/// Number of array elements.
const NUM_ELEMENTS: usize = 8;
/// Element spacing (λ/2 at f₀; λ = c/f₀ = 1.5 mm at 1 MHz in water).
const ELEMENT_SPACING_M: f64 = 7.5e-4;
/// Number of complex baseband snapshots used to estimate the covariance.
const NUM_SNAPSHOTS: usize = 64;
/// Far-field range used to synthesize plane-wave steering (m).
const FAR_FIELD_RANGE_M: f64 = 10.0;
/// Diagonal loading δ applied to the covariance for numerical stability.
const DIAGONAL_LOADING: f64 = 1e-6;

fn main() -> KwaversResult<()> {
    println!("Adaptive Beamforming: MVDR/Capon on a Synthetic Aperture");
    println!("==========================================================");

    let frequency_hz = MHZ_TO_HZ; // 1 MHz
    let sound_speed = SOUND_SPEED_WATER_SIM;

    // Uniform linear array on the x-axis, centred at the origin.
    let sensor_positions: Vec<[f64; 3]> = (0..NUM_ELEMENTS)
        .map(|i| {
            let x = (i as f64 - (NUM_ELEMENTS - 1) as f64 / 2.0) * ELEMENT_SPACING_M;
            [x, 0.0, 0.0]
        })
        .collect();
    let steering = NarrowbandSteering::new(sensor_positions, sound_speed)?;

    // ── Scenario 1: single far-field source at θ₀ = 20° ──
    let theta0_deg = 20.0_f64;
    let (theta_hat, power_at_theta0, distortionless_error) =
        single_source_scan(&steering, frequency_hz, theta0_deg)?;

    println!();
    println!("Scenario 1 — single far-field source at {theta0_deg}°");
    println!("  Capon peak estimated at: {theta_hat:.2}°");
    println!("  spectrum power at θ₀:    {power_at_theta0:.6e}");
    println!("  |a(θ₀)ᴴ w(θ₀) − 1|:      {distortionless_error:.3e}");

    assert!(
        (theta_hat - theta0_deg).abs() <= 3.0,
        "MVDR spectrum peak {theta_hat:.2}° must match the steering oracle {theta0_deg}° within 3°"
    );
    assert!(
        distortionless_error < 1e-6,
        "distortionless constraint violated: |a(θ₀)ᴴ w(θ₀) − 1| = {distortionless_error:.3e}"
    );

    // ── Scenario 2: source at θ₀ plus a strong interferer at θ₁ ──
    let interferer_deg = -15.0_f64;
    let (mvdr_power, das_power) =
        interference_rejection(&steering, frequency_hz, theta0_deg, interferer_deg)?;

    println!();
    println!(
        "Scenario 2 — source at {theta0_deg}° with interferer at {interferer_deg}° (2× amplitude)"
    );
    println!("  MVDR output power at θ₀: {mvdr_power:.6e}");
    println!("  DAS output power at θ₀:  {das_power:.6e}");

    assert!(
        mvdr_power < das_power,
        "MVDR ({mvdr_power:.3e}) must null the interferer below DAS ({das_power:.3e})"
    );

    println!();
    println!("✅ All analytical steering-oracle checks passed.");
    Ok(())
}

/// Scenario 1: scan the Capon spectrum and verify the peak and the
/// distortionless constraint against the oracle direction `theta0_deg`.
fn single_source_scan(
    steering: &NarrowbandSteering,
    frequency_hz: f64,
    theta0_deg: f64,
) -> KwaversResult<(f64, f64, f64)> {
    let snapshots = synthesize_snapshots(steering, frequency_hz, &[(theta0_deg, 1.0)], 0.05, 42)?;
    let covariance = CovarianceEstimator::new(true, NUM_SNAPSHOTS).estimate_complex(&snapshots)?;
    let mvdr = MinimumVariance::with_diagonal_loading(DIAGONAL_LOADING);

    // Capon spectrum over the candidate scan.
    let mut best_angle = 0.0_f64;
    let mut best_power = f64::NEG_INFINITY;
    let mut power_at_theta0 = 0.0_f64;
    for angle_deg in -60..=60 {
        let angle_deg = angle_deg as f64;
        let a = steering
            .steering_vector_point(far_field_candidate(angle_deg), frequency_hz)?
            .into_array();
        let w = mvdr.compute_weights(&covariance, &a)?;
        let power = hermitian_quadratic(&w, &covariance);
        if power > best_power {
            best_power = power;
            best_angle = angle_deg;
        }
        if (angle_deg - theta0_deg).abs() < 1e-9 {
            power_at_theta0 = power;
        }
    }

    // Distortionless constraint at the oracle direction.
    let a0 = steering
        .steering_vector_point(far_field_candidate(theta0_deg), frequency_hz)?
        .into_array();
    let w0 = mvdr.compute_weights(&covariance, &a0)?;
    let distortionless_error = (hermitian_inner(&a0, &w0) - Complex64::new(1.0, 0.0)).norm();

    Ok((best_angle, power_at_theta0, distortionless_error))
}

/// Scenario 2: steer at the source while a strong interferer is present, and
/// compare MVDR output power against delay-and-sum.
fn interference_rejection(
    steering: &NarrowbandSteering,
    frequency_hz: f64,
    theta0_deg: f64,
    interferer_deg: f64,
) -> KwaversResult<(f64, f64)> {
    let snapshots = synthesize_snapshots(
        steering,
        frequency_hz,
        &[(theta0_deg, 1.0), (interferer_deg, 2.0)],
        0.05,
        7,
    )?;
    let covariance = CovarianceEstimator::new(true, NUM_SNAPSHOTS).estimate_complex(&snapshots)?;
    let mvdr = MinimumVariance::with_diagonal_loading(DIAGONAL_LOADING);

    let a0 = steering
        .steering_vector_point(far_field_candidate(theta0_deg), frequency_hz)?
        .into_array();

    // MVDR: adaptive weights minimise output power subject to unit gain at θ₀.
    let w_mvdr = mvdr.compute_weights(&covariance, &a0)?;
    let mvdr_power = hermitian_quadratic(&w_mvdr, &covariance);

    // DAS: normalised conventional steering weights, no interference nulling.
    let w_das = a0.mapv(|v| v / NUM_ELEMENTS as f64);
    let das_power = hermitian_quadratic(&w_das, &covariance);

    Ok((mvdr_power, das_power))
}

/// Far-field candidate point on the direction `angle_deg` (measured from the
/// array-broadside z-axis), at `FAR_FIELD_RANGE_M` range.
fn far_field_candidate(angle_deg: f64) -> [f64; 3] {
    let theta = angle_deg.to_radians();
    [
        FAR_FIELD_RANGE_M * theta.sin(),
        0.0,
        FAR_FIELD_RANGE_M * theta.cos(),
    ]
}

/// Synthesize `NUM_SNAPSHOTS` deterministic complex baseband snapshots.
///
/// Snapshot `n` is `x_n = Σ_k s_k a(θ_k) + noise`, where `a(θ_k)` is the
/// unit-norm phase-only steering vector toward direction `θ_k`, `s_k` is the
/// source amplitude, and the noise has relative magnitude `noise_scale`.
/// All phases come from a deterministic generator so the run is reproducible.
fn synthesize_snapshots(
    steering: &NarrowbandSteering,
    frequency_hz: f64,
    sources: &[(f64, f64)], // (angle_deg, amplitude)
    noise_scale: f64,
    seed: u64,
) -> KwaversResult<Array2<Complex64>> {
    // Steering vectors depend only on the source directions; compute them once.
    let steering_vectors: Vec<Array1<Complex64>> = sources
        .iter()
        .map(|&(angle_deg, _)| {
            steering
                .steering_vector_point(far_field_candidate(angle_deg), frequency_hz)
                .map(NarrowbandSteeringVector::into_array)
        })
        .collect::<KwaversResult<_>>()?;

    let mut snapshots = Array2::<Complex64>::zeros((NUM_ELEMENTS, NUM_SNAPSHOTS));
    for snapshot_index in 0..NUM_SNAPSHOTS {
        // One deterministic complex source term per snapshot.
        let source_term =
            Complex64::new(0.0, deterministic_phase(seed, snapshot_index as u64)).exp();
        for element_index in 0..NUM_ELEMENTS {
            let mut value = Complex64::default();
            for ((angle_deg, amplitude), a) in sources.iter().zip(&steering_vectors) {
                let _ = angle_deg;
                value += Complex64::new(*amplitude, 0.0) * source_term * a[element_index];
            }
            let noise_phase = deterministic_phase(
                seed.wrapping_add(1),
                snapshot_index as u64 * NUM_ELEMENTS as u64 + element_index as u64,
            );
            let noise = Complex64::new(0.0, noise_phase).exp() * noise_scale;
            snapshots[[element_index, snapshot_index]] = value + noise;
        }
    }
    Ok(snapshots)
}

/// `wᴴ R w` — the real MVDR/Capon output power for weights `w` over the
/// Hermitian covariance `R`.
fn hermitian_quadratic(w: &Array1<Complex64>, r: &Array2<Complex64>) -> f64 {
    let n = w.len();
    let mut total = Complex64::default();
    for i in 0..n {
        for j in 0..n {
            total += w[i].conj() * r[[i, j]] * w[j];
        }
    }
    total.re
}

/// `aᴴ w` — Hermitian inner product of two complex vectors.
fn hermitian_inner(a: &Array1<Complex64>, w: &Array1<Complex64>) -> Complex64 {
    a.iter()
        .zip(w.iter())
        .map(|(a_i, w_i)| a_i.conj() * *w_i)
        .sum()
}

/// Deterministic pseudo-random phase in `[0, 2π)` (SplitMix64), so the
/// synthetic aperture needs no external RNG dependency.
fn deterministic_phase(seed: u64, index: u64) -> f64 {
    let mut state = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    state = state.wrapping_mul(6_364_136_223_846_793_005);
    state ^= state >> 29;
    let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
    core::f64::consts::TAU * unit
}
