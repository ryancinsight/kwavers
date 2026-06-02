//! Aanonsen-1984 Fubini-absolute harmonic-ratio test for the Westervelt
//! discrete recurrence, using a clean 1-D harness.
//!
//! # Theorem (Fubini analytical, Hamilton & Blackstock 1998 §4.3.2;
//! Aanonsen et al. 1984 Eq. 6)
//!
//! For a plane wave propagating in a lossless medium with weak nonlinearity:
//! ```text
//!   |P_n(z)| / |P_1(z)| = J_n(n Γ) / (n · J_1(Γ)),    Γ = z / z_shock
//!   z_shock = ρ_0 c_0³ / (β · ω · P_0)
//! ```
//! At `Γ = 0.5`: `J_2(1) / (2·J_1(0.5)) ≈ 0.1149 / (2 · 0.2423) ≈ 0.2371`.
//!
//! # Why 1-D and not 3-D
//!
//! The 3-D `forward_with_schedule` API uses point sources with `1/r`
//! geometric spreading, so local Γ varies along the propagation path and
//! Fubini's constant-amplitude plane-wave assumption is violated. This test
//! uses a **clean 1-D Westervelt FDTD** whose update rule **algebraically
//! matches** the 3-D recurrence in `update_cells`:
//!
//! ```text
//!   p[n+1, i] = sponge[i] · (2 p[n, i] − p[n−1, i]
//!                            + (c·dt)² · ∇²p[n, i]
//!                            + q · ∂²(p²)/∂t²|^n)
//!   q = β · dt² / (ρ · c²)
//!   ∂²(p²)/∂t² ≈ 2 p[n] · d²p/dt² + 2 (dp/dt)²   (product rule)
//! ```
//!
//! The 1-D Laplacian uses the 3-point stencil
//! `(p[i+1] − 2 p[i] + p[i−1]) / dx²`; everything else is bit-identical to
//! the 3-D `update_cells` algebra.
//!
//! # Tier
//!
//! `#[ignore]`'d (~2 s runtime). Run on demand with
//! `cargo test --lib --package kwavers -- --ignored fubini_absolute`.

use super::bessel::{bessel_j1, bessel_j2};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
#[ignore = "Tier 2: Literature validation (Aanonsen 1984 Fubini-absolute, 1-D harness), ~2s runtime"]
fn westervelt_recurrence_fubini_absolute_at_gamma_half_matches_aanonsen_1984() {
    // Physical parameters chosen so Γ = 0.5 lies inside the 1-D domain.
    // Resolution: 30 pts/wavelength at fundamental, 15 pts/wavelength at 2nd
    // harmonic — sufficient to resolve the 2nd harmonic without significant
    // FDTD numerical dispersion bias.
    let nx: usize = 1024;
    let dx: f64 = 5.0e-5; // 0.05 mm → 51.2 mm domain
    let c: f64 = SOUND_SPEED_WATER_SIM;
    let rho: f64 = DENSITY_WATER_NOMINAL;
    let beta: f64 = 10.0;
    let frequency_hz: f64 = MHZ_TO_HZ;
    let omega = std::f64::consts::TAU * frequency_hz;
    let p0: f64 = MPA_TO_PA; // 1 MPa
    let z_shock = rho * c.powi(3) / (beta * omega * p0);
    let target_gamma = 0.5_f64;
    let target_distance_m = target_gamma * z_shock;
    let source_index: usize = 4;
    let receiver_index = source_index + (target_distance_m / dx).round() as usize;
    assert!(
        receiver_index < nx - 16,
        "receiver index {receiver_index} would land inside the far-boundary sponge",
    );

    // CFL-stable timestep matching the 3-D solver's CFL convention.
    let cfl: f64 = 0.5;
    let dt = cfl * dx / c;
    let dt2 = dt * dt;
    let inv_dt = 1.0 / dt;
    let inv_dt2 = 1.0 / dt2;
    let inv_dx2 = 1.0 / (dx * dx);
    let q = beta * dt2 / (rho * c.powi(2));

    // Long burst: chosen so the envelope peak (`t = burst_duration / 2`)
    // is *after* the wave reaches the receiver.
    let cycles: f64 = 80.0;
    let burst_duration = cycles / frequency_hz;
    let travel_time = (receiver_index as f64 - source_index as f64) * dx / c;
    assert!(
        burst_duration / 2.0 > travel_time + 4.0 / frequency_hz,
        "burst envelope peak must be at least 4 cycles after the wavefront \
         arrives at the receiver; got burst peak = {} s, travel time = {} s",
        burst_duration / 2.0,
        travel_time,
    );
    let period_steps = (1.0 / (frequency_hz * dt)).round() as usize;
    let steps = (burst_duration / dt).ceil() as usize + 4 * period_steps;

    // Far-boundary sponge: smooth quadratic ramp over the last 32 cells so
    // the wave is absorbed before reflecting off the boundary.
    let sponge_layer = 32_usize;
    let mut sponge = vec![1.0_f64; nx];
    for i in 0..sponge_layer {
        let edge = i;
        let ratio = (sponge_layer - edge) as f64 / sponge_layer as f64;
        sponge[nx - 1 - i] = (1.0 - 0.18 * ratio * ratio).max(0.0);
    }
    sponge[0] = 0.0;
    sponge[nx - 1] = 0.0;

    let mut p_older = vec![0.0_f64; nx];
    let mut p_prev = vec![0.0_f64; nx];
    let mut p_curr = vec![0.0_f64; nx];
    let mut p_next = vec![0.0_f64; nx];
    let mut traces = Vec::with_capacity(steps);

    for step in 0..steps {
        let t_curr = step as f64 * dt;

        // Interior cell update — algebraically identical to the 3-D
        // `update_cells` (1-D Laplacian instead of 7-point 3-D).
        for i in 1..nx - 1 {
            let center = p_curr[i];
            let prev = p_prev[i];
            let older = p_older[i];
            let lap = (p_curr[i + 1] - 2.0 * center + p_curr[i - 1]) * inv_dx2;
            let dp_dt = (center - prev) * inv_dt;
            let nl = if step >= 2 {
                let d2p_dt2 = (center - 2.0 * prev + older) * inv_dt2;
                2.0_f64.mul_add(center * d2p_dt2, 2.0 * dp_dt * dp_dt)
            } else {
                2.0 * dp_dt * dp_dt
            };
            let raw = 2.0_f64.mul_add(center, -prev) + (c * dt).powi(2) * lap + q * nl;
            p_next[i] = sponge[i] * raw;
        }

        // Hard sinusoidal source with `sin²` envelope ramp.
        let envelope = if t_curr < burst_duration {
            (std::f64::consts::PI * t_curr / burst_duration)
                .sin()
                .powi(2)
        } else {
            0.0
        };
        p_next[source_index] = p0 * (omega * t_curr).sin() * envelope;

        traces.push(p_next[receiver_index]);

        std::mem::swap(&mut p_older, &mut p_prev);
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }

    // Pick a steady-state window centered on the burst-envelope peak.
    let peak_step = ((burst_duration / 2.0) / dt).round() as usize;
    let half_window_periods = 4_usize;
    let window_half_steps = half_window_periods * period_steps;
    assert!(
        peak_step > window_half_steps + 4,
        "burst envelope peak too close to t=0 — increase `cycles`",
    );
    let window_start = peak_step - window_half_steps;
    let window_end = peak_step + window_half_steps;
    assert!(
        window_end < traces.len(),
        "burst window extends past simulation: peak_step = {peak_step}, traces.len() = {}",
        traces.len(),
    );
    let window = &traces[window_start..window_end];

    // Discrete sine/cosine projection at a known frequency.
    let project = |trace: &[f64], freq_hz: f64| -> f64 {
        let omega_proj = std::f64::consts::TAU * freq_hz;
        let n_samples = trace.len() as f64;
        let mut cos_sum = 0.0_f64;
        let mut sin_sum = 0.0_f64;
        for (i, &value) in trace.iter().enumerate() {
            let t = (window_start + i) as f64 * dt;
            cos_sum += value * (omega_proj * t).cos();
            sin_sum += value * (omega_proj * t).sin();
        }
        2.0 * (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n_samples
    };

    let amp_fundamental = project(window, frequency_hz);
    let amp_second_harmonic = project(window, 2.0 * frequency_hz);
    assert!(amp_fundamental.is_finite() && amp_second_harmonic.is_finite());
    assert!(
        amp_fundamental > 0.0,
        "fundamental amplitude must be positive; got |P_1| = {amp_fundamental}",
    );

    let ratio = amp_second_harmonic / amp_fundamental;

    // Empirical Γ from the **observed** `|P_1|` at the receiver.
    let gamma_empirical = beta * omega * target_distance_m * amp_fundamental / (rho * c.powi(3));
    assert!(
        gamma_empirical > 0.05 && gamma_empirical < 1.5,
        "empirical Γ must be in the pre/near-shock regime: got Γ = {gamma_empirical:.4} \
         (|P_1| = {amp_fundamental:.4e} Pa, distance = {target_distance_m:.4e} m)",
    );

    // Fubini analytical at the empirical Γ: `|P_2|/|P_1| = J_2(2Γ) / (2 · J_1(Γ))`.
    let fubini_at_empirical_gamma =
        bessel_j2(2.0 * gamma_empirical) / (2.0 * bessel_j1(gamma_empirical));
    let relative_error = (ratio - fubini_at_empirical_gamma).abs() / fubini_at_empirical_gamma;
    assert!(
        relative_error < 0.15,
        "Aanonsen-1984 Fubini empirical-Γ regression: \
         measured |P_2|/|P_1| = {ratio:.4}; Fubini at empirical Γ = {gamma_empirical:.4} → \
         analytical {fubini_at_empirical_gamma:.4}; relative error = {:.1}% (tolerance: 15%). \
         |P_1| = {amp_fundamental:.4e} Pa, |P_2| = {amp_second_harmonic:.4e} Pa. \
         Geometric distance = {target_distance_m:.4e} m. \
         A relative error > 15% suggests a coefficient error in `q = β·dt²/(ρ·c²)` or in the \
         product-rule `∂²(p²)/∂t² = 2p·d²p/dt² + 2(dp/dt)²` expression.",
        relative_error * 100.0,
    );
}
