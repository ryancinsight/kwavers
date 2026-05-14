//! Power-law absorption decay validation for the 3-D Westervelt FDTD with
//! the Treeby-Cox 2010 fractional-Laplacian operator in [`super::super::absorption`].
//!
//! # Theorem
//!
//! In a homogeneous absorbing medium with power-law attenuation
//! `α(f) = α₀·|f_MHz|^y` Np/m, the ratio of the absorbing-medium pressure
//! envelope to the lossless-medium pressure envelope at the same point
//! and time satisfies
//!
//! ```text
//!   |p_abs(r)| / |p_lossless(r)| = exp(-α(f) · r)
//! ```
//!
//! because the geometric spreading factor `1/r` is identical between
//! the two simulations and divides out exactly. Taking the logarithm of
//! the ratio versus `r` is linear with slope `-α(f)`.
//!
//! # Tier
//!
//! `#[ignore]`'d (~10 s runtime, two forward runs). Run on demand with
//! `cargo test --lib --package kwavers -- --ignored absorption_decay`.

use super::super::super::Point3;
use super::super::encoding::SourceEncoding;
use super::super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::super::types::{GridIndex, Nonlinear3dAperture};
use super::super::Nonlinear3dConfig;
use crate::clinical::therapy::theranostic_guidance::AnatomyKind;

#[test]
#[ignore = "Tier 2: Literature validation (Treeby-Cox 2010 plane-wave decay), ~10s runtime"]
fn fractional_laplacian_absorption_decay_ratio_matches_alpha_omega_y_power_law() {
    let n: usize = 96;
    let cells = n * n * n;
    let spacing_m = 1.5e-4_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let frequency_hz = 1.0e6_f64;
    let alpha0_np_per_m_at_1mhz = 5.8_f64;
    let y_exponent = 1.05_f64;
    let wavelength = c0 / frequency_hz; // 1.5 mm
    let pts_per_wavelength = wavelength / spacing_m; // 10 — adequate

    let speed = vec![c0; cells];
    let density = vec![rho0; cells];
    let beta = vec![0.0_f64; cells]; // linear regime
    let attenuation_alpha0 = vec![alpha0_np_per_m_at_1mhz; cells];
    let attenuation_y = vec![y_exponent; cells];
    let attenuation_zero = vec![0.0_f64; cells];

    let source_x = 8_usize;
    let source_idx = GridIndex {
        x: source_x,
        y: n / 2,
        z: n / 2,
    };
    // Four receivers at r = 4λ, 6λ, 8λ, 10λ → 6 mm, 9 mm, 12 mm, 15 mm.
    let r_voxels: Vec<usize> = [40, 56, 72, 84]
        .iter()
        .copied()
        .filter(|&dx| source_x + dx < n - 8)
        .collect();
    assert!(
        r_voxels.len() >= 3,
        "need at least 3 in-grid far-field receiver positions; got {} \
         (try reducing receiver distances or increasing grid size)",
        r_voxels.len(),
    );
    let receiver_indices: Vec<GridIndex> = r_voxels
        .iter()
        .map(|&dx| GridIndex {
            x: source_x + dx,
            y: n / 2,
            z: n / 2,
        })
        .collect();
    let aperture = Nonlinear3dAperture {
        sources: vec![source_idx],
        receivers: receiver_indices.clone(),
        therapy_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        receiver_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        model_name: "test_absorption_decay".to_owned(),
        focus: *receiver_indices.last().unwrap(),
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = frequency_hz;
    config.source_pressure_pa = 5.0e4; // linear regime
    config.cycles = 6.0;
    config.cfl = 0.4;
    let dt = config.cfl * spacing_m / (c0 * 3.0_f64.sqrt());
    let travel_steps_far =
        ((receiver_indices.last().unwrap().x - source_idx.x) as f64 * spacing_m / (c0 * dt)).ceil()
            as usize;
    let period_steps = (1.0 / (frequency_hz * dt)).round() as usize;
    let pulse_steps = (config.cycles * period_steps as f64).ceil() as usize;
    let steps = travel_steps_far + pulse_steps + 4 * period_steps;
    let schedule = TimeSchedule {
        dt_s: dt,
        time_steps: steps,
    };

    let n_recv = receiver_indices.len();
    let max_step_before_reflection = |recv_index: usize| -> usize {
        let r_to_boundary_voxels = n - receiver_indices[recv_index].x;
        let extra_round_trip_steps = (2 * r_to_boundary_voxels) as f64 * spacing_m / (c0 * dt);
        let pulse_arrival = (receiver_indices[recv_index].x - source_idx.x) as f64 * spacing_m
            / (c0 * dt);
        let reflection_return = pulse_arrival + extra_round_trip_steps;
        (reflection_return - period_steps as f64).floor().max(2.0) as usize
    };

    let run = |alpha_field: &[f64]| -> Vec<f64> {
        let result = forward_with_schedule(ForwardInput {
            speed: &speed,
            density: &density,
            beta: &beta,
            attenuation_np_per_m_mhz: Some(alpha_field),
            attenuation_power_law_y: Some(&attenuation_y),
            n,
            spacing_m,
            aperture: &aperture,
            config: &config,
            schedule,
            encoding: SourceEncoding { index: 0, count: 1 },
            retain_history: false,
        });
        (0..n_recv)
            .map(|recv| {
                let end_step = max_step_before_reflection(recv).min(steps);
                (0..end_step)
                    .map(|step| result.traces[step * n_recv + recv].abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    };

    let peaks_lossless = run(&attenuation_zero);
    let peaks_absorbing = run(&attenuation_alpha0);

    assert!(
        peaks_lossless.iter().all(|p| p.is_finite() && *p > 0.0),
        "lossless peaks must be positive and finite; got {peaks_lossless:?}",
    );
    assert!(
        peaks_absorbing.iter().all(|p| p.is_finite() && *p > 0.0),
        "absorbing peaks must be positive and finite; got {peaks_absorbing:?}",
    );

    let distances_m: Vec<f64> = receiver_indices
        .iter()
        .map(|idx| (idx.x - source_idx.x) as f64 * spacing_m)
        .collect();
    let log_ratio: Vec<f64> = peaks_absorbing
        .iter()
        .zip(peaks_lossless.iter())
        .map(|(pa, pl)| (pa / pl).ln())
        .collect();

    // Least-squares slope of `log(p_abs/p_lossless)` vs `r`. Slope = -α.
    let n_pts = distances_m.len() as f64;
    let mean_r = distances_m.iter().sum::<f64>() / n_pts;
    let mean_y = log_ratio.iter().sum::<f64>() / n_pts;
    let cov_ry: f64 = distances_m
        .iter()
        .zip(log_ratio.iter())
        .map(|(r, y)| (r - mean_r) * (y - mean_y))
        .sum();
    let var_r: f64 = distances_m.iter().map(|r| (r - mean_r).powi(2)).sum();
    let slope = cov_ry / var_r;
    let alpha_fit = -slope;

    let alpha_analytical = alpha0_np_per_m_at_1mhz * (frequency_hz / 1.0e6).powf(y_exponent);
    let relative_error =
        (alpha_fit - alpha_analytical).abs() / alpha_analytical.abs().max(1.0e-12);

    assert!(
        relative_error < 0.35,
        "Treeby-Cox 2010 plane-wave decay regression: \
         fitted α = {alpha_fit:.3} Np/m at {frequency_hz:.2e} Hz; analytical α = {alpha_analytical:.3} Np/m \
         (α₀ = {alpha0_np_per_m_at_1mhz} Np/m at 1 MHz, y = {y_exponent}); \
         relative error = {:.1}% (tolerance: 35%). \
         Distances = {:?} m; lossless peaks = {:?} Pa; absorbing peaks = {:?} Pa; \
         log-ratios = {:?}; grid pts/λ = {pts_per_wavelength:.1}. \
         A larger error suggests an incorrect c-power in the τ coefficient \
         `dt_tau = dt · 2·α₀_ω·c^(y+1)` or a sign error in the spectral \
         filter `|k|^y` weighting.",
        relative_error * 100.0,
        distances_m,
        peaks_lossless,
        peaks_absorbing,
        log_ratio,
    );
}
