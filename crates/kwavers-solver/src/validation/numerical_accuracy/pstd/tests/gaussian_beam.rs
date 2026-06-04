use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use crate::pstd::PSTDConfig as PstdConfig;
use crate::pstd::PSTDSolver;
use std::f64::consts::PI;

#[test]
fn test_gaussian_beam_phase_accuracy() {
    // Phase velocity test for a Gaussian beam propagating in the +x direction.
    //
    // # Mathematical Foundation
    //
    // **Theorem — Acoustic impedance initial condition.**
    // For any rightward-propagating wave p(x,t) = P(x − c₀t), the linearised
    // acoustic momentum equation ρ₀ ∂ux/∂t = −∂p/∂x integrates to
    //   ux = p / (ρ₀ · c₀)
    // This is exact for any pressure profile satisfying the 1D wave equation,
    // including the transverse envelope of a Gaussian beam under the paraxial
    // approximation (valid when w₀ ≫ λ).
    // Reference: Morse & Ingard (1968) *Theoretical Acoustics* §6.2.
    //
    // **Cross-correlation phase velocity (Treeby & Cox 2010, §3.2).**
    // For two sensor signals p₁(t) at x₁ and p₂(t) at x₂, the time-lag τ*
    // at the cross-correlation peak satisfies c_meas = (x₂−x₁)·dx / (τ*·dt).
    // Reference: Treeby & Cox (2010) J. Biomed. Opt. 15(2):021314, Table 1.
    println!("\n=== Gaussian Beam Phase Accuracy Test ===");

    let n = 64;
    let frequency = MHZ_TO_HZ;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let wavelength = c0 / frequency;
    // 16 PPW → < 1% PSTD phase error (Treeby & Cox 2010, Table 1)
    let dx = wavelength / 16.0;

    let mut config = PstdConfig::default();
    config.dt = super::CFL_NUMBER * dx / c0;

    let grid = Grid::new(n, n, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(rho0, c0, &grid);

    // Beam waist = 4λ: paraxial parameter k⊥/k = λ/(2πw₀) ≈ 0.04 → IC error < 0.2%
    let waist_radius: f64 = 4.0 * wavelength;
    let _rayleigh_range = PI * waist_radius.powi(2) / wavelength;

    let mut solver = PSTDSolver::new(
        config.clone(),
        grid.clone(),
        &medium,
        kwavers_domain::source::GridSource::default(),
    )
    .unwrap();

    // Sensor columns for cross-correlation phase velocity measurement
    let sensor_xi = n / 4;
    let sensor_xj = 3 * n / 4;
    let sensor_y = n / 2;
    let mut sensor1: Vec<f64> = Vec::with_capacity(200);
    let mut sensor2: Vec<f64> = Vec::with_capacity(200);

    // Initialize a 1D Gaussian wave packet (constant in y) centered at i_start,
    // left of both sensors.
    //
    // Key design decisions:
    //   1. Constant in y → no transverse diffraction, no energy in ±y or −x.
    //      With ux = p/Z₀ and uy = 0, the IC is a PURE rightward wave. ✓
    //   2. waist_x = λ/2 (= 8 grid cells): narrow enough to avoid aliasing with
    //      periodic images.  The grid has N=64 cells; the Gaussian drops to
    //      e^(−25) ≈ 0 at ±5λ from i_start, so images at i_start±N contribute < 1e-10.
    //      A wider waist (waist_x = 3λ) aliases strongly: at i=50 the image
    //      from i_start−N has amplitude e^(−0.11) ≈ 0.9 — completely corrupt IC.
    //   3. i_start = n/8: pulse centre is well left of sensor1 (i=n/4) so that
    //      sensor1 peaks AFTER the simulation begins, giving a clean rising edge
    //      for the cross-correlation.
    let z0 = rho0 * c0; // acoustic impedance Z₀ = ρ₀c₀
    let i_start = n / 8; // packet centre, left of sensor1 (n/4)
    let waist_x = 0.5 * wavelength; // λ/2 ≈ 8 grid cells (non-aliased)
    for i in 0..n {
        let x = (i as f64 - i_start as f64) * dx;
        let amplitude = (-x * x / waist_x.powi(2)).exp();
        for j in 0..n {
            solver.fields.p[[i, j, 0]] = amplitude;
            solver.fields.ux[[i, j, 0]] = amplitude / z0;
        }
    }

    // Propagation budget: sensor1 at n/4, sensor2 at 3n/4.
    // Steps for pulse centre to reach sensor1: (n/4 − n/8) / CFL = n/8/CFL
    // Expected lag (sensor1 → sensor2): n/2 * dx / (c0 * dt) = n/(2*CFL)
    // Run for steps_to_sensor1 + 3*expected_lag so the lag sits well within lag_max.
    let sensor_sep = (sensor_xj - sensor_xi) as f64 * dx;
    let expected_lag = (sensor_sep / (c0 * config.dt)).round() as usize;
    let steps_to_sensor1 = ((sensor_xi - i_start) as f64 * dx / (c0 * config.dt)).ceil() as usize;
    let total_steps = (steps_to_sensor1 + 3 * expected_lag).min(500);
    println!("Propagating for {} steps (≈1 Rayleigh range)", total_steps);

    for step in 0..total_steps {
        solver.step_forward().unwrap();
        sensor1.push(solver.fields.p[[sensor_xi, sensor_y, 0]]);
        sensor2.push(solver.fields.p[[sensor_xj, sensor_y, 0]]);
        if step % 50 == 0 {
            let max_p = solver
                .fields
                .p
                .iter()
                .map(|&p| p.abs())
                .fold(0.0_f64, f64::max);
            println!("Step {}: max pressure = {:.2e}", step, max_p);
        }
    }

    // ── Cross-correlation phase velocity ──────────────────────────────────────
    // C(τ) = Σₜ p₁(t)·p₂(t+τ);  c_meas = Δx_sensors / (τ*·dt)
    let lag_max = (sensor1.len() / 2).max(1);
    let (best_lag, best_corr) = (1..lag_max)
        .map(|lag| {
            let n_valid = sensor1.len().saturating_sub(lag);
            let corr: f64 = sensor1[..n_valid]
                .iter()
                .zip(&sensor2[lag..lag + n_valid])
                .map(|(&a, &b)| a * b)
                .sum();
            (lag, corr)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((1, 0.0));

    println!(
        "Best lag: {} steps, correlation peak: {:.3e}",
        best_lag, best_corr
    );
    assert!(
        best_corr > 0.0,
        "Cross-correlation non-positive — beam did not propagate from sensor1 to sensor2"
    );

    let sensor_sep = ((sensor_xj - sensor_xi) as f64) * dx;
    let c_meas = sensor_sep / (best_lag as f64 * config.dt);
    let rel_err = (c_meas - c0).abs() / c0;
    println!(
        "c_meas = {:.1} m/s,  c₀ = {:.1} m/s,  error = {:.2}%",
        c_meas,
        c0,
        100.0 * rel_err
    );

    // PSTD at 16 PPW: < 1% dispersion error (Treeby & Cox 2010, Table 1).
    // Allow 5% to account for cross-correlation quantisation and finite propagation.
    assert!(
        rel_err < 0.05,
        "Phase velocity error {:.2}% exceeds 5% (c_meas={:.1}, c₀={:.1})",
        100.0 * rel_err,
        c_meas,
        c0
    );

    // ── Pulse arrival check (secondary) ──────────────────────────────────────
    // The IC is a 1-D Gaussian pulse (constant in y) so there is no transverse
    // profile to check for confinement.  Instead confirm the pulse actually
    // reached sensor2 with measurable amplitude, verifying the propagation loop
    // ran long enough and the field was not attenuated to noise.
    let peak_at_sensor2: f64 = (0..n)
        .map(|j| solver.fields.p[[sensor_xj, j, 0]].abs())
        .fold(0.0_f64, f64::max);
    println!("Peak pressure at sensor2 column: {:.2e}", peak_at_sensor2);
    assert!(
        peak_at_sensor2 > 1e-6,
        "Pulse did not reach sensor2 with measurable amplitude (peak={:.2e})",
        peak_at_sensor2
    );
}
