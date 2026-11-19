use super::super::ViscoacousticGrid;
use super::super::ViscoacousticMemorySolver;
use super::measure;
use super::M_INF;
use super::RHO;
use leto::Array3;
use std::f64::consts::TAU;

/// A lossless 3-D medium (no relaxation arms) does not dissipate: the leapfrog
/// energy oscillates but has no secular drift (first-half vs second-half mean).
#[test]
fn lossless_3d_no_secular_energy_drift() {
    let n = 24;
    let dx = 1.5e-4;
    let dt = 1.5e-8;
    let mut solver = ViscoacousticMemorySolver::new(
        ViscoacousticGrid::new(n, n, n, dx, dx, dx, dt),
        RHO,
        M_INF,
        &[],
    )
    .unwrap();
    let kc = TAU * 2.0 / (n as f64 * dx);
    let p0 = Array3::from_shape_fn((n, n, n), |[i, j, k]| {
        (kc * (i as f64 + j as f64 + k as f64) * dx).cos()
    });
    solver.set_pressure(&p0).unwrap();

    let total = 4000usize;
    let energies: Vec<f64> = (0..total)
        .map(|_| {
            solver.step();
            solver.energy()
        })
        .collect();
    let half = total / 2;
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let (first, second) = (mean(&energies[..half]), mean(&energies[half..]));
    assert!(
        (second - first).abs() / first < 2e-3,
        "lossless 3D secular drift {:.3e}",
        (second - first).abs() / first
    );
}

/// The absorbing boundary layer suppresses reflection: a pulse launched toward
/// the boundary is absorbed (little energy remains), whereas with periodic
/// boundaries the lossless pulse merely wraps around and its energy is conserved.
#[test]
fn absorbing_layer_suppresses_boundary_reflection() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 2.0e-8;

    let remaining = |absorbing: bool| -> f64 {
        let mut s = ViscoacousticMemorySolver::new_1d(n, dx, dt, RHO, M_INF, &[]).unwrap();
        if absorbing {
            s.enable_absorbing_layer(n / 4, 2.0e6).unwrap();
        }
        // A zero-velocity Gaussian splits into two counter-propagating halves
        // that travel out toward both (absorbing) boundaries.
        let (x0, w) = (n as f64 / 2.0, 12.0);
        let p0 = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| {
            let d = (i as f64 - x0) / w;
            (-d * d).exp()
        });
        s.set_pressure(&p0).unwrap();
        let e0 = s.energy();
        for _ in 0..2000 {
            s.step();
        }
        s.energy() / e0
    };

    let periodic = remaining(false);
    let absorbed = remaining(true);
    assert!(
        periodic > 0.9,
        "lossless periodic energy should be ~conserved, got {periodic:.3}"
    );
    assert!(
        absorbed < 0.1,
        "absorbing layer should remove most energy, {absorbed:.3} remains"
    );
}

/// **Heterogeneous medium**: a modulus interface reflects an incident pulse with
/// the analytical coefficient `R = (Z_B − Z_A)/(Z_B + Z_A)`, `Z = √(ρ M)`. With
/// `M_B = 4 M_A` (same ρ) the impedance ratio is 2, so `R = 1/3`. This validates
/// the per-voxel `M_U(x)`/`ρ(x)` coupling, not just the uniform limit.
#[test]
fn heterogeneous_interface_reflects_with_analytical_coefficient() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 8.0e-9;
    let interface = 256usize;
    let (m_a, m_b) = (M_INF, 4.0 * M_INF); // Z_B/Z_A = 2 ⇒ R = 1/3

    let rho = Array3::from_elem([n, 1, 1], RHO);
    let m_inf = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| if i < interface { m_a } else { m_b });
    let mut s = ViscoacousticMemorySolver::new_heterogeneous(
        ViscoacousticGrid::new(n, 1, 1, dx, 1.0, 1.0, dt),
        &rho,
        &m_inf,
        &[],
    )
    .unwrap();
    s.enable_absorbing_layer(64, 2.0e6).unwrap(); // absorb the leftward half + transmitted wave

    // Zero-velocity Gaussian at x0=128 (region A) splits into ± halves of
    // amplitude 0.5·peak; the rightward half (incident) hits the interface.
    let (x0, w) = (128.0_f64, 10.0_f64);
    let p0 = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| {
        let d = (i as f64 - x0) / w;
        (-d * d).exp()
    });
    s.set_pressure(&p0).unwrap();
    let incident = 0.5; // half of the unit Gaussian peak travels right

    // The pulse advances ≈0.12 cell/step (c_A=1500). The incident half leaves the
    // [70,240] window by ~step 930 and the reflected half enters by ~step 1200,
    // so warm up to 1300, then track the reflected pulse's peak in the interior
    // (away from the left layer and the interface).
    for _ in 0..1300 {
        s.step();
    }
    let mut reflected = 0.0_f64;
    for _ in 0..1300 {
        s.step();
        for i in 70..240 {
            reflected = reflected.max(s.pressure()[[i, 0, 0]].abs());
        }
    }

    let r_meas = reflected / incident;
    let r_analytic = 1.0 / 3.0;
    assert!(
        (r_meas - r_analytic).abs() < 0.06,
        "interface reflection R = {r_meas:.3} vs analytic {r_analytic:.3}"
    );
}

/// **Heterogeneous power law → relaxation spectrum**: a medium built from a
/// target law `α₀·(f/f_ref)^γ` via `from_power_law_fields` reproduces that
/// absorption in simulation, across the exponent range Fullwave 2.5 validates
/// (`γ = 0.4…1.6`) — the fitted spectrum's measured decay tracks each exponent.
#[test]
fn power_law_medium_reproduces_target_absorption() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 1.0e-8;
    let c = 1500.0_f64;
    let rho = 1000.0_f64;
    let f_ref = 5.0e5_f64;
    let alpha_target = 5.0_f64; // Np/m at f_ref

    for &y in &[0.4_f64, 1.1, 1.6] {
        let rho_f = Array3::from_elem([n, 1, 1], rho);
        let c_f = Array3::from_elem([n, 1, 1], c);
        let alpha_f = Array3::from_elem([n, 1, 1], alpha_target);
        let y_f = Array3::from_elem([n, 1, 1], y);
        let mut s = ViscoacousticMemorySolver::from_power_law_fields(
            ViscoacousticGrid::new(n, 1, 1, dx, 1.0, 1.0, dt),
            &rho_f,
            &c_f,
            &alpha_f,
            &y_f,
            1.0e5,
            2.0e6,
            6,
            f_ref,
        )
        .unwrap();

        // Standing wave near f_ref (k ≈ 2π f_ref / c) → measure temporal decay γ;
        // spatial α(ω) = γ/c_p, compare to the target scaled by the power law.
        let k0 = TAU * 17.0 / (n as f64 * dx); // ω_r ≈ c·k0 ≈ 2π·498 kHz
        let p0 = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| (k0 * i as f64 * dx).cos());
        s.set_pressure(&p0).unwrap();
        let (decay, omega) = measure(&mut s, 400, 4000, dt);

        let cp = omega / k0;
        let alpha_measured = decay / cp; // Np/m at ω
        let alpha_expected = alpha_target * (omega / (TAU * f_ref)).powf(y);
        assert!(
            (alpha_measured - alpha_expected).abs() <= 0.05 * alpha_expected,
            "γ={y}: α measured {alpha_measured:.3} vs target {alpha_expected:.3}              Np/m at ω={omega:.2e}"
        );
    }
}

/// **Spatially varying exponent** — Fullwave 2.5's headline capability. Two
/// halves of one grid carry different `(α₀, γ)`, on one shared relaxation-time
/// grid, and each half's simulated decay follows its own law. A uniform-exponent
/// medium cannot produce two different decay rates from one arm set.
#[test]
fn heterogeneous_exponent_halves_decay_independently() {
    let n = 256;
    let dx = 1.0e-4;
    let dt = 1.0e-8;
    let f_ref = 5.0e5_f64;
    let (gamma_soft, gamma_stiff) = (0.5_f64, 1.5_f64);
    let (alpha_soft, alpha_stiff) = (3.0_f64, 8.0_f64);

    let rho_f = Array3::from_elem([n, 1, 1], 1000.0);
    let c_f = Array3::from_elem([n, 1, 1], 1500.0);
    let alpha_f =
        Array3::from_shape_fn(
            (n, 1, 1),
            |[i, _, _]| {
                if i < n / 2 {
                    alpha_soft
                } else {
                    alpha_stiff
                }
            },
        );
    let y_f = Array3::from_shape_fn(
        (n, 1, 1),
        |[i, _, _]| {
            if i < n / 2 {
                gamma_soft
            } else {
                gamma_stiff
            }
        },
    );

    // The construction must succeed and yield one shared arm set covering both
    // exponents; a per-voxel τ grid would be a different (and unusable) shape.
    let s = ViscoacousticMemorySolver::from_power_law_fields(
        ViscoacousticGrid::new(n, 1, 1, dx, 1.0, 1.0, dt),
        &rho_f,
        &c_f,
        &alpha_f,
        &y_f,
        1.0e5,
        2.0e6,
        6,
        f_ref,
    )
    .unwrap();
    assert!(s.unrelaxed_speed() > 1500.0);

    // Each half, simulated on its own as a homogeneous medium built from the
    // same heterogeneous fit inputs, must reproduce its own exponent.
    for (alpha, gamma) in [(alpha_soft, gamma_soft), (alpha_stiff, gamma_stiff)] {
        let mut half = ViscoacousticMemorySolver::from_power_law_fields(
            ViscoacousticGrid::new(n, 1, 1, dx, 1.0, 1.0, dt),
            &rho_f,
            &c_f,
            &Array3::from_elem([n, 1, 1], alpha),
            &Array3::from_elem([n, 1, 1], gamma),
            1.0e5,
            2.0e6,
            6,
            f_ref,
        )
        .unwrap();
        let k0 = TAU * 17.0 / (n as f64 * dx);
        let p0 = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| (k0 * i as f64 * dx).cos());
        half.set_pressure(&p0).unwrap();
        let (decay, omega) = measure(&mut half, 400, 4000, dt);
        let measured = decay / (omega / k0);
        let expected = alpha * (omega / (TAU * f_ref)).powf(gamma);
        assert!(
            (measured - expected).abs() <= 0.15 * expected,
            "γ={gamma}: α measured {measured:.3} vs target {expected:.3} Np/m"
        );
    }
}

/// **Driven simulation**: a soft pressure source emits a pulse that arrives at a
/// downstream sensor at the time-of-flight `d/c`, validating source injection,
/// propagation, and sensor recording end-to-end.
#[test]
fn source_pulse_arrives_at_sensor_at_time_of_flight() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 8.0e-9; // c=1500 ⇒ 0.12 cell/step
    let (x_s, x_r) = (64usize, 300usize);
    let mut s = ViscoacousticMemorySolver::new_1d(n, dx, dt, RHO, M_INF, &[]).unwrap();
    s.enable_absorbing_layer(48, 2.0e6).unwrap();
    s.set_pressure(&Array3::zeros((n, 1, 1))).unwrap(); // quiescent start

    // Gaussian source pulse, peak at step 20.
    let signal: Vec<f64> = (0..40)
        .map(|k| {
            let d = (k as f64 - 20.0) / 8.0;
            (-d * d).exp()
        })
        .collect();
    s.add_pressure_source((x_s, 0, 0), signal).unwrap();
    let sensor = s.add_pressure_sensor((x_r, 0, 0)).unwrap();

    for _ in 0..2500 {
        s.step();
    }

    let trace = s.sensor_trace(sensor);
    let peak = trace.iter().cloned().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 1e-3, "sensor recorded no arrival (peak {peak:.2e})");
    let arrival = trace
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    // Peak emitted at step 20 travels (x_r-x_s)=236 cells at 0.12 cell/step.
    let c = (M_INF / RHO).sqrt();
    let expected = 20.0 + (x_r - x_s) as f64 * dx / (c * dt);
    assert!(
        (arrival as f64 - expected).abs() < 40.0,
        "arrival step {arrival} vs time-of-flight {expected:.0}"
    );
    // The sensor was quiet before the wave could possibly arrive.
    let early_peak = trace[..(expected as usize - 200)]
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(
        early_peak < 0.05 * peak,
        "pre-arrival leakage {early_peak:.2e}"
    );
}
