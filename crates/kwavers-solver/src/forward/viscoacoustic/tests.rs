//! Validation of the memory-variable viscoacoustic solver against the analytic
//! complex dispersion relation `ρω² = M(ω)|k|²` of the generalized-Maxwell
//! medium, in 1-D, 2-D, and 3-D.

use super::ViscoacousticMemorySolver;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::TAU;

const RHO: f64 = 1000.0;
const M_INF: f64 = 2.25e9; // ρ·1500² ⇒ relaxed speed 1500 m/s
                           // Two relaxation arms with loss times bracketing the ~MHz test band.
const ARMS: [(f64, f64); 2] = [(1.5e8, 3.2e-7), (8.0e7, 8.0e-8)];

fn m_u() -> f64 {
    M_INF + ARMS.iter().map(|&(dm, _)| dm).sum::<f64>()
}

/// Complex modulus `M(ω) = M_∞ + Σ ΔMₗ iωτₗ/(1+iωτₗ)`.
fn complex_modulus(omega: Complex64) -> Complex64 {
    let mut m = Complex64::new(M_INF, 0.0);
    for &(dm, tau) in &ARMS {
        let iwt = Complex64::new(0.0, 1.0) * omega * tau;
        m += dm * iwt / (1.0 + iwt);
    }
    m
}

/// Solve `ρω² = M(ω)k²` for the complex angular frequency at wavenumber `k`
/// (temporal-decay branch) by Newton iteration from the unrelaxed guess.
/// Returns `(Re ω, |Im ω|)` — oscillation frequency and amplitude decay rate.
fn dispersion_oracle(k: f64) -> (f64, f64) {
    let k2 = k * k;
    let mut omega = Complex64::new(k * (m_u() / RHO).sqrt(), 0.0);
    for _ in 0..60 {
        let f = RHO * omega * omega - complex_modulus(omega) * k2;
        let mut mprime = Complex64::new(0.0, 0.0);
        for &(dm, tau) in &ARMS {
            let denom = 1.0 + Complex64::new(0.0, 1.0) * omega * tau;
            mprime += dm * Complex64::new(0.0, tau) / (denom * denom);
        }
        let fprime = 2.0 * RHO * omega - mprime * k2;
        omega -= f / fprime;
    }
    (omega.re.abs(), omega.im.abs())
}

/// Run a standing-wave initial condition and measure the temporal decay rate
/// (from energy) and oscillation frequency (from the `p` trace at the origin).
fn measure(
    solver: &mut ViscoacousticMemorySolver,
    warmup: usize,
    span: usize,
    dt: f64,
) -> (f64, f64) {
    for _ in 0..warmup {
        solver.step();
    }
    let e_start = solver.energy();
    let mut trace = Vec::with_capacity(span);
    for _ in 0..span {
        solver.step();
        trace.push(solver.pressure()[[0, 0, 0]]);
    }
    let e_end = solver.energy();
    let dt_span = span as f64 * dt;
    let decay = -(e_end / e_start).ln() / (2.0 * dt_span);
    let crossings = trace.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
    let omega = TAU * crossings / 2.0 / dt_span;
    (decay, omega)
}

/// **Broadband 1-D**: across several wavenumbers, the measured decay and
/// frequency match the exact complex dispersion — the memory variables
/// reproduce the whole relaxation spectrum, not a single-frequency fit.
#[test]
fn decay_matches_dispersion_1d() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 1.5e-8;
    for &m in &[6.0, 12.0, 24.0] {
        let mut solver = ViscoacousticMemorySolver::new_1d(n, dx, dt, RHO, M_INF, &ARMS).unwrap();
        let k0 = TAU * m / (n as f64 * dx);
        let p0 = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| (k0 * i as f64 * dx).cos());
        solver.set_pressure(&p0).unwrap();

        let (decay_meas, omega_meas) = measure(&mut solver, 400, 4000, dt);
        let (omega_r, decay) = dispersion_oracle(k0);
        assert!(
            (decay_meas - decay).abs() <= 0.05 * decay,
            "1D mode {m}: decay {decay_meas:.4e} vs oracle {decay:.4e}"
        );
        assert!(
            (omega_meas - omega_r).abs() <= 0.02 * omega_r,
            "1D mode {m}: ω {omega_meas:.4e} vs oracle {omega_r:.4e}"
        );
    }
}

/// **2-D**: a diagonal standing wave `cos(kₓx + k_yy)` exercises both spectral
/// axes and the divergence; its decay matches the isotropic dispersion at
/// `|k| = √(kₓ²+k_y²)`.
#[test]
fn decay_matches_dispersion_2d_diagonal() {
    let n = 96;
    let dx = 1.0e-4;
    let dt = 1.2e-8;
    for &m in &[4.0, 6.0] {
        let mut solver =
            ViscoacousticMemorySolver::new(n, n, 1, dx, dx, dx, dt, RHO, M_INF, &ARMS).unwrap();
        let kc = TAU * m / (n as f64 * dx);
        let kmag = kc * 2.0_f64.sqrt();
        let p0 = Array3::from_shape_fn((n, n, 1), |(i, j, _)| {
            (kc * (i as f64 + j as f64) * dx).cos()
        });
        solver.set_pressure(&p0).unwrap();

        let (decay_meas, omega_meas) = measure(&mut solver, 400, 3000, dt);
        let (omega_r, decay) = dispersion_oracle(kmag);
        assert!(
            (decay_meas - decay).abs() <= 0.06 * decay,
            "2D mode {m}: decay {decay_meas:.4e} vs oracle {decay:.4e}"
        );
        assert!(
            (omega_meas - omega_r).abs() <= 0.03 * omega_r,
            "2D mode {m}: ω {omega_meas:.4e} vs oracle {omega_r:.4e}"
        );
    }
}

/// **3-D**: a body-diagonal standing wave `cos(kₓ(x+y+z))` decays at the
/// isotropic dispersion rate for `|k| = √3 kₓ`.
#[test]
fn decay_matches_dispersion_3d_diagonal() {
    let n = 32;
    let dx = 1.5e-4;
    let dt = 1.5e-8;
    let m = 2.0;
    let mut solver =
        ViscoacousticMemorySolver::new(n, n, n, dx, dx, dx, dt, RHO, M_INF, &ARMS).unwrap();
    let kc = TAU * m / (n as f64 * dx);
    let kmag = kc * 3.0_f64.sqrt();
    let p0 = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
        (kc * (i as f64 + j as f64 + k as f64) * dx).cos()
    });
    solver.set_pressure(&p0).unwrap();

    let (decay_meas, omega_meas) = measure(&mut solver, 300, 2200, dt);
    let (omega_r, decay) = dispersion_oracle(kmag);
    assert!(
        (decay_meas - decay).abs() <= 0.10 * decay,
        "3D: decay {decay_meas:.4e} vs oracle {decay:.4e}"
    );
    assert!(
        (omega_meas - omega_r).abs() <= 0.04 * omega_r,
        "3D: ω {omega_meas:.4e} vs oracle {omega_r:.4e}"
    );
}

/// A lossless 3-D medium (no relaxation arms) does not dissipate: the leapfrog
/// energy oscillates but has no secular drift (first-half vs second-half mean).
#[test]
fn lossless_3d_no_secular_energy_drift() {
    let n = 24;
    let dx = 1.5e-4;
    let dt = 1.5e-8;
    let mut solver =
        ViscoacousticMemorySolver::new(n, n, n, dx, dx, dx, dt, RHO, M_INF, &[]).unwrap();
    let kc = TAU * 2.0 / (n as f64 * dx);
    let p0 = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
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
            s.enable_absorbing_layer(n / 4, 2.0e6);
        }
        // A zero-velocity Gaussian splits into two counter-propagating halves
        // that travel out toward both (absorbing) boundaries.
        let (x0, w) = (n as f64 / 2.0, 12.0);
        let p0 = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| {
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

    let rho = Array3::from_elem((n, 1, 1), RHO);
    let m_inf = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| if i < interface { m_a } else { m_b });
    let mut s =
        ViscoacousticMemorySolver::new_heterogeneous(n, 1, 1, dx, 1.0, 1.0, dt, &rho, &m_inf, &[])
            .unwrap();
    s.enable_absorbing_layer(64, 2.0e6); // absorb the leftward half + transmitted wave

    // Zero-velocity Gaussian at x0=128 (region A) splits into ± halves of
    // amplitude 0.5·peak; the rightward half (incident) hits the interface.
    let (x0, w) = (128.0_f64, 10.0_f64);
    let p0 = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| {
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

/// **CT power-law → relaxation spectrum**: a medium built from a target power-law
/// absorption `α(ω_ref)` via `from_power_law_fields` reproduces that absorption in
/// simulation — the fitted relaxation spectrum's measured decay matches the target.
#[test]
fn power_law_medium_reproduces_target_absorption() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 1.0e-8;
    let c = 1500.0_f64;
    let rho = 1000.0_f64;
    let y = 1.1_f64;
    let f_ref = 5.0e5_f64;
    let alpha_target = 5.0_f64; // Np/m at f_ref

    let rho_f = Array3::from_elem((n, 1, 1), rho);
    let c_f = Array3::from_elem((n, 1, 1), c);
    let alpha_f = Array3::from_elem((n, 1, 1), alpha_target);
    let mut s = ViscoacousticMemorySolver::from_power_law_fields(
        n, 1, 1, dx, 1.0, 1.0, dt, &rho_f, &c_f, &alpha_f, y, 1.0e5, 2.0e6, 6, f_ref,
    )
    .unwrap();

    // Standing wave near f_ref (k ≈ 2π f_ref / c) → measure temporal decay γ;
    // spatial α(ω) = γ/c_p, compare to the target scaled by the power law.
    let k0 = TAU * 17.0 / (n as f64 * dx); // ω_r ≈ c·k0 ≈ 2π·498 kHz
    let p0 = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| (k0 * i as f64 * dx).cos());
    s.set_pressure(&p0).unwrap();
    let (decay, omega) = measure(&mut s, 400, 4000, dt);

    let cp = omega / k0;
    let alpha_measured = decay / cp; // Np/m at ω
    let alpha_expected = alpha_target * (omega / (TAU * f_ref)).powf(y);
    assert!(
        (alpha_measured - alpha_expected).abs() <= 0.15 * alpha_expected,
        "α measured {alpha_measured:.3} vs target {alpha_expected:.3} Np/m at ω={omega:.2e}"
    );
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
    s.enable_absorbing_layer(48, 2.0e6);
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

/// Construction validation and the `GeneralizedMaxwellModel` convenience path.
#[test]
fn construction_validates_and_accepts_model() {
    use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;

    assert!(ViscoacousticMemorySolver::new_1d(0, 1e-4, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(
        ViscoacousticMemorySolver::new(8, 8, 0, 1e-4, 1e-4, 1e-4, 1e-8, RHO, M_INF, &[]).is_err()
    );
    assert!(ViscoacousticMemorySolver::new_1d(64, -1.0, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(
        ViscoacousticMemorySolver::new_1d(64, 1e-4, 1e-8, RHO, M_INF, &[(-1.0, 1e-7)]).is_err()
    );

    let model =
        GeneralizedMaxwellModel::power_law(M_INF, 2.0e8, 1.0e5, 2.0e6, 6, 1.3, RHO).unwrap();
    let solver = ViscoacousticMemorySolver::from_generalized_maxwell(
        &model, 16, 16, 16, 1e-4, 1e-4, 1e-4, 1e-8,
    )
    .expect("model-backed solver");
    assert!(solver.unrelaxed_speed() > (M_INF / RHO).sqrt());
}
