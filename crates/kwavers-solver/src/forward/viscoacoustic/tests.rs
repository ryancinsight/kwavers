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

/// Construction validation and the `GeneralizedMaxwellModel` convenience path.
#[test]
fn construction_validates_and_accepts_model() {
    use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;

    assert!(ViscoacousticMemorySolver::new_1d(0, 1e-4, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(ViscoacousticMemorySolver::new(8, 8, 0, 1e-4, 1e-4, 1e-4, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(
        ViscoacousticMemorySolver::new_1d(64, -1.0, 1e-8, RHO, M_INF, &[]).is_err()
    );
    assert!(
        ViscoacousticMemorySolver::new_1d(64, 1e-4, 1e-8, RHO, M_INF, &[(-1.0, 1e-7)]).is_err()
    );

    let model =
        GeneralizedMaxwellModel::power_law(M_INF, 2.0e8, 1.0e5, 2.0e6, 6, 1.3, RHO).unwrap();
    let solver =
        ViscoacousticMemorySolver::from_generalized_maxwell(&model, 16, 16, 16, 1e-4, 1e-4, 1e-4, 1e-8)
            .expect("model-backed solver");
    assert!(solver.unrelaxed_speed() > (M_INF / RHO).sqrt());
}
