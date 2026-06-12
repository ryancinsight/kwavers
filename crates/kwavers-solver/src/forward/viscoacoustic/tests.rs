//! Validation of the memory-variable viscoacoustic solver against the analytic
//! complex dispersion relation `ρω² = M(ω)k²` of the generalized-Maxwell medium.

use super::ViscoacousticMemorySolver;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::TAU;

const RHO: f64 = 1000.0;
const M_INF: f64 = 2.25e9; // ρ·1500² ⇒ relaxed speed 1500 m/s
// Two relaxation arms with loss times bracketing the ~0.5 MHz test band.
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

/// Solve `ρω² = M(ω)k²` for the complex angular frequency at a real wavenumber
/// `k` (temporal-decay branch) by Newton iteration from the unrelaxed guess.
/// Returns `(Re ω, |Im ω|)` — oscillation frequency and amplitude decay rate.
fn dispersion_oracle(k: f64) -> (f64, f64) {
    let k2 = k * k;
    let mut omega = Complex64::new(k * (m_u() / RHO).sqrt(), 0.0);
    for _ in 0..60 {
        // F(ω) = ρω² − M(ω)k²;  M'(ω) = Σ ΔMₗ iτₗ/(1+iωτₗ)².
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

/// A lossless medium (no relaxation arms) does not dissipate: the leapfrog
/// energy oscillates (v and p are half a step apart) but has no secular drift.
/// Window-averaging removes the bounded oscillation; the time-averaged energy is
/// conserved, so an early window and a late window agree to high precision.
#[test]
fn lossless_medium_conserves_energy() {
    let n = 256;
    let dx = 1.0e-4;
    let dt = 2.0e-8;
    let mut solver = ViscoacousticMemorySolver::new(n, dx, dt, RHO, M_INF, &[]).unwrap();

    let k0 = TAU * 8.0 / (n as f64 * dx);
    let p0 = Array1::from_shape_fn(n, |i| (k0 * i as f64 * dx).cos());
    solver.set_pressure(&p0).unwrap();

    // Collect the energy each step over a long run (tens of periods), then
    // compare the mean of the first half to the mean of the second half. The
    // half-step energy oscillation averages out over many periods, so a
    // remaining difference would be genuine secular dissipation — which a
    // spectral, leapfrog (time-reversible) integrator must not produce.
    let total = 8000usize;
    let energies: Vec<f64> = (0..total)
        .map(|_| {
            solver.step();
            solver.energy()
        })
        .collect();
    let half = total / 2;
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let first = mean(&energies[..half]);
    let second = mean(&energies[half..]);
    assert!(
        (second - first).abs() / first < 1e-3,
        "lossless secular energy drift {:.3e} (no dissipation expected)",
        (second - first).abs() / first
    );
}

/// **Broadband** validation: for several wavenumbers spanning the band, the
/// solver's measured temporal decay rate and oscillation frequency match the
/// exact complex dispersion `ρω² = M(ω)k²` — proving the memory variables
/// reproduce the full relaxation spectrum, not a single-frequency fit.
#[test]
fn temporal_decay_matches_complex_dispersion_across_band() {
    let n = 512;
    let dx = 1.0e-4;
    let dt = 1.5e-8;

    for &m in &[6.0, 12.0, 24.0] {
        let mut solver = ViscoacousticMemorySolver::new(n, dx, dt, RHO, M_INF, &ARMS).unwrap();
        let k0 = TAU * m / (n as f64 * dx);

        let p0 = Array1::from_shape_fn(n, |i| (k0 * i as f64 * dx).cos());
        solver.set_pressure(&p0).unwrap();

        // Settle past the initial transient, then measure over `span` steps.
        let (warmup, span) = (400usize, 4000usize);
        for _ in 0..warmup {
            solver.step();
        }
        let e_start = solver.energy();
        let mut p_trace = Vec::with_capacity(span);
        for _ in 0..span {
            solver.step();
            p_trace.push(solver.pressure()[0]);
        }
        let e_end = solver.energy();

        // Decay rate: E(t) ∝ exp(−2|Im ω| t) ⇒ |Im ω| = −ln(E_end/E_start)/(2 Δt_span).
        let dt_span = span as f64 * dt;
        let decay_meas = -(e_end / e_start).ln() / (2.0 * dt_span);

        // Oscillation frequency: count sign changes of the p[0] trace.
        let crossings = p_trace
            .windows(2)
            .filter(|w| w[0] * w[1] < 0.0)
            .count() as f64;
        let f_meas = crossings / 2.0 / dt_span;
        let omega_meas = TAU * f_meas;

        let (omega_r, decay) = dispersion_oracle(k0);

        assert!(
            (decay_meas - decay).abs() <= 0.05 * decay,
            "k mode {m}: decay measured {decay_meas:.4e} vs oracle {decay:.4e}"
        );
        assert!(
            (omega_meas - omega_r).abs() <= 0.02 * omega_r,
            "k mode {m}: ω measured {omega_meas:.4e} vs oracle {omega_r:.4e}"
        );
    }
}

/// Relaxation stiffening: the phase velocity rises with frequency between the
/// relaxed (`√(M_∞/ρ)`) and unrelaxed (`√(M_U/ρ)`) limits.
#[test]
fn phase_velocity_increases_with_frequency() {
    let c_inf = (M_INF / RHO).sqrt();
    let c_u = (m_u() / RHO).sqrt();
    let n = 512;
    let dx = 1.0e-4;
    let (omega_lo, _) = dispersion_oracle(TAU * 4.0 / (n as f64 * dx));
    let (omega_hi, _) = dispersion_oracle(TAU * 40.0 / (n as f64 * dx));
    let k_lo = TAU * 4.0 / (n as f64 * dx);
    let k_hi = TAU * 40.0 / (n as f64 * dx);
    let cp_lo = omega_lo / k_lo;
    let cp_hi = omega_hi / k_hi;
    assert!(cp_hi > cp_lo, "dispersion: c_p(hi)={cp_hi} ≤ c_p(lo)={cp_lo}");
    assert!(cp_lo >= c_inf - 1.0 && cp_hi <= c_u + 1.0, "speeds out of [c∞,cU]");
}

/// Construction validation and the `GeneralizedMaxwellModel` convenience path.
#[test]
fn construction_validates_and_accepts_model() {
    use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;

    assert!(ViscoacousticMemorySolver::new(0, 1e-4, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(ViscoacousticMemorySolver::new(64, -1.0, 1e-8, RHO, M_INF, &[]).is_err());
    assert!(ViscoacousticMemorySolver::new(64, 1e-4, 1e-8, RHO, M_INF, &[(-1.0, 1e-7)]).is_err());

    let model =
        GeneralizedMaxwellModel::power_law(M_INF, 2.0e8, 1.0e5, 2.0e6, 6, 1.3, RHO).unwrap();
    let solver = ViscoacousticMemorySolver::from_generalized_maxwell(&model, 128, 1e-4, 1e-8)
        .expect("model-backed solver");
    assert!(solver.unrelaxed_speed() > (M_INF / RHO).sqrt());
}
