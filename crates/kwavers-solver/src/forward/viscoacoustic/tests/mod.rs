//! Validation of the memory-variable viscoacoustic solver against the analytic
//! complex dispersion relation `ρω² = M(ω)|k|²` of the generalized-Maxwell
//! medium, in 1-D, 2-D, and 3-D.

use super::ViscoacousticMemorySolver;
use kwavers_math::fft::Complex64;
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

mod dispersion;
mod propagation;
mod validation;
