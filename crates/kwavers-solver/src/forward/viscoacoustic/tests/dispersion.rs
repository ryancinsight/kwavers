use super::super::ViscoacousticGrid;
use super::super::ViscoacousticMemorySolver;
use super::dispersion_oracle;
use super::measure;
use super::ARMS;
use super::M_INF;
use super::RHO;
use kwavers_math::fft::Complex64;
use leto::Array3;
use std::f64::consts::TAU;

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
        let p0 = Array3::from_shape_fn((n, 1, 1), |[i, _, _]| (k0 * i as f64 * dx).cos());
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
        let mut solver = ViscoacousticMemorySolver::new(
            ViscoacousticGrid::new(n, n, 1, dx, dx, dx, dt),
            RHO,
            M_INF,
            &ARMS,
        )
        .unwrap();
        let kc = TAU * m / (n as f64 * dx);
        let kmag = kc * 2.0_f64.sqrt();
        let p0 = Array3::from_shape_fn((n, n, 1), |[i, j, _]| {
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
    let mut solver = ViscoacousticMemorySolver::new(
        ViscoacousticGrid::new(n, n, n, dx, dx, dx, dt),
        RHO,
        M_INF,
        &ARMS,
    )
    .unwrap();
    let kc = TAU * m / (n as f64 * dx);
    let kmag = kc * 3.0_f64.sqrt();
    let p0 = Array3::from_shape_fn((n, n, n), |[i, j, k]| {
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

/// **Discrete dispersion relation.** Von Neumann analysis of the actual update
/// — leapfrog velocity, exponentially integrated `σ`, trapezoidal relaxation
/// term in the pressure update — gives, for `p ∝ z^n e^{ikx}` with
/// `z = e^{-iωΔt}`,
///
/// ```text
///   ρ·(4/Δt²)·sin²(ωΔt/2) = k²·M_eff(z)
///   M_eff(z) = M_U − Σₗ (ΔMₗ/2)(1−e^{-Δt/τₗ})(1+z)/(z − e^{-Δt/τₗ})
/// ```
///
/// against the continuum `ρω² = k²M(ω)`, `M(ω) = M_U − Σₗ ΔMₗ/(1−iωτₗ)` (the
/// `e^{-iωt}` form of the module's `M(ω)`). This pins the scheme's accuracy
/// analytically, independent of any simulation.
///
/// **What sets the error.** The left-hand side is the leapfrog's
/// `ω²·sinc²(ωΔt/2)`, and it dominates: the error is governed by how well `Δt`
/// resolves the **wave**, `ω·Δt`, and is essentially independent of how well it
/// resolves the fastest relaxation, `Δt/τ_min`. The test asserts that
/// independence directly, by scaling every `τ` up a hundredfold and requiring
/// the error not to move.
///
/// That correction matters because the opposite was asserted here previously.
/// The `Δt/τ_min` story was plausible — the trapezoidal relaxation term *is* the
/// only second-order piece besides the leapfrog, since the `σ` integration is
/// exact and the spatial operator spectral — but it was never tested against a
/// `τ` sweep, and it is wrong. Replacing the trapezoid with the closed-form
/// exact integral of `σ/τ` across the step changes the error by under 7 %
/// relative (2.60 % → 2.78 % at `Δt = 1.5e-8`), which is why that avenue was
/// abandoned rather than implemented.
///
/// It also exonerated the scheme when a simulated attenuation measurement
/// disagreed with the prescribed law by 10 % (KW-SOL-072); the artefact was a
/// tapered analysis gate.
#[test]
fn discrete_dispersion_matches_continuum() {
    /// `α` of the continuum relation at real `ω`, for an arbitrary arm set.
    fn alpha_continuum(arms: &[(f64, f64)], omega: f64) -> f64 {
        let mu: f64 = M_INF + arms.iter().map(|&(dm, _)| dm).sum::<f64>();
        let mut m = Complex64::new(mu, 0.0);
        for &(dm, tau) in arms {
            m -= dm / (1.0 - Complex64::new(0.0, omega * tau));
        }
        (Complex64::new(RHO * omega * omega, 0.0) / m)
            .sqrt()
            .im
            .abs()
    }

    /// `α` of the discrete relation at real `ω` and step `dt`.
    fn alpha_discrete(arms: &[(f64, f64)], omega: f64, dt: f64) -> f64 {
        let mu: f64 = M_INF + arms.iter().map(|&(dm, _)| dm).sum::<f64>();
        let z = (Complex64::new(0.0, -omega * dt)).exp();
        let mut m = Complex64::new(mu, 0.0);
        for &(dm, tau) in arms {
            let e = (-dt / tau).exp();
            m -= 0.5 * dm * (1.0 - e) * (1.0 + z) / (z - e);
        }
        let lhs = RHO * 4.0 / (dt * dt) * (0.5 * omega * dt).sin().powi(2);
        (Complex64::new(lhs, 0.0) / m).sqrt().im.abs()
    }

    const F_MAX: f64 = 5.0e6;
    let frequencies = [0.5e6_f64, 1.0e6, 2.0e6, 3.5e6, F_MAX];
    let worst = |arms: &[(f64, f64)], dt: f64| {
        frequencies.iter().fold(0.0_f64, |acc, &f| {
            let omega = TAU * f;
            let exact = alpha_continuum(arms, omega);
            assert!(exact > 0.0, "continuum α must be positive at {f:e} Hz");
            acc.max((alpha_discrete(arms, omega, dt) - exact).abs() / exact)
        })
    };

    // ── The error is a wave-resolution error, not a relaxation-resolution one.
    let dt = 1.5e-8_f64;
    let baseline = worst(&ARMS, dt);
    let slow_arms: Vec<(f64, f64)> = ARMS.iter().map(|&(dm, t)| (dm, t * 100.0)).collect();
    let slowed = worst(&slow_arms, dt);
    assert!(
        (slowed - baseline).abs() < 0.1 * baseline,
        "a 100x reduction in Δt/τ_min moved the error {baseline:.5} → {slowed:.5}; \
         the relaxation term is not supposed to be the driver"
    );

    // ── Points per period at the highest frequency is the governing quantity.
    // The error follows `C·(ω_max·Δt)²`; assert that model directly, since it is
    // the falsifiable claim, and read the practical guidance off it rather than
    // hard-coding step counts.
    let omega_max = TAU * F_MAX;
    let coefficients: Vec<f64> = [0.4_f64, 0.2, 0.1, 0.05]
        .into_iter()
        .map(|omega_dt| worst(&ARMS, omega_dt / omega_max) / (omega_dt * omega_dt))
        .collect();
    let mean = coefficients.iter().sum::<f64>() / coefficients.len() as f64;
    for (i, c) in coefficients.iter().enumerate() {
        assert!(
            (c - mean).abs() < 0.05 * mean,
            "quadratic model does not hold: coefficient {i} is {c:.4} against              mean {mean:.4}"
        );
    }
    assert!(
        (0.10..0.14).contains(&mean),
        "error coefficient C = {mean:.4}, expected ~0.117"
    );

    // With `C ≈ 0.117` and `N = 2π/(ω_max·Δt)` points per period at the highest
    // frequency, the error is `C·(2π/N)²`: 25 points per period holds it under
    // 1 %, and 80 under 0.1 %. Both carry margin, so this is guidance a caller
    // can use rather than a boundary case.
    let step_for = |points_per_period: f64| 1.0 / (F_MAX * points_per_period);
    let coarse = worst(&ARMS, step_for(25.0));
    let fine = worst(&ARMS, step_for(80.0));
    assert!(coarse < 1.0e-2, "25 points per period gave {coarse:.5}");
    assert!(fine < 1.0e-3, "80 points per period gave {fine:.5}");

    // ── Second-order convergence: a 4x smaller step cuts the error by at least
    // 4x (the bound, not the ideal 16x, so this is not a fit to one machine's
    // rounding).
    assert!(
        worst(&ARMS, dt / 4.0) * 4.0 < baseline,
        "refinement did not converge from {baseline:.3e}"
    );
}
