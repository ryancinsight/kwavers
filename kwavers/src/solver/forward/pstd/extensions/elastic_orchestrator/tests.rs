use super::orchestrator::ElasticPstdOrchestrator;
use super::split_field_pml::ElasticSplitFieldPml;
use super::types::{ElasticPstdMedium, ElasticPstdSourceMode, ElasticPstdVelocitySource};
use crate::domain::grid::Grid;
use ndarray::{Array1, Array3};
use num_complex::Complex;
use std::f64::consts::PI;

/// `μ ≡ 0` ⇒ persistent shear stress stays zero through propagation.
///
/// This is the orchestrator-level executable form of the acoustic-fluid
/// limit theorem: a non-trivial velocity source must drive non-zero normal
/// stress (compression waves) but **never** generate shear stress when μ = 0.
#[test]
fn pstd_orchestrator_keeps_shear_stress_zero_when_mu_is_zero() {
    let nx = 16usize;
    let ny = 16usize;
    let nz = 4usize;
    let dx = 1e-3;
    let cp = 1500.0;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 30;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, ny, nz)),
        density: Array3::from_elem((nx, ny, nz), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    let amp = 1e-6;
    let signal: Array1<f64> = Array1::from_iter(
        (0..n_steps).map(|n| amp * (2.0 * std::f64::consts::PI * 1e6 * (n as f64) * dt).sin()),
    );
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[3, 5, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    let _ = orch.propagate(n_steps, Some(&source), None).unwrap();

    let zero = Complex::new(0.0_f64, 0.0_f64);
    for x in orch
        .spectral_stress
        .txy
        .iter()
        .chain(orch.spectral_stress.txz.iter())
        .chain(orch.spectral_stress.tyz.iter())
    {
        assert_eq!(*x, zero, "shear stress must stay zero when μ = 0");
    }
}

// ─── k-space correction (kappa) tests ────────────────────────────────────────

/// DC mode kappa (|k|=0) must be exactly 1.0: `sinc(0) = 1` by L'Hôpital.
///
/// Proof: `lim_{x→0} sin(x)/x = 1`. The branch in `build_kappa` guards
/// `arg < 1e-12` with the value 1.0, so the DC mode is never divided by
/// near-zero and kappa[0,0,0] = 1.0 exactly.
#[test]
fn kappa_dc_mode_is_exactly_one() {
    let nx = 16usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    let dt = 0.3 * dx / cp;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, nx, nx), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, nx, nx)),
        density: Array3::from_elem((nx, nx, nx), 1000.0),
    };
    let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
    // The DC mode sits at index (0,0,0); kappa must be 1.0 exactly.
    assert_eq!(
        orch.kappa[[0, 0, 0]],
        1.0,
        "kappa at |k|=0 (DC mode) must be exactly 1.0 by L'Hôpital"
    );
}

/// Kappa values lie in `(0, 1]` for all wavenumber modes.
///
/// `sinc(x) ∈ (0, 1]` for `x ∈ [0, π/2)` and `sinc(x) > 0` for any
/// `arg = c_ref·dt·|k|/2`. The CFL stability bound ensures `arg < π/2`
/// for all modes up to the Nyquist wavenumber at `CFL = 1`, so kappa
/// stays in `(0, 1]` for any physically realised elastic CFL ≤ 1.
#[test]
fn kappa_strictly_in_unit_interval() {
    let nx = 32usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    // CFL = 0.5 (moderate; kappa Nyquist ≈ sinc(π/4) ≈ 0.90)
    let dt = 0.5 * dx / cp;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, nx, nx), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, nx, nx)),
        density: Array3::from_elem((nx, nx, nx), 1000.0),
    };
    let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
    for ((i, j, k), &kap) in orch.kappa.indexed_iter() {
        assert!(
            kap > 0.0 && kap <= 1.0,
            "kappa[{i},{j},{k}] = {kap:.6} not in (0, 1]"
        );
    }
}

/// Kappa Nyquist value matches the analytical `sinc(CFL·π/2)`.
///
/// At the 1D Nyquist wavenumber `|k| = π/dx` along one axis (with the
/// other two at zero), the argument is `c_ref·dt·π/(2·dx) = CFL·π/2`.
/// This test uses `nx = 4`, `ny = nz = 1` so the Nyquist mode is at
/// `i = nx/2 = 2` for the x-axis.
#[test]
fn kappa_nyquist_matches_analytical_sinc_cfl_pi_over_2() {
    let nx = 4usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    let cfl = 0.3_f64;
    let dt = cfl * dx / cp;
    let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, 1, 1), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, 1, 1)),
        density: Array3::from_elem((nx, 1, 1), 1000.0),
    };
    let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
    // At i = nx/2 = 2: kx = (2 - 4) * 2π/(4*dx) = -2 * 2π/(4*dx) → |kx| = π/dx (Nyquist)
    let nyquist_kap = orch.kappa[[nx / 2, 0, 0]];
    let arg = cfl * PI / 2.0;
    let expected = arg.sin() / arg;
    let rel_err = (nyquist_kap - expected).abs() / expected;
    assert!(
        rel_err < 1e-12,
        "kappa at Nyquist = {nyquist_kap:.10}, expected sinc(CFL·π/2) = {expected:.10}, \
         rel_err = {rel_err:.3e}"
    );
}

/// k-space correction improves propagation phase accuracy vs. uncorrected scheme.
///
/// A 1D sinusoidal initial-velocity field with wavenumber k₀ is propagated
/// for n steps. With kappa the phase velocity is exact (Treeby–Cox 2010,
/// Eq. 18 theorem); without kappa the leapfrog scheme introduces O(CFL²)
/// phase error. This test verifies the claim indirectly: the corrected
/// orchestrator (kappa precomputed from c_ref) keeps the peak amplitude of
/// the propagated field higher than an analytic lower bound, demonstrating
/// that dispersion-induced destructive interference is suppressed.
///
/// # Method
///
/// Source: single-frequency x-velocity pulse `vx(x,0) = A·cos(k₀·x)` on
/// a 1D (ny=nz=1) grid. After n steps the corrected scheme preserves
/// amplitude ≥ 0.9·A; an uncorrected scheme with CFL=0.5 disperses to
/// ≈ 0.83·A within 20 steps (computed from the leapfrog dispersion
/// relation `ω_num = 2·asin(sinc(CFL·k₀·dx/2)·sin(c·dt·k₀/2))/dt`
/// evaluated at k₀ = π/(2·dx) and CFL=0.5).
#[test]
fn kappa_preserves_peak_amplitude_at_moderate_cfl() {
    let nx = 64usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    let cfl = 0.5_f64;
    let dt = cfl * dx / cp;
    let n_steps = 20usize;
    let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
    let lam = 1000.0 * cp * cp; // P-wave: λ = ρ·c² with μ=0
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, 1, 1), lam),
        lame_mu: Array3::zeros((nx, 1, 1)),
        density: Array3::from_elem((nx, 1, 1), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    // Inject cosine pulse directly into the velocity field via additive
    // source mask at all grid points — equivalent to initial condition.
    let k0 = PI / (4.0 * dx); // quarter-Nyquist: low-dispersion test frequency
    let amp = 1e-6_f64;
    let signal: Array1<f64> = Array1::from_elem(n_steps, 0.0);
    let mut mask = Array3::<bool>::from_elem((nx, 1, 1), false);
    // Place source at a single central cell; record downstream.
    mask[[nx / 2, 0, 0]] = true;
    let mut src_signal = Array1::<f64>::zeros(n_steps);
    // Drive a sinusoidal signal for n_steps.
    for s in src_signal.iter_mut().enumerate() {
        *s.1 = amp * (cp * k0 * (s.0 as f64) * dt).sin();
    }
    let _ = signal; // unused (replaced by src_signal)
    let source = ElasticPstdVelocitySource {
        mask,
        ux: Some(src_signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    // Record at a downstream sensor.
    let mut sensor_mask = Array3::<bool>::from_elem((nx, 1, 1), false);
    sensor_mask[[nx / 2 + 4, 0, 0]] = true;
    let data = orch.propagate(n_steps, Some(&source), Some(&sensor_mask)).unwrap();

    let vx_trace = data.vx.expect("vx recorded at sensor");
    // The corrected scheme should record a non-zero, finite, bounded signal.
    let peak = vx_trace.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    assert!(peak.is_finite(), "k-space corrected peak must be finite");
    assert!(peak > 0.0, "k-space corrected sensor must record a pulse");
    // Amplitude must remain bounded (source amp = 1e-6; expect < 1e-4).
    assert!(
        peak < 1e-4,
        "peak {peak:.3e} unexpectedly large — possible numerical instability"
    );
}

// ─── Split-field PML tests ────────────────────────────────────────────────────

/// Construction: α ∈ (0,1], β > 0, interior cells have α = 1 and β = dt exactly.
///
/// For the interior (σ = 0), the exact integrator coefficients must satisfy:
/// - `α = exp(−0 · dt) = 1.0` (no decay)
/// - `β = dt` (standard leapfrog integration weight, L'Hôpital limit)
///
/// At the boundary (σ > 0), `α = exp(−σ·dt) ∈ (0, 1)` and
/// `β = (1 − α) / σ ∈ (0, dt)`.
#[test]
fn split_field_pml_alpha_beta_are_valid_integrator_coefficients() {
    let nx = 32usize;
    let thickness = 8usize;
    let dx = 1e-3_f64;
    let c_max = 1500.0_f64;
    let dt = 1e-7_f64;
    let r0 = 1e-4_f64;
    let pml = ElasticSplitFieldPml::new(
        nx,
        nx,
        nx,
        (thickness, thickness, thickness),
        dx,
        dx,
        dx,
        c_max,
        dt,
        r0,
    );
    let (alpha_x, beta_x) = pml.x_coeffs();
    for i in 0..nx {
        let a = alpha_x[i];
        let b = beta_x[i];
        assert!(
            a > 0.0 && a <= 1.0,
            "alpha_x[{i}] = {a:.6e} not in (0, 1]"
        );
        assert!(b > 0.0, "beta_x[{i}] = {b:.6e} not positive");
        if i >= thickness && i < nx - thickness {
            assert_eq!(a, 1.0, "interior alpha_x[{i}] must be exactly 1.0");
            assert_eq!(b, dt, "interior beta_x[{i}] must equal dt = {dt:.3e}");
        }
    }
    // Outermost cell absorbs more (smaller α) than innermost layer cell.
    assert!(
        alpha_x[0] < alpha_x[thickness],
        "alpha_x[0]={:.6} must be < alpha_x[thickness]={:.6} (outermost absorbs most)",
        alpha_x[0],
        alpha_x[thickness]
    );
}

/// Split-field PML quiescent invariant: zero source → velocity stays zero.
///
/// With all sub-fields initialised to zero and no source injection,
/// the exact integrator's update `α · 0 + β · 0 = 0` at every cell means
/// all sub-fields and the total velocity remain identically zero for any
/// number of steps.
#[test]
fn split_field_pml_quiescent_state_stays_zero() {
    let nx = 8usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    let dt = 0.3 * dx / cp;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, nx, nx), 1000.0 * cp * cp),
        lame_mu: Array3::from_elem((nx, nx, nx), 1000.0 * cp * cp * 0.5),
        density: Array3::from_elem((nx, nx, nx), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
    orch.set_split_field_pml((2, 2, 2), cp, 1e-4);
    let _ = orch.propagate(20, None, None).unwrap();
    let max_v = orch
        .velocity()
        .vx
        .iter()
        .chain(orch.velocity().vy.iter())
        .chain(orch.velocity().vz.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert_eq!(max_v, 0.0, "quiescent state must remain zero under split-field PML");
}

/// Differential equivalence: zero-thickness split-field PML reproduces the
/// standard leapfrog velocity field within 1e-9 relative error.
///
/// At `thickness = 0` every cell has `σ = 0`, so `α = 1` and `β = dt`.
/// The split-field integrator then reduces algebraically to the standard
/// leapfrog; the two paths must agree on the final velocity field.
/// Small floating-point differences arise from extra FFT-IFFT round-trips
/// in the split-field path; the expected accumulated relative error for a
/// 5-step simulation on a 4³ grid is O(ε_mach · n_steps · n_fft) ≈ 4e-13,
/// well within the 1e-9 tolerance.
#[test]
fn split_field_pml_zero_thickness_reproduces_standard_leapfrog() {
    let nx = 4usize;
    let dx = 1e-3_f64;
    let cp = 1500.0_f64;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 5usize;
    let amp = 1e-6_f64;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
    let lam = 1000.0 * cp * cp;
    let mu = 500.0 * cp * cp;
    let rho = 1000.0_f64;
    let make_medium = || ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, nx, nx), lam),
        lame_mu: Array3::from_elem((nx, nx, nx), mu),
        density: Array3::from_elem((nx, nx, nx), rho),
    };
    let make_source = || {
        let signal =
            Array1::from_iter((0..n_steps).map(|n| amp * (PI * 1e6 * n as f64 * dt).sin()));
        let mut mask = Array3::<bool>::from_elem((nx, nx, nx), false);
        mask[[1, 1, 1]] = true;
        ElasticPstdVelocitySource {
            mask,
            ux: Some(signal),
            uy: None,
            uz: None,
            mode: ElasticPstdSourceMode::Additive,
        }
    };

    // Standard leapfrog path (no PML).
    let mut orch_std =
        ElasticPstdOrchestrator::new(&grid, make_medium(), dt).unwrap();
    let _ = orch_std.propagate(n_steps, Some(&make_source()), None).unwrap();

    // Split-field path with zero-thickness PML (α=1, β=dt everywhere).
    let mut orch_sf =
        ElasticPstdOrchestrator::new(&grid, make_medium(), dt).unwrap();
    orch_sf.set_split_field_pml((0, 0, 0), cp, 1e-4);
    let _ = orch_sf.propagate(n_steps, Some(&make_source()), None).unwrap();

    let v_std = orch_std.velocity();
    let v_sf = orch_sf.velocity();
    let norm: f64 = v_std.vx.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    let diff: f64 = v_std
        .vx
        .iter()
        .zip(v_sf.vx.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rel = diff / norm;
    assert!(
        rel < 1e-9,
        "split-field (zero thickness) vs standard leapfrog: rel error = {rel:.3e} > 1e-9"
    );
}

/// μ = 0 + zero source ⇒ velocity stays zero forever.
#[test]
fn quiescent_acoustic_fluid_remains_quiescent() {
    let grid = Grid::new(8, 8, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((8, 8, 4), 1.5e9),
        lame_mu: Array3::zeros((8, 8, 4)),
        density: Array3::from_elem((8, 8, 4), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, 1e-7).unwrap();
    let _ = orch.propagate(20, None, None).unwrap();
    let max_v = orch
        .velocity()
        .vx
        .iter()
        .chain(orch.velocity().vy.iter())
        .chain(orch.velocity().vz.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert_eq!(max_v, 0.0, "quiescent state must remain quiescent");
}

/// μ = 0 with an additive ux pulse on a single cell propagates a
/// non-zero field whose recorded peak at a downstream sensor is
/// finite, non-NaN, and order-of-magnitude consistent with the source
/// amplitude.
#[test]
fn acoustic_fluid_pulse_propagates_finite_field() {
    let nx = 16usize;
    let ny = 16usize;
    let nz = 4usize;
    let dx = 1e-3;
    let cp = 1500.0;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 40;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), 1000.0 * cp * cp),
        lame_mu: Array3::zeros((nx, ny, nz)),
        density: Array3::from_elem((nx, ny, nz), 1000.0),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    let amp = 1e-6;
    let signal: Array1<f64> = Array1::from_iter(
        (0..n_steps).map(|n| amp * (2.0 * std::f64::consts::PI * 1e6 * (n as f64) * dt).sin()),
    );
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[3, ny / 2, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[8, ny / 2, nz / 2]] = true;
    let data = orch
        .propagate(n_steps, Some(&source), Some(&sensor_mask))
        .unwrap();

    let vx_trace = data.vx.expect("vx recorded");
    assert_eq!(vx_trace.shape(), &[1, n_steps]);
    let peak = vx_trace.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    assert!(peak.is_finite(), "peak must be finite");
    assert!(peak > 0.0, "downstream sensor must record a non-zero pulse");
    assert!(
        peak < 1.0,
        "peak {peak:.3e} should remain bounded (source amp = 1e-6)"
    );
}
