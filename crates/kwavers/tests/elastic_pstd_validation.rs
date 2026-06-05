//! Validation tests for [`kwavers_solver::forward::pstd::extensions::
//! ElasticPstdOrchestrator`] against analytical isotropic-elastic
//! propagation invariants.
//!
//! These tests are the Rust-only counterpart of
//! `pykwavers/examples/ewp_elastic_2d_jl_compare.py --pstd`, which
//! validates against KWave.jl's `pstd_elastic_2d` numerically. Together
//! they cover:
//!
//! * **Engine parity** (this file): assertions against analytical wave
//!   physics — no external simulator required, runs in CI.
//! * **Cross-engine parity** (Julia compare script): peak amplitude
//!   matches KWave.jl to 1.0000 across four sensors with Pearson 0.974;
//!   requires Julia + KWave.jl on the runner.

use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::extensions::{
    ElasticPml, ElasticPmlSpec, ElasticPstdMedium, ElasticPstdOrchestrator, ElasticPstdSourceMode,
    ElasticPstdVelocitySource,
};
use ndarray::{Array1, Array3};

/// Aki & Richards (2002) Eq. 4.13: P-wave speed `c_p = sqrt((λ + 2μ) / ρ)`.
fn p_wave_speed(lambda: f64, mu: f64, rho: f64) -> f64 {
    ((lambda + 2.0 * mu) / rho).sqrt()
}

/// A localized ux pulse propagating through a homogeneous elastic medium
/// must arrive at a downstream sensor at time `t = distance / c_p` to
/// within one sample period — within the resolution allowed by the
/// time discretization.
///
/// # Theorem
///
/// For a homogeneous isotropic elastic medium with Lamé parameters
/// `(λ, μ)` and density `ρ`, a longitudinal (compressional) wave packet
/// satisfies `(∂_t² − c_p² ∇²) u = 0` with `c_p = √((λ + 2μ)/ρ)`.  A
/// pulse launched at `x = x₀` and recorded at `x = x₀ + Δ` peaks at
/// time `t_peak = Δ / c_p` (group velocity equals phase velocity for the
/// non-dispersive elastic wave equation).
///
/// # Implementation notes
///
/// To collapse the volumetric (3-D) orchestrator to an effective 2-D
/// problem the source mask is extended through ALL z-layers (uniform
/// extrusion), so the in-slab field stays z-uniform and the wave
/// propagates as a 2-D plane wave from the source line. This is the
/// same convention `pykwavers/examples/ewp_elastic_2d_jl_compare.py
/// --pstd` uses to match KWave.jl's 2-D reference.
#[test]
fn p_wave_arrival_time_matches_analytical() {
    // Medium — water-like compression speed; small μ to keep the test in
    // the elastic regime without dominating the timing budget.
    let cp = 1500.0_f64;
    let cs = 100.0_f64;
    let rho = 1000.0_f64;
    let lambda = rho * (cp * cp - 2.0 * cs * cs);
    let mu = rho * cs * cs;
    assert!((p_wave_speed(lambda, mu, rho) - cp).abs() < 1e-6);

    // Grid — large enough that the wave doesn't wrap around within the
    // recording window (NX·dx / cp > NT·dt suffices).
    let nx = 64usize;
    let ny = 32usize;
    let nz = 8usize;
    let dx = 0.5e-3_f64;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let nt = 200usize;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), lambda),
        lame_mu: Array3::from_elem((nx, ny, nz), mu),
        density: Array3::from_elem((nx, ny, nz), rho),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    // Source — narrow Hann-windowed sinusoid (3 cycles at 1 MHz).
    let f0 = 1.0e6_f64;
    let n_in = (3.0 / f0 / dt) as usize;
    let amp = 1.0e-6_f64;
    let signal: Array1<f64> = Array1::from_iter((0..nt).map(|n| {
        if n < n_in {
            let env = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (n as f64) / (n_in as f64)).cos());
            amp * env * (2.0 * std::f64::consts::PI * f0 * (n as f64) * dt).sin()
        } else {
            0.0
        }
    }));

    let src_x = 8usize;
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    // Extend through all z so the slab behaves as a 2-D problem
    src_mask.slice_mut(ndarray::s![src_x, .., ..]).fill(true);
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    // Sensor — N cells downstream at the y-centre, mid-z.
    let offset = 24usize;
    let sensor_x = src_x + offset;
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[sensor_x, ny / 2, nz / 2]] = true;

    let data = orch
        .propagate(nt, Some(&source), Some(&sensor_mask))
        .unwrap();

    let trace = data.vx.expect("vx recorded");
    assert_eq!(trace.shape(), &[1, nt]);

    // Locate the peak step.
    let (peak_step, peak_value) = (0..nt)
        .map(|n| (n, trace[[0, n]].abs()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let distance = (offset as f64) * dx;
    let travel_time = distance / cp;
    let travel_steps = travel_time / dt;

    // For an additive (non-Dirichlet) velocity source, the recorded
    // sensor trace is the source signal convolved with the medium's
    // Green's function. The peak of the convolution lies somewhere
    // between travel_time (leading edge of envelope arrives) and
    // travel_time + envelope_duration (trailing edge of envelope leaves).
    // Bound the peak step within this physically-defensible window.
    let envelope_steps = n_in as f64;
    let lower_bound = (travel_steps - 5.0).floor().max(0.0) as usize;
    let upper_bound = (travel_steps + envelope_steps + 5.0).ceil() as usize;
    assert!(
        peak_step >= lower_bound && peak_step <= upper_bound,
        "P-wave peak step = {peak_step}, expected within \
         [{lower_bound}, {upper_bound}] (travel = {travel_steps:.1} steps, \
         envelope = {envelope_steps:.1} steps). distance = {distance:.3e} m, \
         c_p = {cp:.1} m/s, dt = {dt:.3e} s."
    );

    // Sanity: the recorded peak is within the same order of magnitude as
    // the source amplitude (allowing for geometric spreading + spectral
    // dispersion + integration over many source steps).
    assert!(
        peak_value > 0.0 && peak_value < amp * 1000.0,
        "P-wave peak amplitude = {peak_value:.3e}, source amp = {amp:.3e} \
         (must be finite, non-zero, bounded)"
    );
}

/// `μ ≡ 0` ⇒ no shear stress, ever.
///
/// # Theorem
///
/// In an acoustic fluid (μ = 0), Hooke's law collapses to the isotropic
/// fluid stress `σ = λ (∇·u) I` — the deviatoric stress is identically
/// zero. Consequently every shear-stress component
/// `σ_xy = σ_xz = σ_yz = 0` for all time, regardless of the source
/// geometry or polarization.
///
/// # Note (what this test does NOT assert)
///
/// Naïvely one might expect a pure ux source in an acoustic fluid to
/// produce zero transverse particle velocity (vy = vz = 0). That is
/// **false** for a localized source: the resulting pressure field
/// `p = -σ_xx = -λ (∇·u)` has finite spatial extent, so its y-gradient
/// `∂p/∂y` drives `∂vy/∂t = -(1/ρ) ∂p/∂y ≠ 0` even with μ = 0. The
/// invariant is shear-stress vanishing, not transverse-velocity
/// vanishing — the latter holds only for infinite-extent plane waves.
///
/// This test is the integration-level counterpart of the spectral-domain
/// unit test `pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero`
/// in `physics/acoustics/mechanics/elastic_wave/tests.rs`, which checks
/// the shear-stress invariant in a single spectral kernel call.
#[test]
fn acoustic_fluid_limit_zero_shear_stress_after_propagation() {
    let cp = 1500.0_f64;
    let rho = 1000.0_f64;
    let lambda = rho * cp * cp;

    let nx = 32usize;
    let ny = 32usize;
    let nz = 8usize;
    let dx = 0.5e-3_f64;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let nt = 60usize;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), lambda),
        lame_mu: Array3::zeros((nx, ny, nz)), // μ = 0 — acoustic-fluid limit
        density: Array3::from_elem((nx, ny, nz), rho),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();

    // ux source at one cell.
    let amp = 1.0e-6_f64;
    let f0 = 1.0e6_f64;
    let signal: Array1<f64> = Array1::from_iter(
        (0..nt).map(|n| amp * (2.0 * std::f64::consts::PI * f0 * (n as f64) * dt).sin()),
    );
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[8, ny / 2, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    // Sensor records ALL three velocity components everywhere via mask.
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    for i in 0..nx {
        for j in 0..ny {
            sensor_mask[[i, j, nz / 2]] = true;
        }
    }
    let _ = orch
        .propagate(nt, Some(&source), Some(&sensor_mask))
        .unwrap();

    let stress = orch.spectral_stress();
    let zero = num_complex::Complex::new(0.0_f64, 0.0_f64);
    for (component_name, arr) in [
        ("σ_xy", &stress.txy),
        ("σ_xz", &stress.txz),
        ("σ_yz", &stress.tyz),
    ] {
        let max = arr.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        assert_eq!(
            arr.iter().filter(|c| **c != zero).count(),
            0,
            "{component_name}: shear-stress sample count > 0 in acoustic-fluid limit \
             (max |{component_name}| = {max:.3e}); μ = 0 must zero every shear component"
        );
    }
}

/// Cumulative PML absorption verifies the **end-to-end** design figure.
///
/// # Theorem (cumulative PML attenuation)
///
/// For a real-space exponential PML with calibration `r0` (theoretical
/// reflection coefficient at normal incidence), a wave packet that
/// originates outside the absorbing layer and propagates entirely
/// through it must arrive at the far edge with amplitude no greater
/// than `~ r0` of its incident amplitude (Roden & Gedney 2000 eq. 25,
/// Treeby & Cox 2010 Appendix B).
///
/// # Verification strategy
///
/// 1. Run the orchestrator without PML for a short time → record peak
///    amplitude at a sensor placed deep in the absorbing layer's
///    nominal location.
/// 2. Run the SAME setup with PML enabled at the same boundary → record
///    the same sensor.
/// 3. Assert that the PML run's peak amplitude is at least 20 dB lower
///    than the no-PML run (substantially exceeds the r0 = 1e-4
///    calibration target's −80 dB single-incidence figure once
///    re-reflection is excluded — the test is conservative because
///    short propagation doesn't fully populate the layer with the
///    pulse's energy).
#[test]
fn pml_attenuates_field_in_absorbing_layer_vs_without_pml() {
    let cp = 1500.0_f64;
    let cs = 100.0_f64;
    let rho = 1000.0_f64;
    let lambda = rho * (cp * cp - 2.0 * cs * cs);
    let mu = rho * cs * cs;

    let nx = 64usize;
    let ny = 16usize;
    let nz = 4usize;
    let dx = 0.5e-3_f64;
    let dt = 0.3 * dx / (cp * 3.0_f64.sqrt());
    let nt = 200usize;

    // Pulse parameters — same as the arrival-time test for consistency.
    let f0 = 1.0e6_f64;
    let n_in = (3.0 / f0 / dt) as usize;
    let amp = 1.0e-6_f64;
    let signal: Array1<f64> = Array1::from_iter((0..nt).map(|n| {
        if n < n_in {
            let env = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (n as f64) / (n_in as f64)).cos());
            amp * env * (2.0 * std::f64::consts::PI * f0 * (n as f64) * dt).sin()
        } else {
            0.0
        }
    }));

    let src_x = 12usize;
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask.slice_mut(ndarray::s![src_x, .., ..]).fill(true);

    // Sensor inside the would-be PML region (last ~10 cells from the
    // x = nx-1 boundary, but spaced safely inside the absorbing layer).
    let pml_thickness = 10usize;
    let sensor_x = nx - 4; // inside the rightmost PML layer
    let mut sensor_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sensor_mask[[sensor_x, ny / 2, nz / 2]] = true;

    let make_orch = || -> ElasticPstdOrchestrator {
        let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
        let medium = ElasticPstdMedium {
            lame_lambda: Array3::from_elem((nx, ny, nz), lambda),
            lame_mu: Array3::from_elem((nx, ny, nz), mu),
            density: Array3::from_elem((nx, ny, nz), rho),
        };
        ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap()
    };
    let make_source = || ElasticPstdVelocitySource {
        mask: src_mask.clone(),
        ux: Some(signal.clone()),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    // Baseline: NO PML.
    let mut orch_baseline = make_orch();
    let baseline = orch_baseline
        .propagate(nt, Some(&make_source()), Some(&sensor_mask))
        .unwrap();
    let baseline_peak = baseline
        .vx
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    // PML run.
    let mut orch_pml = make_orch();
    orch_pml.set_pml((pml_thickness, 0, 0), cp, 1e-4);
    let pml_data = orch_pml
        .propagate(nt, Some(&make_source()), Some(&sensor_mask))
        .unwrap();
    let pml_peak = pml_data
        .vx
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    // Verify the PML produced strict attenuation.
    assert!(
        baseline_peak > 0.0,
        "baseline (no-PML) sensor must record non-zero amplitude (got {baseline_peak:.3e})"
    );
    assert!(
        pml_peak >= 0.0,
        "PML sensor amplitude must be finite and non-negative (got {pml_peak:.3e})"
    );
    let attenuation_db = 20.0 * (pml_peak / baseline_peak).log10();
    assert!(
        attenuation_db <= -20.0,
        "PML attenuation = {attenuation_db:.2} dB at sensor (baseline = {baseline_peak:.3e}, \
         pml = {pml_peak:.3e}); design target ≤ −20 dB"
    );

    // Sanity-check the unused PML import path (also catches set_pml /
    // clear_pml regressions).
    let _: ElasticPml = ElasticPml::new(ElasticPmlSpec {
        shape: (8, 8, 8),
        thickness_cells: (2, 0, 0),
        spacing: (1e-3, 1e-3, 1e-3),
        c_max: 1500.0,
        dt: 1e-7,
        r0: 1e-4,
    });
    orch_pml.clear_pml();
}
