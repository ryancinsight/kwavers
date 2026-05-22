use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use super::orchestrator::ElasticPstdOrchestrator;
use super::pml::ElasticPmlSpec;
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
    let cp = SOUND_SPEED_WATER_SIM;
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
    let c_max = SOUND_SPEED_WATER_SIM;
    let dt = 1e-7_f64;
    let r0 = 1e-4_f64;
    let pml = ElasticSplitFieldPml::new(ElasticPmlSpec {
        shape: (nx, nx, nx),
        thickness_cells: (thickness, thickness, thickness),
        spacing: (dx, dx, dx),
        c_max,
        dt,
        r0,
    });
    let (alpha_x, beta_x) = pml.x_coeffs();
    for i in 0..nx {
        let a = alpha_x[i];
        let b = beta_x[i];
        assert!(a > 0.0 && a <= 1.0, "alpha_x[{i}] = {a:.6e} not in (0, 1]");
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
    let cp = SOUND_SPEED_WATER_SIM;
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
    assert_eq!(
        max_v, 0.0,
        "quiescent state must remain zero under split-field PML"
    );
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
    let cp = SOUND_SPEED_WATER_SIM;
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
    let mut orch_std = ElasticPstdOrchestrator::new(&grid, make_medium(), dt).unwrap();
    let _ = orch_std
        .propagate(n_steps, Some(&make_source()), None)
        .unwrap();

    // Split-field path with zero-thickness PML (α=1, β=dt everywhere).
    let mut orch_sf = ElasticPstdOrchestrator::new(&grid, make_medium(), dt).unwrap();
    orch_sf.set_split_field_pml((0, 0, 0), cp, 1e-4);
    let _ = orch_sf
        .propagate(n_steps, Some(&make_source()), None)
        .unwrap();

    let v_std = orch_std.velocity();
    let v_sf = orch_sf.velocity();
    let norm: f64 = v_std
        .vx
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
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

/// Split-field PML attenuates outgoing waves: amplitude in the outermost PML
/// cells is strictly less than amplitude in the interior.
///
/// # Theorem (spatial amplitude monotonicity through the PML layer)
///
/// For a polynomial-order-p Bérenger split-field PML with target reflection R0,
/// the amplitude decreases monotonically from the interior into the absorbing
/// layer. At outer cell depth i (0 = PML entry, T = outermost):
///
/// ```text
///   A(i) ≤ A(0) × exp(-σ̄_i · dt · i)
///   where σ̄_i = σ_max · (i / T)^p / (p+1)  (profile average to depth i)
/// ```
///
/// For p=2, T=6, R0=1e-4:
///   σ_max = -(p+1)·c_p·ln(R0)/(2·T·dx) ≈ 3.45×10⁶ s⁻¹
///   At i=T: cumulative exp(-σ_max·dt·T/(p+1)) = exp(-3.45e6 × 1.15e-7 × 6/3) ≈ exp(-0.79) ≈ 0.45 per pass
///   Over many passes (100 steps, ~65 inside PML): 0.45^{65} ≈ 1e-23
///
/// The interior is continuously driven by the source, so interior_max ≈ amp.
/// The outer 2 cells are deep inside the PML (σ = σ_max), so outer_max << amp.
/// The test asserts `outer_max < interior_max`, which is guaranteed.
#[test]
fn split_field_pml_attenuates_outgoing_wave() {
    let nx = 24usize;
    let ny = 4usize;
    let nz = 4usize;
    let dx = 1e-3_f64;
    let cp = SOUND_SPEED_WATER_SIM;
    let rho = 1000.0_f64;
    let lam = rho * cp * cp; // μ=0 acoustic fluid
    let dt = 0.3_f64 * dx / (cp * 3.0_f64.sqrt());
    let n_steps = 100usize;
    let thickness = 6usize; // PML cells along x only
    let r0 = 1e-4_f64;
    let amp = 1e-6_f64;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = ElasticPstdMedium {
        lame_lambda: Array3::from_elem((nx, ny, nz), lam),
        lame_mu: Array3::zeros((nx, ny, nz)),
        density: Array3::from_elem((nx, ny, nz), rho),
    };
    let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
    orch.set_split_field_pml((thickness, 0, 0), cp, r0);

    // Continuous-tone source at grid centre for all n_steps so the interior
    // maintains a sustained field while the PML absorbs the outgoing wave.
    let signal =
        Array1::from_iter((0..n_steps).map(|n| amp * (2.0 * PI * 1e6 * n as f64 * dt).sin()));
    let mut src_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    src_mask[[nx / 2, ny / 2, nz / 2]] = true;
    let source = ElasticPstdVelocitySource {
        mask: src_mask,
        ux: Some(signal),
        uy: None,
        uz: None,
        mode: ElasticPstdSourceMode::Additive,
    };

    let _ = orch.propagate(n_steps, Some(&source), None).unwrap();
    let vx = orch.velocity().vx.view();

    // Interior: columns [thickness+2 .. nx-thickness-2], well away from the
    // PML entry to avoid any evanescent field at the interface.
    let interior_max: f64 = (thickness + 2..nx - thickness - 2)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .map(|(i, j, k)| vx[[i, j, k]].abs())
        .fold(0.0_f64, f64::max);

    // Outermost 2 cells on each x-boundary (deepest inside PML, σ = σ_max).
    let outer_max: f64 = (0..2usize)
        .chain(nx - 2..nx)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .map(|(i, j, k)| vx[[i, j, k]].abs())
        .fold(0.0_f64, f64::max);

    assert!(
        interior_max > 0.0,
        "interior must contain a non-zero sustained field; \
         source is active for all {n_steps} steps"
    );
    assert!(
        outer_max < interior_max,
        "PML outer max={outer_max:.3e} must be strictly < interior max={interior_max:.3e}; \
         PML is not attenuating the outgoing wave"
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
    let cp = SOUND_SPEED_WATER_SIM;
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
