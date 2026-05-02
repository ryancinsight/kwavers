use super::super::orchestrator::PSTDSolver;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::forward::pstd::config::{AntiAliasingConfig, BoundaryConfig, PSTDConfig};

const CPML_REFERENCE_STEP2: f64 = 5.344_360e-1;
const CPML_REFERENCE_STEP3: f64 = 1.127_856e-1;
const CPML_REFERENCE_TOL: f64 = 1e-4;

/// Verify that additive pressure source injection produces correct sign pattern.
///
/// Reference: k-Wave Python numpy diagnostic (`diag_source_injection_numpy.py`) confirms
/// that for a point source at [N/2, N/2, N/2] with N=16:
/// - p[N/2, N/2, N/2] (source point) > 0
/// - p[0, N/2, N/2] (off-source) < 0
#[test]
fn test_source_injection_sign_matches_kwave() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = ndarray::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;

    let mut p_signal = ndarray::Array2::<f64>::zeros((1, 2));
    p_signal[[0, 1]] = 1.0;

    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };

    let config = PSTDConfig {
        dt,
        nt: 2,
        boundary: BoundaryConfig::None,
        smooth_sources: false,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

    solver.step_forward().unwrap();
    solver.step_forward().unwrap();

    let p_src = solver.fields.p[[src, src, src]];
    let p_off = solver.fields.p[[0, src, src]];

    assert!(
        p_src > 0.1,
        "p at source [{src},{src},{src}] = {p_src:.6e}, expected ~0.53 Pa (positive)"
    );

    assert!(
        p_off < 0.0,
        "p at [0,{src},{src}] = {p_off:.6e}, expected NEGATIVE (k-Wave: -4.89e-4 Pa). \
         Positive result indicates 3D FFT axis ordering mismatch vs numpy.fftn."
    );
}

/// Verify that free wave propagation does not amplify the injected field.
///
/// Root cause of the 2026-03-27 amplitude bug: Nyquist frequency bin was zeroed in
/// ddx_k_shift_pos/neg operators, which removed ~18% of k-space energy from the
/// velocity/density gradient computation. This caused a 1.64x amplitude amplification
/// per free propagation step. This test guards against that regression.
///
/// Reference values from k-Wave binary (N=16, no PML, signal=[0,1,0]):
///   step 2 (injection): p[8,8,8] = 0.5344 Pa
///   step 3 (free prop): p[8,8,8] = 0.1128 Pa
#[test]
fn test_nyquist_not_zeroed_propagation_amplitude() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = ndarray::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;
    let mut p_signal = ndarray::Array2::<f64>::zeros((1, 4));
    p_signal[[0, 1]] = 1.0;

    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };
    let config = PSTDConfig {
        dt,
        nt: 4,
        boundary: BoundaryConfig::None,
        smooth_sources: false,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    solver.step_forward().unwrap();
    solver.step_forward().unwrap();
    let p_step2 = solver.fields.p[[src, src, src]];
    solver.step_forward().unwrap();
    let p_step3 = solver.fields.p[[src, src, src]];

    assert!(
        (p_step2 - CPML_REFERENCE_STEP2).abs() < CPML_REFERENCE_TOL,
        "step2 p[src] = {p_step2:.6e}, expected {CPML_REFERENCE_STEP2:.6e}"
    );

    assert!(
        (p_step3 - CPML_REFERENCE_STEP3).abs() < CPML_REFERENCE_TOL,
        "step3 p[src] = {p_step3:.6e}, expected {CPML_REFERENCE_STEP3:.6e}"
    );
}

#[test]
fn test_anti_aliasing_runs() {
    let config = PSTDConfig {
        anti_aliasing: AntiAliasingConfig {
            enabled: true,
            cutoff: 0.8,
            order: 4,
        },
        dt: 1e-8,
        nt: 10,
        ..Default::default()
    };

    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let source = GridSource::new_empty();

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

    let result = solver.step_forward();
    assert!(
        result.is_ok(),
        "Step forward failed with anti-aliasing enabled: {:?}",
        result.err()
    );
}

/// Verify propagation amplitude is correct even when CPML boundary is configured.
#[test]
fn test_propagation_amplitude_with_cpml_boundary() {
    use crate::domain::boundary::cpml::CPMLConfig;
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = ndarray::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;
    let mut p_signal = ndarray::Array2::<f64>::zeros((1, 4));
    p_signal[[0, 1]] = 1.0;

    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };

    let cpml_config = CPMLConfig::with_thickness(2);
    let config = PSTDConfig {
        dt,
        nt: 4,
        boundary: BoundaryConfig::CPML(cpml_config),
        smooth_sources: false,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    solver.step_forward().unwrap();
    solver.step_forward().unwrap();
    let p_step2 = solver.fields.p[[src, src, src]];
    solver.step_forward().unwrap();
    let p_step3 = solver.fields.p[[src, src, src]];

    assert!(
        (p_step2 - CPML_REFERENCE_STEP2).abs() < CPML_REFERENCE_TOL,
        "With CPML: step2 p[src] = {p_step2:.6e}, expected {CPML_REFERENCE_STEP2:.6e}"
    );
    assert!(
        (p_step3 - CPML_REFERENCE_STEP3).abs() < CPML_REFERENCE_TOL,
        "With CPML: step3 p[src] = {p_step3:.6e}, expected {CPML_REFERENCE_STEP3:.6e}"
    );
}

/// Theorem: source_kappa = cos(c·dt·k/2) — half-step leapfrog phase factor.
///
/// k-Wave Python kspaceFirstOrder3D.py line 302:
///   source_kappa = ifftshift(cos(c_ref * k * dt / 2))
///
/// At DC (k=0): cos(0) = 1.0.
/// At k_max (CFL=0.5): cos(π/4) ≈ 0.7071.
#[test]
fn test_source_kappa_equals_cosine() {
    use std::f64::consts::PI;

    let n = 32usize;
    let dx = 5e-4_f64;
    let c0 = 1500.0_f64;
    let cfl = 0.5_f64;
    let dt = cfl * dx / c0;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, c0, &grid);

    let source = crate::domain::source::grid_source::GridSource::new_empty();
    let config = PSTDConfig {
        dt,
        nt: 1,
        smooth_sources: false,
        ..Default::default()
    };
    let solver = PSTDSolver::new(config, grid.clone(), &medium, source).unwrap();

    let sc = &solver.source_kappa;

    let dc_val = sc[[0, 0, 0]];
    assert!(
        (dc_val - 1.0).abs() < 1e-12,
        "source_kappa[DC] must be 1.0, got {dc_val}"
    );

    let k_max = PI / dx;
    let arg = 0.5 * c0 * dt * k_max;
    let expected_cos = arg.cos();

    let hi_k_idx = n / 2;
    let hk_val = sc[[hi_k_idx, 0, 0]];
    assert!(
        (hk_val - expected_cos).abs() < 1e-10,
        "source_kappa[{hi_k_idx},0,0]={hk_val} expected cos={expected_cos}"
    );
}
