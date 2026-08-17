use super::super::orchestrator::PSTDSolver;
use crate::forward::pstd::config::{AntiAliasingConfig, BoundaryConfig, PSTDConfig};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::{GridSource, SourceMode};

const CPML_REFERENCE_STEP2: f64 = 5.344_360e-1;
const CPML_REFERENCE_STEP3: f64 = 1.127_856e-1;
const CPML_REFERENCE_TOL: f64 = 1e-4;

/// Verify that additive pressure source injection produces correct sign pattern.
///
/// Reference: k-Wave Python numpy diagnostic (`diag_source_injection_numpy.py`) confirms
/// that for a point source at [N/2, N/2, N/2] with N=16:
/// - p[N/2, N/2, N/2] (source point) > 0
/// - p[0, N/2, N/2] (off-source) < 0
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_source_injection_sign_matches_kwave() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = leto::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;

    let mut p_signal = leto::Array2::<f64>::zeros((1, 2));
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_nyquist_not_zeroed_propagation_amplitude() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = leto::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;
    let mut p_signal = leto::Array2::<f64>::zeros((1, 4));
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
fn test_anti_aliasing_filter_attenuates_nyquist_checkerboard() {
    let config = PSTDConfig {
        anti_aliasing: AntiAliasingConfig {
            enabled: true,
            cutoff: 0.8,
            order: 4,
        },
        dt: 1e-8,
        nt: 1,
        ..Default::default()
    };

    let n = 16usize;
    let grid = Grid::new(n, n, n, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let source = GridSource::new_empty();

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                solver.fields.p[[i, j, k]] = if (i + j + k).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
            }
        }
    }

    let initial_l2 = solver.fields.p.iter().map(|v| v * v).sum::<f64>().sqrt();
    solver.apply_anti_aliasing_filter().unwrap();
    let filtered_l2 = solver.fields.p.iter().map(|v| v * v).sum::<f64>().sqrt();
    let max_abs = solver.fields.p.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));

    assert!(
        filtered_l2 < 0.01 * initial_l2,
        "Nyquist checkerboard should be strongly attenuated: initial_l2={initial_l2:.6e}, filtered_l2={filtered_l2:.6e}"
    );
    assert!(
        max_abs.is_finite(),
        "filtered pressure contains non-finite values"
    );
    assert_eq!(
        solver.time_step_index, 0,
        "direct filter application must not advance the time step"
    );
}

/// Verify propagation amplitude is correct even when CPML boundary is configured.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_propagation_amplitude_with_cpml_boundary() {
    use kwavers_boundary::cpml::CPMLConfig;
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;
    let src = n / 2;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let mut p_mask = leto::Array3::<f64>::zeros((n, n, n));
    p_mask[[src, src, src]] = 1.0;
    let mut p_signal = leto::Array2::<f64>::zeros((1, 4));
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_source_kappa_equals_cosine() {
    use std::f64::consts::PI;

    let n = 32usize;
    let dx = 5e-4_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let cfl = 0.5_f64;
    let dt = cfl * dx / c0;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, c0, &grid);

    let source = kwavers_source::grid_source::GridSource::new_empty();
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

fn pstd_grazing_reflection(pml: usize, sigma_factor: f64) -> (f64, f64, f64) {
    use kwavers_boundary::cpml::{CPMLConfig, PerDimensionAlpha, PerDimensionPML};
    const NX: usize = 160;
    const UPPER: usize = 32;
    const H_REF: usize = 140;
    const F0: f64 = 1.0e6;
    const C: f64 = 1500.0;
    const RHO: f64 = 1000.0;
    const DX: f64 = C / F0 / 8.0;
    const SX: usize = 20;
    const RX: usize = 130;
    const STEPS: usize = 480;

    let run = |h: usize| -> Vec<f64> {
        let ny = pml + h + UPPER + pml;
        let y0 = pml + h;
        let grid = Grid::new(NX, ny, 1, DX, DX, DX).unwrap();
        let medium = HomogeneousMedium::new(RHO, C, 0.0, 0.0, &grid);

        let width = C / F0 / 2.0;
        let mut p0 = leto::Array3::<f64>::zeros((NX, ny, 1));
        for i in 0..NX {
            for j in 0..ny {
                let dx = (i as f64 - SX as f64) * DX;
                let dy = (j as f64 - y0 as f64) * DX;
                p0[[i, j, 0]] = (-((dx * dx + dy * dy) / (width * width))).exp();
            }
        }
        let source = GridSource {
            p0: Some(p0),
            ..GridSource::new_empty()
        };

        let cpml = CPMLConfig {
            per_dimension: PerDimensionPML::new(pml, pml, 0),
            thickness: pml,
            sigma_factor,
            per_dimension_alpha: PerDimensionAlpha::uniform(sigma_factor),
            ..CPMLConfig::default()
        };
        let config = PSTDConfig {
            dt: 0.3 * DX / C,
            nt: STEPS + 1,
            boundary: BoundaryConfig::CPML(cpml),
            smooth_sources: false,
            ..Default::default()
        };
        let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();
        let mut trace = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            solver.step_forward().unwrap();
            trace.push(solver.fields.p[[RX, y0, 0]]);
        }
        trace
    };

    let peak = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let peak_diff =
        |a: &[f64], b: &[f64]| peak(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());

    let reference = run(H_REF);
    let test = run(20);
    let farther = run(H_REF - 30);
    (
        peak(&reference),
        peak_diff(&test, &reference),
        peak_diff(&farther, &reference),
    )
}

/// ## Theorem
/// On the k-Wave split-field PSTD path, `sigma_factor = 3` absorbs grazing
/// incidence far better than k-Wave's `pml_alpha` default of 2.
///
/// ## Measured (2026-08-17, 70° incidence, 20-cell PML)
/// 5.89e-8 of the incident peak against 6.87e-6 — a factor of 117. The
/// convolutional FDTD path agrees at the same thickness (6.84e-9 against
/// 4.26e-7, 62×), which is what licensed raising the shared default; see
/// `sigma_factor_three_outperforms_the_kwave_default_at_grazing_incidence`.
///
/// At 10 cells the two paths disagree on the optimum (FDTD 3, PSTD 4), so this
/// asserts at the shipping thickness, where they agree.
#[test]
fn pstd_sigma_factor_three_outperforms_the_kwave_default() {
    let (direct_default, spurious_default, residual_default) = pstd_grazing_reflection(20, 2.0);
    let (direct_tuned, spurious_tuned, residual_tuned) = pstd_grazing_reflection(20, 3.0);

    // Both references are judged against the *smaller* of the two signals. The
    // tuned configuration is ~100x quieter, so a reference clean enough to
    // resolve the default is not automatically clean enough to resolve it.
    let dirtiest = residual_default.max(residual_tuned);
    assert!(
        dirtiest <= 0.05 * spurious_tuned,
        "reference must resolve the tuned signal: worst residual {dirtiest:.3e}, \
         tuned signal {spurious_tuned:.3e}"
    );
    assert!(
        direct_default > 1e-2,
        "the direct wave must reach the receiver: peak {direct_default:.3e}"
    );

    let ratio_default = spurious_default / direct_default;
    let ratio_tuned = spurious_tuned / direct_tuned;
    assert!(
        ratio_tuned < 0.2 * ratio_default,
        "sigma_factor 3 must beat 2 by >5x on the PSTD path: default {ratio_default:.4e}, \
         tuned {ratio_tuned:.4e} (measured 117x)"
    );
}
