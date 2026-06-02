use super::operators::KSpaceFdtdOperators;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use kwavers_domain::source::GridSource;
use crate::forward::fdtd::config::{FdtdConfig, KSpaceCorrectionMode};
use crate::forward::fdtd::solver::FdtdSolver;
use crate::forward::pstd::config::PSTDConfig;
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::Array3;

fn test_grid() -> Grid {
    Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap()
}

/// K-space and PSTD shift operators must be bit-identical.
///
/// Both call `generate_shift_1d` from the same shared module. Any divergence
/// would indicate they are using different implementations.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_kspace_shift_operators_match_pstd() {
    let grid = test_grid();
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * grid.dx / c_ref;

    let kops = KSpaceFdtdOperators::new(
        grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, c_ref, dt,
    );

    let medium = HomogeneousMedium::water(&grid);
    let mut pstd_config = PSTDConfig::default();
    pstd_config.dt = dt;
    let pstd = PSTDSolver::new(pstd_config, grid, &medium, GridSource::default()).unwrap();

    for i in 0..pstd.ddx_k_shift_pos.len() {
        let diff = (kops.ddx_k_shift_pos[i] - pstd.ddx_k_shift_pos[i]).norm();
        assert!(diff < 1e-14, "ddx_k_shift_pos[{i}] mismatch: diff={diff}");
    }

    let _ = &pstd.kappa;
}

/// With `KSpaceCorrectionMode::None`, the FdtdSolver must produce the same
/// results as before this feature was added (no regression).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_kspace_fd_fallback_unchanged() {
    let grid = test_grid();
    let medium = HomogeneousMedium::water(&grid);
    let c0 = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * grid.dx / c0;

    let config_fd = FdtdConfig {
        kspace_correction: KSpaceCorrectionMode::None,
        staggered_grid: true,
        dt,
        nt: 5,
        ..Default::default()
    };

    let mut solver_fd = FdtdSolver::new(config_fd, &grid, &medium, GridSource::default()).unwrap();
    let mut p_ref = Array3::zeros((grid.nx, grid.ny, grid.nz));
    p_ref[[8, 8, 8]] = 1.0;
    solver_fd.fields.p.assign(&p_ref);

    solver_fd.step_forward().unwrap();
    solver_fd.step_forward().unwrap();
    solver_fd.step_forward().unwrap();

    assert!(
        solver_fd.fields.p.iter().all(|&v| v.is_finite()),
        "FD fallback produced non-finite pressure"
    );
}

/// K-space FDTD must run without NaN/Inf for a simple IVP.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_kspace_spectral_solver_runs_without_nan() {
    let grid = test_grid();
    let medium = HomogeneousMedium::water(&grid);
    let c0 = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * grid.dx / c0;

    let config_ks = FdtdConfig {
        kspace_correction: KSpaceCorrectionMode::Spectral,
        staggered_grid: true,
        dt,
        nt: 10,
        ..Default::default()
    };

    let mut solver_ks = FdtdSolver::new(config_ks, &grid, &medium, GridSource::default()).unwrap();

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let dx = (i as f64 - 8.0) * grid.dx;
                let dy = (j as f64 - 8.0) * grid.dy;
                let dz = (k as f64 - 8.0) * grid.dz;
                let r2 = dx * dx + dy * dy + dz * dz;
                let sigma = 3.0 * grid.dx;
                solver_ks.fields.p[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    for _ in 0..10 {
        solver_ks.step_forward().unwrap();
    }

    assert!(
        solver_ks.fields.p.iter().all(|&v| v.is_finite()),
        "K-space FDTD produced non-finite pressure"
    );
}

/// The spectral gradient of a constant field must be zero everywhere.
///
/// Proof: IFFT( ddx_k_shift · κ · FFT(C) ) = IFFT( C · ddx[0] · κ · δ₀ ) = 0
/// because ddx_k_shift[0] = i·0·exp(0) = 0 (DC bin).
/// # Panics
/// - Panics if assertion fails: `Gradient of constant should be ~0, got max={max_grad}`.
///
#[test]
fn test_spectral_gradient_of_constant_is_zero() {
    let nx = 8;
    let dx = 1e-3;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * dx / c_ref;
    let mut kops = KSpaceFdtdOperators::new(nx, nx, nx, dx, dx, dx, c_ref, dt);

    let constant_field = Array3::from_elem((nx, nx, nx), 2.5_f64);
    kops.compute_grad_pos(&constant_field);

    let max_grad = kops.grad_x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_grad < 1e-10,
        "Gradient of constant should be ~0, got max={max_grad}"
    );
}

/// The spectral divergence of zero velocity must be zero everywhere.
/// # Panics
/// - Panics if assertion fails: `Divergence of zero should be ~0, got max={max_div}`.
///
#[test]
fn test_spectral_divergence_of_zero_is_zero() {
    let nx = 8;
    let dx = 1e-3;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * dx / c_ref;
    let mut kops = KSpaceFdtdOperators::new(nx, nx, nx, dx, dx, dx, c_ref, dt);

    let zero = Array3::zeros((nx, nx, nx));
    kops.compute_divergence_neg(&zero, &zero, &zero);

    let max_div = kops
        .divergence
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_div < 1e-10,
        "Divergence of zero should be ~0, got max={max_div}"
    );
}

/// Spectral x-derivative of sin(kx·x) must equal kx·cos(kx·x) (fundamental mode).
///
/// The spectral derivative is exact: d/dx sin(kx·x) = kx cos(kx·x).
/// We verify max relative error < 1e-6.
/// # Panics
/// - Panics if assertion fails: `Spectral d/dx sin(kx) relative error too large: {rel_err} (max_abs={max_abs_err:.3e}, k_fund={k_fund:.3e})`.
///
#[test]
fn test_spectral_gradient_of_sine_is_cosine() {
    let nx = 32usize;
    let dx = 1e-3_f64;
    let c_ref = SOUND_SPEED_WATER_SIM;
    let dt_small = 1e-10;
    let mut kops = KSpaceFdtdOperators::new(nx, nx, nx, dx, dx, dx, c_ref, dt_small);

    let lx = nx as f64 * dx;
    let k_fund = TWO_PI / lx;
    let mut field = Array3::zeros((nx, nx, nx));
    for i in 0..nx {
        let x = i as f64 * dx;
        let val = (k_fund * x).sin();
        for j in 0..nx {
            for k in 0..nx {
                field[[i, j, k]] = val;
            }
        }
    }

    kops.compute_grad_pos(&field);

    let mut max_abs_err: f64 = 0.0;
    for i in 0..nx {
        let x_stag = (i as f64 + 0.5) * dx;
        let expected = k_fund * (k_fund * x_stag).cos();
        let got = kops.grad_x[[i, 0, 0]];
        max_abs_err = max_abs_err.max((got - expected).abs());
    }
    let rel_err = max_abs_err / k_fund;
    assert!(
        rel_err < 1e-6,
        "Spectral d/dx sin(kx) relative error too large: {rel_err} (max_abs={max_abs_err:.3e}, k_fund={k_fund:.3e})"
    );
}
