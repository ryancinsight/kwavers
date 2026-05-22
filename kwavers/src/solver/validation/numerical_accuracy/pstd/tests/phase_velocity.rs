use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::pstd::PSTDConfig as PstdConfig;
use crate::solver::pstd::PSTDSolver;
use std::f64::consts::PI;

#[test]
fn test_pstd_phase_velocity_accuracy() {
    // Measure numerical phase velocity of the PSTD solver vs. theoretical.
    //
    // # Theory
    //
    // **Theorem** (PSTD phase velocity, Treeby & Cox 2010 §2.2):
    // For a plane wave (wavenumber k, frequency ω = c₀·k) in a homogeneous medium,
    // the k-space PSTD scheme produces numerical dispersion:
    // ```text
    //   ω_num = (2/Δt) · arcsin(c₀ · k · Δt / 2)
    // ```
    // Relative phase velocity error:
    // ```text
    //   |c_num/c₀ − 1| ≈ (c₀ · Δt · k)² / 24 ≪ 10⁻³   for CFL ≤ 0.2, PPW ≥ 20
    // ```
    //
    // # Correct Initial Conditions
    //
    // For a +x-traveling plane wave, the linearized acoustic equations require:
    // ```text
    //   p(x,0)    = sin(kx)
    //   u_x(x,0)  = sin(kx) / (ρ₀ · c₀)   [impedance relation]
    //   ρ′_x(x,0) = p(x,0) / c₀²           [linearized EOS — all in x-component]
    //   ρ′_y = ρ′_z = 0                     [no y/z propagation]
    // ```
    // The previous test used `ρ_y = ρ_z = ρ_total/3` (equal-split), which
    // seeds a standing-wave contamination and is incorrect for a directional wave.
    //
    // # Phase Velocity Measurement
    //
    // Crest-tracking fails on a periodic domain (crest wraps produce spurious
    // velocity spikes). Instead we use cross-correlation of initial and final
    // pressure slices to find the displacement s_peak [grid cells]:
    // ```text
    //   xcorr[s] = Σ_i p₀[i] · p_T[(i+s) mod N]
    // ```
    // Periodic wraps are recovered analytically from the known nominal displacement:
    // ```text
    //   n_wraps = floor(round(c₀·T/Δx) / N)
    //   c_num   = (s_peak + n_wraps·N) · Δx / T
    // ```
    //
    // # Reference
    // Treeby, B.E. & Cox, B.T. (2010) J. Biomed. Opt. 15(2):021314.
    println!("\n=== PSTD Phase Velocity Accuracy Test ===");

    // n=64, nt=800: PPW=20, CFL=0.2 (same physics accuracy as n=128/nt=1000).
    // cells traveled = 0.2·800 = 160; n_wraps = 2; residual s_peak = 32.
    // Quantization error ±0.5/160 = 0.31% < 0.5% tolerance. ~8× faster than n=128.
    let n = 64_usize;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = 1000.0_f64;
    let frequency = 1e6_f64;
    let wavelength = c0 / frequency;
    let dx = wavelength / 20.0; // 20 PPW
    let k = 2.0 * PI / wavelength;
    let dt = 0.2 * dx / c0; // CFL = 0.2
    let nt = 800_usize;

    let mut config = PstdConfig::default();
    config.dt = dt;
    config.nt = nt;
    // No absorbing boundary — periodic domain for clean phase velocity measurement.
    config.boundary = crate::solver::forward::pstd::config::BoundaryConfig::None;

    let grid = Grid::new(n, n, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(rho0, c0, &grid);

    let mut solver = PSTDSolver::new(
        config.clone(),
        grid.clone(),
        &medium,
        crate::domain::source::GridSource::default(),
    )
    .unwrap();

    // Initialize +x traveling wave with correct linearized-EOS density split.
    // ρ_x = p/c₀²; ρ_y = ρ_z = 0 (wave propagates in x only).
    for i in 0..n {
        for j in 0..n {
            let x = i as f64 * dx;
            let p_val = (k * x).sin();
            solver.fields.p[[i, j, 0]] = p_val;
            solver.fields.ux[[i, j, 0]] = p_val / (rho0 * c0);
            solver.rhox[[i, j, 0]] = p_val / (c0 * c0);
            solver.rhoy[[i, j, 0]] = 0.0;
            solver.rhoz[[i, j, 0]] = 0.0;
        }
    }

    // Snapshot initial pressure at y = n/2 slice
    let p_initial: Vec<f64> = (0..n).map(|i| solver.fields.p[[i, n / 2, 0]]).collect();

    // Advance nt steps
    let t_total = nt as f64 * dt;
    for _ in 0..nt {
        solver.step_forward().unwrap();
    }

    // Cross-correlation of initial and final pressure slices
    let p_final: Vec<f64> = (0..n).map(|i| solver.fields.p[[i, n / 2, 0]]).collect();
    let mut xcorr = vec![0.0_f64; n];
    for s in 0..n {
        xcorr[s] = (0..n).map(|i| p_initial[i] * p_final[(i + s) % n]).sum();
    }
    let s_peak = xcorr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Reconstruct total displacement, resolving periodic wrap-arounds.
    // Nominal: c₀·T/Δx = 1500·(1000·0.2·Δx/c₀)/Δx = 200 cells.
    // n_wraps = floor(200/128) = 1, residual = 72, so cells = 72+128 = 200.
    let nominal_cells = (c0 * t_total / dx).round() as usize;
    let n_wraps = nominal_cells / n;
    let cells_traveled = s_peak + n_wraps * n;
    let c_measured = cells_traveled as f64 * dx / t_total;

    let velocity_error = (c_measured - c0).abs() / c0;
    println!("Theoretical c₀       : {:.4} m/s", c0);
    println!("Measured   c_num      : {:.4} m/s", c_measured);
    println!("Relative error        : {:.6}%", velocity_error * 100.0);

    // At 20 PPW / CFL 0.2 the theoretical PSTD dispersion error is ~8×10⁻⁶.
    // The cross-correlation peak quantisation introduces at most ±½ cell ≈ 0.25%.
    // We assert < 0.5% to give margin for absorbing boundary effects.
    assert!(
        velocity_error < 0.005,
        "PSTD phase velocity error {:.4}% exceeds 0.5% budget",
        velocity_error * 100.0
    );
    // Reference CFL_NUMBER to suppress dead_code lint on shared constant
    let _ = super::CFL_NUMBER;
}
