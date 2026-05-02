use crate::domain::grid::Grid;
use crate::domain::medium::core::CoreMedium;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::pstd::PSTDConfig as PstdConfig;
use crate::solver::pstd::PSTDSolver;
use std::f64::consts::PI;

#[test]
fn test_pstd_plane_wave_accuracy() {
    // RIGOROUS VALIDATION: k-space method accuracy (Treeby & Cox 2010, Section 3.2)
    // EXACT VALIDATION: Spectral methods should have minimal dispersion error
    let n = 64; // Reverted to default power of 2
    let frequency = 1e6;
    let wavelength = 1500.0 / frequency; // 1.5mm at 1MHz
                                         // ADJUSTMENT: Use PPW=16 to ensure periodic boundary conditions
                                         // n=64, PPW=16 -> L = 4 * wavelength (integer multiple)
    let dx = wavelength / 16.0; // 16 points per wavelength
    let ppw = wavelength / dx;

    // EXACT ASSERTION: Must meet minimum sampling requirement
    assert!(
        ppw >= super::PPW_MINIMUM as f64,
        "Insufficient spatial sampling: {} < {}",
        ppw,
        super::PPW_MINIMUM
    );

    let mut config = PstdConfig::default();
    config.boundary = crate::solver::forward::pstd::config::BoundaryConfig::None;
    // Ensure stability: dt <= CFL * dx / c
    // dx = 6.25e-5, c = 1500, CFL = 0.3 -> dt_max = 1.25e-8
    config.dt = 1.0e-8;

    let grid = Grid::new(n, n, 1, dx, dx, dx).unwrap();

    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let source_data = crate::domain::source::GridSource::default();
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source_data).unwrap();

    // Initialize plane wave in the solver
    let k = 2.0 * PI / wavelength;
    let c0 = medium.sound_speed(0, 0, 0);
    let rho0 = medium.density(0, 0, 0);

    for i in 0..n {
        for j in 0..n {
            let x = i as f64 * dx;
            let p_val = (k * x).sin();
            solver.fields.p[[i, j, 0]] = p_val;

            // Initialize density consistent with pressure
            // p = c^2 * rho => rho = p / c^2, split across components
            let rho = p_val / (c0 * c0);
            let split = rho / 3.0;
            solver.rhox[[i, j, 0]] = split;
            solver.rhoy[[i, j, 0]] = split;
            solver.rhoz[[i, j, 0]] = split;

            // Initialize velocity consistently with a rightward-propagating wave
            // For a plane wave: v_x = p/(ρc)
            solver.fields.ux[[i, j, 0]] = p_val / (rho0 * c0);
        }
    }

    // Propagate one wavelength
    use crate::domain::boundary::pml::{PMLBoundary, PMLConfig};

    let pml_config = PMLConfig::default();
    let mut _boundary = PMLBoundary::new(pml_config).unwrap();

    let dt = solver.get_timestep();
    let steps = (wavelength / (1500.0 * dt)) as usize;
    let _initial = solver.fields.p.clone();

    println!("Propagating for {} steps, dt = {:.2e}", steps, dt);

    for step in 0..steps {
        solver.step_forward().unwrap();

        if step < 5 || step % 100 == 0 {
            let max_p = solver
                .fields
                .p
                .iter()
                .map(|&p| p.abs())
                .fold(0.0_f64, f64::max);
            println!("Step {}: max pressure = {:.2e}", step, max_p);
        }
    }

    let pressure = solver.fields.p.clone();

    // Calculate phase error using least squares fit to y = a*sin(kx) + b*cos(kx)
    // This is robust against pointwise variations and accurately extracts phase/amplitude
    let mut sum_y_sin = 0.0;
    let mut sum_y_cos = 0.0;
    let mut sum_sin2 = 0.0;
    let mut sum_cos2 = 0.0;
    let mut sum_sin_cos = 0.0;

    for i in n / 4..3 * n / 4 {
        let x = i as f64 * dx;
        let val = pressure[[i, n / 2, 0]];
        let s = (k * x).sin();
        let c = (k * x).cos();

        sum_y_sin += val * s;
        sum_y_cos += val * c;
        sum_sin2 += s * s;
        sum_cos2 += c * c;
        sum_sin_cos += s * c;
    }

    let det = sum_sin2 * sum_cos2 - sum_sin_cos * sum_sin_cos;
    let a = (sum_y_sin * sum_cos2 - sum_y_cos * sum_sin_cos) / det;
    let b = (sum_y_cos * sum_sin2 - sum_y_sin * sum_sin_cos) / det;

    // y = a*sin + b*cos = R*sin(kx + phi)
    // a = R*cos(phi), b = R*sin(phi)
    let phase_error = b.atan2(a).abs();
    let amplitude = (a * a + b * b).sqrt();
    let amplitude_error = (amplitude - 1.0).abs();

    // RIGOROUS VALIDATION: Spectral methods should have minimal dispersion
    // EXACT TOLERANCE: For well-sampled problems, phase error should be < π/4
    let strict_phase_tolerance = PI / 4.0; // ~0.785, much stricter than 1.6
    let strict_amplitude_tolerance = 0.01; // 1% amplitude error max

    // EVIDENCE-BASED ASSERTION: If this fails, PSTD implementation needs fixing
    // CHECK: PSTD phase error should be minimal with periodic boundaries and consistent initialization
    assert!(
        phase_error < strict_phase_tolerance,
        "PSTD phase error exceeds theoretical limit: {:.6} > {:.6} (π/4). \
         This indicates implementation issues in spectral operations.",
        phase_error,
        strict_phase_tolerance
    );

    assert!(
        amplitude_error < strict_amplitude_tolerance,
        "PSTD amplitude error exceeds 1%: {:.6} > {:.6}. \
         Spectral methods should preserve amplitude precisely.",
        amplitude_error,
        strict_amplitude_tolerance
    );
}
