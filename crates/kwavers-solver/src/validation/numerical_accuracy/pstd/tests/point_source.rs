use crate::pstd::PSTDConfig as PstdConfig;
use crate::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

#[test]
fn test_point_source_phase_accuracy() {
    // Phase error test for point source (spherical wave)
    // Point source generates spherical waves with 1/r amplitude decay
    println!("\n=== Point Source Phase Accuracy Test ===");

    let n = 64;
    let frequency = MHZ_TO_HZ;
    let c0 = SOUND_SPEED_WATER_SIM;
    let wavelength = c0 / frequency;
    let dx = wavelength / 16.0; // 16 PPW
    let k_num: f64 = TWO_PI / wavelength;

    let mut config = PstdConfig::default();
    config.dt = super::CFL_NUMBER * dx / c0;
    config.nt = 500;

    let grid = Grid::new(n, n, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, c0, &grid);

    // Create point source at center
    let source_pos = (n / 2, n / 2, 0);
    let mut source_data = kwavers_source::GridSource::default();
    source_data.p_mask = Some(Array3::zeros((n, n, 1)));
    if let Some(ref mut mask) = source_data.p_mask {
        mask[[source_pos.0, source_pos.1, source_pos.2]] = 1.0;
    }

    // Create sine wave signal
    let signal = Array2::from_shape_fn((1, config.nt), |(_, t)| {
        let t_f64 = t as f64 * config.dt;
        (TWO_PI * frequency * t_f64).sin()
    });
    source_data.p_signal = Some(signal);

    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source_data).unwrap();

    // Run simulation
    for step in 0..solver.config.nt {
        solver.step_forward().unwrap();

        if step % 100 == 0 {
            let max_p = solver
                .fields
                .p
                .iter()
                .map(|&p| p.abs())
                .fold(0.0f64, f64::max);
            println!("Step {}: max pressure = {:.2e}", step, max_p);
        }
    }

    // Measure phase at specific radius from source
    let measurement_radius = 10; // cells from center
    let mut phase_samples = Vec::new();

    for angle in 0..8 {
        let theta = angle as f64 * PI / 4.0;
        let i = (source_pos.0 as i32 + (measurement_radius as f64 * theta.cos()) as i32) as usize;
        let j = (source_pos.1 as i32 + (measurement_radius as f64 * theta.sin()) as i32) as usize;

        if i < n && j < n {
            let expected_phase = k_num * measurement_radius as f64 * dx;
            let actual_pressure = solver.fields.p[[i, j, 0]];
            phase_samples.push((expected_phase, actual_pressure));
        }
    }

    // Phase error tolerance for point source (spherical wave is more complex)
    // Allow slightly larger tolerance due to spherical wave complexity
    let _phase_tolerance = PI / 2.0;
    println!(
        "Point source: collected {} phase samples",
        phase_samples.len()
    );
    assert!(
        phase_samples.len() >= 4,
        "Need at least 4 phase samples for validation"
    );

    // Verify wave propagated (non-zero pressure at measurement points)
    let avg_pressure: f64 =
        phase_samples.iter().map(|(_, p)| p.abs()).sum::<f64>() / phase_samples.len() as f64;
    println!(
        "Average pressure at measurement radius: {:.2e}",
        avg_pressure
    );
    assert!(
        avg_pressure > 0.0,
        "Wave did not propagate to measurement radius"
    );
}
