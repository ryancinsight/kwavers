use crate::pstd::PSTDConfig as PstdConfig;
use crate::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::{
    Array2,
    Array3,
};

#[test]
fn test_linear_array_phase_accuracy() {
    // Phase error test for linear array source
    // Validates phase consistency across array elements
    println!("\n=== Linear Array Phase Accuracy Test ===");

    let n = 80;
    let frequency = MHZ_TO_HZ;
    let c0 = SOUND_SPEED_WATER_SIM;
    let wavelength = c0 / frequency;
    let dx = wavelength / 16.0;

    let mut config = PstdConfig::default();
    config.dt = super::CFL_NUMBER * dx / c0;

    let grid = Grid::new(n, n, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, c0, &grid);

    // Create linear array source (multiple point sources along x-axis)
    let num_elements = 8;
    let element_spacing = wavelength; // λ spacing
    let mut source_data = kwavers_source::GridSource::default();
    source_data.p_mask = Some(Array3::zeros((n, n, 1)));

    let center_x = n / 2;
    let center_y = n / 2;

    if let Some(ref mut mask) = source_data.p_mask {
        for elem in 0..num_elements {
            let offset = (elem as f64 - (num_elements - 1) as f64 / 2.0) * element_spacing / dx;
            let i = (center_x as f64 + offset).round() as usize;
            if i < n {
                mask[[i, center_y, 0]] = 1.0;
            }
        }
    }

    // All elements driven in phase
    let signal = Array2::from_shape_fn((1, 500), |[_, t]| {
        let t_f64 = t as f64 * config.dt;
        (TWO_PI * frequency * t_f64).sin()
    });
    source_data.p_signal = Some(signal);

    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source_data).unwrap();

    // Run simulation
    for step in 0..300 {
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

    // Check for constructive interference along beam axis
    let mut on_axis_pressure = 0.0;
    let mut off_axis_pressure = 0.0;

    for i in n / 4..3 * n / 4 {
        on_axis_pressure += solver.fields.p[[i, center_y, 0]].abs();
        off_axis_pressure += solver.fields.p[[i, center_y + 10, 0]].abs();
    }

    let directivity = on_axis_pressure / (off_axis_pressure.max(1e-10));
    println!("Array directivity (on-axis/off-axis): {:.2}", directivity);

    // Array should show directivity (stronger on-axis than off-axis)
    assert!(
        directivity > 2.0,
        "Linear array did not produce directive beam"
    );
}
