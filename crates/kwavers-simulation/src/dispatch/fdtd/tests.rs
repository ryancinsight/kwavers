use super::run;
use crate::configs::NonlinearConfig;
use crate::dispatch::test_support::{centre_source, request};
use crate::types::SimulationRunRequest;
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::config::SolverType;

/// A request without a sensor mask, and one whose mask has no true element,
/// run and record nothing: `(0, time_steps)` for the three steps, and
/// `(0, 2)` when `record_start_index = 2` skips one sample, the columns the
/// same request with a sensor records. The result carries no other record.
#[test]
fn a_request_without_sensors_returns_an_empty_record() {
    let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let nonlinear = NonlinearConfig::default();
    let no_true_element = leto::Array3::from_elem([7, 4, 3], false);

    for (mask, record_start_index, columns) in [
        (None, 0, 3),
        (Some(no_true_element.clone()), 0, 3),
        (Some(no_true_element), 2, 2),
    ] {
        let request = SimulationRunRequest {
            solver_type: SolverType::FDTD,
            time_steps: 3,
            dt: 1.0e-7,
            grid_source: centre_source(),
            sensor_mask: mask,
            record_start_index,
            ..request(&grid, &medium, &nonlinear)
        };

        let result = run(&request, Vec::new()).expect("a request without sensors runs");

        assert_eq!(result.sensor_data.shape(), [0, columns]);
        assert!(result.ux_data.is_none() && result.uy_data.is_none() && result.uz_data.is_none());
        assert!(result.ix_data.is_none() && result.iy_data.is_none() && result.iz_data.is_none());
        assert!(result.stats.is_none() && result.velocity_stats.is_none());
    }
}

/// The empty record is the answer to an absent sensor only: one true mask
/// element keeps one row, so the shape separates the two requests.
#[test]
fn a_request_with_one_sensor_keeps_its_row() {
    let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let nonlinear = NonlinearConfig::default();
    let mut centre = leto::Array3::from_elem([7, 4, 3], false);
    centre[[3, 2, 1]] = true;

    let request = SimulationRunRequest {
        solver_type: SolverType::FDTD,
        time_steps: 3,
        dt: 1.0e-7,
        grid_source: centre_source(),
        sensor_mask: Some(centre),
        ..request(&grid, &medium, &nonlinear)
    };

    let result = run(&request, Vec::new()).expect("a request with one sensor runs");

    assert_eq!(result.sensor_data.shape(), [1, 3]);
}
