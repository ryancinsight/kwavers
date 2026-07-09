use super::*;
use kwavers_grid::Grid;
use leto::Array3;

#[test]
fn test_lithotripsy_simulator_creation() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();

    let params = LithotripsyParameters {
        stone_geometry: Array3::zeros(grid.dimensions()),
        ..Default::default()
    };

    let _simulator = LithotripsySimulator::new(params, grid).unwrap();
}

#[test]
fn test_stone_volume_calculation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

    let mut params = LithotripsyParameters {
        stone_geometry: Array3::zeros(grid.dimensions()),
        ..Default::default()
    };

    // Create a 2x2x2 stone in the center
    for i in 4..6 {
        for j in 4..6 {
            for k in 4..6 {
                params.stone_geometry[[i, j, k]] = 1.0;
            }
        }
    }

    let simulator = LithotripsySimulator::new(params, grid).unwrap();

    // Stone volume should be 8 * voxel_volume = 8e-9 m³
    let expected_volume = 8.0 * 1e-9;
    let actual_volume = simulator.calculate_stone_volume();

    assert!((actual_volume - expected_volume).abs() < 1e-12);
}

#[test]
fn test_simulation_state() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();

    let params = LithotripsyParameters {
        num_shock_waves: 10, // Short simulation for testing
        stone_geometry: Array3::zeros(grid.dimensions()),
        ..Default::default()
    };
    let expected_shocks = params.num_shock_waves;

    let mut simulator = LithotripsySimulator::new(params, grid).unwrap();

    // Run short simulation
    let results = simulator.run_simulation().unwrap();

    assert!(results.shock_waves_delivered <= expected_shocks);
    if results.shock_waves_delivered < expected_shocks {
        assert!(!results.final_safety_assessment.overall_safe);
    }

    // Should have some treatment time
    assert!(results.treatment_time > 0.0);

    assert!((0.0..=1.0).contains(&results.final_safety_assessment.safety_score));
}
