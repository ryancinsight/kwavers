use super::super::types::{ElasticWaveConfig, ElasticWaveField};
use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::Array3;

#[test]
fn test_elastic_wave_solver_recording() -> KwaversResult<()> {
    let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0)?;
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let mut sensor_mask = Array3::from_elem(grid.dimensions(), false);
    sensor_mask[[5, 5, 5]] = true;
    let config = ElasticWaveConfig {
        simulation_time: 1e-4,
        time_step: 1e-5,
        save_every: 2,
        sensor_mask: Some(sensor_mask),
        ..Default::default()
    };
    let mut solver = ElasticWaveSolver::new(&grid, &medium, config)?;
    let mut initial_field = ElasticWaveField::new(10, 10, 10);
    initial_field.uz[[5, 5, 5]] = 1.0;
    let _final_field = solver.propagate(&initial_field, 1e-4, None)?;
    let data = solver.extract_recorded_data().unwrap();
    assert_eq!(data.shape(), [1, 5]);
    Ok(())
}
