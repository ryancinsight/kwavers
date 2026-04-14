use crate::domain::grid::Grid;

use super::{access_serializable_grid, deserialize_grid, serialize_grid, SimulationData};

#[test]
fn grid_serialization_roundtrip() {
    let original = Grid::new(100, 200, 300, 1e-3, 2e-3, 3e-3).unwrap();
    let bytes = serialize_grid(&original).unwrap();
    assert!(!bytes.is_empty());

    let archived = access_serializable_grid(&bytes).unwrap();
    assert_eq!(archived.nx, 100);

    let loaded = deserialize_grid(&bytes).unwrap();
    assert_eq!(original.nx, loaded.nx);
    assert_eq!(original.ny, loaded.ny);
    assert_eq!(original.nz, loaded.nz);
}

#[test]
fn simulation_data_roundtrip_and_access() {
    let grid = Grid::new(10, 20, 30, 0.001, 0.001, 0.001).unwrap();
    let pressure: Vec<f64> = (0..6000).map(|i| i as f64 * 0.1).collect();
    let original = SimulationData::new(1.5, pressure, &grid);

    let bytes = original.to_bytes().unwrap();
    let archived = SimulationData::access(&bytes).unwrap();
    assert_eq!(archived.grid.nx, 10);

    let loaded = SimulationData::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.grid.nx, 10);
    assert!((loaded.time - 1.5).abs() < 1e-10);
}
