use crate::domain::grid::Grid;
use crate::domain::medium::{core::CoreMedium, viscous::ViscousProperties};

use super::HomogeneousMedium;

#[test]
fn test_water_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let water = HomogeneousMedium::water(&grid);

    assert_eq!(water.density(0, 0, 0), 998.0);
    assert_eq!(water.sound_speed(0, 0, 0), 1482.0);
    assert_eq!(water.viscosity(0.0, 0.0, 0.0, &grid), 1.0e-3);
}

#[test]
fn test_blood_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let blood = HomogeneousMedium::blood(&grid);

    assert_eq!(blood.density(0, 0, 0), 1060.0);
    assert_eq!(blood.sound_speed(0, 0, 0), 1570.0);
    assert_eq!(blood.viscosity(0.0, 0.0, 0.0, &grid), 3.5e-3);
}

#[test]
fn test_air_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let air = HomogeneousMedium::air(&grid);

    assert_eq!(air.density(0, 0, 0), 1.204);
    assert_eq!(air.sound_speed(0, 0, 0), 343.0);
}
