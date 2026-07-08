use super::types::ElectromagneticFdtdSolver;
use kwavers_field::{ArrayD, EMFields, VecStorage};
use kwavers_grid::Grid;
use kwavers_physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};

fn seeded_solver() -> ElectromagneticFdtdSolver {
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let materials = EMMaterialDistribution::vacuum(&[4, 4, 4]);
    let mut solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

    solver.ex.fill(4.0);
    solver.ey.fill(8.0);
    solver.ez.fill(12.0);
    solver.hx.fill(3.0);
    solver.hy.fill(5.0);
    solver.hz.fill(7.0);

    solver
}

fn output_fields(shape: &[usize]) -> EMFields {
    EMFields::new(
        ArrayD::<f64, VecStorage<f64>>::from_elem(shape, -1.0).unwrap(),
        ArrayD::<f64, VecStorage<f64>>::from_elem(shape, -2.0).unwrap(),
    )
}

#[test]
fn test_em_fdtd_creation() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

    // Use canonical domain composition pattern
    let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

    let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();
    assert_eq!(solver.em_dimension(), EMDimension::Three);
}

#[test]
fn test_maxwell_time_step() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();

    // Use canonical domain composition pattern
    let materials = EMMaterialDistribution::vacuum(&[32, 32, 32]);

    let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

    // Speed of light in vacuum (normalized units)
    let c = 1.0;
    let dt = solver.max_stable_dt(c);

    // Check that time step is reasonable
    assert!(dt > 0.0);
    assert!(dt < 1e-3); // Should be smaller than spatial step
}

#[test]
fn boundary_conditions_reuse_shape_compatible_output_storage() {
    let mut solver = seeded_solver();
    let mut fields = EMFields::new(
        ArrayD::<f64, VecStorage<f64>>::from_elem(&[4, 4, 4, 3], -1.0).unwrap(),
        ArrayD::<f64, VecStorage<f64>>::from_elem(&[4, 4, 4, 3], -2.0).unwrap(),
    );
    fields.displacement =
        Some(ArrayD::<f64, VecStorage<f64>>::from_elem(&[4, 4, 4, 3], 1.0).unwrap());
    fields.flux_density =
        Some(ArrayD::<f64, VecStorage<f64>>::from_elem(&[4, 4, 4, 3], 2.0).unwrap());

    let electric_ptr = fields.electric.iter().next().map(|x| x as *const f64);
    let magnetic_ptr = fields.magnetic.iter().next().map(|x| x as *const f64);

    solver.apply_em_boundary_conditions(&mut fields);

    // Verify no reallocation occurred (pointer to first element is unchanged).
    assert_eq!(
        fields.electric.iter().next().map(|x| x as *const f64),
        electric_ptr
    );
    assert_eq!(
        fields.magnetic.iter().next().map(|x| x as *const f64),
        magnetic_ptr
    );
    assert!(fields.displacement.is_none());
    assert!(fields.flux_density.is_none());
    assert_eq!(*fields.electric.get(&[1, 1, 1, 0]).unwrap(), 4.0);
    assert_eq!(*fields.electric.get(&[1, 1, 1, 1]).unwrap(), 8.0);
    assert_eq!(*fields.electric.get(&[1, 1, 1, 2]).unwrap(), 12.0);
    assert_eq!(*fields.magnetic.get(&[1, 1, 1, 0]).unwrap(), 3.0);
    assert_eq!(*fields.magnetic.get(&[1, 1, 1, 1]).unwrap(), 5.0);
    assert_eq!(*fields.magnetic.get(&[1, 1, 1, 2]).unwrap(), 7.0);
    // Check fields match the cache in the solver
    let e_match = fields
        .electric
        .iter()
        .zip(solver.em_fields().electric.iter())
        .all(|(a, b)| a == b);
    assert!(e_match, "electric fields must match solver cache");
    let m_match = fields
        .magnetic
        .iter()
        .zip(solver.em_fields().magnetic.iter())
        .all(|(a, b)| a == b);
    assert!(m_match, "magnetic fields must match solver cache");
}

#[test]
fn boundary_conditions_repair_mismatched_output_shape() {
    let mut solver = seeded_solver();
    let mut fields = output_fields(&[1, 1, 1, 3]);

    solver.apply_em_boundary_conditions(&mut fields);

    assert_eq!(fields.electric.shape(), &[4, 4, 4, 3]);
    assert_eq!(fields.magnetic.shape(), &[4, 4, 4, 3]);
    assert_eq!(*fields.electric.get(&[1, 1, 1, 2]).unwrap(), 12.0);
    assert_eq!(*fields.magnetic.get(&[1, 1, 1, 2]).unwrap(), 7.0);
    fields.validate_shapes().unwrap();
}
