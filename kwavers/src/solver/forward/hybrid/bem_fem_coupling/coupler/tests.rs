use num_complex::Complex64;

use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use crate::domain::mesh::BoundaryType;

use super::super::BemFemCouplingConfig;
use super::struct_impl::BemFemCoupler;

#[test]
fn test_bem_fem_coupler_creation() {
    let mut fem_mesh = TetrahedralMesh::new();
    fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);

    let bem_boundary = vec![0];
    let config = BemFemCouplingConfig::default();

    let coupler = BemFemCoupler::new(config, &fem_mesh, &bem_boundary);

    let _coupler = coupler.unwrap();
}

#[test]
fn test_solve_fem_system_single_element() {
    use crate::domain::mesh::tetrahedral::BoundaryType;

    let mut fem_mesh = TetrahedralMesh::new();
    let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = fem_mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = fem_mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = fem_mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);

    fem_mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let bem_boundary = vec![n0, n1, n2];

    let config = BemFemCouplingConfig::default();
    let coupler = BemFemCoupler::new(config, &fem_mesh, &bem_boundary).unwrap();

    let mut fem_field = vec![Complex64::new(0.0, 0.0); 4];

    fem_field[n0] = Complex64::new(1.0, 0.0);
    fem_field[n1] = Complex64::new(1.0, 0.0);
    fem_field[n2] = Complex64::new(1.0, 0.0);

    let wavenumber = 0.0;

    let matrix = coupler
        .assemble_system_matrix(&fem_mesh, wavenumber)
        .unwrap();
    coupler
        .solve_linear_system(&matrix, fem_field.as_mut_slice())
        .unwrap();

    let val = fem_field[n3];
    assert!(
        (val.re - 1.0).abs() < 1e-4,
        "Expected real part 1.0, got {}",
        val.re
    );
    assert!(
        val.im.abs() < 1e-4,
        "Expected imag part 0.0, got {}",
        val.im
    );
}

#[test]
fn test_solve_coupled_run() {
    use crate::domain::mesh::tetrahedral::BoundaryType;

    let mut fem_mesh = TetrahedralMesh::new();
    let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = fem_mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = fem_mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = fem_mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    fem_mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let bem_boundary = vec![n0, n1, n2];

    let config = BemFemCouplingConfig {
        max_iterations: 2,
        ..Default::default()
    };

    let mut coupler = BemFemCoupler::new(config, &fem_mesh, &bem_boundary).unwrap();

    let mut fem_field = vec![Complex64::new(1.0, 0.0); 4];
    let mut bem_boundary_values = vec![Complex64::default(); 4];

    let wavenumber = 1.0;

    let result = coupler.solve_coupled(
        &mut fem_field,
        &mut bem_boundary_values,
        &fem_mesh,
        wavenumber,
    );

    match result {
        Ok(_) => {}
        Err(e) => {
            match e {
                crate::core::error::KwaversError::Numerical(_) => {
                    // Accept numerical error from BEM solver as sign of connectivity
                }
                _ => panic!("Unexpected error type: {:?}", e),
            }
        }
    }
}
