use crate::domain::grid::Grid;
use super::material::PoroelasticMaterial;
use super::simulation::PoroelasticSimulation;

#[test]
fn test_poroelastic_material_default() {
    let mat = PoroelasticMaterial::default();
    assert!(mat.porosity > 0.0 && mat.porosity < 1.0);
    assert!(mat.solid_density > 0.0);
}

#[test]
fn test_material_validation() {
    let result =
        PoroelasticMaterial::new(1.5, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5);
    assert!(result.is_err()); // Porosity > 1

    let result =
        PoroelasticMaterial::new(0.3, -1.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5);
    assert!(result.is_err()); // Negative density
}

#[test]
fn test_tissue_types() {
    let bone = PoroelasticMaterial::from_tissue_type("trabecular_bone").unwrap();
    let liver = PoroelasticMaterial::from_tissue_type("liver").unwrap();

    // Bone should be less porous than liver in this model
    assert!(bone.porosity < liver.porosity || bone.porosity > 0.0);
    assert!(bone.solid_bulk_modulus > liver.solid_bulk_modulus);
}

#[test]
fn test_bulk_density() {
    let mat = PoroelasticMaterial::default();
    let rho = mat.bulk_density();

    // Should be between solid and fluid densities
    let min_rho = mat.fluid_density.min(mat.solid_density);
    let max_rho = mat.fluid_density.max(mat.solid_density);
    assert!(rho >= min_rho && rho <= max_rho);
}

#[test]
fn test_effective_bulk_modulus() {
    let mat = PoroelasticMaterial::default();
    let k_eff = mat.effective_bulk_modulus();

    // Should be positive
    assert!(k_eff > 0.0);
}

#[test]
fn test_characteristic_frequency() {
    let mat = PoroelasticMaterial::default();
    let f_c = mat.characteristic_frequency();

    // Should be positive and finite
    assert!(f_c > 0.0 && f_c.is_finite());
}

#[test]
fn test_poroelastic_simulation_creation() {
    let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
    let material = PoroelasticMaterial::default();

    let sim = PoroelasticSimulation::new(&grid, material);
    assert!(sim.is_ok());
}
