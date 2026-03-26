use super::super::materials::EMMaterialDistribution;
use super::maxwell::ElectromagneticWaveEquation;
use crate::domain::field::EMFields;
use crate::physics::electromagnetic::equations::types::EMDimension;

// Mock implementation for testing traits
struct MockEMSolver {
    materials: EMMaterialDistribution,
    fields: EMFields,
}

impl MockEMSolver {
    fn new() -> Self {
        let vacuum_props =
            crate::domain::medium::properties::ElectromagneticPropertyData::vacuum();
        let materials = crate::physics::electromagnetic::equations::materials::EMMaterialUtils::create_uniform_distribution(&[10, 10], vacuum_props);
        let electric = ndarray::ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
        let magnetic = ndarray::ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
        let fields = EMFields::new(electric, magnetic);

        Self { materials, fields }
    }
}

impl ElectromagneticWaveEquation for MockEMSolver {
    fn em_dimension(&self) -> EMDimension {
        EMDimension::Two
    }
    fn material_properties(&self) -> &EMMaterialDistribution {
        &self.materials
    }
    fn em_fields(&self) -> &EMFields {
        &self.fields
    }

    fn step_maxwell(&mut self, _dt: f64) -> Result<(), String> {
        Ok(())
    }
    fn apply_em_boundary_conditions(&mut self, _fields: &mut EMFields) {}
    fn check_em_constraints(&self, _fields: &EMFields) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn test_wave_impedance_calculation() {
    let solver = MockEMSolver::new();
    let impedance = solver.wave_impedance();

    // Vacuum impedance should be approximately 377 Ω
    let vacuum_impedance = impedance.iter().next().unwrap();
    assert!((vacuum_impedance - 377.0).abs() < 1.0);
}

#[test]
fn test_skin_depth_insulator() {
    let solver = MockEMSolver::new();
    let skin_depth = solver.skin_depth(1e9); // 1 GHz

    // Insulator should have infinite skin depth
    assert!(skin_depth.iter().all(|&d| d.is_infinite()));
}
