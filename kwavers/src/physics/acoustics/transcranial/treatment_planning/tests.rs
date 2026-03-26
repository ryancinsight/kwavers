use super::*;
use crate::domain::grid::Grid;
use ndarray::Array3;

#[test]
fn test_treatment_planner_creation() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let ct_data = Array3::from_elem((64, 64, 64), 0.0); // Air

    let planner = TreatmentPlanner::new(&grid, &ct_data);
    assert!(planner.is_ok());
}

#[test]
fn test_treatment_plan_generation() {
    let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
    let ct_data = Array3::from_elem((32, 32, 32), 400.0); // Bone-like HU
    let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

    let target = TargetVolume {
        center: [0.016, 0.016, 0.016],
        dimensions: [0.004, 0.004, 0.004],
        shape: TargetShape::Ellipsoidal,
        priority: 8,
        max_temperature: 45.0,
        required_intensity: 100.0,
    };

    let spec = TransducerSpecification::default();
    let plan = planner.generate_plan("test_patient", &[target], &spec);

    assert!(plan.is_ok());
}

#[test]
fn test_skull_properties_analysis() {
    let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
    let ct_data = Array3::from_elem((16, 16, 16), 800.0); // Dense bone
    let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

    let properties = planner.analyze_skull_properties();
    assert!(properties.is_ok());

    let props = properties.unwrap();
    assert_eq!(props.sound_speed.dim(), (16, 16, 16));
    assert_eq!(props.density.dim(), (16, 16, 16));
    assert_eq!(props.attenuation.dim(), (16, 16, 16));
}
