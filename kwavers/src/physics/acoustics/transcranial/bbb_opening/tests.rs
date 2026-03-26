use super::simulator::BBBOpening;
use super::types::BBBParameters;
use ndarray::Array3;

#[test]
fn test_bbb_opening_creation() {
    let pressure = Array3::from_elem((8, 8, 8), 1e5); // 100 kPa
    let bubbles = Array3::from_elem((8, 8, 8), 1e7); // 10^7 bubbles/mL
    let params = BBBParameters::default();

    let bbb = BBBOpening::new(pressure, bubbles, params);
    assert_eq!(bbb.acoustic_pressure.dim(), (8, 8, 8));
}

#[test]
fn test_permeability_calculation() {
    let pressure = Array3::from_elem((4, 4, 4), 3e5); // 0.3 MPa
    let bubbles = Array3::from_elem((4, 4, 4), 5e7); // Optimal concentration
    let params = BBBParameters::default();

    let mut bbb = BBBOpening::new(pressure, bubbles, params);
    let result = bbb.simulate_opening();

    assert!(result.is_ok());
    assert!(bbb
        .permeability
        .permeability_factor
        .iter()
        .any(|&x| x > 1.0));
}

#[test]
fn test_mechanical_index_calculation() {
    use super::models::PermeabilityModels;
    let _pressure = Array3::from_elem((4, 4, 4), 1e5);
    let _bubbles = Array3::from_elem((4, 4, 4), 1e7);
    let params = BBBParameters::default();

    let models = PermeabilityModels::new(&params);
    let mi = models.calculate_mechanical_index(1e5);
    assert!(mi > 0.05 && mi < 0.2);
}

#[test]
fn test_safety_validation() {
    let pressure = Array3::from_elem((4, 4, 4), 3e5); // Safe pressure
    let bubbles = Array3::from_elem((4, 4, 4), 5e7);
    let params = BBBParameters::default();

    let mut bbb = BBBOpening::new(pressure, bubbles, params);
    bbb.simulate_opening().unwrap();

    let validation = bbb.validate_safety();
    assert!(validation.is_safe || !validation.warnings.is_empty());
}

#[test]
fn test_treatment_protocol() {
    let pressure = Array3::from_elem((4, 4, 4), 1e5);
    let bubbles = Array3::from_elem((4, 4, 4), 1e7);
    let params = BBBParameters::default();

    let bbb = BBBOpening::new(pressure, bubbles, params);
    let protocol = bbb.generate_protocol();

    assert!(protocol.frequency > 0.0);
    assert!(protocol.duration > 0.0);
    assert!(!protocol.safety_checks.is_empty());
}
