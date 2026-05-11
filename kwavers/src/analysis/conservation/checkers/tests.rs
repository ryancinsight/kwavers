//! Unit tests for `ConservationChecker` and associated types.

use std::collections::HashMap;

use ndarray::Array3;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

use super::checker::ConservationChecker;
use super::types::ConservationLaw;

#[test]
fn test_conservation_checker_creation() -> KwaversResult<()> {
    let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1)?;
    let checker = ConservationChecker::new(grid, 1e-6);

    assert_eq!(checker.check_count, 0);

    Ok(())
}

#[test]
fn test_conservation_law_display() {
    assert_eq!(ConservationLaw::Mass.to_string(), "Mass");
    assert_eq!(ConservationLaw::Momentum.to_string(), "Momentum");
    assert_eq!(ConservationLaw::Energy.to_string(), "Energy");
    assert_eq!(ConservationLaw::Charge.to_string(), "Charge");
}

#[test]
fn test_initialization() -> KwaversResult<()> {
    let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let mut checker = ConservationChecker::new(grid, 1e-6);

    let mut fields = HashMap::new();
    fields.insert("pressure".to_string(), Array3::ones((8, 8, 8)));

    let result = checker.initialize(&fields, &["pressure"])?;
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("pressure"));

    Ok(())
}

#[test]
fn test_perfect_conservation() -> KwaversResult<()> {
    let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let mut checker = ConservationChecker::new(grid, 1e-10);

    let mut fields = HashMap::new();
    let pressure = Array3::ones((8, 8, 8));
    fields.insert("pressure".to_string(), pressure.clone());

    checker.initialize(&fields, &["pressure"])?;

    let results = checker.check(&fields, 0.0)?;
    let pressure_result = results.get("pressure").unwrap();

    assert_eq!(pressure_result.relative_error, 0.0);
    assert!(pressure_result.passed);

    Ok(())
}

#[test]
fn test_conservation_violation_detection() -> KwaversResult<()> {
    let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let mut checker = ConservationChecker::new(grid, 1e-10);

    let mut fields = HashMap::new();
    fields.insert("pressure".to_string(), Array3::ones((8, 8, 8)));

    checker.initialize(&fields, &["pressure"])?;

    let mut pressure_modified = Array3::ones((8, 8, 8));
    pressure_modified[[0, 0, 0]] = 1.5;
    fields.insert("pressure".to_string(), pressure_modified);

    let results = checker.check(&fields, 0.0)?;
    let pressure_result = results.get("pressure").unwrap();

    assert!(pressure_result.relative_error > 0.0);
    assert!(!pressure_result.passed);
    let msg = pressure_result
        .error_message
        .as_deref()
        .expect("error_message must be Some when check fails");
    assert!(!msg.is_empty(), "error_message must be non-empty when check fails");

    Ok(())
}

#[test]
fn test_infer_law_from_name() -> KwaversResult<()> {
    let grid = Grid::new(8, 8, 8, 0.1, 0.1, 0.1)?;
    let checker = ConservationChecker::new(grid, 1e-6);

    assert_eq!(checker.infer_law_from_name("pressure"),    ConservationLaw::Charge);
    assert_eq!(checker.infer_law_from_name("density"),     ConservationLaw::Mass);
    assert_eq!(checker.infer_law_from_name("velocity_x"),  ConservationLaw::Momentum);
    assert_eq!(checker.infer_law_from_name("thermal_energy"), ConservationLaw::Energy);

    Ok(())
}
