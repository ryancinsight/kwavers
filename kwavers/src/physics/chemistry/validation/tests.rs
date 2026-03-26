use super::arrhenius::ArrheniusValidator;
use super::kinetics_database::ValidatedKinetics;
use super::literature::LiteratureValue;

#[test]
fn test_literature_value_creation() {
    let lit = LiteratureValue::new(5.5e9, 1e9);
    assert_eq!(lit.nominal, 5.5e9);
    assert!(lit.min < lit.nominal);
    assert!(lit.max > lit.nominal);
}

#[test]
fn test_literature_value_from_range() {
    let lit = LiteratureValue::from_range(4.5e9, 5.5e9);
    assert_eq!(lit.nominal, 5.0e9);
    assert_eq!(lit.min, 4.5e9);
    assert_eq!(lit.max, 5.5e9);
}

#[test]
fn test_within_range_check() {
    let lit = LiteratureValue::from_range(1e7, 2e7);
    assert!(lit.is_within_range(1.5e7));
    assert!(!lit.is_within_range(0.5e7));
    assert!(!lit.is_within_range(3e7));
}

#[test]
fn test_percent_difference() {
    let lit = LiteratureValue::new(1e8, 1e7);
    let diff = lit.percent_difference(1.1e8);
    assert!((diff - 10.0).abs() < 0.1);
}

#[test]
fn test_validated_kinetics_oh_recombination() {
    let kinetics = ValidatedKinetics::new();
    let result = kinetics.validate("oh_recombination", 5.5e9);
    assert!(result.is_ok());
    let res = result.unwrap();
    assert!(res.within_range);
}

#[test]
fn test_validated_kinetics_out_of_range() {
    let kinetics = ValidatedKinetics::new();
    let result = kinetics.validate("oh_recombination", 1e10); // Too high
    assert!(result.is_ok());
    let res = result.unwrap();
    assert!(!res.within_range);
}

#[test]
fn test_validation_result_report() {
    let kinetics = ValidatedKinetics::new();
    let result = kinetics.validate("oh_recombination", 5.5e9).unwrap();
    let report = result.report();
    assert!(!report.is_empty());
    assert!(report.contains("Reaction:"));
    assert!(report.contains("Within range:"));
}

#[test]
fn test_arrhenius_temperature_dependence() {
    let validator = ArrheniusValidator::new(50000.0, 298.15); // 50 kJ/mol

    let k_298 = validator.rate_constant_at_temperature(1e8, 298.15);
    let k_308 = validator.rate_constant_at_temperature(1e8, 308.15);

    assert!(k_308 > k_298);
}

#[test]
fn test_q10_factor() {
    let validator = ArrheniusValidator::new(50000.0, 298.15);
    let q10 = validator.q10_factor(298.15);

    assert!(q10 > 1.0);
    assert!(q10 < 10.0);
}

#[test]
fn test_q10_reasonableness() {
    let validator = ArrheniusValidator::new(50000.0, 298.15);
    assert!(validator.is_reasonable_q10(298.15));
}

#[test]
fn test_high_activation_energy_q10() {
    let validator = ArrheniusValidator::new(100000.0, 298.15); // 100 kJ/mol
    let q10 = validator.q10_factor(298.15);

    let validator_low = ArrheniusValidator::new(20000.0, 298.15);
    let q10_low = validator_low.q10_factor(298.15);

    assert!(q10 > q10_low);
}

#[test]
fn test_kinetics_database_completeness() {
    let kinetics = ValidatedKinetics::new();

    assert!(kinetics.validate("oh_recombination", 5e9).is_ok());
    assert!(kinetics.validate("superoxide_dismutation", 1.5e8).is_ok());
    assert!(kinetics.validate("peroxide_hydroxyl", 2.7e7).is_ok());
    assert!(kinetics.validate("ozone_hydroxyl", 1e8).is_ok());
}

#[test]
fn test_unknown_reaction() {
    let kinetics = ValidatedKinetics::new();
    let result = kinetics.validate("unknown_reaction", 1e8);
    assert!(result.is_err());
}

#[test]
fn test_case_insensitivity() {
    let kinetics = ValidatedKinetics::new();
    let result1 = kinetics.validate("OH_RECOMBINATION", 5.5e9);
    let result2 = kinetics.validate("oh_recombination", 5.5e9);

    assert!(result1.is_ok());
    assert!(result2.is_ok());
    assert_eq!(result1.unwrap().within_range, result2.unwrap().within_range);
}
