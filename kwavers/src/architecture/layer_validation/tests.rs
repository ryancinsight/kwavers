use super::types::{ArchitectureLayer, ViolationType};
use super::validator::ArchitectureValidator;

#[test]
fn test_layer_ordering() {
    assert!(ArchitectureLayer::can_depend_on(
        ArchitectureLayer::Solver,
        ArchitectureLayer::Physics
    ));
    assert!(!ArchitectureLayer::can_depend_on(
        ArchitectureLayer::Physics,
        ArchitectureLayer::Solver
    ));
}

#[test]
fn test_max_dependency() {
    assert_eq!(
        ArchitectureLayer::max_dependency(ArchitectureLayer::Solver),
        ArchitectureLayer::Physics
    );
    assert_eq!(
        ArchitectureLayer::max_dependency(ArchitectureLayer::Clinical),
        ArchitectureLayer::Simulation
    );
}

#[test]
fn test_valid_architecture() {
    let mut validator = ArchitectureValidator::new();

    validator.register_module("core::error", ArchitectureLayer::Core);
    validator.register_module("math::linear_algebra", ArchitectureLayer::Math);
    validator.register_module("solver::fdtd", ArchitectureLayer::Solver);

    validator.add_dependency("math::linear_algebra", "core::error");
    validator.add_dependency("solver::fdtd", "math::linear_algebra");
    validator.add_dependency("solver::fdtd", "core::error");

    let result = validator.validate().unwrap();
    assert!(result.passed);
    assert_eq!(result.violations.len(), 0);
}

#[test]
fn test_upward_dependency_violation() {
    let mut validator = ArchitectureValidator::new();

    validator.register_module("core::error", ArchitectureLayer::Core);
    validator.register_module("solver::fdtd", ArchitectureLayer::Solver);

    validator.add_dependency("core::error", "solver::fdtd");

    let result = validator.validate().unwrap();
    assert!(!result.passed);
    assert_eq!(result.violations.len(), 1);
    assert_eq!(
        result.violations[0].violation_type,
        ViolationType::UpwardDependency
    );
}

#[test]
fn test_validation_report() {
    let mut validator = ArchitectureValidator::new();
    validator.register_module("core::error", ArchitectureLayer::Core);
    validator.register_module("solver::fdtd", ArchitectureLayer::Solver);
    validator.add_dependency("core::error", "solver::fdtd");

    let result = validator.validate().unwrap();
    let report = validator.report(&result);

    assert!(report.contains("VALIDATION REPORT"));
    assert!(report.contains("FAILED"));
    assert!(report.contains("Upward Dependency"));
}

#[test]
fn test_layer_display() {
    assert_eq!(ArchitectureLayer::Core.as_str(), "Core (Layer 0)");
    assert_eq!(ArchitectureLayer::Solver.as_str(), "Solver (Layer 4)");
    assert_eq!(
        ArchitectureLayer::Infrastructure.as_str(),
        "Infrastructure (Layer 8)"
    );
}
