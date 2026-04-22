//! Real grid validation backed by pykwavers parity tests.

use std::sync::OnceLock;

use crate::python_validation::{
    run_pytest_validation, PytestValidationTarget, PythonValidationResult,
};

pub fn generate_grid_validation_report() -> Vec<PythonValidationResult> {
    static RESULTS: OnceLock<Vec<PythonValidationResult>> = OnceLock::new();

    RESULTS
        .get_or_init(|| {
            [
                PytestValidationTarget {
                    test_name: "grid_parity_suite",
                    args: &["pykwavers/tests/test_grid_parity.py"],
                },
                PytestValidationTarget {
                    test_name: "grid_initialization_basic",
                    args: &["pykwavers/tests/test_basic.py::test_grid_creation"],
                },
            ]
            .into_iter()
            .map(|target| run_pytest_validation(&target))
            .collect()
        })
        .clone()
}
