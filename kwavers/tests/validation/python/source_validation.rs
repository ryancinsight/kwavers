//! Real source validation backed by pykwavers parity tests.

use std::sync::OnceLock;

use crate::python_validation::{
    run_pytest_validation, PytestValidationTarget, PythonValidationResult,
};

pub fn generate_source_validation_report() -> Vec<PythonValidationResult> {
    static RESULTS: OnceLock<Vec<PythonValidationResult>> = OnceLock::new();

    RESULTS
        .get_or_init(|| {
            [
                PytestValidationTarget {
                    test_name: "source_creation_surface",
                    args: &["pykwavers/tests/test_source_parity.py::TestSourceCreation"],
                },
                PytestValidationTarget {
                    test_name: "source_mask_injection",
                    args: &["pykwavers/tests/test_source_parity.py::TestMaskSourceSignalMatching"],
                },
                PytestValidationTarget {
                    test_name: "source_initial_pressure_and_velocity",
                    args: &[
                        "pykwavers/tests/test_source_parity.py::TestInitialPressureSource",
                        "pykwavers/tests/test_source_parity.py::TestVelocitySource",
                    ],
                },
            ]
            .into_iter()
            .map(|target| run_pytest_validation(&target))
            .collect()
        })
        .clone()
}
