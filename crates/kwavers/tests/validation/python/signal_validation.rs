//! Real signal validation backed by pykwavers parity tests.

use std::sync::OnceLock;

use crate::python_validation::{
    run_pytest_validation, PytestValidationTarget, PythonValidationResult,
};

pub fn generate_signal_validation_report() -> Vec<PythonValidationResult> {
    static RESULTS: OnceLock<Vec<PythonValidationResult>> = OnceLock::new();

    RESULTS
        .get_or_init(|| {
            [
                PytestValidationTarget {
                    test_name: "signal_generation_parity",
                    args: &["pykwavers/tests/test_source_parity.py::TestSignalGenerationParityWithKWave"],
                },
                PytestValidationTarget {
                    test_name: "tone_burst_parity",
                    args: &["pykwavers/tests/test_solver_parity.py::TestToneBurstParity"],
                },
                PytestValidationTarget {
                    test_name: "signal_generation",
                    args: &["pykwavers/tests/test_source_parity.py::TestSignalGeneration"],
                },
            ]
            .into_iter()
            .map(|target| run_pytest_validation(&target))
            .collect()
        })
        .clone()
}
