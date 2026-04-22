//! Real sensor validation backed by pykwavers parity tests.

use std::sync::OnceLock;

use crate::python_validation::{
    run_pytest_validation, PytestValidationTarget, PythonValidationResult,
};

pub fn generate_sensor_validation_report() -> Vec<PythonValidationResult> {
    static RESULTS: OnceLock<Vec<PythonValidationResult>> = OnceLock::new();

    RESULTS
        .get_or_init(|| {
            [
                PytestValidationTarget {
                    test_name: "sensor_creation_surface",
                    args: &["pykwavers/tests/test_sensor_parity.py::TestSensorCreation"],
                },
                PytestValidationTarget {
                    test_name: "sensor_recording_contract",
                    args: &["pykwavers/tests/test_sensor_parity.py::TestSensorRecording"],
                },
                PytestValidationTarget {
                    test_name: "sensor_statistics_and_arrays",
                    args: &[
                        "pykwavers/tests/test_sensor_parity.py::TestArraySensor",
                        "pykwavers/tests/test_sensor_parity.py::TestRecordingModes",
                    ],
                },
            ]
            .into_iter()
            .map(|target| run_pytest_validation(&target))
            .collect()
        })
        .clone()
}
