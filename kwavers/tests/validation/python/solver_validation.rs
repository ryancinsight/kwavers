//! Real solver validation backed by pykwavers solver tests.

use std::sync::OnceLock;

use crate::python_validation::{
    run_pytest_validation, PytestValidationTarget, PythonValidationResult,
};

pub fn generate_solver_validation_report() -> Vec<PythonValidationResult> {
    static RESULTS: OnceLock<Vec<PythonValidationResult>> = OnceLock::new();

    RESULTS
        .get_or_init(|| {
            [
                PytestValidationTarget {
                    test_name: "solver_cpu_surface",
                    args: &[
                        "pykwavers/tests/test_bindings_surface.py::test_simulation_cpu_surface_runs_end_to_end",
                    ],
                },
                PytestValidationTarget {
                    test_name: "solver_fdtd_pstd_nonzero",
                    args: &[
                        "pykwavers/tests/test_solver_parity.py::TestSolverConsistency",
                        "-k",
                        "not correlated",
                    ],
                },
                PytestValidationTarget {
                    test_name: "solver_heterogeneous_and_absorption",
                    args: &[
                        "pykwavers/tests/test_solver_parity.py::TestHeterogeneousMediumSolver",
                        "pykwavers/tests/test_solver_parity.py::TestAbsorbingMediumSolver",
                    ],
                },
                PytestValidationTarget {
                    test_name: "solver_convergence_and_p0",
                    args: &[
                        "pykwavers/tests/test_solver_parity.py::TestSolverConvergence",
                        "pykwavers/tests/test_solver_parity.py::TestP0InitialPressureParity",
                    ],
                },
            ]
            .into_iter()
            .map(|target| run_pytest_validation(&target))
            .collect()
        })
        .clone()
}
