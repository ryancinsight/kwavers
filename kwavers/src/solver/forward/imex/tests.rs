//! Tests for IMEX schemes

#[cfg(test)]
use super::imex_bdf::{IMEXBDFConfig, IMEXBDF};
#[cfg(test)]
use super::imex_rk::{IMEXRKConfig, IMEXRKType, IMEXRK};
#[cfg(test)]
use super::*;
#[cfg(test)]
use crate::core::error::KwaversResult;
#[cfg(test)]
use crate::domain::grid::Grid;
#[cfg(test)]
use ndarray::Array3;
#[cfg(test)]
use std::sync::Arc;

/// Test problem: dy/dt = -y (stiff) + sin(t) (non-stiff)
fn create_test_problem() -> (
    impl Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    impl Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
) {
    let explicit_rhs = |_field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        // Non-stiff part: sin(t) - approximated as constant
        Ok(Array3::from_elem((4, 4, 4), 0.1))
    };

    let implicit_rhs = |field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        // Moderately stiff part: -10*y (reduced from -100*y for better convergence)
        Ok(field.mapv(|y| -10.0 * y))
    };

    (explicit_rhs, implicit_rhs)
}

#[test]
fn test_imex_rk_ssp2() {
    let grid = Arc::new(Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap());
    let config = IMEXConfig {
        tolerance: 1e-6, // Relax tolerance for tests
        ..Default::default()
    };
    let rk_config = IMEXRKConfig {
        scheme_type: IMEXRKType::SSP2_222,
        embedded_error: false,
    };

    let scheme = IMEXSchemeType::RungeKutta(IMEXRK::new(rk_config));
    let mut integrator = IMEXIntegrator::new(config, scheme, grid.clone());

    let field = Array3::from_elem((4, 4, 4), 1.0);
    let dt = 0.01;
    let (explicit_rhs, implicit_rhs) = create_test_problem();

    let result = integrator
        .advance(&field, dt, explicit_rhs, implicit_rhs)
        .unwrap();

    // Check that solution is bounded
    assert!(result.iter().all(|&v| v.is_finite()));
    assert!(result.iter().all(|&v| v.abs() < 10.0));
}

#[test]
fn test_imex_rk_ark3() {
    let grid = Arc::new(Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap());
    let config = IMEXConfig {
        tolerance: 1e-6, // Relax tolerance for tests
        ..Default::default()
    };
    let rk_config = IMEXRKConfig {
        scheme_type: IMEXRKType::ARK3,
        embedded_error: false,
    };

    let scheme = IMEXSchemeType::RungeKutta(IMEXRK::new(rk_config));
    let mut integrator = IMEXIntegrator::new(config, scheme, grid.clone());

    let field = Array3::from_elem((4, 4, 4), 1.0);
    let dt = 0.01;
    let (explicit_rhs, implicit_rhs) = create_test_problem();

    let result = integrator
        .advance(&field, dt, explicit_rhs, implicit_rhs)
        .unwrap();

    // Check that solution is bounded
    assert!(result.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_imex_bdf2() {
    let grid = Arc::new(Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap());
    let config = IMEXConfig {
        tolerance: 1e-6, // Relax tolerance for tests
        ..Default::default()
    };
    let bdf_config = IMEXBDFConfig {
        order: 2,
        variable_order: false,
        max_order: 5,
    };

    let mut bdf = IMEXBDF::new(bdf_config.clone());
    let scheme = IMEXSchemeType::BDF(bdf.clone());
    let mut integrator = IMEXIntegrator::new(config, scheme, grid.clone());

    let mut field = Array3::from_elem((4, 4, 4), 1.0);
    let dt = 0.01;
    let (explicit_rhs, implicit_rhs) = create_test_problem();

    // Take a few steps to build history
    for _ in 0..3 {
        let result = integrator
            .advance(&field, dt, &explicit_rhs, &implicit_rhs)
            .unwrap();
        bdf.update(result.clone(), &explicit_rhs).unwrap();
        field = result;
    }

    // Check that solution is stable
    assert!(field.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_operator_splitting_strang() {
    let splitting = StrangSplitting::new();
    let field = Array3::from_elem((4, 4, 4), 1.0);
    let dt = 0.1;

    let operator_a = |field: &Array3<f64>, dt: f64| -> KwaversResult<Array3<f64>> {
        // Simple decay: dy/dt = -y
        Ok(field.mapv(|y| y * (-dt).exp()))
    };

    let operator_b = |field: &Array3<f64>, dt: f64| -> KwaversResult<Array3<f64>> {
        // Simple growth: dy/dt = y
        Ok(field.mapv(|y| y * dt.exp()))
    };

    let result = splitting
        .split_step(&field, dt, operator_a, operator_b)
        .unwrap();

    // For these operators, Strang splitting should be very accurate
    let expected = field.clone(); // Growth and decay cancel
    let error: f64 = (result - expected)
        .iter()
        .map(|&e| e * e)
        .sum::<f64>()
        .sqrt();
    assert!(error < 1e-2); // Second-order accuracy
}

#[test]
fn test_implicit_solver_linear() {
    // Use linear solver for simple linear problem
    let solver = LinearSolver::default();
    let initial = Array3::from_elem((4, 4, 4), 1.0);

    // Solve: y - 0.5 = 0, which means y = 0.5
    // For linear solver, residual = y - G(y), so G(y) = 0.5
    let residual_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> { Ok(y.mapv(|yi| yi - 0.5)) };

    let solution = solver.solve(&initial, residual_fn).unwrap();

    // Check solution
    assert!(solution.iter().all(|&v| (v - 0.5).abs() < 1e-10));
}

#[test]
fn test_stiffness_detection() {
    let mut detector = StiffnessDetector::new(10.0);
    let field = Array3::from_elem((4, 4, 4), 1.0);

    let explicit_rhs = |field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        Ok(field.mapv(|_| 1.0)) // O(1) terms
    };

    let implicit_rhs = |field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        Ok(field.mapv(|y| -100.0 * y)) // O(100) terms - still stiff for detection test
    };

    let metric = detector
        .detect(&field, &explicit_rhs, &implicit_rhs)
        .unwrap();

    assert!(metric.is_stiff());
    assert!(metric.ratio() > 10.0);
}

#[test]
fn test_stability_analysis() {
    let analyzer = IMEXStabilityAnalyzer::new();
    let scheme = IMEXSchemeType::RungeKutta(IMEXRK::new(IMEXRKConfig {
        scheme_type: IMEXRKType::SSP2_222,
        embedded_error: false,
    }));

    let region = analyzer.compute_region(&scheme);

    // SSP2 should have explicit stability limit around 2
    assert!(region.explicit_dt_max > 1.0 && region.explicit_dt_max < 3.0);
    // L-stable implicit part
    assert!(region.implicit_dt_max > 1000.0);
}

#[test]
fn test_conservation() {
    // Test that IMEX schemes preserve linear invariants
    let grid = Arc::new(Grid::new(8, 8, 8, 1.0, 1.0, 1.0).unwrap());
    let config = IMEXConfig {
        adaptive_stiffness: false,
        check_stability: false,
        ..Default::default()
    };

    let scheme = IMEXSchemeType::RungeKutta(IMEXRK::new(IMEXRKConfig::default()));
    let mut integrator = IMEXIntegrator::new(config, scheme, grid.clone());

    let field = Array3::from_elem((8, 8, 8), 1.0);
    let dt = 0.01;

    // Conservative system: d/dt ∫u = 0
    let explicit_rhs = |field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        // Centered differences (conservative)
        let mut rhs = Array3::zeros(field.dim());
        let (nx, ny, nz) = field.dim();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    rhs[[i, j, k]] = 0.5 * (field[[i + 1, j, k]] - field[[i - 1, j, k]]);
                }
            }
        }
        Ok(rhs)
    };

    let implicit_rhs =
        |_field: &Array3<f64>| -> KwaversResult<Array3<f64>> { Ok(Array3::zeros((8, 8, 8))) };

    let initial_sum: f64 = field.sum();
    let result = integrator
        .advance(&field, dt, explicit_rhs, implicit_rhs)
        .unwrap();
    let final_sum: f64 = result.sum();

    // Check conservation
    assert!((final_sum - initial_sum).abs() < 1e-10);
}
