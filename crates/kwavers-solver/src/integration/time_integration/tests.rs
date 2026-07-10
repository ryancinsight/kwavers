//! Value-semantic tests for multi-rate time integration.
//!
//! The test equations use analytical references:
//! - `du/dt = lambda u` for one-step RK4 stability-polynomial validation.
//! - `du/dt = c` for Adams-Bashforth constant-derivative exactness.
//! - CFL and subcycle formulas from the production controllers.

use super::multi_rate_controller::MultiRateController;
use super::stability::{CFLCondition, StabilityAnalyzer};
use super::time_scale_separation::TimeScaleSeparator;
use super::time_stepper::{AdamsBashforth, AdamsBashforthConfig, RK4Config, RungeKutta4};
use super::traits::{MultiRateConfig, TimeStepper};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::{Array3, Array4};
use std::collections::HashMap;

#[test]
fn rk4_matches_fourth_order_stability_polynomial_for_linear_growth() -> KwaversResult<()> {
    let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0)?;
    let mut stepper = RungeKutta4::new(RK4Config::default());
    let mut field = Array3::from_elem((1, 1, 1), 1.0);
    let lambda = -2.0_f64;
    let dt = 0.1_f64;

    stepper.step(
        &mut field,
        |u| Ok(u.mapv(|value| lambda * value)),
        dt,
        &grid,
    )?;

    let z = lambda * dt;
    let expected = 1.0 + z + z.powi(2) / 2.0 + z.powi(3) / 6.0 + z.powi(4) / 24.0;
    assert!(
        (field[[0, 0, 0]] - expected).abs() < 1e-15,
        "RK4 one-step value {}, expected stability polynomial {}",
        field[[0, 0, 0]],
        expected
    );
    Ok(())
}

#[test]
fn adams_bashforth2_is_exact_for_constant_derivative_after_startup() -> KwaversResult<()> {
    let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0)?;
    let mut stepper = AdamsBashforth::new(AdamsBashforthConfig {
        order: 2,
        startup_steps: 1,
    });
    let mut field = Array3::from_elem((1, 1, 1), 3.0);
    let derivative = 2.5_f64;
    let dt = 0.2_f64;

    for _ in 0..3 {
        stepper.step(
            &mut field,
            |u| Ok(Array3::from_elem(u.shape(), derivative)),
            dt,
            &grid,
        )?;
    }

    let expected = 3.0 + 3.0 * dt * derivative;
    assert!(
        (field[[0, 0, 0]] - expected).abs() < 1e-15,
        "AB2 constant-derivative value {}, expected {}",
        field[[0, 0, 0]],
        expected
    );
    Ok(())
}

#[test]
fn adams_bashforth3_is_exact_for_constant_derivative_after_startup() -> KwaversResult<()> {
    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0)?;
    let mut stepper = AdamsBashforth::new(AdamsBashforthConfig {
        order: 3,
        startup_steps: 2,
    });
    let mut field = Array3::from_elem((2, 2, 2), 3.0);
    let derivative = -1.25_f64;
    let dt = 0.2_f64;

    for _ in 0..4 {
        stepper.step(
            &mut field,
            |u| Ok(Array3::from_elem(u.shape(), derivative)),
            dt,
            &grid,
        )?;
    }

    let expected = 3.0 + 4.0 * dt * derivative;
    assert!(
        field.iter().all(|&value| (value - expected).abs() < 1e-15),
        "AB3 constant-derivative field {field:?}, expected all {expected}"
    );
    Ok(())
}

#[test]
fn stability_analyzer_uses_acoustic_and_diffusion_bounds() -> KwaversResult<()> {
    let grid = Grid::new(4, 4, 4, 0.002, 0.001, 0.004)?;
    let analyzer = StabilityAnalyzer::new(0.5);
    let field = Array3::from_elem((4, 4, 4), 1.0);
    let constraints = HashMap::from([
        ("max_wave_speed".to_string(), 2_000.0),
        ("diffusion_coefficient".to_string(), 1.25e-3),
    ]);

    let dt = analyzer.compute_stable_dt_from_constraints(&field, &grid, &constraints)?;

    let dx_min = 0.001_f64;
    let acoustic_dt = 0.5 * dx_min / 2_000.0;
    let diffusion_dt = 0.5 * dx_min * dx_min / (2.0 * 1.25e-3);
    let expected = acoustic_dt.min(diffusion_dt);
    assert_eq!(dt, expected);
    Ok(())
}

#[test]
fn cfl_condition_reports_value_contract() -> KwaversResult<()> {
    let grid = Grid::new(2, 2, 2, 0.001, 0.002, 0.003)?;
    let condition = CFLCondition::new(2.0e-7, 1_500.0, &grid, 0.5);

    assert_eq!(condition.min_dx, 0.001);
    assert_eq!(condition.cfl_number, 0.3);
    assert_eq!(condition.max_dt, 0.5 * 0.001 / 1_500.0);
    assert!(condition.is_stable);
    assert!(condition.report().contains("stable=true"));
    Ok(())
}

#[test]
fn multi_rate_controller_selects_slowest_global_step_and_fast_subcycles() -> KwaversResult<()> {
    let mut controller = MultiRateController::new(MultiRateConfig {
        max_subcycles: 8,
        min_dt: 1.0e-9,
        ..Default::default()
    });
    let component_dt = HashMap::from([
        ("acoustic".to_string(), 1.0e-6),
        ("thermal".to_string(), 5.0e-6),
        ("chemical".to_string(), 2.0e-6),
    ]);

    let (global_dt, subcycles) = controller.determine_time_steps(&component_dt, 10.0e-6)?;

    assert_eq!(global_dt, 5.0e-6);
    assert_eq!(subcycles["thermal"], 1);
    assert_eq!(subcycles["chemical"], 3);
    assert_eq!(subcycles["acoustic"], 5);
    assert_eq!(controller.total_steps(), 1);
    assert_eq!(controller.subcycle_counts()["acoustic"], 5);
    assert_eq!(controller.efficiency_ratio(), 15.0 / 9.0);
    Ok(())
}

#[test]
fn time_scale_separator_matches_quadratic_closed_form() -> KwaversResult<()> {
    let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
    let mut fields = Array4::zeros((1, 5, 5, 5));

    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                fields[[0, i, j, k]] = (i * i + j * j + k * k) as f64;
            }
        }
    }

    let mut separator = TimeScaleSeparator::new(&grid);
    let scales = separator.analyze(&fields, 1e-12)?;

    let grad_max = 6.0 * 3.0_f64.sqrt();
    let laplacian_max = 6.0_f64;
    let expected_diffusive = 1.0 / grad_max;
    let expected_acoustic = 1.0 / laplacian_max.sqrt();

    assert_eq!(scales.len(), 2);
    assert!(
        (scales[0] - expected_diffusive).abs() < 1e-15,
        "diffusive scale {}, expected {}",
        scales[0],
        expected_diffusive
    );
    assert!(
        (scales[1] - expected_acoustic).abs() < 1e-15,
        "acoustic scale {}, expected {}",
        scales[1],
        expected_acoustic
    );
    assert!(!separator.is_stiff());
    Ok(())
}

#[test]
fn time_scale_separator_handles_domains_without_central_stencil() -> KwaversResult<()> {
    let grid = Grid::new(2, 3, 3, 1.0, 1.0, 1.0)?;
    let fields = Array4::from_elem((2, 2, 3, 3), 7.0);
    let mut separator = TimeScaleSeparator::new(&grid);

    let scales = separator.analyze(&fields, 1e-12)?;

    assert!(scales.is_empty());
    assert!(!separator.is_stiff());
    Ok(())
}
