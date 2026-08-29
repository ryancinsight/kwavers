use super::*;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::Array3;

const GRID_SHAPE: [usize; 3] = [2, 3, 4];

fn solver_with_config(
    config: super::super::super::super::types::ElasticWaveConfig,
) -> ElasticWaveSolver {
    let grid = Grid::new(2, 3, 4, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
    let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    ElasticWaveSolver::new(&grid, &medium, config).expect("valid test solver")
}

fn base_config() -> super::super::super::super::types::ElasticWaveConfig {
    let mut sensor_mask = Array3::from_elem(GRID_SHAPE, false);
    sensor_mask[[0, 0, 0]] = true;
    super::super::super::super::types::ElasticWaveConfig {
        time_step: 1.0e-6,
        simulation_time: 1.0e-6,
        save_every: 1,
        pml_thickness: 1,
        sensor_mask: Some(sensor_mask),
        ..Default::default()
    }
}

fn invalid_force() -> ElasticBodyForceConfig {
    ElasticBodyForceConfig::GaussianImpulse {
        center_m: [0.0; 3],
        sigma_m: [0.0; 3],
        direction: [0.0; 3],
        t0_s: 0.0,
        sigma_t_s: 0.0,
        impulse_n_per_m3_s: 1.0,
    }
}

fn assert_field_unchanged(actual: &ElasticWaveField, expected: &ElasticWaveField) {
    assert_eq!(actual.ux, expected.ux);
    assert_eq!(actual.uy, expected.uy);
    assert_eq!(actual.uz, expected.uz);
    assert_eq!(actual.vx, expected.vx);
    assert_eq!(actual.vy, expected.vy);
    assert_eq!(actual.vz, expected.vz);
    assert_same_float(actual.time, expected.time);
}

fn assert_dimension_error(error: KwaversError, component: &str, actual_shape: [usize; 3]) {
    match error {
        KwaversError::Validation(ValidationError::DimensionMismatch { expected, actual }) => {
            assert_eq!(
                expected,
                format!("ElasticWaveField.{component} shape {GRID_SHAPE:?}")
            );
            assert_eq!(actual, format!("{actual_shape:?}"));
        }
        other => panic!("expected component dimension mismatch, got {other}"),
    }
}

fn invalid_value(error: KwaversError) -> (String, f64, String) {
    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => (parameter, value, reason),
        other => panic!("expected invalid-value error, got {other}"),
    }
}

fn assert_same_float(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual, expected);
    }
}

fn assert_preflight_invalid(
    solver: &mut ElasticWaveSolver,
    field: &ElasticWaveField,
    duration: f64,
    parameter: &str,
    value: f64,
    reason: &str,
) {
    let field_before = field.clone();
    let recorder_before = solver.extract_recorded_data();
    let error = solver
        .propagate(field, duration, Some(&invalid_force()))
        .expect_err("invalid propagation input must be rejected");
    let (actual_parameter, actual_value, actual_reason) = invalid_value(error);
    assert_eq!(actual_parameter, parameter);
    assert_same_float(actual_value, value);
    assert_eq!(actual_reason, reason);
    assert_field_unchanged(field, &field_before);
    assert_eq!(solver.extract_recorded_data(), recorder_before);
}

fn assert_history_entries_invalid(
    config: super::super::super::super::types::ElasticWaveConfig,
    parameter: &str,
    value: f64,
    reason: &str,
) {
    let solver = solver_with_config(config);
    let recorder_before = solver.extract_recorded_data();
    let displacement = Array3::zeros(GRID_SHAPE);
    let errors = [
        solver
            .propagate_waves(&displacement)
            .expect_err("invalid history input must be rejected"),
        solver
            .propagate_waves_with_body_force_only_override(Some(&invalid_force()))
            .expect_err("invalid override input must be rejected"),
    ];
    for error in errors {
        let (actual_parameter, actual_value, actual_reason) = invalid_value(error);
        assert_eq!(actual_parameter, parameter);
        assert_same_float(actual_value, value);
        assert_eq!(actual_reason, reason);
    }
    assert_eq!(solver.extract_recorded_data(), recorder_before);
}

type ReplaceComponent = fn(&mut ElasticWaveField, Array3<f64>);

fn replace_ux(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.ux = replacement;
}

fn replace_uy(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.uy = replacement;
}

fn replace_uz(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.uz = replacement;
}

fn replace_vx(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.vx = replacement;
}

fn replace_vy(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.vy = replacement;
}

fn replace_vz(field: &mut ElasticWaveField, replacement: Array3<f64>) {
    field.vz = replacement;
}

#[test]
fn propagate_rejects_every_malformed_component_before_mutation() {
    let components: [(&str, ReplaceComponent); 6] = [
        ("ux", replace_ux),
        ("uy", replace_uy),
        ("uz", replace_uz),
        ("vx", replace_vx),
        ("vy", replace_vy),
        ("vz", replace_vz),
    ];
    let malformed_shapes = [[4, 3, 2], [2, 3, 3], [2, 3, 5]];

    for (component, replace) in components {
        for shape in malformed_shapes {
            let mut solver = solver_with_config(base_config());
            let mut field = ElasticWaveField::new(2, 3, 4);
            replace(&mut field, Array3::zeros(shape));
            let field_before = field.clone();
            let recorder_before = solver.extract_recorded_data();

            let error = solver
                .propagate(&field, 1.0e-6, Some(&invalid_force()))
                .expect_err("malformed component must be rejected");

            assert_dimension_error(error, component, shape);
            assert_field_unchanged(&field, &field_before);
            assert_eq!(solver.extract_recorded_data(), recorder_before);
        }
    }
}

#[test]
fn all_propagation_entries_reject_invalid_durations() {
    for duration in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut direct_solver = solver_with_config(base_config());
        let field = ElasticWaveField::new(2, 3, 4);
        assert_preflight_invalid(
            &mut direct_solver,
            &field,
            duration,
            "duration_s",
            duration,
            "must be finite and greater than zero",
        );

        let mut config = base_config();
        config.simulation_time = duration;
        assert_history_entries_invalid(
            config,
            "duration_s",
            duration,
            "must be finite and greater than zero",
        );
    }
}

#[test]
fn configured_time_step_domain_is_validated() {
    let field = ElasticWaveField::new(2, 3, 4);
    for time_step in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let mut config = base_config();
        config.time_step = time_step;
        let mut solver = solver_with_config(config.clone());
        assert_preflight_invalid(
            &mut solver,
            &field,
            1.0e-6,
            "ElasticWaveConfig.time_step",
            time_step,
            "must be finite and non-negative; zero selects automatic CFL",
        );
        assert_history_entries_invalid(
            config,
            "ElasticWaveConfig.time_step",
            time_step,
            "must be finite and non-negative; zero selects automatic CFL",
        );
    }

    for time_step in [0.0, -0.0, 1.0e-6] {
        let mut config = base_config();
        config.time_step = time_step;
        let mut solver = solver_with_config(config);
        let result = solver
            .propagate(&field, 1.0e-9, None)
            .expect("zero selects CFL and a positive finite step is valid");
        assert!(result.time.is_finite() && result.time > 0.0);
    }
}

#[test]
fn automatic_cfl_rejects_invalid_effective_steps() {
    let field = ElasticWaveField::new(2, 3, 4);
    for cfl_factor in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut config = base_config();
        config.time_step = 0.0;
        config.cfl_factor = cfl_factor;
        let mut solver = solver_with_config(config.clone());
        let recorder_before = solver.extract_recorded_data();
        let error = solver
            .propagate(&field, 1.0e-6, Some(&invalid_force()))
            .expect_err("invalid automatic CFL result must be rejected");
        let (parameter, value, reason) = invalid_value(error);
        assert_eq!(parameter, "effective_time_step");
        assert!(!value.is_finite() || value <= 0.0);
        assert_eq!(reason, "must be finite and greater than zero");
        assert_eq!(solver.extract_recorded_data(), recorder_before);
        assert_history_entries_invalid(
            config,
            "effective_time_step",
            value,
            "must be finite and greater than zero",
        );
    }
}

#[test]
fn initial_and_derived_time_domains_are_validated() {
    for initial_time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut solver = solver_with_config(base_config());
        let mut field = ElasticWaveField::new(2, 3, 4);
        field.time = initial_time;
        assert_preflight_invalid(
            &mut solver,
            &field,
            1.0e-6,
            "ElasticWaveField.time",
            initial_time,
            "must be finite",
        );
    }

    let mut negative_solver = solver_with_config(base_config());
    let mut negative_field = ElasticWaveField::new(2, 3, 4);
    negative_field.time = -1.0;
    let negative_result = negative_solver
        .propagate(&negative_field, 1.0e-9, None)
        .expect("finite negative initial time remains valid");
    assert!(negative_result.time > -1.0 && negative_result.time.is_finite());

    let mut count_config = base_config();
    count_config.time_step = f64::MIN_POSITIVE;
    let mut count_solver = solver_with_config(count_config);
    let field = ElasticWaveField::new(2, 3, 4);
    let count_error = count_solver
        .propagate(&field, 1.0, Some(&invalid_force()))
        .expect_err("unrepresentable step count must be rejected");
    let (parameter, value, reason) = invalid_value(count_error);
    assert_eq!(parameter, "simulation_step_count");
    assert!(value.is_finite() && value >= usize::MAX as f64);
    assert_eq!(
        reason,
        "must be finite, positive, and representable as usize"
    );

    let mut end_config = base_config();
    end_config.time_step = f64::MAX;
    let mut end_solver = solver_with_config(end_config);
    let mut late_field = ElasticWaveField::new(2, 3, 4);
    late_field.time = f64::MAX;
    let end_error = end_solver
        .propagate(&late_field, f64::MAX, Some(&invalid_force()))
        .expect_err("non-finite end time must be rejected");
    let (parameter, value, reason) = invalid_value(end_error);
    assert_eq!(parameter, "simulation_end_time");
    assert!(value.is_infinite());
    assert_eq!(reason, "must be finite");
}

#[test]
fn duration_shorter_than_step_still_executes_once() {
    for (duration, time_step) in [(1.0e-6, 1.0e-3), (f64::MIN_POSITIVE, f64::MAX)] {
        let mut config = base_config();
        config.time_step = time_step;
        let mut solver = solver_with_config(config);
        let field = ElasticWaveField::new(2, 3, 4);

        let result = solver
            .propagate(&field, duration, None)
            .expect("positive sub-step duration must execute one step");

        assert_eq!(result.time, time_step);
    }
}

#[test]
fn propagate_waves_reports_structural_displacement_mismatch() {
    let solver = solver_with_config(base_config());
    let actual_shape = [4, 3, 2];
    let error = solver
        .propagate_waves(&Array3::zeros(actual_shape))
        .expect_err("mismatched displacement shape must be rejected");

    match error {
        KwaversError::Validation(ValidationError::DimensionMismatch { expected, actual }) => {
            assert_eq!(
                expected,
                format!("initial_displacement shape {GRID_SHAPE:?}")
            );
            assert_eq!(actual, format!("{actual_shape:?}"));
        }
        other => panic!("expected displacement dimension mismatch, got {other}"),
    }
}

#[test]
fn history_schedule_preserves_initial_final_and_saved_times() {
    const DT: f64 = 9.536_743_164_062_5e-7;
    for (steps, save_every, saved_steps, nonzero) in [
        (4, 2, &[0, 2, 4][..], false),
        (5, 2, &[0, 2, 4, 5][..], true),
        (3, 4, &[0, 3][..], true),
    ] {
        let mut config = base_config();
        config.time_step = DT;
        config.simulation_time = steps as f64 * DT;
        config.save_every = save_every;
        let displacement = if nonzero {
            Array3::from_shape_fn(GRID_SHAPE, |[i, j, k]| {
                (i * 12 + j * 4 + k + 1) as f64 * 1.0e-12
            })
        } else {
            Array3::zeros(GRID_SHAPE)
        };

        let history_solver = solver_with_config(config.clone());
        let history = history_solver
            .propagate_waves(&displacement)
            .expect("valid history propagation");
        assert_eq!(history.len(), saved_steps.len());
        for (field, &saved_step) in history.iter().zip(saved_steps) {
            assert_eq!(field.time, saved_step as f64 * DT);
        }

        let mut initial = ElasticWaveField::new(2, 3, 4);
        initial.uz.assign(&displacement);
        assert_field_unchanged(history.first().expect("initial snapshot"), &initial);
        let mut direct_solver = solver_with_config(config);
        let final_field = direct_solver
            .propagate(&initial, steps as f64 * DT, None)
            .expect("valid direct propagation");
        assert_field_unchanged(history.last().expect("final snapshot"), &final_field);
    }
}
