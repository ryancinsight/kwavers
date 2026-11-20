use super::super::run;
use crate::configs::NonlinearConfig;
use crate::dispatch::test_support::{centre_source, request};
use crate::types::SimulationRunRequest;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::config::SolverType;
use leto::Array3;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace_test() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn fixture() -> (Grid, NonlinearConfig) {
    (
        Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid"),
        NonlinearConfig::default(),
    )
}

fn fdtd_request<'a>(
    grid: &'a Grid,
    medium: &'a HomogeneousMedium,
    nonlinear: &'a NonlinearConfig,
) -> SimulationRunRequest<'a> {
    SimulationRunRequest {
        solver_type: SolverType::FDTD,
        time_steps: 3,
        dt: 1.0e-7,
        grid_source: centre_source(),
        ..request(grid, medium, nonlinear)
    }
}

#[test]
fn explicit_order_and_statistics_match_independent_mask_recordings() {
    let _trace = trace_test();
    let (grid, nonlinear) = tracing::info_span!("arrange").in_scope(fixture);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let ordered = [(3, 2, 1), (1, 1, 1)];

    for record_modes in [Vec::new(), vec!["p_max".to_owned()]] {
        let (result, references) = tracing::info_span!("act").in_scope(|| {
            let result = run(
                &SimulationRunRequest {
                    transducer_ordered_indices: Some(ordered.to_vec()),
                    record_modes: record_modes.clone(),
                    ..fdtd_request(&grid, &medium, &nonlinear)
                },
                Vec::new(),
            )
            .expect("ordered recordings are supported");
            let references = ordered.map(|(x, y, z)| {
                let mut mask = Array3::from_elem([7, 4, 3], false);
                mask[[x, y, z]] = true;
                run(
                    &SimulationRunRequest {
                        sensor_mask: Some(mask),
                        record_modes: record_modes.clone(),
                        ..fdtd_request(&grid, &medium, &nonlinear)
                    },
                    Vec::new(),
                )
                .expect("single-point mask records one sensor")
            });
            (result, references)
        });

        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(result.sensor_data.shape(), [2, 3], "modes {record_modes:?}");
        assert!((0..3).any(|step| {
            references[0].sensor_data[[0, step]].to_bits()
                != references[1].sensor_data[[0, step]].to_bits()
        }));
        for (row, reference) in references.iter().enumerate() {
            for step in 0..3 {
                assert_eq!(
                    result.sensor_data[[row, step]].to_bits(),
                    reference.sensor_data[[0, step]].to_bits(),
                );
            }
            if record_modes.is_empty() {
                assert!(result.stats.is_none());
            } else {
                let actual = result
                    .stats
                    .as_ref()
                    .expect("ordered result retains statistics");
                let expected = reference
                    .stats
                    .as_ref()
                    .expect("mask result retains statistics");
                assert_eq!(actual.num_sensors(), 2);
                assert_eq!(actual.p_max[row].to_bits(), expected.p_max[0].to_bits());
                assert_eq!(actual.p_rms[row].to_bits(), expected.p_rms[0].to_bits());
            }
        }
    }
}

#[test]
fn explicit_empty_order_overrides_the_mask_and_retains_field_statistics() {
    let _trace = trace_test();
    let (grid, nonlinear) = tracing::info_span!("arrange").in_scope(fixture);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let mut mask = Array3::from_elem([7, 4, 3], false);
    mask[[3, 2, 1]] = true;
    let result = tracing::info_span!("act").in_scope(|| {
        run(
            &SimulationRunRequest {
                sensor_mask: Some(mask),
                transducer_ordered_indices: Some(Vec::new()),
                record_modes: vec!["p_max".to_owned()],
                ..fdtd_request(&grid, &medium, &nonlinear)
            },
            Vec::new(),
        )
        .expect("empty selection with field statistics is valid")
    });

    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(result.sensor_data.shape(), [0, 3]);
    assert_eq!(result.stats.expect("sampled statistics").num_sensors(), 0);
    let (peak, _, _, _) = result.full_grid_stats.expect("full grid statistics");
    assert!(peak[[3, 2, 1]] > 0.0);
}

#[test]
fn explicit_indices_are_validated_in_each_dimension() {
    let _trace = trace_test();
    let (grid, nonlinear) = tracing::info_span!("arrange").in_scope(fixture);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    for invalid in [(7, 2, 1), (3, 4, 1), (3, 2, 3)] {
        let error = tracing::info_span!("act").in_scope(|| {
            run(
                &SimulationRunRequest {
                    transducer_ordered_indices: Some(vec![(3, 2, 1), invalid]),
                    record_modes: vec!["p_max".to_owned()],
                    ..fdtd_request(&grid, &medium, &nonlinear)
                },
                Vec::new(),
            )
            .expect_err("out-of-grid recording index must be rejected")
        });
        let _assert = tracing::info_span!("assert").entered();
        let KwaversError::Validation(ValidationError::ConstraintViolation { message }) = error
        else {
            panic!("expected an index validation error, got {error:?}");
        };
        assert!(message.contains(&format!("{invalid:?}")));
    }
}
