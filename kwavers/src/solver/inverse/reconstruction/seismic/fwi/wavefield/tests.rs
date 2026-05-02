use ndarray::Array3;

use super::{WavefieldConfig, WavefieldModeler};

#[test]
fn test_forward_model_rejects_invalid_velocity() {
    let mut modeler = WavefieldModeler::new();
    let velocity_model = Array3::zeros((4, 4, 4));

    let err = modeler
        .forward_model(&velocity_model)
        .expect_err("zero velocity must fail");

    assert!(format!("{err:?}").contains("strictly positive maximum"));
}

#[test]
fn test_forward_model_rejects_out_of_bounds_geometry() {
    let mut modeler = WavefieldModeler::with_config(WavefieldConfig {
        source_position: Some((4, 0, 0)),
        receivers: vec![(0, 0, 0)],
        ..WavefieldConfig::default()
    });
    let velocity_model = Array3::from_elem((4, 4, 4), 1500.0);

    let err = modeler
        .forward_model(&velocity_model)
        .expect_err("out-of-bounds source must fail");

    assert!(format!("{err:?}").contains("Source position out of bounds"));
}

#[test]
fn test_adjoint_model_uses_checkpointed_replay() {
    let mut modeler = WavefieldModeler::with_config(WavefieldConfig {
        dx: 2.0,
        dt: 1.0,
        max_time: 3.0,
        peak_frequency: 1.0,
        source_position: Some((0, 0, 0)),
        receivers: vec![(0, 0, 0)],
    });
    modeler.pml_width = 0;

    let velocity_model = Array3::from_elem((4, 4, 4), 1.0);
    let synthetic = modeler
        .forward_model(&velocity_model)
        .expect("forward model must succeed");
    assert_eq!(synthetic.shape(), &[1, 3]);
    assert!((synthetic[[0, 1]] - 1.0).abs() < f64::EPSILON);
    let replay_cache = modeler
        .forward_replay
        .as_ref()
        .expect("forward replay cache must exist");
    assert_eq!(replay_cache.stride, 2);
    assert_eq!(replay_cache.checkpoints.len(), 2);
    assert!(replay_cache.checkpoints[0]
        .current
        .iter()
        .all(|&v| v.abs() < f64::EPSILON));

    let residual = synthetic.clone();
    let gradient = modeler
        .adjoint_model(&velocity_model, &residual)
        .expect("adjoint model must succeed");

    let expected = (0..3)
        .map(|t| {
            let tau = t as f64 - 1.0;
            let a = std::f64::consts::PI * tau;
            let value = (1.0 - 2.0 * a * a) * (-a * a).exp();
            value * value
        })
        .sum::<f64>();

    assert!((gradient[[0, 0, 0]] - expected).abs() < 1e-12);
    assert_eq!(gradient.sum(), expected);
    assert!(modeler.forward_replay.is_none());
}
