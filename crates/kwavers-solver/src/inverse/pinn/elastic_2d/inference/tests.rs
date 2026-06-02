use super::*;
use crate::inverse::pinn::elastic_2d::Config;
use burn::backend::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_predictor_creation() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    assert!(predictor.model().num_parameters() > 0);
}

#[test]
fn test_single_point_prediction() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let result = predictor.predict_point(0.5, 0.5, 0.1);

    let displacement = result.unwrap();
    assert_eq!(displacement.len(), 2);
}

#[test]
fn test_batch_prediction() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let points = vec![(0.5, 0.5, 0.1), (0.6, 0.6, 0.2), (0.7, 0.7, 0.3)];
    let result = predictor.predict_batch(&points);

    let displacements = result.unwrap();
    assert_eq!(displacements.dim(), (3, 2));
}

#[test]
fn test_empty_batch_error() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let points: Vec<(f64, f64, f64)> = vec![];
    let result = predictor.predict_batch(&points);
    assert!(result.is_err());
}

#[test]
fn test_field_evaluation() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let x_grid = Array1::linspace(0.0, 1.0, 5);
    let y_grid = Array1::linspace(0.0, 1.0, 5);
    let t = 0.1;

    let result = predictor.evaluate_field(&x_grid, &y_grid, t);

    let field = result.unwrap();
    assert_eq!(field.dim(), (5, 5, 2));
}

#[test]
fn test_time_series() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let times = Array1::linspace(0.0, 1.0, 10);
    let result = predictor.time_series(0.5, 0.5, &times);

    let time_series = result.unwrap();
    assert_eq!(time_series.dim(), (10, 2));
}

#[test]
fn test_magnitude_field() {
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let x_grid = Array1::linspace(0.0, 1.0, 5);
    let y_grid = Array1::linspace(0.0, 1.0, 5);
    let t = 0.1;

    let result = predictor.magnitude_field(&x_grid, &y_grid, t);

    let magnitude = result.unwrap();
    assert_eq!(magnitude.dim(), (5, 5));

    // All magnitudes should be non-negative
    for val in magnitude.iter() {
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_material_parameters_inverse() {
    let config = Config::inverse_problem(1e9, 5e8, 1000.0);
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let (lambda, mu, rho) = predictor.material_parameters();
    assert!((lambda.unwrap() - 1e9).abs() < 1e-3);
    assert!((mu.unwrap() - 5e8).abs() < 1e-3);
    assert!((rho.unwrap() - 1000.0).abs() < 1e-3);
}

#[test]
fn test_material_parameters_forward() {
    let config = Config::forward_problem(1e9, 5e8, 1000.0);
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
    let predictor = ElasticPinnPredictor::new(model);

    let (lambda, mu, rho) = predictor.material_parameters();
    assert!(lambda.is_none());
    assert!(mu.is_none());
    assert!(rho.is_none());
}
