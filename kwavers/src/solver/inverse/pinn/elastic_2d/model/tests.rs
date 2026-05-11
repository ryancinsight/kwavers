use super::*;
use burn::backend::NdArray;
use burn::tensor::Tensor;

type TestBackend = NdArray<f32>;

#[test]
fn test_model_creation() {
    let config = crate::solver::inverse::pinn::elastic_2d::Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device);
    let _model = model.unwrap();
}

#[test]
fn test_model_forward_pass() {
    let config = crate::solver::inverse::pinn::elastic_2d::Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let y = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let output = model.forward(x, y, t);
    let dims = output.dims();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 2);
}

#[test]
fn test_model_batch_forward() {
    let config = crate::solver::inverse::pinn::elastic_2d::Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    let batch_size = 10;
    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device).repeat(&[batch_size, 1]);
    let y = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device).repeat(&[batch_size, 1]);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device).repeat(&[batch_size, 1]);

    let output = model.forward(x, y, t);
    let dims = output.dims();
    assert_eq!(dims[0], batch_size);
    assert_eq!(dims[1], 2);
}

#[test]
fn test_inverse_problem_parameters() {
    let config =
        crate::solver::inverse::pinn::elastic_2d::Config::inverse_problem(1e9, 5e8, 1000.0);
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    let (lambda_est, mu_est, rho_est) = model.estimated_parameters();

    assert!((lambda_est.unwrap() - 1e9).abs() < 1e-3);
    assert!((mu_est.unwrap() - 5e8).abs() < 1e-3);
    assert!((rho_est.unwrap() - 1000.0).abs() < 1e-3);
}

#[test]
fn test_forward_problem_no_learnable_params() {
    let config =
        crate::solver::inverse::pinn::elastic_2d::Config::forward_problem(1e9, 5e8, 1000.0);
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    assert!(model.lambda.is_none());
    assert!(model.mu.is_none());
    assert!(model.rho.is_none());

    let (lambda_est, mu_est, rho_est) = model.estimated_parameters();
    assert!(lambda_est.is_none());
    assert!(mu_est.is_none());
    assert!(rho_est.is_none());
}

#[test]
fn test_parameter_count() {
    let config = crate::solver::inverse::pinn::elastic_2d::Config {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    let count = model.num_parameters();

    assert_eq!(count, 172);
}

#[test]
fn test_activation_functions() {
    let device = Default::default();

    let x = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);

    let y_tanh = x.clone().tanh();
    let tanh_val = y_tanh.to_data().as_slice::<f32>().unwrap()[0];
    assert!((tanh_val - 0.76).abs() < 0.1);

    let y_sin = x.clone().sin();
    let sin_val = y_sin.to_data().as_slice::<f32>().unwrap()[0];
    assert!((sin_val - 0.84).abs() < 0.1);

    let neg_x = x.clone().neg();
    let exp_neg_x = neg_x.exp();
    let one = Tensor::<TestBackend, 2>::ones_like(&x);
    let sigmoid_x = one.clone() / (one + exp_neg_x);
    let y_swish = x.clone() * sigmoid_x;
    let swish_val = y_swish.to_data().as_slice::<f32>().unwrap()[0];
    assert!((swish_val - 0.73).abs() < 0.1);
}

#[test]
fn test_get_material_parameters() {
    let config =
        crate::solver::inverse::pinn::elastic_2d::Config::inverse_problem(2e9, 1e9, 2000.0);
    let device = Default::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

    let lambda = model.get_lambda(1e9);
    let mu = model.get_mu(5e8);
    let rho = model.get_rho(1000.0);

    let lambda_val = lambda.to_data().as_slice::<f32>().unwrap()[0] as f64;
    let mu_val = mu.to_data().as_slice::<f32>().unwrap()[0] as f64;
    let rho_val = rho.to_data().as_slice::<f32>().unwrap()[0] as f64;

    assert!((lambda_val - 2e9).abs() < 1e-3);
    assert!((mu_val - 1e9).abs() < 1e-3);
    assert!((rho_val - 2000.0).abs() < 1e-3);
}
