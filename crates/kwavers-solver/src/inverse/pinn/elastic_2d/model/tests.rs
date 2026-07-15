use super::*;
use coeus_autograd::Var;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

type TestBackend = coeus_core::MoiraiBackend;

fn var_scalar(v: f32) -> Var<f32, TestBackend> {
    let backend = TestBackend::default();
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[v], &backend),
        false,
    )
}

#[test]
fn test_model_creation() {
    let config = crate::inverse::pinn::elastic_2d::Config::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config);
    let _model = model.unwrap();
}

#[test]
fn test_model_forward_pass() {
    let config = crate::inverse::pinn::elastic_2d::Config::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

    let x = var_scalar(0.5);
    let y = var_scalar(0.5);
    let t = var_scalar(0.1);

    let output = model.forward(&x, &y, &t);
    let dims = output.tensor.shape();
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 2);
}

#[test]
fn test_model_batch_forward() {
    let config = crate::inverse::pinn::elastic_2d::Config::default();
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

    let batch_size = 10;
    let backend = TestBackend::default();
    let mk = |v: f32| {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(
                vec![batch_size, 1],
                &vec![v; batch_size],
                &backend,
            ),
            false,
        )
    };
    let x = mk(0.5);
    let y = mk(0.5);
    let t = mk(0.1);

    let output = model.forward(&x, &y, &t);
    let dims = output.tensor.shape();
    assert_eq!(dims[0], batch_size);
    assert_eq!(dims[1], 2);
}

#[test]
fn test_inverse_problem_parameters() {
    let config =
        crate::inverse::pinn::elastic_2d::Config::inverse_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

    let (lambda_est, mu_est, rho_est) = model.estimated_parameters();

    assert!((lambda_est.unwrap() - 1e9).abs() < 1e-3);
    assert!((mu_est.unwrap() - 5e8).abs() < 1e-3);
    assert!((rho_est.unwrap() - DENSITY_WATER_NOMINAL).abs() < 1e-3);
}

#[test]
fn test_forward_problem_no_learnable_params() {
    let config =
        crate::inverse::pinn::elastic_2d::Config::forward_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

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
    let config = crate::inverse::pinn::elastic_2d::Config {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

    let count = model.num_parameters();

    assert_eq!(count, 172);
}

#[test]
fn test_activation_functions() {
    let x = var_scalar(1.0);

    let y_tanh = coeus_autograd::tanh(&x);
    let tanh_val = y_tanh.tensor.as_slice()[0];
    assert!((tanh_val - 0.76).abs() < 0.1);

    let y_sin = coeus_autograd::sin(&x);
    let sin_val = y_sin.tensor.as_slice()[0];
    assert!((sin_val - 0.84).abs() < 0.1);

    // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let neg_x = coeus_autograd::scalar_mul(&x, -1.0);
    let backend = TestBackend::default();
    let one = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[1.0_f32], &backend),
        false,
    );
    let exp_neg_x = coeus_autograd::exp(&neg_x);
    let sigmoid_x = coeus_autograd::div(&one, &coeus_autograd::add(&one, &exp_neg_x));
    let y_swish = coeus_autograd::mul(&x, &sigmoid_x);
    let swish_val = y_swish.tensor.as_slice()[0];
    assert!((swish_val - 0.73).abs() < 0.1);
}

#[test]
fn test_get_material_parameters() {
    let config = crate::inverse::pinn::elastic_2d::Config::inverse_problem(2e9, 1e9, 2000.0);
    let model = ElasticPINN2D::<TestBackend>::new(&config).unwrap();

    let lambda = model.get_lambda(1e9);
    let mu = model.get_mu(5e8);
    let rho = model.get_rho(1000.0);

    let lambda_val = lambda.as_slice()[0] as f64;
    let mu_val = mu.as_slice()[0] as f64;
    let rho_val = rho.as_slice()[0] as f64;

    assert!((lambda_val - 2e9).abs() < 1e-3);
    assert!((mu_val - 1e9).abs() < 1e-3);
    assert!((rho_val - 2000.0).abs() < 1e-3);
}
