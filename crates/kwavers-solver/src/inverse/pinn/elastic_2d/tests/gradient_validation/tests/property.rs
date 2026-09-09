use super::helpers::{autodiff_gradient_x, autodiff_gradient_y};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use coeus_autograd::Var;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

type B = super::TestBackend;

#[test]
fn test_gradient_batch_consistency() {
    // Property: Gradient should be consistent across batch processing
    let config = Config::default();
    let model = ElasticPINN2D::<B>::new(&config).unwrap();
    let backend = B::default();

    let x = 0.5;
    let y = 0.5;
    let t = 0.5;

    let single_grad = autodiff_gradient_x(&model, x, y, t, 0).unwrap();

    let x_batch = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![2, 1], &[x as f32, x as f32], &backend),
        true,
    );
    let y_batch = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![2, 1], &[y as f32, y as f32], &backend),
        false,
    );
    let t_batch = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![2, 1], &[t as f32, t as f32], &backend),
        false,
    );

    let u_batch = model
        .forward(&x_batch, &y_batch, &t_batch)
        .expect("elastic batch forward");
    let u0 = coeus_autograd::slice(&u_batch, &[(0, 1), (0, 1)]);

    coeus_autograd::sum(&u0)
        .backward()
        .expect("elastic batch backward");
    let du_dx = x_batch.grad().expect("Batch gradient should exist");
    let batch_grad = du_dx.as_slice()[0] as f64;

    let rel_error = ((single_grad - batch_grad).abs()) / (single_grad.abs() + 1e-10);

    assert!(
        rel_error < 1e-5,
        "a gradient must not depend on the batch it was computed in:          single={single_grad:.6e}, batch={batch_grad:.6e}, rel_err={rel_error:.6e}"
    );
}

/// The forward pass and its gradient stay finite at an interior point.
///
/// Named for what it asserts. It was `test_pde_residual_components`, which
/// computes no residual -- non-finite values are the classic autodiff failure
/// and that is the real property here.
#[test]
fn forward_and_gradient_stay_finite() {
    let config = Config::forward_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    let model = ElasticPINN2D::<B>::new(&config).unwrap();
    let backend = B::default();

    let mk = |v: f32| {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[v], &backend),
            true,
        )
    };
    let x = mk(0.5);
    let y = mk(0.5);
    let t = mk(0.1);

    let u = model.forward(&x, &y, &t).expect("elastic PDE forward");

    for &val in u.tensor.as_slice() {
        assert!(val.is_finite(), "Forward pass should produce finite values");
    }

    let u_x_component = coeus_autograd::slice(&u, &[(0, 1), (0, 1)]);
    coeus_autograd::sum(&u_x_component)
        .backward()
        .expect("elastic PDE backward");

    let du_dx = x.grad().expect("PDE residual gradient should exist");
    let du_dx_val = du_dx.as_slice()[0];

    assert!(
        du_dx_val.is_finite(),
        "Gradient should be finite, got {}",
        du_dx_val
    );
}
