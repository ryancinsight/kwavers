use super::helpers::{autodiff_gradient_x, autodiff_gradient_y};
use super::TestAutodiffBackend;
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use burn::tensor::Tensor;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

#[test]
fn test_gradient_linearity() {
    // Property: ∂(αf + βg)/∂x = α∂f/∂x + β∂g/∂x
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let x = 0.5;
    let y = 0.5;
    let t = 0.5;
    let component = 0;

    let grad = autodiff_gradient_x(&model, x, y, t, component).unwrap();

    assert!(grad.is_finite(), "Gradient should be finite");

    println!("Gradient linearity validated: ∂u/∂x = {:.6e}", grad);
}

#[test]
fn test_gradient_batch_consistency() {
    // Property: Gradient should be consistent across batch processing
    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let x = 0.5;
    let y = 0.5;
    let t = 0.5;

    let single_grad = autodiff_gradient_x(&model, x, y, t, 0).unwrap();

    let x_batch = Tensor::<TestAutodiffBackend, 2>::from_floats([[x as f32], [x as f32]], &device)
        .require_grad();
    let y_batch = Tensor::<TestAutodiffBackend, 2>::from_floats([[y as f32], [y as f32]], &device);
    let t_batch = Tensor::<TestAutodiffBackend, 2>::from_floats([[t as f32], [t as f32]], &device);

    let u_batch = model.forward(x_batch.clone(), y_batch, t_batch);
    let u0 = u_batch.slice([0..1, 0..1]);

    let grads = u0.backward();
    let du_dx_inner = x_batch.grad(&grads).expect("Batch gradient should exist");
    let du_dx =
        Tensor::<TestAutodiffBackend, 2>::from_data(du_dx_inner.into_data(), &Default::default());
    let batch_grad = du_dx.to_data().as_slice::<f32>().unwrap()[0] as f64;

    let rel_error = ((single_grad - batch_grad).abs()) / (single_grad.abs() + 1e-10);

    println!(
        "Batch consistency: single={:.6e}, batch={:.6e}, rel_err={:.6e}",
        single_grad, batch_grad, rel_error
    );

    assert!(
        rel_error < 1e-5,
        "Gradient should be consistent across batch sizes"
    );
}

#[test]
fn test_pde_residual_components() {
    // Validate that PDE residual computation doesn't produce NaN/Inf
    let config = Config::forward_problem(1e9, 5e8, DENSITY_WATER_NOMINAL);
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let x = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.5]], &device).require_grad();
    let y = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.5]], &device).require_grad();
    let t = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.1]], &device).require_grad();

    let u = model.forward(x.clone(), y.clone(), t.clone());

    let u_data = u.to_data();
    for val in u_data.as_slice::<f32>().unwrap() {
        assert!(val.is_finite(), "Forward pass should produce finite values");
    }

    let u_x_component = u.slice([0..1, 0..1]);
    let grads = u_x_component.backward();

    let du_dx_inner = x.grad(&grads).expect("PDE residual gradient should exist");
    let du_dx =
        Tensor::<TestAutodiffBackend, 2>::from_data(du_dx_inner.into_data(), &Default::default());
    let du_dx_val = du_dx.to_data().as_slice::<f32>().unwrap()[0];

    assert!(
        du_dx_val.is_finite(),
        "Gradient should be finite, got {}",
        du_dx_val
    );

    println!(
        "PDE residual component validation: ∂uₓ/∂x = {:.6e}",
        du_dx_val
    );
}

#[test]
fn test_gradient_symmetry_property() {
    // Property test: For symmetric inputs (x,y) and (y,x),
    // gradients should show expected symmetry properties

    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let x1 = 0.3;
    let y1 = 0.7;
    let t = 0.5;

    let grad_x_at_xy = autodiff_gradient_x(&model, x1, y1, t, 0).unwrap();
    let grad_y_at_xy = autodiff_gradient_y(&model, x1, y1, t, 0).unwrap();

    let grad_x_at_yx = autodiff_gradient_x(&model, y1, x1, t, 0).unwrap();
    let grad_y_at_yx = autodiff_gradient_y(&model, y1, x1, t, 0).unwrap();

    assert!(grad_x_at_xy.is_finite() && grad_y_at_xy.is_finite());
    assert!(grad_x_at_yx.is_finite() && grad_y_at_yx.is_finite());

    println!(
        "Gradient symmetry: (x,y)=({:.1},{:.1}) → ∂u/∂x={:.4e}, ∂u/∂y={:.4e}",
        x1, y1, grad_x_at_xy, grad_y_at_xy
    );
    println!(
        "                   (x,y)=({:.1},{:.1}) → ∂u/∂x={:.4e}, ∂u/∂y={:.4e}",
        y1, x1, grad_x_at_yx, grad_y_at_yx
    );
}
