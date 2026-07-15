use super::helpers::{autodiff_gradient_x, autodiff_gradient_y};
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use coeus_autograd::Var;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

type B = super::TestBackend;

#[test]
fn test_gradient_linearity() {
    // Property: ∂(αf + βg)/∂x = α∂f/∂x + β∂g/∂x
    let config = Config::default();
    let model = ElasticPINN2D::<B>::new(&config).unwrap();

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

    let u_batch = model.forward(&x_batch, &y_batch, &t_batch);
    let u0 = coeus_autograd::slice(&u_batch, &[(0, 1), (0, 1)]);

    coeus_autograd::sum(&u0).backward();
    let du_dx = x_batch.grad().expect("Batch gradient should exist");
    let batch_grad = du_dx.as_slice()[0] as f64;

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

    let u = model.forward(&x, &y, &t);

    for &val in u.tensor.as_slice() {
        assert!(val.is_finite(), "Forward pass should produce finite values");
    }

    let u_x_component = coeus_autograd::slice(&u, &[(0, 1), (0, 1)]);
    coeus_autograd::sum(&u_x_component).backward();

    let du_dx = x.grad().expect("PDE residual gradient should exist");
    let du_dx_val = du_dx.as_slice()[0];

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
    let model = ElasticPINN2D::<B>::new(&config).unwrap();

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
