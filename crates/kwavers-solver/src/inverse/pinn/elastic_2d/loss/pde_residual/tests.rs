use super::compute_elastic_wave_pde_residual;
use crate::inverse::pinn::elastic_2d::config::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

type B = coeus_core::MoiraiBackend;

fn var_col(backend: &B, values: &[f32]) -> Var<f32, B> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![values.len(), 1], values, backend),
        false,
    )
}

#[test]
fn test_residual_is_finite() -> KwaversResult<()> {
    let backend = B::default();
    let config = Config {
        hidden_layers: vec![8],
        ..Config::default()
    };
    let model = ElasticPINN2D::<B>::new(&config)?;

    let x = var_col(&backend, &[0.1, 0.3]);
    let y = var_col(&backend, &[0.2, 0.4]);
    let t = var_col(&backend, &[0.05, 0.1]);

    let (residual_x, residual_y) =
        compute_elastic_wave_pde_residual(&model, &x, &y, &t, 1000.0, 2.25e9, 0.0)?;

    for v in residual_x.tensor.as_slice() {
        assert!(v.is_finite(), "residual_x contains non-finite value: {v}");
    }
    for v in residual_y.tensor.as_slice() {
        assert!(v.is_finite(), "residual_y contains non-finite value: {v}");
    }
    Ok(())
}

/// Regression test for the weight-gradient-detachment defect discovered
/// during this migration (see `ml::autodiff_utils::second_order`'s
/// module-level weight-gradient contract): the PDE residual must backprop
/// into every network parameter, not just be a training-inert value.
#[test]
fn test_residual_gradient_reaches_network_weights() -> KwaversResult<()> {
    let backend = B::default();
    let config = Config {
        hidden_layers: vec![8],
        ..Config::default()
    };
    let model = ElasticPINN2D::<B>::new(&config)?;

    for p in model.parameters() {
        p.zero_grad();
    }

    let x = var_col(&backend, &[0.1, 0.3]);
    let y = var_col(&backend, &[0.2, 0.4]);
    let t = var_col(&backend, &[0.05, 0.1]);

    let (residual_x, residual_y) =
        compute_elastic_wave_pde_residual(&model, &x, &y, &t, 1000.0, 2.25e9, 5.0e8)?;

    let loss = coeus_autograd::add(
        &coeus_autograd::sum(&residual_x),
        &coeus_autograd::sum(&residual_y),
    );
    loss.backward();

    let any_param_has_nonzero_grad = model.parameters().iter().any(|p| {
        p.grad()
            .map(|g| g.as_slice().iter().any(|&v| v != 0.0))
            .unwrap_or(false)
    });
    assert!(
        any_param_has_nonzero_grad,
        "PDE residual loss produced zero gradient for every network parameter — \
         the physics loss term is training-inert"
    );
    Ok(())
}
