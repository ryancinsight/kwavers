//! Tests for PINN optimizers.

#[cfg(feature = "pinn")]
use super::pinn_optimizer::PINNOptimizer;
use super::types::OptimizerAlgorithm;

#[cfg(feature = "pinn")]
#[test]
fn test_optimizer_creation() {
    type TestBackend = burn::backend::Autodiff<burn::backend::NdArray>;
    let sgd_opt = PINNOptimizer::<TestBackend>::sgd(0.01, 0.0001);
    assert_eq!(sgd_opt.algorithm, OptimizerAlgorithm::SGD);
    assert_eq!(sgd_opt.learning_rate, 0.01);
    assert_eq!(sgd_opt.weight_decay, 0.0001);
}

#[test]
fn test_optimizer_algorithm_enum() {
    assert_eq!(OptimizerAlgorithm::SGD as u32, 0);
    assert_eq!(OptimizerAlgorithm::Adam as u32, 2);
    assert_eq!(OptimizerAlgorithm::AdamW as u32, 3);
}

/// Value-semantic convergence test for the real (burn-backed) `SGDMomentum`,
/// `Adam`, and `AdamW` optimizers — guards against the previous no-op regression
/// (Adam/AdamW) and the lr-scaling momentum approximation (SGDMomentum).
///
/// Fits a tiny `ElasticPINN2D` to a constant displacement target with a few
/// hundred steps on the CPU autodiff backend (no GPU). Asserts (a) the MSE loss
/// drops well below its initial value and (b) the model output actually changes
/// (a no-op optimizer would leave both invariant).
#[cfg(feature = "pinn")]
#[test]
fn adam_and_adamw_reduce_loss_and_update_parameters() {
    use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
    use crate::inverse::pinn::elastic_2d::Config;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;

    type AB = Autodiff<NdArray<f32>>;
    let device = Default::default();

    // Tiny network for a fast deterministic test.
    let config = Config {
        hidden_layers: vec![8, 8],
        ..Config::default()
    };

    // Fixed collocation batch and a constant (u_x, u_y) target.
    let x = Tensor::<AB, 2>::from_floats([[0.1], [0.4], [0.7]], &device);
    let y = Tensor::<AB, 2>::from_floats([[0.2], [0.5], [0.8]], &device);
    let t = Tensor::<AB, 2>::from_floats([[0.0], [0.1], [0.2]], &device);
    let target = Tensor::<AB, 2>::from_floats([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]], &device);

    let mse = |model: &ElasticPINN2D<AB>| -> f32 {
        let pred = model.forward(x.clone(), y.clone(), t.clone());
        let diff = pred - target.clone();
        (diff.clone() * diff).mean().into_scalar()
    };
    let probe = |model: &ElasticPINN2D<AB>| -> f32 {
        model
            .forward(x.clone(), y.clone(), t.clone())
            .sum()
            .into_scalar()
    };

    for algorithm in ["sgd_momentum", "adam", "adamw"] {
        let mut model = ElasticPINN2D::<AB>::new(&config, &device).unwrap();
        let mut optimizer = match algorithm {
            "sgd_momentum" => PINNOptimizer::sgd_momentum(&model, 5e-3, 0.0, 0.9),
            "adam" => PINNOptimizer::adam(&model, 1e-2, 0.0, 0.9, 0.999, 1e-8),
            _ => PINNOptimizer::adamw(&model, 1e-2, 1e-4, 0.9, 0.999, 1e-8),
        };

        let initial_loss = mse(&model);
        let initial_probe = probe(&model);
        assert!(
            initial_loss > 1e-4,
            "{algorithm}: initial loss must be non-trivial; got {initial_loss}"
        );

        for _ in 0..200 {
            let pred = model.forward(x.clone(), y.clone(), t.clone());
            let diff = pred - target.clone();
            let loss = (diff.clone() * diff).mean();
            let grads = loss.backward();
            model = optimizer.step(model.clone(), grads);
        }

        let final_loss = mse(&model);
        let final_probe = probe(&model);

        // (a) loss must drop substantially (defeats a no-op, which keeps it flat).
        assert!(
            final_loss < 0.5 * initial_loss,
            "{algorithm}: loss must decrease; initial = {initial_loss}, final = {final_loss}"
        );
        // (b) parameters must have changed (output is not invariant).
        assert!(
            (final_probe - initial_probe).abs() > 1e-5,
            "{algorithm}: model output must change after optimization \
             (initial {initial_probe}, final {final_probe})"
        );
    }
}
