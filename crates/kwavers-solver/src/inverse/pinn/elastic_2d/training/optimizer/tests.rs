//! Tests for PINN optimizers.

use super::pinn_optimizer::PINNOptimizer;
use super::types::OptimizerAlgorithm;

type B = coeus_core::MoiraiBackend;

#[test]
fn test_optimizer_creation() {
    use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
    use crate::inverse::pinn::elastic_2d::Config;

    let config = Config {
        hidden_layers: vec![4],
        ..Config::default()
    };
    let model = ElasticPINN2D::<B>::new(&config).unwrap();
    let sgd_opt = PINNOptimizer::sgd(&model, 0.01, 0.0001);
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

/// Value-semantic convergence test for `SGDMomentum`, `Adam`, and `AdamW` —
/// guards against a no-op optimizer regression.
///
/// Fits a tiny `ElasticPINN2D` to a constant displacement target with a few
/// hundred steps. Asserts (a) the MSE loss drops well below its initial
/// value and (b) the model output actually changes (a no-op optimizer
/// would leave both invariant).
#[test]
fn adam_and_adamw_reduce_loss_and_update_parameters() {
    use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
    use crate::inverse::pinn::elastic_2d::Config;
    use coeus_autograd::Var;

    let backend = B::default();

    let config = Config {
        hidden_layers: vec![8, 8],
        ..Config::default()
    };

    let var_col = |values: &[f32]| {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![(values.len()), 1], values, &backend),
            false,
        )
    };
    let x = var_col(&[0.1, 0.4, 0.7]);
    let y = var_col(&[0.2, 0.5, 0.8]);
    let t = var_col(&[0.0, 0.1, 0.2]);
    let target = Var::new(
        coeus_tensor::Tensor::from_slice_on(
            vec![3, 2],
            &[1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0],
            &backend,
        ),
        false,
    );

    let mse = |model: &ElasticPINN2D<B>| -> f32 {
        let pred = model.forward(&x, &y, &t);
        let diff = coeus_autograd::sub(&pred, &target);
        coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff))
            .tensor
            .as_slice()[0]
    };
    let probe = |model: &ElasticPINN2D<B>| -> f32 {
        coeus_autograd::sum(&model.forward(&x, &y, &t))
            .tensor
            .as_slice()[0]
    };

    for algorithm in ["sgd_momentum", "adam", "adamw"] {
        let mut model = ElasticPINN2D::<B>::new(&config).unwrap();
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
            for p in model.parameters() {
                p.zero_grad();
            }
            let pred = model.forward(&x, &y, &t);
            let diff = coeus_autograd::sub(&pred, &target);
            let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
            loss.backward();
            optimizer.step(&mut model);
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
