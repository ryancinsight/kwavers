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
