//! Persistent optimizer state — moment buffers for Adam and SGD-Momentum.

#[cfg(feature = "pinn")]
use burn::module::{Module, Param};
#[cfg(feature = "pinn")]
use burn::tensor::{backend::Backend, Tensor};

/// Persistent state for Adam optimizer with moment buffers.
///
/// Maintains first and second moment estimates for all parameters by mirroring
/// the model structure, enabling mathematically complete Adam optimization with
/// exponential moving averages across training steps.
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct PersistentAdamState<B: Backend> {
    /// First moment estimates (exponential moving average of gradients).
    pub first_moments: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
    /// Second moment estimates (exponential moving average of squared gradients).
    pub second_moments: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
    /// Global timestep counter (for bias correction).
    pub timestep: usize,
    /// Beta1 hyperparameter (first moment decay rate, typically 0.9).
    pub beta1: f64,
    /// Beta2 hyperparameter (second moment decay rate, typically 0.999).
    pub beta2: f64,
    /// Epsilon for numerical stability (typically 1e-8).
    pub epsilon: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> PersistentAdamState<B> {
    /// Initialize Adam state with zero moment buffers.
    pub fn new(
        model: &crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let first_moments = model.clone().map(&mut ZeroInitMapper);
        let second_moments = model.clone().map(&mut ZeroInitMapper);

        Self {
            first_moments,
            second_moments,
            timestep: 0,
            beta1,
            beta2,
            epsilon,
        }
    }

    /// Increment timestep counter.
    pub fn step(&mut self) {
        self.timestep += 1;
    }
}

/// Mapper to zero-initialize all parameters in a model.
#[cfg(feature = "pinn")]
pub(super) struct ZeroInitMapper;

#[cfg(feature = "pinn")]
impl<B: Backend> burn::module::ModuleMapper<B> for ZeroInitMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let shape = param.shape();
        let device = param.device();
        let zeros = Tensor::<B, D>::zeros(shape, &device);

        if param.is_require_grad() {
            Param::from_tensor(zeros.require_grad())
        } else {
            Param::from_tensor(zeros)
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

/// Persistent velocity state for SGD with momentum.
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct MomentumState<B: Backend> {
    /// Velocity buffers (exponential moving average of gradients).
    pub velocity: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
    /// Momentum coefficient (typically 0.9).
    pub beta: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> MomentumState<B> {
    /// Initialize momentum state with zero velocity buffers.
    pub fn new(
        model: &crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        beta: f64,
    ) -> Self {
        let velocity = model.clone().map(&mut ZeroInitMapper);
        Self { velocity, beta }
    }
}
