#[cfg(feature = "pinn")]
use crate::ml::uncertainty::PinnUncertaintyPredictor;
use kwavers_core::error::KwaversResult;
#[cfg(not(feature = "pinn"))]
use leto::Array2;
#[cfg(feature = "pinn")]
use leto::Array2;
#[cfg(feature = "pinn")]
use tyche_core::{
    sampling::{Counter, UserDomain},
    Seed, SplitMix64,
};

/// Individual ensemble model
#[derive(Debug, Clone)]
pub(super) struct EnsembleModel {
    pub(super) _random_seed: u64,
    pub(super) weight: f64,
    pub(super) performance_score: f64,
}

impl EnsembleModel {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            _random_seed: seed,
            weight: 1.0,
            performance_score: 0.0,
        }
    }

    #[cfg(feature = "pinn")]
    pub(super) fn predict_with_noise<P: PinnUncertaintyPredictor + ?Sized>(
        &self,
        predictor: &P,
        inputs: &Array2<f32>,
    ) -> KwaversResult<Array2<f32>> {
        let seed = Seed::new(self._random_seed);
        let mut noisy_inputs = inputs.clone();

        for (idx, elem) in noisy_inputs.iter_mut().enumerate() {
            let address = u64::try_from(idx).expect("invariant: usize fits in u64");
            let unit =
                Counter::<UserDomain<0>, SplitMix64>::unit::<f32>(seed, address, 0).clamp(0.0, 1.0);
            let noise = unit.mul_add(0.02, -0.01);
            *elem += noise;
        }

        predictor.predict_inputs(&noisy_inputs)
    }

    pub(super) fn train(
        &mut self,
        data: &[Array2<f32>],
        targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        let mut total_error = 0.0;
        let mut count = 0;

        for (input, target) in data.iter().zip(targets.iter()) {
            let error = (input - target).mapv(|x| x * x).iter().sum::<f32>();
            total_error += error;
            count += input.len();
        }

        if count > 0 {
            self.performance_score = 1.0 / (1.0 + total_error as f64 / count as f64);
        }

        Ok(())
    }
}
