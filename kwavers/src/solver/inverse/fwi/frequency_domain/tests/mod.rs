use std::sync::Arc;

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

use super::gradient::objective_and_gradient;
use super::{
    invert, simulate_frequency_observation, AbsorbingBoundary, Config, DenseConvergentBornOperator,
    FrequencyObservation, PstdSpectralConvergentBornOperator, SingleScatterBornOperator,
    SpectralConvergentBornOperator,
};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};
use ndarray::Array3;

mod forward;
mod gradient_fd;
mod inversion;

pub(super) fn test_config() -> Config {
    Config {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: 0.005,
        iterations: 6,
        initial_step_s_per_m: 3.0e-6,
        min_sound_speed_m_s: 1450.0,
        max_sound_speed_m_s: 1560.0,
        estimate_source_scaling: false,
        tikhonov_weight: 0.0,
        forward_operator: Arc::new(SingleScatterBornOperator),
    }
}

pub(super) fn test_array() -> MultiRowRingArray {
    MultiRowRingArray::new(6, 2, 0.08, 0.01).expect("ring array")
}
