use std::sync::Arc;

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

use super::gradient::objective_and_gradient;
use super::{
    invert, simulate_frequency_observation, AbsorbingBoundary, Config, DenseConvergentBornOperator,
    FrequencyObservation, PstdFiniteWindowBornOperator, PstdSpectralConvergentBornOperator,
    SingleScatterBornOperator, SpectralConvergentBornOperator,
};
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};
use leto::{Array2, Array3};

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

pub(super) fn first_rows(array: &Array2<Complex64>, rows: usize) -> Array2<Complex64> {
    let [total_rows, cols] = array.shape();
    assert!(rows <= total_rows);
    let values = array
        .as_slice()
        .expect("frequency observation matrix must be contiguous")[..rows * cols]
        .to_vec();
    Array2::from_shape_vec([rows, cols], values).expect("row-prefix shape matches copied data")
}
