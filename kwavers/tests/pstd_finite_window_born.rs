use kwavers::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers::solver::inverse::fwi::frequency_domain::{
    simulate_pstd_finite_window_born_observation, PstdFiniteWindowBornConfig,
};
use ndarray::Array3;

#[test]
fn finite_window_born_is_linear_in_slowness_squared_contrast() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = test_config();
    let base = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let small = contrast_volume(0.01);
    let double = contrast_volume(0.02);

    let base_data =
        simulate_pstd_finite_window_born_observation(&base, &array, 200_000.0, config, 4)
            .expect("base data");
    let small_data =
        simulate_pstd_finite_window_born_observation(&small, &array, 200_000.0, config, 4)
            .expect("small contrast");
    let double_data =
        simulate_pstd_finite_window_born_observation(&double, &array, 200_000.0, config, 4)
            .expect("double contrast");

    let mut max_reference: f64 = 0.0;
    let mut max_error: f64 = 0.0;
    for ((&base_value, &small_value), &double_value) in base_data
        .iter()
        .zip(small_data.iter())
        .zip(double_data.iter())
    {
        let expected = (small_value - base_value) * 2.0;
        let got = double_value - base_value;
        max_reference = max_reference.max(expected.norm());
        max_error = max_error.max((got - expected).norm());
    }

    assert!(
        max_reference > f64::EPSILON.sqrt(),
        "contrast increment must be nonzero"
    );
    assert!(
        max_error <= f64::EPSILON.sqrt() * max_reference.max(1.0),
        "finite-window Born increment must be linear in chi: max_error={max_error}, max_reference={max_reference}"
    );
}

#[test]
fn finite_window_born_rejects_off_grid_ring_geometry() {
    let array = MultiRowRingArray::new(4, 1, 0.012, 0.0).expect("ring array");
    let model = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);

    let error =
        simulate_pstd_finite_window_born_observation(&model, &array, 200_000.0, test_config(), 4)
            .expect_err("off-grid geometry must fail");

    assert!(
        error.to_string().contains("not on the centered grid axis"),
        "{error}"
    );
}

fn test_config() -> PstdFiniteWindowBornConfig {
    PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: 0.005,
        time_step_s: 1.0e-7,
        source_amplitude_pa: 1.0e3,
        cycles_per_frequency: 3,
        frequency_bin_cycles: 1,
    }
}

fn contrast_volume(normalized_slowness_squared_delta: f64) -> Array3<f64> {
    let mut sound_speed = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let reference_slowness = 1.0 / SOUND_SPEED_WATER_SIM;
    let target_slowness_squared =
        reference_slowness * reference_slowness * (1.0 + normalized_slowness_squared_delta);
    sound_speed[[1, 1, 0]] = 1.0 / target_slowness_squared.sqrt();
    sound_speed
}
