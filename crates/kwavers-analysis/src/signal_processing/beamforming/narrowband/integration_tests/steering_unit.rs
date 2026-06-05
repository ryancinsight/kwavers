//! Steering-vector unit-magnitude integration tests.

use super::super::steering::NarrowbandSteering;
use super::helpers::generate_ula_positions;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn steering_vector_has_unit_magnitude_for_all_sensors() {
    // Integration test verifying steering vector normalization across the pipeline.

    let n_sensors = 6;
    let spacing_m = 0.01;
    let positions = generate_ula_positions(n_sensors, spacing_m);
    let c = SOUND_SPEED_WATER_SIM;
    let f0 = 50_000.0;

    let steering = NarrowbandSteering::new(positions.clone(), c).expect("steering init");

    let candidates = vec![[0.0, 0.0, 0.05], [0.02, 0.0, 0.05], [-0.01, 0.01, 0.08]];

    for candidate in candidates {
        let sv = steering
            .steering_vector_point(candidate, f0)
            .expect("steering vector");

        assert_eq!(sv.as_array().len(), n_sensors);

        for &element in sv.as_array().iter() {
            let magnitude: f64 = element.norm();
            assert!(
                (magnitude - 1.0).abs() < 1e-10,
                "Steering vector element should have unit magnitude, got {:.6}",
                magnitude
            );
        }
    }
}
