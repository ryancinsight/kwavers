//! Exact finite-difference identities at the configured model bounds.

use super::super::super::acquisition::RingAcquisition;
use super::super::super::forward::simulate_frequency_observation;
use super::super::super::types::{Config, FrequencyObservation};
use super::*;
use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};

#[test]
fn clamped_probe_matches_full_gradient_difference() {
    let _subscriber = tracing::subscriber::set_default(super::super::test_support::subscriber());
    let _span = tracing::info_span!("clamped_probe_matches_full_gradient_difference").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let config = Config {
        estimate_source_scaling: false,
        ..Config::default()
    };
    let ring = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.02),
        Length::from_unit::<Meter>(0.0),
    )
    .unwrap();
    let acquisition = RingAcquisition::new(&ring);
    let truth = Array3::from_elem([2, 3, 1], config.reference_sound_speed_m_s);
    let observations = [FrequencyObservation::new(
        180_000.0,
        simulate_frequency_observation(&truth, &acquisition, 180_000.0, &config).unwrap(),
    )];
    let model =
        sound_speed_to_slowness(&Array3::from_elem([2, 3, 1], config.min_sound_speed_m_s)).unwrap();
    let objective = Objective::new(&model, &observations, &acquisition, &config).unwrap();
    let (_, gradient) = objective.evaluate(&model).unwrap();
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let fd_epsilon = super::super::super::gauss_newton::GaussNewtonConfig::default().fd_epsilon;
    drop(arrange);
    for sign in [1.0, -1.0] {
        let direction = Array3::from_elem(model.shape(), sign);
        let actual = {
            let _act = tracing::info_span!("act", sign).entered();
            hessian_vector(
                &model,
                &gradient,
                &direction,
                &objective,
                reference_slowness,
                fd_epsilon,
            )
            .unwrap()
        };
        let _assert = tracing::info_span!("assert").entered();
        let eps = fd_epsilon * reference_slowness;
        let mut trial = model.mapv(|value| value + eps * sign);
        super::super::super::inversion::clamp_slowness(&mut trial, &config);
        let (_, expected) = objective.evaluate(&trial).unwrap();
        for ((&actual, &next), &previous) in actual.iter().zip(expected.iter()).zip(gradient.iter())
        {
            assert_eq!(actual.to_bits(), ((next - previous) / eps).to_bits());
        }
        if sign > 0.0 {
            assert!(actual
                .iter()
                .all(|&value| value.to_bits() == 0.0_f64.to_bits()));
        } else {
            assert!(actual.iter().any(|&value| value != 0.0));
        }
    }
}
