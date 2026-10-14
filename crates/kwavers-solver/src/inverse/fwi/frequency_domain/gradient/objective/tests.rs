//! Prepared geometry must never retain model-dependent results.

use super::super::super::acquisition::RingAcquisition;
use super::super::super::forward::{predict_frequency_rows, simulate_frequency_observation};
use super::*;
use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    complex_l2_objective, complex_source_scale, sound_speed_to_slowness, MultiRowRingArray,
};

#[test]
fn prepared_objective_recomputes_changed_models_and_rejects_new_shape() {
    let _subscriber = tracing::subscriber::set_default(super::super::test_support::subscriber());
    let _span = tracing::info_span!("prepared_objective_recomputes_changed_models").entered();
    let _arrange = tracing::info_span!("arrange").entered();
    let array = MultiRowRingArray::new(
        4,
        2,
        Length::from_unit::<Meter>(0.02),
        Length::from_unit::<Meter>(0.001),
    )
    .unwrap();
    let acquisition = RingAcquisition::new(&array);
    let config = Config::default();
    let background = Array3::from_elem([4, 3, 2], config.reference_sound_speed_m_s);
    let observations: Vec<_> = [180_000.0, 240_000.0]
        .iter()
        .map(|&frequency| {
            FrequencyObservation::new(
                frequency,
                simulate_frequency_observation(&background, &acquisition, frequency, &config)
                    .unwrap(),
            )
        })
        .collect();
    let baseline = sound_speed_to_slowness(&background).unwrap();
    let mut inclusion = background.clone();
    // Same +60 m/s lesion contrast as the unchanged full reconstruction cases.
    inclusion[[1, 1, 0]] += 60.0;
    let inclusion = sound_speed_to_slowness(&inclusion).unwrap();
    drop(_arrange);
    for estimate_source_scaling in [false, true] {
        let config = Config {
            estimate_source_scaling,
            ..config.clone()
        };
        let experiment = Objective::new(&baseline, &observations, &acquisition, &config).unwrap();
        for (model, changed) in [(&baseline, false), (&inclusion, true), (&baseline, false)] {
            let (actual, gradient) = {
                let _act = tracing::info_span!("act", estimate_source_scaling).entered();
                experiment.evaluate(model).unwrap()
            };
            let _assert = tracing::info_span!("assert").entered();
            let mut expected = 0.0;
            for observation in &observations {
                let predicted = predict_frequency_rows(
                    model,
                    &acquisition,
                    observation.frequency_hz,
                    &config,
                    acquisition.transmission_count(),
                )
                .unwrap();
                for transmit in 0..acquisition.transmission_count() {
                    let predicted = predicted
                        .index_axis::<1>(0, transmit)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    let observed = observation
                        .observed_pressure
                        .index_axis::<1>(0, transmit)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    let scale = if estimate_source_scaling {
                        complex_source_scale(&predicted, &observed).unwrap()
                    } else {
                        Complex64::new(1.0, 0.0)
                    };
                    let predicted = predicted
                        .iter()
                        .map(|&value| scale * value)
                        .collect::<Vec<_>>();
                    expected += complex_l2_objective(&predicted, &observed).unwrap();
                }
            }
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(gradient.shape(), model.shape());
            if changed {
                assert!(actual > 0.0);
                assert!(gradient.iter().any(|&value| value != 0.0));
            }
        }
        let different = Array3::from_elem([3, 4, 2], baseline[[0, 0, 0]]);
        match experiment.evaluate(&different).unwrap_err() {
            KwaversError::DimensionMismatch(message) => assert_eq!(
                message,
                format!(
                    "prepared objective shape {:?}, model shape {:?}",
                    baseline.shape(),
                    different.shape()
                )
            ),
            error => panic!("unexpected error: {error}"),
        }
    }
}
