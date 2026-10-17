//! Bitwise differential contract for real ring and rotating acquisitions.

use super::super::acquisition::{RingAcquisition, RotatingAcquisition};
use super::super::born_kernel::cell_volume;
use super::super::cbs::real_scattering_potential;
use super::super::forward::simulate_frequency_observation;
use super::*;
use aequitas::systems::si::quantities::Length;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    MultiRowRingArray, RotatingOpposedLinearArray,
};
use leto::Array3;

fn assert_complex_bits(actual: Complex64, expected: Complex64) {
    assert_eq!(actual.re.to_bits(), expected.re.to_bits());
    assert_eq!(actual.im.to_bits(), expected.im.to_bits());
}

/// Geometry and frequency do not depend on slowness. Prepared coefficients
/// must equal the original Green/incident equations bit for bit, including
/// acquisitions whose receiver coordinates change with the transmit.
#[test]
fn prepared_coefficients_match_uncached_equations() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("prepared_coefficients_match_uncached_equations").entered();
    let ring = MultiRowRingArray::new(
        4,
        2,
        Length::from_unit::<Meter>(0.02),
        Length::from_unit::<Meter>(0.001),
    )
    .unwrap();
    let rotating = RotatingOpposedLinearArray::new(3, 0.002, 0.03, 3).unwrap();
    let ring = RingAcquisition::new(&ring);
    let rotating = RotatingAcquisition::new(&rotating);
    for (acquisition, shared_receivers) in [
        (
            &ring as &dyn TransmissionAcquisition,
            Some(ring.receiver_count()),
        ),
        (&rotating as &dyn TransmissionAcquisition, None),
    ] {
        let config = Config::default();
        let mut model = Array3::from_elem([4, 3, 2], config.reference_sound_speed_m_s);
        model[[1, 1, 0]] += 60.0;
        let centers = voxel_centers((4, 3, 2), config.spacing_m);
        for frequency_hz in [180_000.0, 240_000.0] {
            let _arrange = tracing::info_span!("arrange", frequency_hz).entered();
            let observed =
                simulate_frequency_observation(&model, acquisition, frequency_hz, &config).unwrap();
            drop(_arrange);
            let (geometry, retained) = {
                let _act = tracing::info_span!("act").entered();
                let mut geometry = BornGeometry::new(
                    model.shape(),
                    frequency_hz,
                    acquisition,
                    &config,
                    acquisition.transmission_count(),
                );
                let retained = geometry.prepare_products(products::WORKSPACE_BYTES);
                (geometry, retained)
            };
            let _assert = tracing::info_span!("assert").entered();
            let wavenumber = TWO_PI * frequency_hz * (1.0 / config.reference_sound_speed_m_s);
            let min_distance = 0.5 * config.spacing_m;
            let slowness = model.mapv(|value| 1.0 / value);
            let potential = real_scattering_potential(
                TWO_PI * frequency_hz,
                &slowness,
                1.0 / config.reference_sound_speed_m_s,
            )
            .unwrap();
            let volume = cell_volume(config.spacing_m);
            assert_eq!(geometry.transmits.len(), acquisition.transmission_count());
            assert!(retained > 0 && retained <= products::WORKSPACE_BYTES);
            let products = geometry.products.as_ref().unwrap();
            assert_eq!(products.len(), acquisition.transmission_count());
            for (transmit, row) in geometry.transmits.iter().enumerate() {
                let sources = acquisition.sources(transmit);
                let reference = incident_field(sources, &centers, wavenumber, min_distance);
                assert_eq!(row.incident.len(), reference.len());
                for (&actual, &expected) in row.incident.iter().zip(&reference) {
                    assert_complex_bits(actual, expected);
                }
                assert_eq!(row.receivers.len(), acquisition.receiver_count());
                for (receiver_index, &receiver) in
                    acquisition.receivers(transmit).iter().enumerate()
                {
                    let field = &geometry.fields[row.receivers[receiver_index]];
                    assert_eq!(field.len(), centers.len());
                    for (&actual, (_, point)) in field.iter().zip(centers.iter()) {
                        assert_complex_bits(
                            actual,
                            outgoing_green(*point, receiver, wavenumber, min_distance),
                        );
                    }
                    let unscaled = products[transmit].fields::<false>(receiver_index);
                    let unit_scaled = products[transmit].fields::<true>(receiver_index);
                    assert_eq!(unscaled.len(), centers.len());
                    assert_eq!(unit_scaled.len(), centers.len());
                    for index in 0..centers.len() {
                        assert_complex_bits(unscaled[index], reference[index] * field[index]);
                        assert_complex_bits(
                            unit_scaled[index],
                            (Complex64::new(1.0, 0.0) * reference[index]) * field[index],
                        );
                    }
                    let direct = sources
                        .iter()
                        .map(|&source| outgoing_green(source, receiver, wavenumber, min_distance))
                        .sum();
                    assert_complex_bits(row.direct[receiver_index], direct);
                    let scattered: Complex64 = centers
                        .iter()
                        .enumerate()
                        .map(|(index, (_, point))| {
                            reference[index]
                                * outgoing_green(*point, receiver, wavenumber, min_distance)
                                * (potential[index] * volume)
                        })
                        .sum();
                    assert_complex_bits(observed[[transmit, receiver_index]], direct + scattered);
                }
            }
            if let Some(count) = shared_receivers {
                assert_eq!(geometry.fields.len(), count);
            }
        }
    }
}
