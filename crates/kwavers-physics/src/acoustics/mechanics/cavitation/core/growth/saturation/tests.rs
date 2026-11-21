use super::*;
use crate::acoustics::mechanics::cavitation::core::rectified_diffusion_rate;
use eunomia::assert_relative_eq;
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

const P: f64 = 101_325.0;
const R: f64 = 42.5e-6;
const D: f64 = 2e-9;
const C: f64 = 0.26;

fn with_trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

#[test]
fn far_field_changes_signed_flux_at_fixed_solubility() {
    with_trace(|| {
        // Zhang Eqs. (3.6), (3.17)–(3.19). Independent complex response
        // and binomial moments give flux references on both sides of zero.
        let fixtures = info_span!("arrange").in_scope(|| {
            [
                (0.99, -4.035_273_487_577_627_4e-15),
                (1.00, -1.245_070_203_197_059_7e-15),
                (1.01, 1.545_133_081_183_508_5e-15),
                (1.05, 1.270_594_621_870_578e-14),
            ]
        });
        let actual = info_span!("act").in_scope(|| {
            fixtures.map(|(saturation, _)| {
                rectified_diffusion_rate_with_concentrations(
                    R,
                    P,
                    20e3,
                    26.6e3,
                    D,
                    C,
                    C * saturation,
                )
                .unwrap()
            })
        });
        info_span!("assert").in_scope(|| {
            for ((_, expected), flux) in fixtures.into_iter().zip(actual) {
                assert_relative_eq!(flux, expected, max_relative = 2e-11, epsilon = 1e-25);
            }
            assert_relative_eq!(
                actual[1],
                rectified_diffusion_rate(R, P, 20e3, 26.6e3, D, C),
                epsilon = 1e-27
            );
        });
    });
}

#[test]
fn acoustic_off_recovers_spherical_henry_laplace_diffusion() {
    with_trace(|| {
        let (wall, conductance, bulk) = info_span!("arrange").in_scope(|| {
            let wall = C * (1.0 + 2.0 * SURFACE_TENSION_WATER / (R * P));
            let conductance = 4.0 * std::f64::consts::PI * R * D;
            let below = f64::from_bits(wall.to_bits() - 1);
            let above = f64::from_bits(wall.to_bits() + 1);
            (wall, conductance, [0.0, below, wall, above, 0.30])
        });
        let (actual, no_transfer) = info_span!("act").in_scope(|| {
            let actual = bulk.map(|bulk| {
                rectified_diffusion_rate_with_concentrations(R, P, 0.0, 26.6e3, D, C, bulk).unwrap()
            });
            let no_transfer =
                rectified_diffusion_rate_with_concentrations(R, P, 20e3, 26.6e3, 0.0, C, 0.0)
                    .unwrap();
            (actual, no_transfer)
        });
        info_span!("assert").in_scope(|| {
            for (bulk, actual) in bulk.into_iter().zip(actual) {
                assert_relative_eq!(actual, conductance * (bulk - wall), epsilon = 1e-28);
            }
            assert!(actual[0] < 0.0 && actual[1] < 0.0 && actual[3] > 0.0);
            assert_eq!(no_transfer, 0.0);
        });
    });
}

#[test]
fn rejects_invalid_concentration_and_large_amplitude() {
    with_trace(|| {
        let bulk = info_span!("arrange").in_scope(|| [-0.001, f64::NAN, f64::INFINITY]);
        let (invalid_bulk, high_drive, large_amplitude) = info_span!("act").in_scope(|| {
            let invalid_bulk = bulk.map(|bulk| {
                rectified_diffusion_rate_with_concentrations(R, P, 20e3, 26.6e3, D, C, bulk)
            });
            let high_drive =
                rectified_diffusion_rate_with_concentrations(R, P, 6.7e6, 26.6e3, D, C, C);
            let large_amplitude =
                rectified_diffusion_rate_with_concentrations(120e-6, P, 100e3, 26.6e3, D, C, C);
            (invalid_bulk, high_drive, large_amplitude)
        });
        info_span!("assert").in_scope(|| {
            for result in invalid_bulk {
                assert!(matches!(
                    result,
                    Err(KwaversError::Physics(PhysicsError::InvalidParameter { parameter, .. }))
                        if parameter == "far_field_concentration"
                ));
            }
            assert!(matches!(
                high_drive,
                Err(KwaversError::Physics(PhysicsError::InvalidParameter { parameter, .. }))
                    if parameter == "acoustic_pressure / ambient_pressure"
            ));
            assert!(matches!(
                large_amplitude,
                Err(KwaversError::Physics(PhysicsError::InvalidParameter { parameter, value, .. }))
                    if parameter == "epsilon" && value > 3.0
            ));
        });
    });
}
