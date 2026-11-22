//! Physical invariants of the homogeneous absorption step.
use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Length, ReciprocalLength, Time, Velocity, Wavenumber,
};
use aequitas::systems::si::units::{Millimeter, Millisecond};
use eunomia::Complex64;
use kwavers_medium::absorption::{AbsorptionCalculator, AbsorptionModel, FractionalLaplacian};
use leto::Array3;
use std::f64::consts::TAU;

fn trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

#[test]
fn calibrated_gain_is_unit_invariant_and_composes() {
    trace(|| {
        let second = Time::from_base(1.0);
        let metre = Length::from_base(1.0);
        let alternate_time = Time::from_unit::<Millisecond>(1000.0);
        let alternate_length = Length::from_unit::<Millimeter>(1000.0);
        let one = Dimensionless::from_base(1.0);
        for (length, time) in [(metre, second), (alternate_length, alternate_time)] {
            let (model, ks, mut field) = tracing::info_span!("arrange").in_scope(|| {
                let model = FractionalLaplacian::new(
                    one / (length * 2.0),
                    one / (time * TAU),
                    Dimensionless::from_base(1.5),
                )
                .unwrap();
                let ks = Array3::from_shape_fn([1, 2, 2], |[_, j, k]| {
                    one / length * [0., 1., 4., 1.][2 * j + k]
                });
                (
                    model,
                    ks,
                    Array3::from_elem([1, 2, 2], Complex64::new(1.0, -2.0)),
                )
            });
            tracing::info_span!("act").in_scope(|| {
                let half = model.step(length / time, time / 2.).unwrap();
                half.apply_k_space(&mut field, &ks).unwrap();
                half.apply_k_space(&mut field, &ks).unwrap();
            });
            tracing::info_span!("assert").in_scope(|| {
                let calc = AbsorptionCalculator::new(AbsorptionModel::Fractional(model));
                assert_eq!(
                    calc.absorption_coefficient(one / (time * TAU)).unwrap(),
                    ReciprocalLength::from_base(0.5)
                );
                for (got, loss) in field.iter().zip([0.0_f64, 0.5, 4., 0.5]) {
                    let expected = Complex64::new(1., -2.) * (-loss).exp();
                    assert!((*got - expected).norm() <= 8.0 * f64::EPSILON * expected.norm());
                }
            });
        }
    });
}

#[test]
fn malformed_modes_are_atomic_and_constant_loss_includes_dc() {
    trace(|| {
        let (model, mut field) = tracing::info_span!("arrange").in_scope(|| {
            (
                FractionalLaplacian::new(
                    ReciprocalLength::from_base(0.25),
                    Frequency::from_base(1.),
                    Dimensionless::from_base(0.),
                )
                .unwrap(),
                Array3::from_elem([2, 1, 1], Complex64::new(-2., 3.)),
            )
        });
        let speed = Velocity::from_base(2.);
        let step =
            tracing::info_span!("act").in_scope(|| model.step(speed, Time::from_base(2.)).unwrap());
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                step.gain(Wavenumber::from_base(0.)).unwrap().into_base(),
                (-1.0_f64).exp()
            );
            assert_eq!(
                model
                    .step(speed, Time::from_base(0.))
                    .unwrap()
                    .gain(Wavenumber::from_base(2.))
                    .unwrap()
                    .into_base(),
                1.
            );
            assert!(model.step(speed, Time::from_base(-1.)).is_err());
            assert!(model
                .step(Velocity::from_base(0.), Time::from_base(1.))
                .is_err());
            assert!(FractionalLaplacian::new(
                ReciprocalLength::from_base(-1.),
                Frequency::from_base(1.),
                Dimensionless::from_base(2.)
            )
            .is_err());
            let before = field.clone();
            for axis in [
                vec![1.],
                vec![1., f64::NAN],
                vec![1., -1.],
                vec![1., f64::INFINITY],
            ] {
                let ks = Array3::from_shape_fn([axis.len(), 1, 1], |[i, _, _]| {
                    Wavenumber::from_base(axis[i])
                });
                assert!(step.apply_k_space(&mut field, &ks).is_err());
                assert_eq!(field, before);
            }
        });
    });
}

#[test]
fn representable_gain_survives_an_out_of_range_power() {
    trace(|| {
        // (2^±600)^2 alone over/underflows. The reciprocal calibration
        // and path length make the loss exponent exactly one in both cases.
        for power in [-600, 600] {
            let (model, frequency, distance) =
                tracing::info_span!("arrange", power).in_scope(|| {
                    let frequency = 2.0_f64.powi(power);
                    let model = FractionalLaplacian::new(
                        ReciprocalLength::from_base(frequency.recip()),
                        Frequency::from_base(1.),
                        Dimensionless::from_base(2.),
                    )
                    .unwrap();
                    (model, frequency, frequency.recip())
                });
            let gain = tracing::info_span!("act").in_scope(|| {
                model
                    .step(Velocity::from_base(1.), Time::from_base(distance))
                    .unwrap()
                    .gain(Wavenumber::from_base(TAU * frequency))
                    .unwrap()
            });
            tracing::info_span!("assert")
                .in_scope(|| assert_eq!(gain.into_base(), (-1.0_f64).exp()));
        }
    });
}
