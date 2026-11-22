//! Analytic group slowness and signed spectral phase for the typed law.
use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, ReciprocalLength, Time, Velocity, Wavenumber,
};
use eunomia::Complex64;
use kwavers_medium::absorption::{AbsorptionDispersionCorrection, PowerLawAbsorption};
use leto::Array3;
use std::f64::consts::{FRAC_PI_2, TAU};

fn trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

fn law(loss: f64, exponent: f64) -> PowerLawAbsorption {
    PowerLawAbsorption::new(
        ReciprocalLength::from_base(loss),
        Frequency::from_base(1.0),
        Dimensionless::from_base(exponent),
        true,
    )
    .unwrap()
}

#[test]
fn group_velocity_at_dc_for_lossless_medium_is_sound_speed() {
    trace(|| {
        let model = tracing::info_span!("arrange").in_scope(|| law(0.0, 1.5));
        let speed = Velocity::from_base(1500.0);
        let group = tracing::info_span!("act").in_scope(|| {
            model
                .group_velocity(Frequency::from_base(0.0), speed)
                .unwrap()
        });
        tracing::info_span!("assert").in_scope(|| assert_eq!(group, speed));
    });
}

#[test]
fn phase_and_group_match_the_dispersion_relation_and_its_derivative() {
    trace(|| {
        let (speed, f) = tracing::info_span!("arrange")
            .in_scope(|| (Velocity::from_base(TAU), Frequency::from_base(1.0)));
        for (y, loss, q) in [(0.5, 1.0, 1.0), (1.5, 0.5, -0.5), (1.5, 0.75, -0.75)] {
            let model = law(loss, y);
            let (phase, group) = tracing::info_span!("act", y, loss).in_scope(|| {
                (
                    model.phase_velocity(f, speed).unwrap(),
                    model.group_velocity(f, speed).unwrap(),
                )
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(phase, speed / (1.0 + q));
                assert_eq!(group, speed / (1.0 + y * q));
            });
        }
        let dc = Frequency::from_base(0.0);
        let zero = Velocity::from_base(0.0);
        assert_eq!(law(1.0, 0.5).phase_velocity(dc, speed).unwrap(), zero);
        assert_eq!(law(1.0, 0.5).group_velocity(dc, speed).unwrap(), zero);
        assert_eq!(law(1.0, 1.5).group_velocity(dc, speed).unwrap(), speed);
        assert!(law(1.0, 1.0).phase_velocity(f, speed).is_err());
        assert!(law(1.0, 1.5).phase_velocity(f, speed).is_err());
    });
}

#[test]
fn spectral_rotation_keeps_norm_and_opposite_wavenumbers_conjugate() {
    trace(|| {
        let (operator, mut spectrum, mut ks) = tracing::info_span!("arrange").in_scope(|| {
            let operator =
                AbsorptionDispersionCorrection::new(law(1.0, 0.5), Velocity::from_base(TAU))
                    .unwrap();
            (
                operator,
                Array3::from_elem([1, 1, 3], Complex64::new(2.0, 0.0)),
                Array3::from_shape_fn([1, 1, 3], |[_, _, k]| Wavenumber::from_base(k as f64 - 1.0)),
            )
        });
        // |k|=1 gives f=1 and cₚ=c₀/2, so Δt=1/2 rotates by ∓π/2.
        tracing::info_span!("act").in_scope(|| {
            operator
                .apply_k_space(&mut spectrum, &ks, Time::from_base(0.5))
                .unwrap()
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                operator
                    .dispersion_relation(Frequency::from_base(1.0))
                    .unwrap(),
                Wavenumber::from_base(2.0)
            );
            assert_eq!(
                operator
                    .dispersion_relation(Frequency::from_base(0.0))
                    .unwrap(),
                Wavenumber::from_base(0.0)
            );
            for (k, angle) in [FRAC_PI_2, 0.0, -FRAC_PI_2].into_iter().enumerate() {
                let v = spectrum[[0, 0, k]];
                assert!((v.re - 2.0 * angle.cos()).abs() <= 4.0 * f64::EPSILON);
                assert!((v.im - 2.0 * angle.sin()).abs() <= 4.0 * f64::EPSILON);
            }
            ks[[0, 0, 2]] = Wavenumber::from_base(f64::NAN);
            let before = spectrum.clone();
            assert!(operator
                .apply_k_space(&mut spectrum, &ks, Time::from_base(0.5))
                .is_err());
            assert_eq!(spectrum, before);
        });
    });
}

#[test]
fn phase_velocity_survives_overflow_of_the_dimensionless_correction() {
    trace(|| {
        // At the reference and y=1/2, cₚ = τ/(τ/c₀ + α_ref).
        let (model, speed) = tracing::info_span!("arrange")
            .in_scope(|| (law(f64::MAX, 0.5), Velocity::from_base(f64::MAX)));
        let cp = tracing::info_span!("act").in_scope(|| {
            model
                .phase_velocity(Frequency::from_base(1.0), speed)
                .unwrap()
        });
        tracing::info_span!("assert").in_scope(|| {
            let expected = TAU / f64::MAX;
            // Four rounded logarithms/additions of size at most 2 ln(MAX)
            // govern the relative forward error after the final exponential.
            let bound = 8.0 * f64::EPSILON * f64::MAX.ln();
            assert!((cp.into_base() / expected - 1.0).abs() <= bound);
        });
    });
}

#[test]
fn group_velocity_survives_an_unweighted_correction_above_maximum() {
    trace(|| {
        let (model, speed) = tracing::info_span!("arrange")
            .in_scope(|| (law(1.5 * TAU, 0.5), Velocity::from_base(f64::MAX)));
        let group = tracing::info_span!("act")
            .in_scope(|| model.group_velocity(Frequency::from_base(1.0), speed));
        tracing::info_span!("assert").in_scope(|| {
            // q = 1.5 c₀ is outside binary64, but y q = 0.75 c₀ fits.
            let expected = 1.0 / (1.0 / f64::MAX + 0.75);
            let bound = 8.0 * f64::EPSILON * f64::MAX.ln();
            assert!((group.unwrap().into_base() / expected - 1.0).abs() <= bound);
        });
    });
}
