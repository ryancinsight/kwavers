//! Analytic oracles for the physical causal and periodic derivatives.
use aequitas::systems::si::{
    dimensions,
    quantities::{Dimensionless, Frequency, Pressure, Time},
};
use aequitas_runtime::{Exponent, RuntimeDimension};
use eunomia::Complex;
use kwavers_medium::absorption::FractionalDerivative;

fn trace(f: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, f);
}

#[test]
fn half_derivative_has_physical_units_and_composes() {
    trace(|| {
        let d = tracing::info_span!("arrange")
            .in_scope(|| FractionalDerivative::new(Dimensionless::from_base(0.5_f64), 3).unwrap());
        let input = [2., 0., 0., 0.];
        let dt = Time::from_base(4.);
        let y = tracing::info_span!("act")
            .in_scope(|| d.apply(Pressure::from_base(&input[..]), dt).unwrap());
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(y.as_base(), &[1., -0.5, -0.125, -0.0625]);
            assert_eq!(y.dimension().axes()[2], Exponent::ratio(-5, 2).unwrap());
            let twice = d.apply(y.as_deref(), dt).unwrap();
            assert_eq!(twice.as_base(), &[0.5, -0.5, 0., 0.]);
            assert_eq!(
                twice.dimension(),
                &RuntimeDimension::of::<dimensions::Pressure>()
                    .quotient(&RuntimeDimension::of::<dimensions::Time>())
            );
        });
    });
}

#[test]
fn signed_spectrum_dc_and_mismatch_are_checked_before_returning_data() {
    trace(|| {
        let d = tracing::info_span!("arrange")
            .in_scope(|| FractionalDerivative::new(Dimensionless::from_base(0.5_f32), 2).unwrap());
        let bins = [
            Complex::new(3., 0.),
            Complex::new(2., 0.),
            Complex::new(2., 0.),
        ];
        let signal = Pressure::from_base(&bins[..]);
        let axis = [0., 1., -1.].map(Frequency::from_base);
        let y = tracing::info_span!("act")
            .in_scope(|| d.apply_frequency_domain(signal, &axis).unwrap());
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(y.dimension().axes()[2], Exponent::ratio(-5, 2).unwrap());
            assert_eq!(y.as_base()[0], Complex::new(0., 0.));
            let expected = 2. * std::f32::consts::PI.sqrt();
            for (bin, sign) in y.as_base()[1..].iter().zip([1., -1.]) {
                let bound = 4. * f32::EPSILON * expected;
                assert!(
                    (bin.re - expected).abs() <= bound && (bin.im - sign * expected).abs() <= bound
                );
            }
            assert!(d.apply_frequency_domain(signal, &axis[..2]).is_err());
            assert!(d
                .apply_frequency_domain(Pressure::from_base(&bins[..2]), &axis)
                .is_err());
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut bad = axis;
                bad[2] = Frequency::from_base(invalid);
                assert!(d.apply_frequency_domain(signal, &bad).is_err());
            }
            assert_eq!(bins[2], Complex::new(2., 0.));
        });
    });
}

#[test]
fn identity_integral_and_integer_quadrants() {
    trace(|| {
        let bins = [Complex::new(2., -3.)];
        let signal = Pressure::from_base(&bins[..]);
        let dc = [Frequency::from_base(-0.)];
        let unit = [Frequency::from_base(1. / std::f64::consts::TAU)];
        for (order, expected) in [
            (0., bins[0]),
            (1., Complex::new(3., 2.)),
            (2., Complex::new(-2., 3.)),
            (3., Complex::new(-3., -2.)),
            (-1., Complex::new(-3., -2.)),
        ] {
            let d = FractionalDerivative::new(Dimensionless::from_base(order), 3).unwrap();
            assert_eq!(
                d.apply_frequency_domain(signal, &unit).unwrap().as_base(),
                &[expected]
            );
            if order < 0. {
                assert!(d.apply_frequency_domain(signal, &dc).is_err());
                let x = [1., 2., 3.];
                assert_eq!(
                    d.apply(Pressure::from_base(&x[..]), Time::from_base(2.))
                        .unwrap()
                        .as_base(),
                    &[2., 6., 12.]
                );
            } else if order == 0. {
                assert_eq!(
                    d.apply_frequency_domain(signal, &dc).unwrap().as_base(),
                    &bins
                );
            }
        }
        for order in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(FractionalDerivative::new(Dimensionless::from_base(order), 3).is_err());
        }
        let d = FractionalDerivative::new(Dimensionless::from_base(1.), 0).unwrap();
        let empty = Pressure::from_base(&[][..]);
        for dt in [0., -1., f64::INFINITY, f64::NAN] {
            assert!(d.apply(empty, Time::from_base(dt)).is_err());
        }
        assert!(d
            .apply(empty, Time::from_base(1.))
            .unwrap()
            .as_base()
            .is_empty());
        assert_eq!(
            d.apply(Pressure::from_base(&[2., 4.][..]), Time::from_base(2.))
                .unwrap()
                .as_base(),
            &[1., 2.]
        );
    });
}

#[test]
fn representable_outputs_survive_intermediate_range_limits() {
    trace(|| {
        let half = FractionalDerivative::new(Dimensionless::from_base(0.5_f64), 0).unwrap();
        let tiny = f64::from_bits(1);
        let bins = [Complex::new(1.0, 0.0)];
        let result = half
            .apply_frequency_domain(
                Pressure::from_base(&bins[..]),
                &[Frequency::from_base(tiny)],
            )
            .unwrap();
        // real multiplier = sqrt(π) sqrt(f). Both factors are normal.
        let expected = std::f64::consts::PI.sqrt() * tiny.sqrt();
        assert!(
            (result.as_base()[0].re / expected - 1.).abs() <= 8. * f64::EPSILON * tiny.ln().abs()
        );
        // Rotating a least-subnormal input first would round each component
        // to one whole subnormal. The large derivative scale revives both.
        let small = [Complex::new(tiny, 0.)];
        let big = f64::MAX;
        let out = half
            .apply_frequency_domain(
                Pressure::from_base(&small[..]),
                &[Frequency::from_base(big)],
            )
            .unwrap();
        let expected = (tiny * big.sqrt()) * std::f64::consts::PI.sqrt();
        assert!((out.as_base()[0].re / expected - 1.).abs() <= 8. * f64::EPSILON * big.ln());
        let causal = FractionalDerivative::new(Dimensionless::from_base(0.5), 1).unwrap();
        let out = causal
            .apply(Pressure::from_base(&[tiny, 0.][..]), Time::from_base(0.25))
            .unwrap();
        assert_eq!(out.as_base()[1], -tiny);

        let bins = [Complex::new(f64::MAX, f64::MAX)];
        let result = half
            .apply_frequency_domain(
                Pressure::from_base(&bins[..]),
                &[Frequency::from_base(0.125 / std::f64::consts::TAU)],
            )
            .unwrap();
        let expected = f64::MAX * 0.5;
        assert!((result.as_base()[0].im / expected - 1.).abs() <= 8. * f64::EPSILON);

        let first = FractionalDerivative::new(Dimensionless::from_base(1.0), 1).unwrap();
        let out = first
            .apply(
                Pressure::from_base(&[-f64::MAX, f64::MAX][..]),
                Time::from_base(2.0),
            )
            .unwrap();
        assert_eq!(out.as_base()[1], f64::MAX);
    });
}

proptest::proptest! {
    #[test]
    fn two_spectral_half_derivatives_match_one_time_derivative(
        re in -1e3_f64..1e3_f64, im in -1e3_f64..1e3_f64, hz in -1e4_f64..1e4_f64
    ) {
        let half = FractionalDerivative::new(Dimensionless::from_base(0.5), 0).unwrap();
        let data = [Complex::new(re, im)];
        let frequencies = [Frequency::from_base(hz)];
        let first = half.apply_frequency_domain(Pressure::from_base(&data[..]), &frequencies).unwrap();
        let second = half.apply_frequency_domain(first.as_deref(), &frequencies).unwrap();
        let omega = std::f64::consts::TAU * hz;
        let z = second.as_base()[0];
        // Two rounded rotations and magnitude products. Use the input
        // one-norm to bound cancellation in either Cartesian component.
        let bound = 32. * f64::EPSILON * omega.abs() * (re.abs() + im.abs());
        proptest::prop_assert!((z.re + omega * im).abs() <= bound);
        proptest::prop_assert!((z.im - omega * re).abs() <= bound);
        proptest::prop_assert_eq!(second.dimension(), &RuntimeDimension::of::<dimensions::Pressure>()
            .quotient(&RuntimeDimension::of::<dimensions::Time>()));
    }
}

#[test]
fn causal_history_crosses_output_tiles_at_the_correct_lag() {
    trace(|| {
        for n in [1, 255, 256, 257, 511, 512, 513] {
            let input: Vec<f32> = (1..=n)
                .map(|i| f32::from(u16::try_from(i).unwrap()))
                .collect();
            for lag in [0, 1, 32, 256, usize::MAX] {
                let d = FractionalDerivative::new(Dimensionless::from_base(-1.0), lag).unwrap();
                let out = d
                    .apply(Pressure::from_base(&input[..]), Time::from_base(0.5))
                    .unwrap();
                for (i, &value) in out.as_base().iter().enumerate() {
                    let first = i.saturating_sub(lag);
                    // Order −1 weights are all unity. A finite-memory integral
                    // of this ramp is the arithmetic sum times the timestep.
                    let start = input[first];
                    let end = input[i];
                    let count = f32::from(u16::try_from(i - first + 1).unwrap());
                    assert_eq!(value, (start + end) * count * 0.25);
                }
            }
        }
    });
}

#[test]
fn ordinary_bins_agree_between_bulk_and_extended_range_paths() {
    trace(|| {
        let samples = [
            Complex::new(1.25, -4.5),
            Complex::new(-2.0, 0.5),
            Complex::new(3.0, 2.0),
            Complex::new(f64::MAX, 0.0),
        ];
        let frequencies = [3.25, -7.5, 0.0, 0.0].map(Frequency::from_base);
        for alpha in [0.25, 0.5, 1.0] {
            let derivative = FractionalDerivative::new(Dimensionless::from_base(alpha), 8).unwrap();
            let bulk = derivative
                .apply_frequency_domain(Pressure::from_base(&samples[..3]), &frequencies[..3])
                .unwrap();
            let extended = derivative
                .apply_frequency_domain(Pressure::from_base(&samples[..]), &frequencies)
                .unwrap();
            assert_eq!(bulk.as_base(), &extended.as_base()[..3]);
            assert_eq!(extended.as_base()[3], Complex::new(0.0, 0.0));
            assert_eq!(bulk.dimension(), extended.dimension());
        }
    });
}
