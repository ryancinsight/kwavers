use crate::signal_processing::b_mode::TgcConfig;
use aequitas::systems::si::quantities::{Frequency, Length, ReciprocalLength, Velocity};
use eunomia::RealField;
use kwavers_medium::absorption::power_law::reference_loss_from_decibels_per_centimetre;
use leto::{Array1, Layout, VecStorage};
use tracing_subscriber::fmt::format::FmtSpan;

fn trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

#[test]
fn tgc_flattens_attenuated_echoes() {
    trace(|| {
        let (cfg, rf) = tracing::info_span!("arrange").in_scope(|| {
            let reference = Frequency::from_base(1e6);
            let cfg = TgcConfig::new(
                reference_loss_from_decibels_per_centimetre(0.5) / reference,
                Frequency::from_base(5e6),
                Velocity::from_base(1540.0),
                Frequency::from_base(40e6),
            )
            .unwrap();
            // Independent dB-domain echoes at uniform signed reflectivity.
            let rf = Array1::from_shape_fn(8000, |[i]| {
                let depth_cm = 100.0 * 1540.0 * i as f64 / 80e6;
                let amplitude = if i % 2 == 0 { 2.0 } else { -0.5 };
                amplitude * 10.0_f64.powf(-2.0 * 0.5 * 5.0 * depth_cm / 20.0)
            });
            (cfg, rf)
        });
        let (corrected, curve) = tracing::info_span!("act")
            .in_scope(|| (cfg.apply(&rf).unwrap(), cfg.gain_curve(rf.len()).unwrap()));
        tracing::info_span!("assert").in_scope(|| {
            // The largest natural exponent here is < 9. A relative error of
            // eight epsilons across conversions and products perturbs exp by
            // at most exp(9·8ε)-1, plus the final multiply and oracle rounding.
            for i in 0..rf.len() {
                let expected: f64 = if i % 2 == 0 { 2.0 } else { -0.5 };
                assert!((corrected[i] - expected).abs() <= 80.0 * f64::EPSILON * expected.abs());
                assert_eq!(curve[i], cfg.gain(i).unwrap().into_base());
            }
            assert_eq!(cfg.depth(0).unwrap(), Length::from_base(0.0));
            assert_eq!(cfg.depth(4000).unwrap(), Length::from_base(0.077));
            assert_eq!(curve[0], 1.0);
        });
    });
}

#[test]
fn tgc_rejects_nonfinite_parameters_and_negative_absorption() {
    trace(|| {
        let invalid = tracing::info_span!("arrange")
            .in_scope(|| [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0]);
        for v in invalid {
            for slot in 0..4 {
                let mut inputs = [1.0; 4];
                inputs[slot] = v;
                let [a, f, c, r] = inputs;
                let result = tracing::info_span!("act", slot, v).in_scope(|| {
                    TgcConfig::new(
                        ReciprocalLength::from_base(a) / Frequency::from_base(1.0),
                        Frequency::from_base(f),
                        Velocity::from_base(c),
                        Frequency::from_base(r),
                    )
                });
                tracing::info_span!("assert").in_scope(|| {
                    if slot == 0 && v == 0.0 {
                        assert_eq!(result.unwrap().gain(usize::MAX).unwrap().into_base(), 1.0);
                    } else {
                        assert!(matches!(
                            result,
                            Err(kwavers_core::error::KwaversError::InvalidInput(_))
                        ));
                    }
                });
            }
        }
    });
}

#[test]
fn tgc_in_place_uses_logical_order_and_rejects_overflow_atomically() {
    fn check<T: RealField>() {
        let (cfg, mut line) = tracing::info_span!("arrange").in_scope(|| {
            let f = Frequency::from_base(T::ONE);
            let cfg = TgcConfig::new(
                ReciprocalLength::from_base(T::LN_2) / f,
                f,
                Velocity::from_base(T::ONE),
                f,
            )
            .unwrap();
            let line = Array1::new(
                Layout::try_new([3], [-1], 2).unwrap(),
                VecStorage::new(vec![T::ONE; 3]),
            )
            .unwrap();
            (cfg, line)
        });
        tracing::info_span!("act").in_scope(|| cfg.apply_in_place(&mut line).unwrap());
        tracing::info_span!("assert").in_scope(|| {
            for (i, expected) in [T::ONE, T::from_count(2), T::from_count(4)]
                .into_iter()
                .enumerate()
            {
                assert!((line[i] - expected).abs() <= expected * T::EPSILON);
            }
            let mut huge = Array1::from(vec![-T::ONE; 2048]);
            assert!(cfg.apply_in_place(&mut huge).is_err());
            assert!(huge.iter().all(|&v| v == -T::ONE));
            assert!(cfg.gain_curve(2048).is_err());
            assert!(cfg.gain_curve(0).unwrap().is_empty());
        });
    }
    trace(|| {
        check::<f32>();
        check::<f64>();
    });
}

#[test]
fn tgc_calibration_keeps_representable_products_at_extreme_scales() {
    trace(|| {
        let (fast, slow) =
            tracing::info_span!("arrange").in_scope(|| (f64::MAX, f64::MIN_POSITIVE));
        let f = Frequency::from_base(1.0);
        tracing::info_span!("act").in_scope(|| {
            for scale in [fast, slow] {
                let cfg = TgcConfig::new(
                    ReciprocalLength::from_base(scale) / f,
                    Frequency::from_base(scale.recip()),
                    Velocity::from_base(scale),
                    Frequency::from_base(scale),
                )
                .unwrap();
                tracing::info_span!("assert").in_scope(|| {
                    assert_eq!(cfg.depth(2).unwrap(), Length::from_base(1.0));
                    // Rounding the reciprocal and exponential each costs at most a few ulps.
                    let e = std::f64::consts::E;
                    assert!((cfg.gain(1).unwrap().into_base() - e).abs() <= 4.0 * f64::EPSILON * e);
                });
            }
        });
    });
}
