use super::*;
use eunomia::assert_relative_eq;
use eunomia::Complex;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use leto::Array3;

#[test]
fn apply_frequency_domain_scales_frequency_planes() {
    let config = PowerLawAbsorption::new(
        ReciprocalLength::from_base(2.0),
        Frequency::from_base(MHZ_TO_HZ),
        Dimensionless::from_base(1.0),
        false,
    )
    .expect("valid loss");
    let model = PowerLawModel::new(config);
    let initial = Complex::new(2.0, -1.0);
    let mut spectrum = Array3::from_elem([2, 2, 1], initial);
    let distance = 0.25;

    model
        .apply_frequency_domain(
            &mut spectrum,
            &[0.0, MHZ_TO_HZ].map(Frequency::from_base),
            Length::from_base(distance),
        )
        .expect("valid spectrum");

    assert_eq!(spectrum[[0, 0, 0]], initial);
    assert_eq!(spectrum[[0, 1, 0]], initial);

    let attenuation = (-2.0 * distance).exp();
    let expected = initial * attenuation;
    for j in 0..2 {
        assert_relative_eq!(spectrum[[1, j, 0]].re, expected.re, epsilon = 1e-12);
        assert_relative_eq!(spectrum[[1, j, 0]].im, expected.im, epsilon = 1e-12);
    }
}

#[test]
fn apply_frequency_domain_rejects_frequency_axis_mismatch() {
    let model = PowerLawModel::new(PowerLawAbsorption::default());
    let mut spectrum = Array3::from_elem([2, 1, 1], Complex::new(1.0, 0.0));

    assert!(model
        .apply_frequency_domain(
            &mut spectrum,
            &[Frequency::from_base(MHZ_TO_HZ)],
            Length::from_base(1.0)
        )
        .is_err());
}

#[test]
fn np_m_conversion_round_trips_against_its_inverse() {
    for &(alpha_db, y, f) in &[
        (0.5_f64, 1.1_f64, 1.0e6_f64),
        (0.75, 1.5, 3.0e6),
        (0.25, 0.4, 5.0e5),
    ] {
        let np_m = power_law_db_cm_to_np_m(alpha_db, y, f);
        let back = np_m_to_power_law_db_cm(np_m, f, y);
        assert!(
            (back - alpha_db).abs() < 1e-12 * alpha_db,
            "round trip {alpha_db} -> {np_m} -> {back}"
        );
    }
    // At the reference frequency the exponent drops out and the conversion
    // is exactly dB/cm -> Np/m.
    let at_ref = power_law_db_cm_to_np_m(1.0, 1.7, 1.0e6);
    assert!((at_ref - 100.0 / 8.685_889_638_065_035).abs() < 1e-12);
    assert_eq!(power_law_db_cm_to_np_m(1.0, 1.0, 0.0), 0.0);
}

#[test]
fn test_power_law_db_cm_to_np_omega_m_matches_kwave_reference() {
    let got = power_law_db_cm_to_np_omega_m(0.75, 1.5);
    let expected: f64 = 5.482_481_235_081_536e-10;
    // The conversion contains six rounded elementary operations and a
    // libm `powf` call. Sixteen ulps bounds this composed forward error
    // at the reference magnitude while retaining k-Wave as the oracle.
    let tolerance = 16.0 * f64::EPSILON * expected.abs().max(got.abs());
    assert!(
        (got - expected).abs() <= tolerance,
        "conversion mismatch: got {got}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn np_m_to_db_cm_round_trips_at_frequency() {
    // A target attenuation [Np/m] at a drive frequency must convert to a
    // power-law prefactor that reproduces exactly that attenuation at f.
    let alpha_target_np_m = 37.0;
    let f = 0.5e6;
    let y = 1.1;
    let db_cm = np_m_to_power_law_db_cm(alpha_target_np_m, f, y);
    let alpha_si = power_law_db_cm_to_np_omega_m(db_cm, y);
    let omega = 2.0 * std::f64::consts::PI * f;
    let recovered = alpha_si * omega.powf(y);
    assert!(
        (recovered - alpha_target_np_m).abs() <= 1e-9 * alpha_target_np_m,
        "round-trip: target={alpha_target_np_m} Np/m, recovered={recovered} Np/m"
    );
}

#[test]
fn np_m_to_db_cm_guards_nonpositive_frequency() {
    assert_eq!(np_m_to_power_law_db_cm(10.0, 0.0, 1.1), 0.0);
    assert_eq!(np_m_to_power_law_db_cm(10.0, -1.0, 1.1), 0.0);
}

#[test]
fn omega_inverse_round_trips_against_forward() {
    for &(db, y) in &[(0.75_f64, 1.5_f64), (0.5, 1.1), (2.0, 2.0)] {
        let si = power_law_db_cm_to_np_omega_m(db, y);
        let back = np_omega_m_to_power_law_db_cm(si, y);
        assert!(
            (back - db).abs() <= 8.0 * f64::EPSILON * db.abs().max(1e-300),
            "omega round trip {db} -> {si} -> {back}"
        );
    }
}

#[test]
fn hz_prefactor_matches_at_frequency_form_at_1hz() {
    // The Np/(m·Hz^y) prefactor is the at-frequency form at 1 Hz exactly.
    for &(alpha_db, y) in &[(0.5_f64, 1.1_f64), (0.75, 1.5), (2.0, 2.0)] {
        let prefac = power_law_db_cm_to_np_m_hz(alpha_db, y);
        let at_1hz = power_law_db_cm_to_np_m(alpha_db, y, 1.0);
        assert!(
            (prefac - at_1hz).abs() <= 4.0 * f64::EPSILON * at_1hz.abs().max(1e-300),
            "Hz prefactor {prefac} != at-1Hz {at_1hz}"
        );
    }
}
#[test]
fn fractional_absorption_uses_declared_reference_and_rejects_invalid_axis_atomically() {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let (law, reference) = tracing::info_span!("arrange").in_scope(|| {
            let reference = Frequency::from_base(2.0 * MHZ_TO_HZ);
            let law = PowerLawAbsorption::new(
                ReciprocalLength::from_base(6.0),
                reference,
                Dimensionless::from_base(1.5),
                false,
            )
            .expect("valid reference loss");
            (law, reference)
        });
        let (loss_at_reference, loss_at_four_times_reference, untouched, failure) =
            tracing::info_span!("act").in_scope(|| {
                let mut wave = Array3::from_elem([2, 1, 1], Complex::new(2.0, -1.0));
                let result = PowerLawModel::new(law).apply_frequency_domain(
                    &mut wave,
                    &[reference, Frequency::from_base(f64::NAN)],
                    Length::from_base(0.125),
                );
                (
                    law.absorption_at_frequency(reference)
                        .expect("at reference"),
                    law.absorption_at_frequency(reference * 4.0)
                        .expect("above reference"),
                    wave,
                    result,
                )
            });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(loss_at_reference, ReciprocalLength::from_base(6.0));
            assert_eq!(
                loss_at_four_times_reference,
                ReciprocalLength::from_base(48.0)
            );
            let error = failure.expect_err("a NaN frequency must be rejected");
            assert!(matches!(
                error,
                KwaversError::InvalidInput(message)
                    if message == "Frequency must be finite and nonnegative"
            ));
            assert!(untouched
                .iter()
                .all(|value| *value == Complex::new(2.0, -1.0)));
        });
    });
}

mod dispersion;

#[test]
fn spectrum_scales_each_logical_plane_in_both_storage_orders() {
    use leto::{Layout, VecStorage};
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let reference = Frequency::from_base(1.0);
        let model = tracing::info_span!("arrange").in_scope(|| {
            PowerLawModel::new(
                PowerLawAbsorption::new(
                    ReciprocalLength::from_base(std::f64::consts::LN_2),
                    reference,
                    Dimensionless::from_base(1.0),
                    false,
                )
                .unwrap(),
            )
        });
        let frequencies = [reference * 0.0, reference, reference * 2.0];
        for layout in [
            Layout::c_contiguous([3, 2, 4]).unwrap(),
            Layout::f_contiguous([3, 2, 4]).unwrap(),
        ] {
            let mut spectrum = Array3::new(
                layout,
                VecStorage::new(
                    (0..24)
                        .map(|i| Complex::new(f64::from(i + 1), -f64::from(i + 2)))
                        .collect(),
                ),
            )
            .unwrap();
            let input = spectrum.clone();
            tracing::info_span!("act").in_scope(|| {
                model
                    .apply_frequency_domain(&mut spectrum, &frequencies, Length::from_base(1.0))
                    .unwrap()
            });
            tracing::info_span!("assert").in_scope(|| {
                for (i, gain) in [1.0, 0.5, 0.25].into_iter().enumerate() {
                    for j in 0..2 {
                        for k in 0..4 {
                            let expected = input[[i, j, k]] * gain;
                            let actual = spectrum[[i, j, k]];
                            assert!(
                                (actual.re - expected.re).abs()
                                    <= 2.0 * f64::EPSILON * expected.re.abs()
                            );
                            assert!(
                                (actual.im - expected.im).abs()
                                    <= 2.0 * f64::EPSILON * expected.im.abs()
                            );
                        }
                    }
                }
            });
        }
    });
}
