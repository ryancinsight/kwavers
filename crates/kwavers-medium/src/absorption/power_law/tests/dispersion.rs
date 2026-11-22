use super::*;
use aequitas::systems::si::quantities::Velocity;

#[test]
fn dispersion_resolves_even_exponents_and_adjacent_odd_poles() {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let (frequency, speed) = tracing::info_span!("arrange")
            .in_scope(|| (Frequency::from_base(1.0), Velocity::from_base(TWO_PI)));
        // At the reference frequency c₀ = 2π and α = 1, so c₀ α/ω = 1.
        // Below the pole, cₚ/c₀ = tan(δπ/2)/(1 + tan(δπ/2)).
        for y in [
            0.0,
            2.0,
            4.0,
            2.0_f64.powi(53),
            1.0_f64.next_down(),
            1.0,
            1.0_f64.next_up(),
        ] {
            let result = tracing::info_span!("act", y).in_scope(|| {
                PowerLawAbsorption::new(
                    ReciprocalLength::from_base(1.0),
                    frequency,
                    Dimensionless::from_base(y),
                    true,
                )
                .unwrap()
                .phase_velocity(frequency, speed)
            });
            tracing::info_span!("assert", y).in_scope(|| {
                if y % 2.0 == 0.0 {
                    assert_eq!(result.unwrap(), speed);
                } else if y < 1.0 {
                    let delta = (1.0 - y) * std::f64::consts::FRAC_PI_2;
                    // tan(δ) differs from δ by O(δ³), below half an ulp here.
                    let expected = TWO_PI * delta / (1.0 + delta);
                    assert!(
                        (result.unwrap().into_base() - expected).abs()
                            <= 4.0 * f64::EPSILON * expected
                    );
                } else {
                    assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
                }
            });
        }
    });
}
