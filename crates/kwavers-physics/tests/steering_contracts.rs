use std::f64::consts::PI;

use kwavers_physics::analytical::transducer::{
    circular_piston_directivity, linear_array_factor, steered_beam_pattern_1d,
    steering_grating_lobe_ratio_1d,
};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn steering_magnitude_and_secondary_ratio_include_negative_pressure_lobes() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let (angle, ka, signed_directivity) = info_span!("arrange").in_scope(|| {
            let angle = 0.5_f64.asin();
            let ka = 10.0;
            (angle, ka, circular_piston_directivity(&[angle], ka)[0])
        });

        let (pattern, ratios) = info_span!("act").in_scope(|| {
            (
                steered_beam_pattern_1d(&[0.0], &[0.0, angle], 1.0, 0.0, ka),
                steering_grating_lobe_ratio_1d(
                    &[0.0],
                    &[0.0, angle],
                    &[0.0, angle],
                    1.0,
                    ka,
                    angle / 2.0,
                ),
            )
        });

        info_span!("assert").in_scope(|| {
            // One centered radiator has unit array factor at every angle.
            // The signed piston pressure at k a sin(theta) = 5 is negative.
            let lobe = signed_directivity.abs();
            assert!(signed_directivity < 0.0);
            assert!((pattern[0] - 1.0).abs() <= f64::EPSILON);
            assert!(
                (pattern[1] - lobe).abs() <= 4.0 * f64::EPSILON,
                "pressure magnitude {} must equal {lobe}",
                pattern[1]
            );
            assert!(
                (ratios[0] - lobe).abs() <= 4.0 * f64::EPSILON,
                "broadside ratio {} must include the negative-pressure lobe {lobe}",
                ratios[0]
            );
            assert!(
                (ratios[1] * lobe - 1.0).abs() <= 4.0 * f64::EPSILON,
                "off-axis ratio {} must use command pressure magnitude {lobe}",
                ratios[1]
            );
        });
    });
}

#[test]
fn uniform_array_factor_matches_coherent_sum_on_and_around_replicas() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let (wavenumber, direction) = info_span!("arrange").in_scope(|| {
            let wavenumber = 8.0 * PI;
            let mut direction = Vec::new();
            for order in -3..=3 {
                let replica = (f64::from(order) / 4.0).asin();
                direction.extend([replica.next_down(), replica, replica.next_up()]);
            }
            direction.extend([0.13, -0.31, 0.67]);
            (wavenumber, direction)
        });
        for n in [1, 2, 3, 10, 32] {
            let pattern = info_span!("act", elements = n)
                .in_scope(|| linear_array_factor(&direction, wavenumber, 1.0, n, 0.0));
            info_span!("assert", elements = n).in_scope(|| {
                for (&theta, actual) in direction.iter().zip(pattern) {
                    let half_phase = wavenumber * theta.sin() / 2.0;
                    let exact = (0..n)
                        .map(|i| ((2.0 * i as f64 - (n - 1) as f64) * half_phase).cos())
                        .sum::<f64>()
                        / n as f64;
                    // Bound phase rounding in the reference's n products plus
                    // the sum of n terms of modulus <= 1 and normalization.
                    let bound = 32.0 * n as f64 * f64::EPSILON;
                    assert!(
                        (actual - exact).abs() <= bound,
                        "N={n} theta={theta:.17e}: {actual} differs from coherent pressure {exact}"
                    );
                }
            });
        }
    });
}
