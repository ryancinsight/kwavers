//! Value-semantic sensitivity-screening tests.

use super::{Parameter, ParameterSpace, Seed, SensitivityAnalyzer, SensitivityConfig};
use core::num::NonZeroU32;

#[test]
fn analyzer_preserves_validated_configuration() {
    let config = SensitivityConfig {
        sample_count: NonZeroU32::new(100).expect("non-zero fixture"),
        seed: Seed::new(17),
    };
    let analyzer = SensitivityAnalyzer::new(config.clone()).unwrap();
    assert_eq!(analyzer.config, config);
}

#[test]
fn affine_response_has_unit_squared_correlation() {
    let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
        sample_count: NonZeroU32::new(128).expect("non-zero fixture"),
        seed: Seed::new(23),
    })
    .unwrap();
    let space =
        ParameterSpace::new(
            [Parameter::borrowed("amplitude", -2.0, 3.0).expect("valid parameter")],
        )
        .expect("valid space");

    let report = analyzer
        .analyze(|parameters| 2.0f64.mul_add(parameters[0], 0.5), &space)
        .unwrap();

    assert_eq!(report.sample_count(), 128);
    let error = (report.squared_correlations()[0] - 1.0).abs();
    // Welford correlation accumulates O(n) rounded operations on O(1) data.
    assert!(
        error <= 512.0 * f64::EPSILON,
        "affine one-parameter screening must satisfy r²=1; error={error:e}"
    );
}

#[test]
fn replay_is_bitwise_deterministic() {
    let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
        sample_count: NonZeroU32::new(64).expect("non-zero fixture"),
        seed: Seed::new(91),
    })
    .unwrap();
    let space = ParameterSpace::new([
        Parameter::borrowed("left", 0.0, 1.0).expect("valid parameter"),
        Parameter::borrowed("right", 1.0, 2.0).expect("valid parameter"),
    ])
    .expect("valid space");

    let first = analyzer
        .analyze(|parameters| parameters[0] * parameters[1], &space)
        .unwrap();
    let second = analyzer
        .analyze(|parameters| parameters[0] * parameters[1], &space)
        .unwrap();

    assert_eq!(
        first.squared_correlations().map(f64::to_bits),
        second.squared_correlations().map(f64::to_bits)
    );
}

#[test]
fn singleton_and_non_finite_responses_are_rejected() {
    let singleton_error = SensitivityAnalyzer::new(SensitivityConfig {
        sample_count: NonZeroU32::new(1).expect("non-zero fixture"),
        seed: Seed::new(0),
    })
    .unwrap_err();
    assert!(
        format!("{singleton_error:?}").contains("at least two samples"),
        "singleton rejection must state the correlation cardinality"
    );

    let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default()).unwrap();
    let space = ParameterSpace::new([Parameter::borrowed("x", 0.0, 1.0).expect("valid parameter")])
        .expect("valid space");
    let finite_error = analyzer.analyze(|_| f64::NAN, &space).unwrap_err();
    assert!(
        format!("{finite_error:?}").contains("sample 0: NaN"),
        "non-finite response rejection must identify the sample and value"
    );
}
