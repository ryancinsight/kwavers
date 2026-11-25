use super::support::{pair, trace};
use kwavers_driver::{audit, error::Geometry, geom::Nm, rules::DesignRules, Error, GridSpec};

#[test]
fn pair_skew_and_layer_balance() {
    let _trace = trace("routed pair");
    let rules = tracing::info_span!("arrange").in_scope(DesignRules::holohv);
    assert_eq!(rules.diff_pair_length_tolerance, Nm(500_000));
    for (end_n, faults) in [
        (4_000_000, 1),
        (8_000_000, 0),
        (8_300_000, 0),
        (8_499_999, 0),
        (8_500_000, 0),
        (8_500_001, 1),
    ] {
        let board = pair(&[2_000_000, 8_000_000], &[2_000_000, end_n]);
        let report = tracing::info_span!("act", end_n).in_scope(|| audit(&board, &[], &[], &rules));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                report.diff_pair_length_mismatch_violations, faults,
                "N endpoint {end_n} nm"
            );
            if faults != 0 {
                assert!(!report.hard_drc_clean());
            }
        });
    }
    let board = pair(
        &[2_000_000, 8_000_000, 10_000_000],
        &[2_000_000, 4_000_000, 10_000_000],
    );
    let report = tracing::info_span!("act").in_scope(|| audit(&board, &[], &[], &rules));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            (
                report.diff_pair_length_mismatch_violations,
                report.diff_pair_via_count_violations,
                report.diff_pair_segment_length_mismatch_violations
            ),
            (0, 0, 1)
        );
        assert!(!report.hard_drc_clean());
    });
}

#[test]
fn grid_rejects_invalid_pitch_with_typed_error() {
    let _trace = trace("invalid grid");
    let pitches = tracing::info_span!("arrange").in_scope(|| [Nm(-1), Nm(0)]);
    for pitch in pitches {
        let result = tracing::info_span!("act", ?pitch)
            .in_scope(|| GridSpec::cover(Nm(1000), Nm(1000), pitch, 2));
        tracing::info_span!("assert")
            .in_scope(|| assert!(matches!(result, Err(Error::Geometry(Geometry::EmptyGrid)))));
    }
}
