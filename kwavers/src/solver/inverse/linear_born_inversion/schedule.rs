//! Row scheduling for frequency-continuation linear Born solves.
//!
//! Rows are source-major, offset-major, frequency-major, harmonic-major. A
//! continuation stage with prefix `p` keeps rows whose frequency index is less
//! than `p`; the final stage is therefore the full row set.

use super::LinearBornInversionConfig;

/// Build low-to-high frequency row schedules.
pub(crate) fn continuation_rows(
    config: &LinearBornInversionConfig,
    nrows: usize,
) -> Vec<Vec<usize>> {
    let nf = config.frequencies_hz.len();
    let harmonic_count = config.harmonic_count();
    let stage_count = if config.frequency_continuation { nf } else { 1 };
    let mut stages = Vec::with_capacity(stage_count);
    for stage in 0..stage_count {
        let prefix = if config.frequency_continuation {
            stage + 1
        } else {
            nf
        };
        let rows = (0..nrows)
            .filter(|row| (row / harmonic_count) % nf < prefix)
            .collect();
        stages.push(rows);
    }
    stages
}

/// Split a total iteration budget across continuation stages.
pub(crate) fn stage_iteration_count(total: usize, stages: usize, stage_idx: usize) -> usize {
    let base = total / stages;
    let extra = usize::from(stage_idx < total % stages);
    base + extra
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_rows_expand_by_frequency_prefix() {
        let config = LinearBornInversionConfig {
            frequencies_hz: vec![1.0, 2.0, 3.0],
            receiver_offsets: vec![1],
            nonlinear_harmonic_model: true,
            frequency_continuation: true,
            ..LinearBornInversionConfig::default()
        };

        let rows = continuation_rows(&config, 12);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![0, 1, 6, 7]);
        assert_eq!(rows[1], vec![0, 1, 2, 3, 6, 7, 8, 9]);
        assert_eq!(rows[2], (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn stage_iteration_count_distributes_remainder_to_early_stages() {
        assert_eq!(stage_iteration_count(8, 3, 0), 3);
        assert_eq!(stage_iteration_count(8, 3, 1), 3);
        assert_eq!(stage_iteration_count(8, 3, 2), 2);
    }
}
