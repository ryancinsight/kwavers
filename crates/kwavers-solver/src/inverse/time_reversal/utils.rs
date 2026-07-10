//! Time reversal utilities for acoustic focusing

use leto::{
    Array2,
    Array3,
};

/// Time reversal processing utilities
#[derive(Debug)]
pub struct TimeReversalUtils;

impl TimeReversalUtils {
    /// Apply time reversal to recorded signals
    ///
    /// For `signals[s, t]`, the reversed output is
    /// `signals[s, n_t - 1 - t]`. The implementation constructs the output
    /// directly from that involution, avoiding a full matrix clone followed by
    /// per-row swaps.
    #[must_use]
    pub fn time_reverse_signals(signals: &Array2<f64>) -> Array2<f64> {
        let [n_sensors, n_samples] = signals.shape();
        Array2::from_shape_fn((n_sensors, n_samples), |[sensor, sample]| {
            signals[[sensor, n_samples - 1 - sample]]
        })
    }

    /// Focus calculation for time reversal
    #[must_use]
    pub fn calculate_focus_quality(field: &Array3<f64>) -> f64 {
        let max_val = field.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mean_val = field.iter().map(|x| x.abs()).sum::<f64>() / (field.len()) as f64;
        if mean_val > 0.0 {
            max_val / mean_val
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TimeReversalUtils;
    use leto::Array2;

    #[test]
    fn time_reverse_signals_reverses_each_sensor_row() {
        let signals = Array2::from_shape_fn((2, 4), |[s, t]| {
            if s == 0 { (t + 1) as f64 } else { ((t + 1) * 10) as f64 }
        });

        let reversed = TimeReversalUtils::time_reverse_signals(&signals);

        assert_eq!(reversed[[0, 0]], 4.0);
        assert_eq!(reversed[[0, 3]], 1.0);
        assert_eq!(reversed[[1, 0]], 40.0);
        assert_eq!(reversed[[1, 3]], 10.0);
        assert_eq!(signals[[0, 0]], 1.0);
        assert_eq!(signals[[1, 3]], 40.0);
    }

    #[test]
    fn time_reverse_signals_is_involution_for_rectangular_data() {
        let signals =
            Array2::from_shape_fn((3, 5), |[sensor, sample]| (10 * sensor + sample) as f64);

        let reversed = TimeReversalUtils::time_reverse_signals(&signals);
        let restored = TimeReversalUtils::time_reverse_signals(&reversed);

        assert_eq!(restored, signals);
    }

    #[test]
    fn time_reverse_signals_handles_single_sample_and_empty_sensors() {
        let single_sample = Array2::from_shape_fn((3, 1), |[s, _]| [5.0, 7.0, 11.0][s]);
        let empty_sensors = Array2::<f64>::zeros((0, 4));

        assert_eq!(
            TimeReversalUtils::time_reverse_signals(&single_sample),
            single_sample
        );
        let [nr, ns] = TimeReversalUtils::time_reverse_signals(&empty_sensors).shape();
        assert_eq!((nr, ns), (0, 4));
    }
}
