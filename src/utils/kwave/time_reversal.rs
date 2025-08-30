//! Time reversal utilities for acoustic focusing
//!
//! Implements time reversal processing and reconstruction

use ndarray::{Array2, Array3};

/// Time reversal processing utilities
#[derive(Debug, Debug)]
pub struct TimeReversalUtils;

impl TimeReversalUtils {
    /// Apply time reversal to recorded signals
    pub fn time_reverse_signals(signals: &Array2<f64>) -> Array2<f64> {
        let mut reversed = signals.clone();
        for mut row in reversed.rows_mut() {
            let n = row.len();
            for i in 0..n / 2 {
                row.swap(i, n - 1 - i);
            }
        }
        reversed
    }

    /// Focus calculation for time reversal
    pub fn calculate_focus_quality(field: &Array3<f64>) -> f64 {
        let max_val = field.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mean_val = field.iter().map(|x| x.abs()).sum::<f64>() / field.len() as f64;
        if mean_val > 0.0 {
            max_val / mean_val
        } else {
            0.0
        }
    }
}
