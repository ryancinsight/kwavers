use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::signal::window as window_coeff;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalWindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian,
    Tukey { alpha: f64 },
}

#[must_use]
pub fn window_value(window: SignalWindowType, normalized_time: f64) -> f64 {
    if !(0.0..=1.0).contains(&normalized_time) {
        return 0.0;
    }

    match window {
        SignalWindowType::Rectangular => 1.0,
        // Hann/Hamming/Blackman delegate to the kwavers-math window-coefficient SSOT.
        SignalWindowType::Hann => window_coeff::hann(normalized_time),
        SignalWindowType::Hamming => window_coeff::hamming(normalized_time),
        SignalWindowType::Blackman => window_coeff::blackman(normalized_time),
        SignalWindowType::Gaussian => {
            let sigma = 0.4;
            let arg = (normalized_time - 0.5) / sigma;
            (-0.5 * arg * arg).exp()
        }
        // Tukey delegates to the kwavers-math window SSOT (`alpha` = cosine
        // fraction `r`, clamped to [0, 1]: rectangular at 0, Hann at 1).
        SignalWindowType::Tukey { alpha } => window_coeff::tukey(normalized_time, alpha),
    }
}

/// Get win.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[must_use]
pub fn get_win(window: SignalWindowType, n: usize, symmetric: bool) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }

    let denom = if symmetric { (n - 1) as f64 } else { n as f64 };
    (0..n)
        .map(|i| window_value(window, i as f64 / denom))
        .collect()
}
/// Apply window.
/// # Errors
/// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn apply_window(signal: &[f64], window: &[f64]) -> KwaversResult<Vec<f64>> {
    if signal.len() != window.len() {
        return Err(KwaversError::InvalidInput(format!(
            "window length mismatch: signal has {}, window has {}",
            signal.len(),
            window.len()
        )));
    }

    Ok(signal
        .iter()
        .zip(window.iter())
        .map(|(&x, &w)| x * w)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn hann_symmetric_has_zero_endpoints() {
        let w = get_win(SignalWindowType::Hann, 8, true);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 0.0).abs() < 1e-12);
        assert!((w[7] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn hann_periodic_does_not_repeat_endpoint() {
        let w = get_win(SignalWindowType::Hann, 8, false);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 0.0).abs() < 1e-12);
        assert!(w[7] > 0.0);
    }

    #[test]
    fn apply_window_rejects_mismatched_lengths() {
        let err = apply_window(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(err.to_string().contains("window length mismatch"));
    }

    proptest! {
        #[test]
        fn get_win_returns_expected_length(n in 0usize..256) {
            let w = get_win(SignalWindowType::Hann, n, true);
            prop_assert_eq!(w.len(), n);
        }

        #[test]
        fn window_value_is_zero_outside_unit_interval(x in any::<f64>()) {
            prop_assume!(x.is_finite());
            prop_assume!(!(0.0..=1.0).contains(&x));
            let y = window_value(SignalWindowType::Hann, x);
            prop_assert_eq!(y, 0.0);
        }
    }
}