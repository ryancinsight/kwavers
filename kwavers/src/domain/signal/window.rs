use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian,
    Tukey { alpha: f64 },
}

#[must_use]
pub fn window_value(window: WindowType, normalized_time: f64) -> f64 {
    if !(0.0..=1.0).contains(&normalized_time) {
        return 0.0;
    }

    match window {
        WindowType::Rectangular => 1.0,
        WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * normalized_time).cos()),
        WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * normalized_time).cos(),
        WindowType::Blackman => {
            0.42 - 0.5 * (2.0 * PI * normalized_time).cos()
                + 0.08 * (4.0 * PI * normalized_time).cos()
        }
        WindowType::Gaussian => {
            let sigma = 0.4;
            let arg = (normalized_time - 0.5) / sigma;
            (-0.5 * arg * arg).exp()
        }
        WindowType::Tukey { alpha } => {
            if alpha <= 0.0 {
                1.0
            } else if alpha >= 1.0 {
                0.5 * (1.0 - (2.0 * PI * normalized_time).cos())
            } else if normalized_time < alpha / 2.0 {
                0.5 * (1.0 + (2.0 * PI * normalized_time / alpha - PI).cos())
            } else if normalized_time <= 1.0 - alpha / 2.0 {
                1.0
            } else {
                0.5 * (1.0 + (2.0 * PI * (normalized_time - 1.0) / alpha + PI).cos())
            }
        }
    }
}

#[must_use]
pub fn get_win(window: WindowType, n: usize, symmetric: bool) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }

    let denom = if symmetric { (n - 1) as f64 } else { n as f64 };
    (0..n)
        .map(|i| window_value(window, i as f64 / denom))
        .collect()
}

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
        let w = get_win(WindowType::Hann, 8, true);
        assert_eq!(w.len(), 8);
        assert!((w[0] - 0.0).abs() < 1e-12);
        assert!((w[7] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn hann_periodic_does_not_repeat_endpoint() {
        let w = get_win(WindowType::Hann, 8, false);
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
            let w = get_win(WindowType::Hann, n, true);
            prop_assert_eq!(w.len(), n);
        }

        #[test]
        fn window_value_is_zero_outside_unit_interval(x in any::<f64>()) {
            prop_assume!(x.is_finite());
            prop_assume!(!(0.0..=1.0).contains(&x));
            let y = window_value(WindowType::Hann, x);
            prop_assert_eq!(y, 0.0);
        }
    }
}
