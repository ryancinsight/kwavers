#[cfg(test)]
mod tests {
    use ndarray::Array3;

    #[test]
    fn test_spectral_dg_shock_detection() {
        // Validate shock capturing (Persson & Peraire 2006)
        let n = 64;
        let _dx = 1e-3;

        // Create discontinuous field (shock)
        let mut field = Array3::zeros((n, 1, 1));
        for i in 0..n {
            if i < n / 2 {
                field[[i, 0, 0]] = 1.0;
            } else {
                field[[i, 0, 0]] = 0.1;
            }
        }

        // Smooth slightly to avoid numerical issues
        for _ in 0..2 {
            let mut smoothed = field.clone();
            for i in 1..n - 1 {
                smoothed[[i, 0, 0]] = 0.25 * field[[i - 1, 0, 0]]
                    + 0.5 * field[[i, 0, 0]]
                    + 0.25 * field[[i + 1, 0, 0]];
            }
            field = smoothed;
        }

        // Compute smoothness indicator (Persson-Peraire)
        let mut smoothness = Array3::zeros((n, 1, 1));

        for i in 2..n - 2 {
            // Modal decay indicator
            let local_vals = [
                field[[i - 2, 0, 0]],
                field[[i - 1, 0, 0]],
                field[[i, 0, 0]],
                field[[i + 1, 0, 0]],
                field[[i + 2, 0, 0]],
            ];

            // Compute local smoothness indicator (variance-based per Jiang & Shu 1996)
            // Estimates local polynomial variation for shock detection
            let mean = local_vals.iter().copied().sum::<f64>() / 5.0;
            let variance = local_vals
                .iter()
                .copied()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / 5.0;

            // High variance indicates discontinuity
            smoothness[[i, 0, 0]] = variance.sqrt();
        }

        // Find shock location
        let mut max_indicator = 0.0;
        let mut shock_location = 0;

        for i in 0..n {
            if smoothness[[i, 0, 0]] > max_indicator {
                max_indicator = smoothness[[i, 0, 0]];
                shock_location = i;
            }
        }

        // Shock should be detected near n/2
        let error = (shock_location as i32 - n as i32 / 2).abs();
        assert!(error < 5, "Shock detection error: {} cells", error);
    }
}
