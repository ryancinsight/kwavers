//! Synthetic RF volumes and frames for the real-time 3D beamforming demo.

use leto::{Array3, Array4};

/// Generate synthetic RF data for testing
pub(crate) fn generate_synthetic_rf_data(
    frames: usize,
    channels: usize,
    samples: usize,
) -> Array4<f32> {
    use std::f32::consts::PI;

    let mut rf_data = Array4::<f32>::zeros((frames, channels, samples, 1));

    // Generate synthetic ultrasound RF signals
    for f in 0..frames {
        for c in 0..channels {
            for s in 0..samples {
                let t = s as f32 / 50_000_000.0; // 50MHz sampling
                let freq = 2_500_000.0; // 2.5MHz center frequency

                // Add some tissue-like scattering
                let scattering = (c as f32 * 0.1).sin() * (s as f32 * 0.01).cos();
                let signal = (2.0 * PI * freq * t).sin() * scattering.exp();

                // Add noise
                let noise = (rand::random::<f32>() - 0.5) * 0.1;
                rf_data[[f, c, s, 0]] = signal + noise;
            }
        }
    }

    rf_data
}

/// Generate synthetic frame data for streaming
pub(crate) fn generate_synthetic_frame(channels: usize, samples: usize) -> Array3<f32> {
    use std::f32::consts::PI;

    let mut frame = Array3::<f32>::zeros((channels, samples, 1));

    for c in 0..channels {
        for s in 0..samples {
            let t = s as f32 / 50_000_000.0;
            let freq = 2_500_000.0;

            // Simple synthetic signal
            let signal = (2.0 * PI * freq * t).sin() * (c as f32 * 0.05).cos();
            let noise = (rand::random::<f32>() - 0.5) * 0.05;

            frame[[c, s, 0]] = signal + noise;
        }
    }

    frame
}
