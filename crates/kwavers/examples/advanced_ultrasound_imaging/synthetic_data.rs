//! Synthetic RF data and image grids for the advanced ultrasound imaging demo.

use kwavers_physics::acoustics::imaging::modalities::ultrasound::advanced::SyntheticApertureConfig;
use leto::{Array1, Array2, Array3};

/// Generate synthetic RF data for SA imaging
pub(crate) fn generate_synthetic_sa_rf_data(
    n_samples: usize,
    n_rx: usize,
    n_tx: usize,
    config: &SyntheticApertureConfig,
) -> Array3<f64> {
    let mut rf_data = Array3::<f64>::zeros((n_samples, n_rx, n_tx));

    // Create a simple point scatterer at (0, 30mm)
    let scatterer_x = 0.0f64;
    let scatterer_z = 30e-3f64;

    for tx in 0..n_tx {
        for rx in 0..n_rx {
            // Calculate transmit and receive element positions
            let tx_x = (tx as f64 - (n_tx - 1) as f64 / 2.0) * config.element_spacing;
            let rx_x = (rx as f64 - (n_rx - 1) as f64 / 2.0) * config.element_spacing;

            // Calculate round-trip delay to scatterer
            let tx_distance = ((scatterer_x - tx_x).powi(2) + scatterer_z.powi(2)).sqrt();
            let rx_distance = ((scatterer_x - rx_x).powi(2) + scatterer_z.powi(2)).sqrt();
            let total_delay = (tx_distance + rx_distance) / config.sound_speed;

            // Convert to sample index
            let sample_idx = (total_delay * config.sampling_frequency) as usize;
            if sample_idx < n_samples {
                // Add a simple pulse (in practice, this would be more complex)
                let amplitude = 1.0 / ((tx_distance + rx_distance) * 10.0); // Attenuation
                rf_data[[sample_idx, rx, tx]] = amplitude;
            }
        }
    }

    rf_data
}

/// Generate synthetic RF data for plane wave imaging
pub(crate) fn generate_synthetic_pw_rf_data(
    n_samples: usize,
    n_elements: usize,
    _tx_angle: f64,
) -> Array2<f64> {
    let mut rf_data = Array2::<f64>::zeros((n_samples, n_elements));

    // Create a simple point scatterer at (5mm, 30mm)
    let scatterer_x = 5e-3f64;
    let scatterer_z = 30e-3f64;
    let sound_speed = 1540.0;
    let sampling_frequency = 40e6;

    for elem in 0..n_elements {
        // Calculate element position
        let elem_x = (elem as f64 - (n_elements - 1) as f64 / 2.0) * 0.3e-3;

        // For plane wave, transmit delay is incorporated in steering
        // Receive delay is distance from scatterer to element
        let rx_distance = ((scatterer_x - elem_x).powi(2) + scatterer_z.powi(2)).sqrt();
        let rx_delay = rx_distance / sound_speed;

        // Convert to sample index
        let sample_idx = (rx_delay * sampling_frequency) as usize;
        if sample_idx < n_samples {
            let amplitude = 1.0 / (rx_distance * 10.0); // Attenuation
            rf_data[[sample_idx, elem]] = amplitude;
        }
    }

    rf_data
}

/// Create image grid coordinates
pub(crate) fn create_image_grid(width: usize, height: usize, max_depth: f64) -> Array3<f64> {
    let mut grid = Array3::<f64>::zeros((2, height, width)); // [x/z, height, width]

    let x_range = 40e-3; // ±20mm lateral
    let z_range = max_depth;

    for i in 0..height {
        for j in 0..width {
            // X coordinate (lateral)
            grid[[0, i, j]] = (j as f64 - width as f64 / 2.0) * x_range / width as f64;
            // Z coordinate (depth)
            grid[[1, i, j]] = (i as f64) * z_range / height as f64;
        }
    }

    grid
}

/// Generate noisy received signal for coded excitation testing
pub(crate) fn generate_noisy_received_signal(
    code: &Array1<eunomia::Complex64>,
    noise_level: f64,
) -> Array1<f64> {
    use rand::prelude::*;

    let mut rng = rand::thread_rng();
    let mut signal = Array1::<f64>::zeros(code.len() * 4); // Longer to show compression

    // Add the code with some delay and noise
    let delay = 50;
    for i in 0..code.len() {
        if delay + i < signal.len() {
            signal[delay + i] = code[i].re + rng.gen::<f64>() * noise_level;
        }
    }

    signal
}
