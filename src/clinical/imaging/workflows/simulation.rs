use super::config::{ElastographyConfig, PhotoacousticConfig};
use crate::core::error::KwaversResult;
use crate::domain::sensor::beamforming::BeamformingConfig3D;
use ndarray::Array3;
// use ndarray_stats::QuantileExt; // Removed - not in dependencies

/// Generate realistic RF data for ultrasound simulation
pub fn generate_realistic_rf_data(config: &BeamformingConfig3D) -> Array3<f64> {
    let (num_depth, num_lat, num_elev) = config.volume_dims;

    let mut rf_data = Array3::zeros((num_depth, num_lat, num_elev));

    // Generate realistic ultrasound RF signals
    // This simulates backscattered echoes with tissue-like properties
    for elev in 0..num_elev {
        for lat in 0..num_lat {
            for depth in 0..num_depth {
                // Distance from transducer element to voxel
                let distance = ((depth as f64 * config.sound_speed / config.sampling_frequency)
                    + (lat as f64 - num_lat as f64 / 2.0).powi(2) * 0.0001
                    + (elev as f64 - num_elev as f64 / 2.0).powi(2) * 0.0001)
                    .sqrt();

                // Attenuation with depth
                let attenuation = (-0.5 * distance * 100.0).exp(); // 0.5 dB/cm/MHz

                // Tissue scattering with some randomness
                let scattering = (rand::random::<f64>() - 0.5) * 0.1;

                // Generate RF signal with realistic envelope
                let t = depth as f64 / config.sampling_frequency;
                let envelope = (-((t - distance / config.sound_speed)
                    * config.center_frequency
                    * 2.0
                    * std::f64::consts::PI)
                    .powi(2)
                    * 0.5)
                    .exp();
                let rf_signal = envelope
                    * (2.0 * std::f64::consts::PI * config.center_frequency * t).sin()
                    * attenuation
                    * (1.0 + scattering);

                rf_data[[depth, lat, elev]] = rf_signal;
            }
        }
    }

    rf_data
}

/// Generate realistic photoacoustic data
pub fn generate_realistic_pa_data(_config: &PhotoacousticConfig) -> (Vec<Array3<f64>>, Vec<f64>) {
    // Generate time-resolved pressure fields
    let time_points = vec![0.0, 2e-6, 4e-6, 6e-6, 8e-6]; // 5 time points
    let mut pressure_fields = Vec::new();

    for &t in &time_points {
        let mut field = Array3::from_elem((128, 128, 64), 0.0);

        // Generate realistic PA wave propagation
        for z in 0..64 {
            for y in 0..128 {
                for x in 0..128 {
                    let r = ((x as f64 - 64.0).powi(2)
                        + (y as f64 - 64.0).powi(2)
                        + (z as f64 - 32.0).powi(2))
                    .sqrt()
                        * 0.001; // distance in meters
                    let propagation_time = r / 1540.0; // speed of sound

                    if t >= propagation_time {
                        // Gaussian pulse with spherical spreading
                        let amplitude =
                            1e5 * (-((t - propagation_time) * 5e6).powi(2)).exp() / (r + 0.01); // 100 kPa peak
                        let variation = (rand::random::<f64>()).max(0.1);
                        field[[x, y, z]] = amplitude * variation; // Add some variation
                    }
                }
            }
        }

        pressure_fields.push(field);
    }

    (pressure_fields, time_points)
}

/// Reconstruct photoacoustic image using time-reversal
pub fn reconstruct_pa_image(
    pressure_fields: &[Array3<f64>],
    _config: &PhotoacousticConfig,
) -> KwaversResult<Array3<f64>> {
    // Simple back-projection reconstruction
    // In practice, this would use proper time-reversal algorithms
    let mut reconstructed = Array3::zeros(pressure_fields[0].dim());

    for field in pressure_fields {
        reconstructed += field;
    }

    // Normalize
    let max_val = reconstructed.iter().cloned().fold(0.0f64, f64::max);
    if max_val > 0.0 {
        reconstructed.mapv_inplace(|x| x / max_val);
    }

    Ok(reconstructed)
}

/// Compute photoacoustic SNR
pub fn compute_pa_snr(image: &Array3<f64>) -> f64 {
    let mean = image.mean().unwrap_or(0.0);
    let variance = image.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / image.len() as f64;
    let std = variance.sqrt();

    if std > 0.0 {
        20.0 * (mean / std).log10() // SNR in dB
    } else {
        0.0
    }
}

/// Generate realistic elastography data
pub fn generate_realistic_elastography_data(
    _config: &ElastographyConfig,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let dims = (128, 128, 64);

    // Generate realistic tissue properties
    let mut youngs_modulus = Array3::zeros(dims);
    let mut shear_modulus = Array3::zeros(dims);
    let mut shear_wave_speed = Array3::zeros(dims);

    for z in 0..dims.2 {
        for y in 0..dims.1 {
            for x in 0..dims.0 {
                // Create layered tissue structure
                let depth = z as f64 * 0.001; // depth in meters

                // Base properties (soft tissue)
                let mut e_mod = 10e3; // 10 kPa
                let mut g_mod = 5e3; // 5 kPa

                // Add inclusions (harder regions)
                if (x as f64 - 64.0).powi(2) + (y as f64 - 64.0).powi(2) < 100.0
                    && depth > 0.02
                    && depth < 0.04
                {
                    e_mod = 50e3; // 50 kPa inclusion
                    g_mod = 25e3;
                }

                // Add some spatial variation
                let variation = (rand::random::<f64>() - 0.5) * 0.2;
                e_mod *= (1.0f64 + variation).max(0.5f64);
                g_mod *= (1.0f64 + variation).max(0.5f64);

                youngs_modulus[[x, y, z]] = e_mod;
                shear_modulus[[x, y, z]] = g_mod;

                // Shear wave speed from modulus and density (ρ ≈ 1000 kg/m³)
                shear_wave_speed[[x, y, z]] = (g_mod / 1000.0f64).sqrt();
            }
        }
    }

    (youngs_modulus, shear_modulus, shear_wave_speed)
}
