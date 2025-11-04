//! Advanced Ultrasound Imaging Example
//!
//! This example demonstrates advanced ultrasound imaging techniques:
//! - Synthetic Aperture (SA) imaging
//! - Plane Wave Imaging (PWI) with compounding
//! - Coded Excitation with pulse compression
//!
//! Showcases the implementation of modern ultrasound imaging methods
//! for improved resolution, frame rate, and penetration.

use kwavers::physics::imaging::ultrasound::advanced::{
    SyntheticApertureConfig, SyntheticApertureReconstruction,
    PlaneWaveConfig, PlaneWaveReconstruction, PlaneWaveCompounding,
    CodedExcitationConfig, ExcitationCode, CodedExcitationProcessor,
};
use ndarray::{Array1, Array2, Array3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🩺 Advanced Ultrasound Imaging Demonstration");
    println!("===========================================");

    // Demonstrate synthetic aperture imaging
    demonstrate_synthetic_aperture()?;

    // Demonstrate plane wave imaging with compounding
    demonstrate_plane_wave_imaging()?;

    // Demonstrate coded excitation
    demonstrate_coded_excitation()?;

    println!("\n✅ Advanced ultrasound imaging demonstration completed!");
    println!("   Demonstrated: Synthetic aperture, plane wave imaging, coded excitation");

    Ok(())
}

/// Demonstrate synthetic aperture imaging
fn demonstrate_synthetic_aperture() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Synthetic Aperture (SA) Imaging Demonstration");
    println!("-----------------------------------------------");

    // Configure SA imaging
    let sa_config = SyntheticApertureConfig {
        num_tx_elements: 32,
        num_rx_elements: 32,
        element_spacing: 0.3e-3, // 0.3mm
        sound_speed: 1540.0,
        frequency: 5e6,
        sampling_frequency: 40e6,
        num_tx_angles: 1,
    };

    println!("SA Configuration:");
    println!("  TX Elements: {}", sa_config.num_tx_elements);
    println!("  RX Elements: {}", sa_config.num_rx_elements);
    println!("  Element Spacing: {:.1} mm", sa_config.element_spacing * 1e3);
    println!("  Center Frequency: {:.1} MHz", sa_config.frequency / 1e6);

    // Create SA reconstruction processor
    let sa_reconstruction = SyntheticApertureReconstruction::new(sa_config.clone());

    // Create synthetic RF data for demonstration
    // In practice, this would come from actual transducer data
    let n_samples = 2048;
    let n_rx = sa_config.num_rx_elements;
    let n_tx = sa_config.num_tx_elements;

    println!("  Generating synthetic RF data...");
    let rf_data = generate_synthetic_sa_rf_data(n_samples, n_rx, n_tx, &sa_config);

    // Create image grid
    let image_width = 100;
    let image_height = 100;
    let image_depth = 50e-3; // 50mm depth
    let image_grid = create_image_grid(image_width, image_height, image_depth);

    println!("  Reconstructing SA image...");
    let sa_image = sa_reconstruction.reconstruct(&rf_data, &image_grid);

    // Analyze image quality
    let image_stats = analyze_image_quality(&sa_image);
    println!("  SA Image Statistics:");
    println!("    Image Size: {} x {}", sa_image.nrows(), sa_image.ncols());
    println!("    Max Value: {:.3}", image_stats.max_value);
    println!("    Mean Value: {:.3}", image_stats.mean_value);
    println!("    Dynamic Range: {:.1} dB", image_stats.dynamic_range);

    // SA provides excellent resolution but requires many transmissions
    let total_transmissions = sa_config.num_tx_elements * sa_config.num_rx_elements;
    println!("    Total TX-RX Pairs: {}", total_transmissions);

    Ok(())
}

/// Demonstrate plane wave imaging with multi-angle compounding
fn demonstrate_plane_wave_imaging() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Plane Wave Imaging (PWI) with Compounding");
    println!("-------------------------------------------");

    // Configure plane wave imaging
    let base_config = PlaneWaveConfig {
        tx_angle: 0.0,
        num_elements: 64,
        element_spacing: 0.3e-3,
        sound_speed: 1540.0,
        frequency: 5e6,
        sampling_frequency: 40e6,
    };

    // Define compounding angles
    let angles = vec![
        -20.0f64.to_radians(), // -20 degrees
        -10.0f64.to_radians(), // -10 degrees
         0.0,                  // 0 degrees
         10.0f64.to_radians(), // 10 degrees
         20.0f64.to_radians(), // 20 degrees
    ];

    println!("PWI Configuration:");
    println!("  Elements: {}", base_config.num_elements);
    println!("  Compounding Angles: {}", angles.len());
    println!("  Frequency: {:.1} MHz", base_config.frequency / 1e6);

    // Create individual plane wave reconstructions
    let mut pw_images = Vec::new();

    for (i, &angle) in angles.iter().enumerate() {
        println!("  Processing angle {}: {:.0}°", i + 1, angle.to_degrees());

        let pw_config = PlaneWaveConfig {
            tx_angle: angle,
            ..base_config.clone()
        };

        let pw_reconstruction = PlaneWaveReconstruction::new(pw_config);

        // Generate synthetic RF data for this angle
        let n_samples = 2048;
        let rf_data = generate_synthetic_pw_rf_data(n_samples, base_config.num_elements, angle);

        // Create image grid
        let image_width = 100;
        let image_height = 100;
        let image_depth = 50e-3;
        let image_grid = create_image_grid(image_width, image_height, image_depth);

        let pw_image = pw_reconstruction.reconstruct(&rf_data, &image_grid);
        pw_images.push(pw_image);
    }

    // Perform multi-angle compounding
    println!("  Compounding {} angle images...", angles.len());

    let pw_compounding = PlaneWaveCompounding::new(&angles, base_config);
    let compounded_images = Array3::from_shape_vec(
        (pw_images.len(), pw_images[0].nrows(), pw_images[0].ncols()),
        pw_images.into_iter().flatten().collect(),
    )?;

    let compounded_image = pw_compounding.compound(&compounded_images);

    // Analyze compounded image
    let image_stats = analyze_image_quality(&compounded_image);
    println!("  Compounded Image Statistics:");
    println!("    Image Size: {} x {}", compounded_image.nrows(), compounded_image.ncols());
    println!("    Max Value: {:.3}", image_stats.max_value);
    println!("    Mean Value: {:.3}", image_stats.mean_value);
    println!("    Dynamic Range: {:.1} dB", image_stats.dynamic_range);

    // PWI provides high frame rates with good image quality through compounding
    let frame_rate = 1.0 / (angles.len() as f64 * 100e-6); // Assuming 100μs per angle
    println!("    Estimated Frame Rate: {:.0} fps", frame_rate);

    Ok(())
}

/// Demonstrate coded excitation with pulse compression
fn demonstrate_coded_excitation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 Coded Excitation with Pulse Compression");
    println!("-----------------------------------------");

    // Configure coded excitation
    let codes = vec![
        ("Chirp", ExcitationCode::Chirp {
            start_freq: 2e6,
            end_freq: 8e6,
            length: 256,
        }),
        ("Barker-7", ExcitationCode::Barker { length: 7 }),
        ("Barker-13", ExcitationCode::Barker { length: 13 }),
    ];

    for (name, code) in codes {
        println!("  Testing {} Code:", name);

        let config = CodedExcitationConfig {
            code: code.clone(),
            sound_speed: 1540.0,
            sampling_frequency: 20e6,
        };

        let processor = CodedExcitationProcessor::new(config);

        // Generate excitation code
        let excitation_code = processor.generate_code();
        println!("    Code Length: {} samples", excitation_code.len());

        // Calculate theoretical SNR improvement
        let snr_improvement = processor.theoretical_snr_improvement();
        println!("    Theoretical SNR Improvement: {:.1} dB", 20.0 * snr_improvement.log10());

        // Simulate received signal with noise
        let received_signal = generate_noisy_received_signal(&excitation_code, 0.1);

        // Apply matched filtering
        let compressed_signal = processor.matched_filter(&received_signal, &excitation_code);

        // Analyze compression results
        let compression_stats = analyze_pulse_compression(&received_signal, &compressed_signal);
        println!("    Compression Ratio: {:.1}", compression_stats.compression_ratio);
        println!("    Peak Sidelobe Level: {:.1} dB", compression_stats.peak_sidelobe_db);
        println!("    Main Lobe Width: {:.0} samples", compression_stats.main_lobe_width);
    }

    Ok(())
}

/// Generate synthetic RF data for SA imaging
fn generate_synthetic_sa_rf_data(
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
fn generate_synthetic_pw_rf_data(n_samples: usize, n_elements: usize, _tx_angle: f64) -> Array2<f64> {
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
fn create_image_grid(width: usize, height: usize, max_depth: f64) -> Array3<f64> {
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
fn generate_noisy_received_signal(code: &Array1<num_complex::Complex64>, noise_level: f64) -> Array1<f64> {
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

/// Analyze image quality metrics
struct ImageStats {
    max_value: f64,
    mean_value: f64,
    dynamic_range: f64,
}

fn analyze_image_quality(image: &Array2<f64>) -> ImageStats {
    let mut max_val = 0.0f64;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for &val in image.iter() {
        max_val = max_val.max(val);
        sum += val;
        count += 1;
    }

    let mean_val = sum / count as f64;
    let dynamic_range = if mean_val > 0.0 {
        20.0 * (max_val / mean_val).log10()
    } else {
        0.0
    };

    ImageStats {
        max_value: max_val,
        mean_value: mean_val,
        dynamic_range,
    }
}

/// Analyze pulse compression results
struct CompressionStats {
    compression_ratio: f64,
    peak_sidelobe_db: f64,
    main_lobe_width: usize,
}

fn analyze_pulse_compression(original: &Array1<f64>, compressed: &Array1<f64>) -> CompressionStats {
    // Find peak in compressed signal
    let mut peak_idx = 0;
    let mut peak_value = 0.0f64;

    for (i, &val) in compressed.iter().enumerate() {
        if val > peak_value {
            peak_value = val;
            peak_idx = i;
        }
    }

    // Find main lobe width (points above half maximum)
    let half_max = peak_value / 2.0;
    let mut start_idx = peak_idx;
    let mut end_idx = peak_idx;

    while start_idx > 0 && compressed[start_idx] > half_max {
        start_idx -= 1;
    }
    while end_idx < compressed.len() - 1 && compressed[end_idx] > half_max {
        end_idx += 1;
    }

    let main_lobe_width = end_idx - start_idx;

    // Find peak sidelobe
    let mut max_sidelobe = 0.0f64;
    for (i, &val) in compressed.iter().enumerate() {
        if i < start_idx || i > end_idx {
            max_sidelobe = max_sidelobe.max(val);
        }
    }

    let peak_sidelobe_db = if max_sidelobe > 0.0 {
        20.0 * (max_sidelobe / peak_value).log10()
    } else {
        -100.0
    };

    let compression_ratio = original.len() as f64 / main_lobe_width as f64;

    CompressionStats {
        compression_ratio,
        peak_sidelobe_db,
        main_lobe_width,
    }
}
