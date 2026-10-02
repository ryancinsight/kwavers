//! Advanced Ultrasound Imaging Example
//!
//! This example demonstrates advanced ultrasound imaging techniques:
//! - Synthetic Aperture (SA) imaging
//! - Plane Wave Imaging (PWI) with compounding
//! - Coded Excitation with pulse compression
//!
//! Showcases the implementation of modern ultrasound imaging methods
//! for improved resolution, frame rate, and penetration.

use kwavers_physics::acoustics::imaging::modalities::ultrasound::advanced::{
    CodedExcitationConfig, CodedExcitationProcessor, ExcitationCode, PlaneWaveCompounding,
    PlaneWaveReconstruction, SyntheticApertureConfig, SyntheticApertureReconstruction,
    UltrasoundPlaneWaveConfig,
};
use leto::Array3;
use std::io::Write;

#[path = "advanced_ultrasound_imaging/quality_metrics.rs"]
mod quality_metrics;
#[path = "advanced_ultrasound_imaging/synthetic_data.rs"]
mod synthetic_data;
use quality_metrics::{analyze_image_quality, analyze_pulse_compression};
use synthetic_data::{
    create_image_grid, generate_noisy_received_signal, generate_synthetic_pw_rf_data,
    generate_synthetic_sa_rf_data,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🩺 Advanced Ultrasound Imaging Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "==========================================="
    );

    // Demonstrate synthetic aperture imaging
    demonstrate_synthetic_aperture()?;

    // Demonstrate plane wave imaging with compounding
    demonstrate_plane_wave_imaging()?;

    // Demonstrate coded excitation
    demonstrate_coded_excitation()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n✅ Advanced ultrasound imaging demonstration completed!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Demonstrated: Synthetic aperture, plane wave imaging, coded excitation"
    );

    Ok(())
}

/// Demonstrate synthetic aperture imaging
fn demonstrate_synthetic_aperture() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n🎯 Synthetic Aperture (SA) Imaging Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-----------------------------------------------"
    );

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

    let _ = writeln!(std::io::stdout().lock(), "SA Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  TX Elements: {}",
        sa_config.num_tx_elements
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  RX Elements: {}",
        sa_config.num_rx_elements
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Element Spacing: {:.1} mm",
        sa_config.element_spacing * 1e3
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Center Frequency: {:.1} MHz",
        sa_config.frequency / 1e6
    );

    // Create SA reconstruction processor
    let sa_reconstruction = SyntheticApertureReconstruction::new(sa_config.clone());

    // Create synthetic RF data for demonstration
    // In practice, this would come from actual transducer data
    let n_samples = 2048;
    let n_rx = sa_config.num_rx_elements;
    let n_tx = sa_config.num_tx_elements;

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Generating synthetic RF data..."
    );
    let rf_data = generate_synthetic_sa_rf_data(n_samples, n_rx, n_tx, &sa_config);

    // Create image grid
    let image_width = 100;
    let image_height = 100;
    let image_depth = 50e-3; // 50mm depth
    let image_grid = create_image_grid(image_width, image_height, image_depth);

    let _ = writeln!(std::io::stdout().lock(), "  Reconstructing SA image...");
    let sa_image = sa_reconstruction.reconstruct(&rf_data, &image_grid);

    // Analyze image quality
    let image_stats = analyze_image_quality(&sa_image);
    let _ = writeln!(std::io::stdout().lock(), "  SA Image Statistics:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Image Size: {} x {}",
        sa_image.shape()[0],
        sa_image.shape()[1]
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Max Value: {:.3}",
        image_stats.max_value
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Mean Value: {:.3}",
        image_stats.mean_value
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Dynamic Range: {:.1} dB",
        image_stats.dynamic_range
    );

    // SA provides excellent resolution but requires many transmissions
    let total_transmissions = sa_config.num_tx_elements * sa_config.num_rx_elements;
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Total TX-RX Pairs: {}",
        total_transmissions
    );

    Ok(())
}

/// Demonstrate plane wave imaging with multi-angle compounding
fn demonstrate_plane_wave_imaging() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n🌊 Plane Wave Imaging (PWI) with Compounding"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-------------------------------------------"
    );

    // Configure plane wave imaging
    let base_config = UltrasoundPlaneWaveConfig {
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
        0.0,                   // 0 degrees
        10.0f64.to_radians(),  // 10 degrees
        20.0f64.to_radians(),  // 20 degrees
    ];

    let _ = writeln!(std::io::stdout().lock(), "PWI Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Elements: {}",
        base_config.num_elements
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Compounding Angles: {}",
        angles.len()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Frequency: {:.1} MHz",
        base_config.frequency / 1e6
    );

    // Create individual plane wave reconstructions
    let mut pw_images = Vec::new();

    for (i, &angle) in angles.iter().enumerate() {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Processing angle {}: {:.0}°",
            i + 1,
            angle.to_degrees()
        );

        let pw_config = UltrasoundPlaneWaveConfig {
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
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Compounding {} angle images...",
        angles.len()
    );

    let pw_compounding = PlaneWaveCompounding::new(&angles, base_config);
    let compounded_images = Array3::from_shape_vec(
        (
            pw_images.len(),
            pw_images[0].shape()[0],
            pw_images[0].shape()[1],
        ),
        pw_images.iter().flat_map(|a| a.iter().copied()).collect(),
    )?;

    let compounded_image = pw_compounding.compound(&compounded_images);

    // Analyze compounded image
    let image_stats = analyze_image_quality(&compounded_image);
    let _ = writeln!(std::io::stdout().lock(), "  Compounded Image Statistics:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Image Size: {} x {}",
        compounded_image.shape()[0],
        compounded_image.shape()[1]
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Max Value: {:.3}",
        image_stats.max_value
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Mean Value: {:.3}",
        image_stats.mean_value
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Dynamic Range: {:.1} dB",
        image_stats.dynamic_range
    );

    // PWI provides high frame rates with good image quality through compounding
    let frame_rate = 1.0 / (angles.len() as f64 * 100e-6); // Assuming 100μs per angle
    let _ = writeln!(
        std::io::stdout().lock(),
        "    Estimated Frame Rate: {:.0} fps",
        frame_rate
    );

    Ok(())
}

/// Demonstrate coded excitation with pulse compression
fn demonstrate_coded_excitation() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n📡 Coded Excitation with Pulse Compression"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-----------------------------------------"
    );

    // Configure coded excitation
    let codes: Vec<(&str, ExcitationCode)> = vec![
        (
            "Chirp",
            ExcitationCode::Chirp {
                start_freq: 2e6,
                end_freq: 8e6,
                length: 256,
            },
        ),
        ("Barker-7", ExcitationCode::Barker { length: 7 }),
        ("Barker-13", ExcitationCode::Barker { length: 13 }),
    ];

    for (name, code) in codes {
        let _ = writeln!(std::io::stdout().lock(), "  Testing {} Code:", name);

        let config = CodedExcitationConfig {
            code: code.clone(),
            sound_speed: 1540.0,
            sampling_frequency: 20e6,
        };

        let processor = CodedExcitationProcessor::new(config);

        // Generate excitation code
        let excitation_code = processor.generate_code();
        let _ = writeln!(
            std::io::stdout().lock(),
            "    Code Length: {} samples",
            excitation_code.len()
        );

        // Calculate theoretical SNR improvement
        let snr_improvement = processor.theoretical_snr_improvement();
        let _ = writeln!(
            std::io::stdout().lock(),
            "    Theoretical SNR Improvement: {:.1} dB",
            20.0 * snr_improvement.log10()
        );

        // Simulate received signal with noise
        let received_signal = generate_noisy_received_signal(&excitation_code, 0.1);

        // Apply matched filtering
        let compressed_signal = processor.matched_filter(&received_signal, &excitation_code);

        // Analyze compression results
        let compression_stats = analyze_pulse_compression(&received_signal, &compressed_signal);
        let _ = writeln!(
            std::io::stdout().lock(),
            "    Compression Ratio: {:.1}",
            compression_stats.compression_ratio
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "    Peak Sidelobe Level: {:.1} dB",
            compression_stats.peak_sidelobe_db
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "    Main Lobe Width: {:.0} samples",
            compression_stats.main_lobe_width
        );
    }

    Ok(())
}
