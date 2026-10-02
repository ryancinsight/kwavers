//! Real-Time 3D Beamforming Example
//!
//! This example demonstrates the real-time 3D beamforming capabilities implemented
//! in Sprint 164. It showcases GPU-accelerated volumetric ultrasound imaging with
//! streaming data processing for 4D ultrasound applications.
//!
//! # Features Demonstrated
//! - GPU-accelerated 3D delay-and-sum beamforming
//! - Real-time streaming data processing
//! - Dynamic focusing and apodization
//! - Performance benchmarking vs CPU
//! - 4D ultrasound visualization
//!
//! # Performance Targets
//! - Reconstruction time: <10ms per volume
//! - Speedup: 10-100× vs CPU implementation
//! - Memory efficiency: Streaming processing

#[cfg(feature = "gpu")]
use kwavers_analysis::signal_processing::beamforming::three_dimensional::{
    Beamforming3dApodizationWindow as ApodizationWindow, BeamformingAlgorithm3D,
    BeamformingConfig3D, BeamformingProcessor3D,
};
#[cfg(feature = "gpu")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "gpu")]
use kwavers_gpu::beamforming::three_dimensional::WgpuBeamformingProvider;
use std::io::Write;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
#[path = "real_time_3d_beamforming/benchmarking.rs"]
mod benchmarking;
#[cfg(feature = "gpu")]
#[path = "real_time_3d_beamforming/synthetic_data.rs"]
mod synthetic_data;
#[cfg(feature = "gpu")]
use benchmarking::demonstrate_performance_benchmarking;
#[cfg(feature = "gpu")]
use synthetic_data::{generate_synthetic_frame, generate_synthetic_rf_data};

#[cfg(feature = "gpu")]
fn main() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🚀 Real-Time 3D Beamforming Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "=========================================="
    );

    // Demonstrate different beamforming configurations
    demonstrate_delay_and_sum()?;
    demonstrate_dynamic_focusing()?;
    demonstrate_streaming_processing()?;
    demonstrate_performance_benchmarking()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n✅ Real-time 3D beamforming demonstration completed successfully!"
    );
    let _ = writeln!(std::io::stdout().lock(), "   Demonstrated: GPU acceleration, streaming processing, dynamic focusing, performance benchmarking");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🚫 Real-Time 3D Beamforming Example"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "===================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "This example requires GPU acceleration."
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Run with: cargo run --example real_time_3d_beamforming --features gpu"
    );
}

#[cfg(feature = "gpu")]
/// Demonstrate basic delay-and-sum beamforming
fn demonstrate_delay_and_sum() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n📊 Delay-and-Sum 3D Beamforming Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------------------"
    );

    // Create configuration
    let config = BeamformingConfig3D {
        volume_dims: (64, 64, 64),    // Smaller for demo
        num_elements_3d: (16, 16, 8), // 2,048 elements
        ..Default::default()
    };

    let _ = writeln!(std::io::stdout().lock(), "Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Volume: {}×{}×{}",
        config.volume_dims.0,
        config.volume_dims.1,
        config.volume_dims.2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Elements: {}×{}×{} = {}",
        config.num_elements_3d.0,
        config.num_elements_3d.1,
        config.num_elements_3d.2,
        config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Voxel spacing: {:.1}mm",
        config.voxel_spacing.0 * 1000.0
    );

    // Create processor
    let mut processor =
        BeamformingProcessor3D::with_provider(config.clone(), WgpuBeamformingProvider::new()?)?;

    // Generate synthetic RF data (simulating ultrasound acquisition)
    let rf_data = generate_synthetic_rf_data(
        4, // frames
        config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2,
        512, // samples per channel
    );

    // Define beamforming algorithm
    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: false,
        apodization: ApodizationWindow::Hamming,
        sub_volume_size: None,
    };

    // Process volume
    let start_time = Instant::now();
    let volume = processor.process_volume(&rf_data, &algorithm)?;
    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Display results
    let _ = writeln!(std::io::stdout().lock(), "Processing Results:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Processing time: {:.2}ms",
        processing_time
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Reconstruction rate: {:.1} volumes/sec",
        1000.0 / processing_time
    );
    let [nx, ny, nz] = volume.shape();
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Volume dimensions: {}×{}×{}",
        nx,
        ny,
        nz
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Volume range: {:.3} to {:.3}",
        volume.iter().copied().fold(f32::INFINITY, f32::min),
        volume.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Get performance metrics
    let metrics = processor.metrics();
    let _ = writeln!(std::io::stdout().lock(), "Performance Metrics:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  GPU memory usage: {:.1} MB",
        metrics.gpu_memory_mb
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  CPU memory usage: {:.1} MB",
        metrics.cpu_memory_mb
    );

    Ok(())
}

/// Demonstrate dynamic focusing capabilities
#[cfg(feature = "gpu")]
fn demonstrate_dynamic_focusing() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n🎯 Dynamic Focusing 3D Beamforming Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "------------------------------------------------"
    );

    let config = BeamformingConfig3D {
        volume_dims: (32, 32, 32), // Even smaller for quick demo
        num_elements_3d: (8, 8, 4),
        ..Default::default()
    };

    let mut processor =
        BeamformingProcessor3D::with_provider(config.clone(), WgpuBeamformingProvider::new()?)?;

    // Generate RF data
    let rf_data = generate_synthetic_rf_data(2, 256, 256);

    // Compare static vs dynamic focusing
    let algorithms = vec![
        (
            "Static Focusing",
            BeamformingAlgorithm3D::DelayAndSum {
                dynamic_focusing: false,
                apodization: ApodizationWindow::Hamming,
                sub_volume_size: None,
            },
        ),
        (
            "Dynamic Focusing",
            BeamformingAlgorithm3D::DelayAndSum {
                dynamic_focusing: true,
                apodization: ApodizationWindow::Hamming,
                sub_volume_size: None,
            },
        ),
    ];

    for (name, algorithm) in algorithms {
        let start_time = Instant::now();
        let volume = processor.process_volume(&rf_data, &algorithm)?;
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Calculate basic quality metrics
        let mean_signal = leto::mean_all(&volume)
            .expect("invariant: beamforming configuration produces a non-empty volume");
        let max_signal = volume.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let dynamic_range = if mean_signal != 0.0 {
            20.0 * (max_signal / mean_signal.abs()).log10()
        } else {
            0.0
        };

        let _ = writeln!(std::io::stdout().lock(), "{}:", name);
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Processing time: {:.2}ms",
            processing_time
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Dynamic range: {:.1} dB",
            dynamic_range
        );
        let _ = writeln!(std::io::stdout().lock(), "  Peak signal: {:.3}", max_signal);
    }

    Ok(())
}

/// Demonstrate real-time streaming processing for 4D ultrasound
#[cfg(feature = "gpu")]
fn demonstrate_streaming_processing() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n📺 Real-Time Streaming 4D Ultrasound Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "--------------------------------------------------"
    );

    let config = BeamformingConfig3D {
        volume_dims: (32, 32, 32),
        num_elements_3d: (8, 8, 4),
        enable_streaming: true,
        streaming_buffer_size: 8,
        ..Default::default()
    };

    let mut processor =
        BeamformingProcessor3D::with_provider(config.clone(), WgpuBeamformingProvider::new()?)?;

    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: true,
        apodization: ApodizationWindow::Hann,
        sub_volume_size: None,
    };

    let _ = writeln!(std::io::stdout().lock(), "Streaming Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Buffer size: {} frames",
        config.streaming_buffer_size
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Volume size: {}×{}×{}",
        config.volume_dims.0,
        config.volume_dims.1,
        config.volume_dims.2
    );

    // Simulate streaming data acquisition
    let total_frames = 24;
    let mut processed_volumes = 0;
    let mut total_processing_time = 0.0;

    let _ = writeln!(std::io::stdout().lock(), "Processing streaming frames...");

    for frame_idx in 0..total_frames {
        // Generate synthetic frame data
        let frame_data = generate_synthetic_frame(
            config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2,
            256,
        );

        let start_time = Instant::now();

        // Process streaming frame
        if let Some(_volume) = processor.process_streaming(&frame_data, &algorithm)? {
            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
            total_processing_time += processing_time;
            processed_volumes += 1;

            if processed_volumes <= 3 || processed_volumes % 5 == 0 {
                let _ = writeln!(
                    std::io::stdout().lock(),
                    "  Frame {}: Processed volume in {:.2}ms",
                    frame_idx,
                    processing_time
                );
            }
        } else {
            let _ = writeln!(
                std::io::stdout().lock(),
                "  Frame {}: Added to buffer (buffer not full yet)",
                frame_idx
            );
        }
    }

    if processed_volumes > 0 {
        let avg_processing_time = total_processing_time / processed_volumes as f64;
        let _ = writeln!(std::io::stdout().lock(), "Streaming Results:");
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Total volumes processed: {}",
            processed_volumes
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Average processing time: {:.2}ms per volume",
            avg_processing_time
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Effective frame rate: {:.1} volumes/sec",
            1000.0 / avg_processing_time
        );
    }

    Ok(())
}
