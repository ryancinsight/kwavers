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
use kwavers::{
    sensor::beamforming::{
        ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D,
        BeamformingProcessor3D,
    },
    KwaversResult,
};
use ndarray::{Array3, Array4};
use std::time::Instant;

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> KwaversResult<()> {
    println!("🚀 Real-Time 3D Beamforming Demonstration");
    println!("==========================================");

    // Demonstrate different beamforming configurations
    demonstrate_delay_and_sum().await?;
    demonstrate_dynamic_focusing().await?;
    demonstrate_streaming_processing().await?;
    demonstrate_performance_benchmarking().await?;

    println!("\n✅ Real-time 3D beamforming demonstration completed successfully!");
    println!("   Demonstrated: GPU acceleration, streaming processing, dynamic focusing, performance benchmarking");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("🚫 Real-Time 3D Beamforming Example");
    println!("====================================");
    println!("This example requires GPU acceleration.");
    println!("Run with: cargo run --example real_time_3d_beamforming --features gpu");
    println!("Or: cargo run --example real_time_3d_beamforming --features gpu,wgpu");
}

#[cfg(feature = "gpu")]
/// Demonstrate basic delay-and-sum beamforming
async fn demonstrate_delay_and_sum() -> KwaversResult<()> {
    println!("\n📊 Delay-and-Sum 3D Beamforming Demonstration");
    println!("---------------------------------------------");

    // Create configuration
    let config = BeamformingConfig3D {
        volume_dims: (64, 64, 64), // Smaller for demo
        num_elements_3d: (16, 16, 8), // 2,048 elements
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Volume: {}×{}×{}", config.volume_dims.0, config.volume_dims.1, config.volume_dims.2);
    println!("  Elements: {}×{}×{} = {}", config.num_elements_3d.0, config.num_elements_3d.1, config.num_elements_3d.2,
             config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2);
    println!("  Voxel spacing: {:.1}mm", config.voxel_spacing.0 * 1000.0);

    // Create processor
    let mut processor = BeamformingProcessor3D::new(config).await?;

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
    println!("Processing Results:");
    println!("  Processing time: {:.2}ms", processing_time);
    println!("  Reconstruction rate: {:.1} volumes/sec", 1000.0 / processing_time);
    println!("  Volume dimensions: {}×{}×{}", volume.nrows(), volume.ncols(), volume.npages());
    println!("  Volume range: {:.3} to {:.3}", volume.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             volume.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

    // Get performance metrics
    let metrics = processor.metrics();
    println!("Performance Metrics:");
    println!("  GPU memory usage: {:.1} MB", metrics.gpu_memory_mb);
    println!("  CPU memory usage: {:.1} MB", metrics.cpu_memory_mb);

    Ok(())
}

/// Demonstrate dynamic focusing capabilities
#[cfg(feature = "gpu")]
async fn demonstrate_dynamic_focusing() -> KwaversResult<()> {
    println!("\n🎯 Dynamic Focusing 3D Beamforming Demonstration");
    println!("------------------------------------------------");

    let config = BeamformingConfig3D {
        volume_dims: (32, 32, 32), // Even smaller for quick demo
        num_elements_3d: (8, 8, 4),
        ..Default::default()
    };

    let mut processor = BeamformingProcessor3D::new(config).await?;

    // Generate RF data
    let rf_data = generate_synthetic_rf_data(2, 256, 256);

    // Compare static vs dynamic focusing
    let algorithms = vec![
        ("Static Focusing", BeamformingAlgorithm3D::DelayAndSum {
            dynamic_focusing: false,
            apodization: ApodizationWindow::Hamming,
            sub_volume_size: None,
        }),
        ("Dynamic Focusing", BeamformingAlgorithm3D::DelayAndSum {
            dynamic_focusing: true,
            apodization: ApodizationWindow::Hamming,
            sub_volume_size: None,
        }),
    ];

    for (name, algorithm) in algorithms {
        let start_time = Instant::now();
        let volume = processor.process_volume(&rf_data, &algorithm)?;
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Calculate basic quality metrics
        let mean_signal = volume.mean().unwrap_or(0.0);
        let max_signal = volume.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let dynamic_range = if mean_signal != 0.0 {
            20.0 * (max_signal / mean_signal.abs()).log10()
        } else {
            0.0
        };

        println!("{}:", name);
        println!("  Processing time: {:.2}ms", processing_time);
        println!("  Dynamic range: {:.1} dB", dynamic_range);
        println!("  Peak signal: {:.3}", max_signal);
    }

    Ok(())
}

/// Demonstrate real-time streaming processing for 4D ultrasound
#[cfg(feature = "gpu")]
async fn demonstrate_streaming_processing() -> KwaversResult<()> {
    println!("\n📺 Real-Time Streaming 4D Ultrasound Demonstration");
    println!("--------------------------------------------------");

    let config = BeamformingConfig3D {
        volume_dims: (32, 32, 32),
        num_elements_3d: (8, 8, 4),
        enable_streaming: true,
        streaming_buffer_size: 8,
        ..Default::default()
    };

    let mut processor = BeamformingProcessor3D::new(config).await?;

    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: true,
        apodization: ApodizationWindow::Hann,
        sub_volume_size: None,
    };

    println!("Streaming Configuration:");
    println!("  Buffer size: {} frames", config.streaming_buffer_size);
    println!("  Volume size: {}×{}×{}", config.volume_dims.0, config.volume_dims.1, config.volume_dims.2);

    // Simulate streaming data acquisition
    let total_frames = 24;
    let mut processed_volumes = 0;
    let mut total_processing_time = 0.0;

    println!("Processing streaming frames...");

    for frame_idx in 0..total_frames {
        // Generate synthetic frame data
        let frame_data = generate_synthetic_frame(
            config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2,
            256,
        );

        let start_time = Instant::now();

        // Process streaming frame
        if let Some(volume) = processor.process_streaming(&frame_data, &algorithm)? {
            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
            total_processing_time += processing_time;
            processed_volumes += 1;

            if processed_volumes <= 3 || processed_volumes % 5 == 0 {
                println!("  Frame {}: Processed volume in {:.2}ms", frame_idx, processing_time);
            }
        } else {
            println!("  Frame {}: Added to buffer (buffer not full yet)", frame_idx);
        }
    }

    if processed_volumes > 0 {
        let avg_processing_time = total_processing_time / processed_volumes as f64;
        println!("Streaming Results:");
        println!("  Total volumes processed: {}", processed_volumes);
        println!("  Average processing time: {:.2}ms per volume", avg_processing_time);
        println!("  Effective frame rate: {:.1} volumes/sec", 1000.0 / avg_processing_time);
    }

    Ok(())
}

/// Demonstrate performance benchmarking against CPU implementation
#[cfg(feature = "gpu")]
async fn demonstrate_performance_benchmarking() -> KwaversResult<()> {
    println!("\n⚡ Performance Benchmarking Demonstration");
    println!("-----------------------------------------");

    let config = BeamformingConfig3D {
        volume_dims: (16, 16, 16), // Very small for benchmarking
        num_elements_3d: (4, 4, 2),
        ..Default::default()
    };

    let mut processor = BeamformingProcessor3D::new(config).await?;

    // Generate test data
    let rf_data = generate_synthetic_rf_data(1, 32, 128);

    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: false,
        apodization: ApodizationWindow::Rectangular,
        sub_volume_size: None,
    };

    // Benchmark GPU implementation
    println!("Benchmarking GPU implementation...");
    let mut gpu_times = Vec::new();

    for _ in 0..5 {
        let start = Instant::now();
        let _volume = processor.process_volume(&rf_data, &algorithm)?;
        gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let gpu_avg = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
    let gpu_min = gpu_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Simulate CPU implementation (simplified)
    println!("Benchmarking CPU implementation (simplified)...");
    let mut cpu_times = Vec::new();

    for _ in 0..5 {
        let start = Instant::now();
        let _volume = cpu_beamforming_delay_and_sum(&rf_data, &config)?;
        cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let cpu_min = cpu_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Calculate speedup
    let speedup_avg = cpu_avg / gpu_avg;
    let speedup_best = cpu_min / gpu_min;

    println!("Performance Results:");
    println!("  GPU average: {:.3}ms", gpu_avg);
    println!("  GPU best: {:.3}ms", gpu_min);
    println!("  CPU average: {:.3}ms", cpu_avg);
    println!("  CPU best: {:.3}ms", cpu_min);
    println!("  Speedup (average): {:.1}x", speedup_avg);
    println!("  Speedup (best): {:.1}x", speedup_best);

    // Check against targets
    let target_time = 10.0; // 10ms target
    let target_speedup = 10.0; // 10x speedup target

    println!("Target Analysis:");
    println!("  Target time: <{}ms per volume", target_time);
    println!("  Achieved: {:.1}ms {}", gpu_avg, if gpu_avg < target_time { "✅" } else { "❌" });
    println!("  Target speedup: >{}x", target_speedup);
    println!("  Achieved: {:.1}x {}", speedup_avg, if speedup_avg > target_speedup { "✅" } else { "❌" });

    Ok(())
}

/// Generate synthetic RF data for testing
fn generate_synthetic_rf_data(frames: usize, channels: usize, samples: usize) -> Array4<f32> {
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
fn generate_synthetic_frame(channels: usize, samples: usize) -> Array3<f32> {
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

/// Simplified CPU implementation for benchmarking
#[cfg(feature = "gpu")]
fn cpu_beamforming_delay_and_sum(rf_data: &Array4<f32>, config: &BeamformingConfig3D) -> KwaversResult<Array3<f32>> {
    let (frames, channels, samples, _) = rf_data.dim();
    let (vol_x, vol_y, vol_z) = config.volume_dims;

    let mut volume = Array3::<f32>::zeros((vol_x, vol_y, vol_z));

    // Simplified CPU delay-and-sum (much slower than optimized version)
    for x in 0..vol_x {
        for y in 0..vol_y {
            for z in 0..vol_z {
                let voxel_pos = (
                    x as f32 * config.voxel_spacing.0,
                    y as f32 * config.voxel_spacing.1,
                    z as f32 * config.voxel_spacing.2,
                );

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Process subset of elements for speed
                let elements_to_process = (channels / 16).min(64); // Process fewer elements

                for e in 0..elements_to_process {
                    // Simplified delay calculation
                    let element_pos = (
                        ((e % 16) as f32 - 7.5) * config.element_spacing.0,
                        (((e / 16) % 16) as f32 - 7.5) * config.element_spacing.1,
                        ((e / 256) as f32 - 3.5) * config.element_spacing.2,
                    );

                    let distance = ((voxel_pos.0 - element_pos.0).powi(2) +
                                  (voxel_pos.1 - element_pos.1).powi(2) +
                                  (voxel_pos.2 - element_pos.2).powi(2)).sqrt();

                    let delay_samples = (distance / config.sound_speed * config.sampling_frequency) as usize;
                    let sample_idx = delay_samples.min(samples - 1);

                    // Sum across frames
                    for f in 0..frames {
                        sum += rf_data[[f, e, sample_idx, 0]] as f64;
                    }
                    weight_sum += 1.0;
                }

                volume[[x, y, z]] = (sum / weight_sum) as f32;
            }
        }
    }

    Ok(volume)
}
