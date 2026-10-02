//! GPU-versus-CPU delay-and-sum benchmark for the real-time 3D beamforming demo.

use super::synthetic_data::generate_synthetic_rf_data;
use kwavers_analysis::signal_processing::beamforming::three_dimensional::{
    Beamforming3dApodizationWindow as ApodizationWindow, BeamformingAlgorithm3D,
    BeamformingConfig3D, BeamformingProcessor3D,
};
use kwavers_core::error::KwaversResult;
use kwavers_gpu::beamforming::three_dimensional::WgpuBeamformingProvider;
use leto::{Array3, Array4};
use std::io::Write;
use std::time::Instant;

/// Demonstrate performance benchmarking against CPU implementation
pub(crate) fn demonstrate_performance_benchmarking() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n⚡ Performance Benchmarking Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-----------------------------------------"
    );

    let config = BeamformingConfig3D {
        volume_dims: (16, 16, 16), // Very small for benchmarking
        num_elements_3d: (4, 4, 2),
        ..Default::default()
    };

    let mut processor =
        BeamformingProcessor3D::with_provider(config.clone(), WgpuBeamformingProvider::new()?)?;

    // Generate test data
    let rf_data = generate_synthetic_rf_data(1, 32, 128);

    let algorithm = BeamformingAlgorithm3D::DelayAndSum {
        dynamic_focusing: false,
        apodization: ApodizationWindow::Rectangular,
        sub_volume_size: None,
    };

    // Benchmark GPU implementation
    let _ = writeln!(
        std::io::stdout().lock(),
        "Benchmarking GPU implementation..."
    );
    let mut gpu_times = Vec::new();

    for _ in 0..5 {
        let start = Instant::now();
        let _volume = processor.process_volume(&rf_data, &algorithm)?;
        gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let gpu_avg = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
    let gpu_min = gpu_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Simulate CPU implementation (simplified)
    let _ = writeln!(
        std::io::stdout().lock(),
        "Benchmarking CPU implementation (simplified)..."
    );
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

    let _ = writeln!(std::io::stdout().lock(), "Performance Results:");
    let _ = writeln!(std::io::stdout().lock(), "  GPU average: {:.3}ms", gpu_avg);
    let _ = writeln!(std::io::stdout().lock(), "  GPU best: {:.3}ms", gpu_min);
    let _ = writeln!(std::io::stdout().lock(), "  CPU average: {:.3}ms", cpu_avg);
    let _ = writeln!(std::io::stdout().lock(), "  CPU best: {:.3}ms", cpu_min);
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Speedup (average): {:.1}x",
        speedup_avg
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Speedup (best): {:.1}x",
        speedup_best
    );

    // Check against targets
    let target_time = 10.0; // 10ms target
    let target_speedup = 10.0; // 10x speedup target

    let _ = writeln!(std::io::stdout().lock(), "Target Analysis:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Target time: <{}ms per volume",
        target_time
    );
    eprintln!(
        "  Achieved: {:.1}ms {}",
        gpu_avg,
        if gpu_avg < target_time { "✅" } else { "❌" }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Target speedup: >{}x",
        target_speedup
    );
    eprintln!(
        "  Achieved: {:.1}x {}",
        speedup_avg,
        if speedup_avg > target_speedup {
            "✅"
        } else {
            "❌"
        }
    );

    Ok(())
}

/// Simplified CPU implementation for benchmarking
fn cpu_beamforming_delay_and_sum(
    rf_data: &Array4<f32>,
    config: &BeamformingConfig3D,
) -> KwaversResult<Array3<f32>> {
    let [frames, channels, samples, _] = rf_data.shape();
    let (vol_x, vol_y, vol_z) = config.volume_dims;

    let mut volume = Array3::<f32>::zeros((vol_x, vol_y, vol_z));

    // Simplified CPU delay-and-sum (much slower than optimized version)
    for x in 0..vol_x {
        for y in 0..vol_y {
            for z in 0..vol_z {
                let voxel_pos = (
                    x as f32 * config.voxel_spacing.0 as f32,
                    y as f32 * config.voxel_spacing.1 as f32,
                    z as f32 * config.voxel_spacing.2 as f32,
                );

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Process subset of elements for speed
                let elements_to_process = (channels / 16).min(64); // Process fewer elements

                for e in 0..elements_to_process {
                    // Simplified delay calculation
                    let element_pos = (
                        ((e % 16) as f32 - 7.5) * config.element_spacing_3d.0 as f32,
                        (((e / 16) % 16) as f32 - 7.5) * config.element_spacing_3d.1 as f32,
                        ((e / 256) as f32 - 3.5) * config.element_spacing_3d.2 as f32,
                    );

                    let distance = ((voxel_pos.0 - element_pos.0).powi(2)
                        + (voxel_pos.1 - element_pos.1).powi(2)
                        + (voxel_pos.2 - element_pos.2).powi(2))
                    .sqrt();

                    let delay_samples = (distance / config.sound_speed as f32
                        * config.sampling_frequency as f32)
                        as usize;
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
