//! Doppler Velocity Estimation Example
//!
//! Demonstrates how to use the Doppler ultrasound module for blood flow
//! velocity estimation using the autocorrelation method.
//!
//! This example shows:
//! - Setting up Doppler imaging parameters for vascular imaging
//! - Generating synthetic I/Q data representing blood flow
//! - Estimating velocity using the Kasai autocorrelation method
//! - Applying wall filters to remove clutter
//! - Creating 2D color flow images
//!
//! # Usage
//!
//! ```bash
//! cargo run --example doppler_velocity_estimation
//! ```

use kwavers::clinical::imaging::doppler::{
    AutocorrelationConfig, AutocorrelationEstimator, ColorFlowConfig, ColorFlowImaging,
};
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Kwavers: Doppler Velocity Estimation Example          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Configure Doppler Imaging for Vascular Application
    // ========================================================================

    println!("📋 Configuration:");
    println!("  └─ Creating vascular imaging parameters (7.5 MHz, 5 kHz PRF)");

    let doppler_config = AutocorrelationConfig::vascular();

    println!(
        "  └─ Center frequency: {} MHz",
        doppler_config.center_frequency / 1e6
    );
    println!("  └─ PRF: {} kHz", doppler_config.prf / 1e3);
    println!(
        "  └─ Ensemble size: {} pulses",
        doppler_config.ensemble_size
    );
    println!(
        "  └─ Nyquist velocity: ±{:.2} m/s\n",
        doppler_config.nyquist_velocity()
    );

    // ========================================================================
    // 2. Create Synthetic I/Q Data
    // ========================================================================

    println!("🔬 Generating Synthetic I/Q Data:");

    let ensemble_size = doppler_config.ensemble_size;
    let n_depths = 128; // Imaging depth samples
    let n_beams = 64; // Lateral beams

    println!(
        "  └─ Dimensions: {} pulses × {} depths × {} beams",
        ensemble_size, n_depths, n_beams
    );

    // Generate synthetic blood flow signal with known velocity
    let target_velocity = 0.3; // m/s (typical arterial flow)
    let doppler_frequency =
        2.0 * doppler_config.center_frequency * target_velocity / doppler_config.speed_of_sound;

    println!("  └─ Simulated flow velocity: {:.2} m/s", target_velocity);
    println!("  └─ Doppler shift: {:.1} Hz\n", doppler_frequency);

    let mut iq_data = Array3::<Complex64>::zeros((ensemble_size, n_depths, n_beams));

    // Simulate blood flow in center region with Doppler shift
    for n in 0..ensemble_size {
        for depth in 0..n_depths {
            for beam in 0..n_beams {
                // Create flow region (vessel in center)
                let is_vessel = (depth > 40 && depth < 90) && (beam > 20 && beam < 45);

                if is_vessel {
                    // Blood flow signal with Doppler shift
                    let t = n as f64 / doppler_config.prf;
                    let phase = 2.0 * PI * doppler_frequency * t;
                    let amplitude = 0.8; // Strong scatterer signal

                    iq_data[[n, depth, beam]] =
                        Complex64::new(amplitude * phase.cos(), amplitude * phase.sin());
                } else {
                    // Background noise
                    iq_data[[n, depth, beam]] = Complex64::new(0.01, 0.01);
                }
            }
        }
    }

    // ========================================================================
    // 3. Estimate Velocity using Autocorrelation Method
    // ========================================================================

    println!("🎯 Estimating Velocity (Kasai Autocorrelation Method):");

    let estimator = AutocorrelationEstimator::new(doppler_config.clone());
    let (velocity, variance) = estimator.estimate(&iq_data.view())?;

    println!(
        "  └─ Velocity map computed: {} × {}",
        velocity.dim().0,
        velocity.dim().1
    );

    // Find peak velocity in flow region
    let mut max_velocity = 0.0;
    let mut max_pos = (0, 0);

    for depth in 40..90 {
        for beam in 20..45 {
            let v = velocity[[depth, beam]].abs();
            if v > max_velocity {
                max_velocity = v;
                max_pos = (depth, beam);
            }
        }
    }

    println!(
        "  └─ Peak velocity: {:.3} m/s at depth={}, beam={}",
        max_velocity, max_pos.0, max_pos.1
    );
    println!("  └─ Target velocity: {:.3} m/s", target_velocity);
    println!(
        "  └─ Estimation error: {:.1}%\n",
        ((max_velocity - target_velocity) / target_velocity * 100.0).abs()
    );

    // ========================================================================
    // 4. Apply Variance-Based Quality Filtering
    // ========================================================================

    println!("🔍 Applying Quality Filter:");

    let filtered_velocity = estimator.filter_by_variance(&velocity, &variance);

    let mut high_confidence_pixels = 0;
    for var in variance.iter() {
        if *var <= doppler_config.variance_threshold {
            high_confidence_pixels += 1;
        }
    }

    let total_pixels = n_depths * n_beams;
    let confidence_pct = (high_confidence_pixels as f64 / total_pixels as f64) * 100.0;

    println!(
        "  └─ Variance threshold: {}",
        doppler_config.variance_threshold
    );
    println!(
        "  └─ High-confidence pixels: {} / {} ({:.1}%)\n",
        high_confidence_pixels, total_pixels, confidence_pct
    );

    // ========================================================================
    // 5. Generate Color Flow Image
    // ========================================================================

    println!("🎨 Generating Color Flow Image:");

    let color_flow_config = ColorFlowConfig {
        autocorrelation: doppler_config,
        wall_filter: Default::default(),
        spatial_averaging: Some((3, 3)), // 3×3 spatial smoothing
    };

    let color_flow = ColorFlowImaging::new(color_flow_config);
    let flow_result = color_flow.process(&iq_data.view())?;

    println!("  └─ Color flow map generated");
    println!("  └─ Spatial averaging: 3×3 kernel");
    println!(
        "  └─ Center frequency: {:.1} MHz",
        flow_result.center_frequency / 1e6
    );
    println!("  └─ PRF: {:.1} kHz\n", flow_result.prf / 1e3);

    // ========================================================================
    // 6. Display Statistics
    // ========================================================================

    println!("📊 Velocity Statistics:");

    // Compute statistics in vessel region
    let mut vessel_velocities = Vec::new();
    for depth in 40..90 {
        for beam in 20..45 {
            if variance[[depth, beam]] < 0.3 {
                vessel_velocities.push(filtered_velocity[[depth, beam]]);
            }
        }
    }

    if !vessel_velocities.is_empty() {
        let mean_velocity: f64 =
            vessel_velocities.iter().sum::<f64>() / vessel_velocities.len() as f64;
        let std_dev = (vessel_velocities
            .iter()
            .map(|v| (v - mean_velocity).powi(2))
            .sum::<f64>()
            / vessel_velocities.len() as f64)
            .sqrt();

        println!(
            "  └─ Mean velocity (vessel): {:.3} ± {:.3} m/s",
            mean_velocity, std_dev
        );
        println!("  └─ Samples used: {}", vessel_velocities.len());
    }

    println!("\n✅ Doppler velocity estimation complete!");
    println!("\n💡 Clinical Applications:");
    println!("  • Vascular stenosis detection");
    println!("  • Cardiac valve flow assessment");
    println!("  • Fetal umbilical artery monitoring");
    println!("  • Perfusion analysis");

    Ok(())
}
