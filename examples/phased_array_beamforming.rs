// examples/phased_array_beamforming.rs
//! Advanced Phased Array Beamforming Example
//!
//! This example demonstrates the advanced phased array transducer capabilities:
//! - Electronic beam focusing at specific points
//! - Beam steering to different angles
//! - Custom phase delay patterns
//! - Element cross-talk modeling
//! - Performance analysis and comparison
//!
//! Showcases the proper use of kwavers factory patterns with advanced transducer modeling.

use kwavers::{
    grid::Grid,
    medium::homogeneous::HomogeneousMedium,
    signal::SineWave,
    source::{BeamformingMode, PhasedArrayConfig, PhasedArrayTransducer, Source},
    KwaversResult,
};
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    println!("🎯 Advanced Phased Array Beamforming Demonstration");
    println!("=====================================================");

    // Demonstrate different beamforming modes
    demonstrate_focus_beamforming()?;
    demonstrate_beam_steering()?;
    demonstrate_custom_patterns()?;
    demonstrate_cross_talk_effects()?;

    println!("\n✅ Phased array beamforming demonstration completed successfully!");
    println!(
        "   Demonstrated: Electronic focusing, beam steering, custom patterns, cross-talk modeling"
    );

    Ok(())
}

/// Demonstrate electronic beam focusing capabilities
fn demonstrate_focus_beamforming() -> KwaversResult<()> {
    println!("\n📍 Electronic Beam Focusing Demonstration");
    println!("-----------------------------------------");

    // Create grid and medium
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    // Configure phased array
    let array_config = PhasedArrayConfig {
        num_elements: 32,
        element_spacing: 0.3e-3, // λ/2 at 2.5 MHz
        element_width: 0.25e-3,
        element_height: 10e-3,
        center_position: (0.0, 0.0, 0.0),
        frequency: 2.5e6,
        enable_crosstalk: true,
        crosstalk_coefficient: 0.05,
    };

    // Create signal
    let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));

    // Create phased array transducer
    let mut array = PhasedArrayTransducer::create(array_config.clone(), signal, &medium, &grid)?;

    println!("Array Configuration:");
    println!("  Elements: {}", array_config.num_elements);
    println!("  Spacing: {:.1} mm", array_config.element_spacing * 1e3);
    println!("  Frequency: {:.1} MHz", array_config.frequency / 1e6);

    // Test different focus depths
    let focus_depths = [20e-3, 40e-3, 60e-3, 80e-3]; // 20, 40, 60, 80 mm

    for &depth in &focus_depths {
        let focus_target = (0.0, 0.0, depth);
        array.set_beamforming(BeamformingMode::Focus {
            target: focus_target,
        });

        let delays = array.element_delays();
        let max_delay = delays.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let delay_range = delays
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - delays
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

        println!(
            "  Focus at {:.0}mm: Max delay = {:.2} rad, Range = {:.2} rad",
            depth * 1e3,
            max_delay,
            delay_range
        );

        // Calculate focal quality metric
        let center_idx = delays.len() / 2;
        let edge_delay_avg = (delays[0] + delays[delays.len() - 1]) / 2.0;
        let focus_quality = (edge_delay_avg - delays[center_idx]).abs();
        println!("    Focus quality: {:.2} rad", focus_quality);
    }

    Ok(())
}

/// Demonstrate beam steering capabilities
fn demonstrate_beam_steering() -> KwaversResult<()> {
    println!("\n🎯 Electronic Beam Steering Demonstration");
    println!("-----------------------------------------");

    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    let array_config = PhasedArrayConfig {
        num_elements: 64,
        element_spacing: 0.3e-3,
        frequency: 2.5e6,
        ..Default::default()
    };

    let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
    let mut array = PhasedArrayTransducer::create(array_config, signal, &medium, &grid)?;

    // Test different steering angles
    let steering_angles: [f64; 4] = [0.0, 15.0, 30.0, 45.0]; // degrees

    for &angle_deg in &steering_angles {
        let theta = angle_deg.to_radians();
        let phi = 0.0; // Steering in x-z plane

        array.set_beamforming(BeamformingMode::Steer { theta, phi });

        let delays = array.element_delays();
        let delay_gradient = (delays[delays.len() - 1] - delays[0]) / (delays.len() - 1) as f64;

        println!(
            "  Steer to {:.0}°: Delay gradient = {:.3} rad/element",
            angle_deg, delay_gradient
        );

        // Calculate steering accuracy
        let wavelength = 1500.0 / 2.5e6; // c/f
        let expected_gradient = -2.0 * std::f64::consts::PI * theta.sin() / wavelength * 0.3e-3;

        let accuracy_metric = if expected_gradient.abs() < 1e-9 {
            // For 0-degree case, use absolute error metric
            let absolute_error = delay_gradient.abs();
            if absolute_error < 1e-6 {
                format!("within {:.2e} rad/element tolerance", absolute_error)
            } else {
                format!("absolute error: {:.3e} rad/element", absolute_error)
            }
        } else {
            // For non-zero angles, use relative error
            let relative_error =
                ((delay_gradient - expected_gradient) / expected_gradient * 100.0).abs();
            if relative_error < 0.1 {
                format!("within {:.2}% of theoretical", relative_error)
            } else {
                format!("error: {:.1}% from theoretical", relative_error)
            }
        };
        println!("    Steering accuracy: {}", accuracy_metric);
    }

    Ok(())
}

/// Demonstrate custom beamforming patterns
fn demonstrate_custom_patterns() -> KwaversResult<()> {
    println!("\n🎨 Custom Beamforming Patterns Demonstration");
    println!("--------------------------------------------");

    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    let array_config = PhasedArrayConfig {
        num_elements: 32,
        frequency: 2.5e6,
        ..Default::default()
    };

    let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
    let mut array = PhasedArrayTransducer::create(array_config.clone(), signal, &medium, &grid)?;

    // Pattern 1: Dual focus (split beam)
    println!("  Pattern 1: Dual Focus (Split Beam)");
    let mut dual_focus_delays = Vec::new();
    for i in 0..array_config.num_elements {
        let element_pos = -((array_config.num_elements - 1) as f64) / 2.0 + i as f64;
        let delay = if element_pos < 0.0 {
            // Left half focuses at (-10mm, 0, 40mm)
            calculate_focus_delay(
                element_pos * array_config.element_spacing,
                (-10e-3, 0.0, 40e-3),
                1500.0,
                2.5e6,
            )
        } else {
            // Right half focuses at (10mm, 0, 40mm)
            calculate_focus_delay(
                element_pos * array_config.element_spacing,
                (10e-3, 0.0, 40e-3),
                1500.0,
                2.5e6,
            )
        };
        dual_focus_delays.push(delay);
    }
    array.set_beamforming(BeamformingMode::Custom {
        delays: dual_focus_delays.clone(),
    });
    println!(
        "    Applied {} custom delays for dual focus",
        dual_focus_delays.len()
    );

    // Pattern 2: Gaussian apodization with linear phase
    println!("  Pattern 2: Gaussian Apodization with Linear Phase");
    let mut gaussian_delays = Vec::new();
    for i in 0..array_config.num_elements {
        let element_pos = -((array_config.num_elements - 1) as f64) / 2.0 + i as f64;
        let linear_phase = element_pos * 0.1; // Linear phase progression
        let gaussian_weight = (-0.5 * (element_pos / 8.0).powi(2)).exp(); // Gaussian envelope
        gaussian_delays.push(linear_phase * gaussian_weight);
    }
    array.set_beamforming(BeamformingMode::Custom {
        delays: gaussian_delays.clone(),
    });
    println!("    Applied Gaussian-weighted linear phase pattern");

    // Pattern 3: Sinusoidal phase pattern
    println!("  Pattern 3: Sinusoidal Phase Pattern");
    let mut sinusoidal_delays = Vec::new();
    for i in 0..array_config.num_elements {
        let phase = 2.0 * std::f64::consts::PI * i as f64 / array_config.num_elements as f64;
        sinusoidal_delays.push((phase * 2.0).sin() * 0.5); // Sinusoidal pattern
    }
    array.set_beamforming(BeamformingMode::Custom {
        delays: sinusoidal_delays,
    })?;
    println!("    Applied sinusoidal phase pattern for side lobe control");

    Ok(())
}

/// Demonstrate cross-talk effects
fn demonstrate_cross_talk_effects() -> KwaversResult<()> {
    println!("\n🔗 Element Cross-talk Effects Demonstration");
    println!("-------------------------------------------");

    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    // Compare arrays with and without cross-talk
    let crosstalk_coefficients = [0.0, 0.05, 0.1, 0.2];

    for &coefficient in &crosstalk_coefficients {
        let array_config = PhasedArrayConfig {
            num_elements: 32,
            frequency: 2.5e6,
            enable_crosstalk: coefficient > 0.0,
            crosstalk_coefficient: coefficient,
            ..Default::default()
        };

        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        let mut array = PhasedArrayTransducer::create(array_config, signal, &medium, &grid)?;

        // Set focus at 50mm depth
        let focus_target = (0.0, 0.0, 50e-3);
        array.set_beamforming(BeamformingMode::Focus {
            target: focus_target,
        });

        // Calculate source term at focus point
        let source_at_focus =
            array.get_source_term(0.0, focus_target.0, focus_target.1, focus_target.2, &grid);

        // Calculate source term off-axis (5mm lateral)
        let source_off_axis =
            array.get_source_term(0.0, 5e-3, focus_target.1, focus_target.2, &grid);

        const EPSILON: f64 = 1e-10;
        let contrast_ratio = if source_off_axis.abs() > EPSILON {
            20.0 * (source_at_focus.abs() / source_off_axis.abs()).log10()
        } else {
            f64::INFINITY
        };

        println!(
            "  Cross-talk {:.1}: Focus/Off-axis = {:.1} dB",
            coefficient, contrast_ratio
        );
    }

    Ok(())
}

/// Helper function to calculate focus delay for a single element
fn calculate_focus_delay(
    element_x: f64,
    target: (f64, f64, f64),
    sound_speed: f64,
    frequency: f64,
) -> f64 {
    let distance = ((element_x - target.0).powi(2) + target.1.powi(2) + target.2.powi(2)).sqrt();
    let reference_distance = (target.0.powi(2) + target.1.powi(2) + target.2.powi(2)).sqrt();
    let path_difference = distance - reference_distance;
    let wavelength = sound_speed / frequency;
    -2.0 * std::f64::consts::PI * path_difference / wavelength
}
