//! Signal Behavior Test
//!
//! Validates that signal generators (SineWave, etc.) produce correct
//! amplitudes at t=0 and throughout the simulation.
//!
//! Mathematical Specification:
//! - SineWave: A·sin(2πf·t + φ)
//! - At t=0 with φ=0: amplitude should be 0
//! - At t=T/4: amplitude should be A (where T=1/f)
//! - At t=T/2: amplitude should be 0
//!
//! This test ensures sources don't inject spurious energy at t=0.
//!
//! Author: Ryan Clanton (@ryancinsight)
//! Date: 2025-01-20

use kwavers::domain::signal::{Signal, SineWave};
use std::f64::consts::PI;

#[test]
fn test_sinewave_initial_amplitude() {
    // Create sine wave with zero phase
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa
    let phase = 0.0;

    let signal = SineWave::new(frequency, amplitude, phase);

    println!("\n=== SineWave Initial Amplitude Test ===");
    println!("Frequency: {} Hz", frequency);
    println!("Amplitude: {} Pa", amplitude);
    println!("Phase: {} rad", phase);

    // Test at t=0
    let amp_t0 = signal.amplitude(0.0);
    println!("\nAt t=0:");
    println!("  amplitude(0) = {:.6e}", amp_t0);
    println!("  Expected: ~0 (sin(0) = 0)");

    // Should be exactly zero for zero phase sine wave
    assert!(
        amp_t0.abs() < 1e-10,
        "SineWave with zero phase should have zero amplitude at t=0, got {:.3e}",
        amp_t0
    );

    println!("  ✓ Amplitude at t=0 is zero");

    // Test at t=T/4 (quarter period - peak)
    let period = 1.0 / frequency;
    let t_quarter = period / 4.0;
    let amp_t_quarter = signal.amplitude(t_quarter);

    println!("\nAt t=T/4 ({:.3e} s):", t_quarter);
    println!("  amplitude(T/4) = {:.6e}", amp_t_quarter);
    println!("  Expected: {:.6e} (sin(π/2) = 1)", amplitude);

    assert!(
        (amp_t_quarter - amplitude).abs() < 1e-6,
        "Amplitude at T/4 should be A, got {:.3e} (expected {:.3e})",
        amp_t_quarter,
        amplitude
    );

    println!("  ✓ Amplitude at T/4 is A");

    // Test at t=T/2 (half period - zero crossing)
    let t_half = period / 2.0;
    let amp_t_half = signal.amplitude(t_half);

    println!("\nAt t=T/2 ({:.3e} s):", t_half);
    println!("  amplitude(T/2) = {:.6e}", amp_t_half);
    println!("  Expected: ~0 (sin(π) = 0)");

    assert!(
        amp_t_half.abs() < 1e-6,
        "Amplitude at T/2 should be zero, got {:.3e}",
        amp_t_half
    );

    println!("  ✓ Amplitude at T/2 is zero");

    println!("\n✓ SineWave initial amplitude test PASSED");
}

#[test]
fn test_sinewave_with_phase_offset() {
    // Test sine wave with π/2 phase offset (becomes cosine)
    let frequency = 1e6;
    let amplitude = 1e5;
    let phase = PI / 2.0; // cos(ωt) = sin(ωt + π/2)

    let signal = SineWave::new(frequency, amplitude, phase);

    println!("\n=== SineWave with Phase Offset Test ===");
    println!("Phase: {} rad (π/2)", phase);

    // At t=0, cos(0) = 1
    let amp_t0 = signal.amplitude(0.0);
    println!("\nAt t=0:");
    println!("  amplitude(0) = {:.6e}", amp_t0);
    println!("  Expected: {:.6e} (cos(0) = 1)", amplitude);

    assert!(
        (amp_t0 - amplitude).abs() < 1e-6,
        "Cosine (phase=π/2) should have amplitude A at t=0, got {:.3e}",
        amp_t0
    );

    println!("  ✓ Cosine starts at maximum");

    // At t=T/4, cos(π/2) = 0
    let period = 1.0 / frequency;
    let t_quarter = period / 4.0;
    let amp_t_quarter = signal.amplitude(t_quarter);

    println!("\nAt t=T/4:");
    println!("  amplitude(T/4) = {:.6e}", amp_t_quarter);
    println!("  Expected: ~0 (cos(π/2) = 0)");

    assert!(
        amp_t_quarter.abs() < 1e-6,
        "Cosine at T/4 should be zero, got {:.3e}",
        amp_t_quarter
    );

    println!("  ✓ Cosine has zero crossing at T/4");

    println!("\n✓ Phase offset test PASSED");
}

#[test]
fn test_sinewave_negative_time() {
    // Verify signal handles negative times correctly (should extend smoothly)
    let signal = SineWave::new(1e6, 1e5, 0.0);

    println!("\n=== SineWave Negative Time Test ===");

    // Test at small negative time
    let t_neg = -1e-7; // -0.1 µs
    let amp_neg = signal.amplitude(t_neg);

    println!("At t={:.3e} s:", t_neg);
    println!("  amplitude = {:.6e}", amp_neg);

    // sin(-x) = -sin(x), so should be negative small value
    assert!(
        amp_neg < 0.0,
        "Amplitude at negative time should be negative (sin is odd function)"
    );

    // Check symmetry: amplitude(-t) ≈ -amplitude(t)
    let t_pos = 1e-7;
    let amp_pos = signal.amplitude(t_pos);

    let symmetry_error = (amp_neg + amp_pos).abs();
    println!("\nSymmetry check:");
    println!("  amplitude(-t) = {:.6e}", amp_neg);
    println!("  amplitude(+t) = {:.6e}", amp_pos);
    println!("  amplitude(-t) + amplitude(+t) = {:.6e}", symmetry_error);

    assert!(
        symmetry_error < 1e-6,
        "Sine wave should be antisymmetric: sin(-t) = -sin(t)"
    );

    println!("  ✓ Antisymmetry verified");

    println!("\n✓ Negative time test PASSED");
}

#[test]
fn test_sinewave_amplitude_range() {
    // Verify amplitude never exceeds specified maximum
    let frequency = 1e6;
    let max_amplitude = 1e5;
    let signal = SineWave::new(frequency, max_amplitude, 0.0);

    println!("\n=== SineWave Amplitude Range Test ===");
    println!("Maximum amplitude: {:.3e} Pa", max_amplitude);

    let period = 1.0 / frequency;
    let num_samples = 100;
    let mut measured_max: f64 = 0.0;
    let mut measured_min: f64 = 0.0;

    for i in 0..num_samples {
        let t = (i as f64 / num_samples as f64) * period;
        let amp = signal.amplitude(t);
        measured_max = measured_max.max(amp);
        measured_min = measured_min.min(amp);
    }

    println!("\nMeasured over one period:");
    println!("  Max amplitude: {:.6e}", measured_max);
    println!("  Min amplitude: {:.6e}", measured_min);
    println!("  Peak-to-peak: {:.6e}", measured_max - measured_min);

    assert!(
        (measured_max - max_amplitude).abs() < 1e-6,
        "Maximum should be A = {:.3e}, got {:.3e}",
        max_amplitude,
        measured_max
    );

    assert!(
        (measured_min + max_amplitude).abs() < 1e-6,
        "Minimum should be -A = {:.3e}, got {:.3e}",
        -max_amplitude,
        measured_min
    );

    println!("  ✓ Amplitude range correct");

    println!("\n✓ Amplitude range test PASSED");
}

#[test]
fn test_sinewave_frequency() {
    // Verify frequency by counting zero crossings
    let frequency = 1e6; // 1 MHz
    let signal = SineWave::new(frequency, 1e5, 0.0);

    println!("\n=== SineWave Frequency Test ===");
    println!("Expected frequency: {:.3e} Hz", frequency);

    let period = 1.0 / frequency;
    let total_time = 10.0 * period; // 10 periods
    let dt = period / 100.0; // 100 samples per period
    let num_samples = (total_time / dt) as usize;

    let mut zero_crossings = 0;
    let mut last_sign = 0;

    for i in 0..num_samples {
        let t = i as f64 * dt;
        let amp = signal.amplitude(t);
        let sign = if amp > 0.0 {
            1
        } else if amp < 0.0 {
            -1
        } else {
            0
        };

        if sign != 0 && last_sign != 0 && sign != last_sign {
            zero_crossings += 1;
        }
        if sign != 0 {
            last_sign = sign;
        }
    }

    let measured_periods = zero_crossings as f64 / 2.0;
    let measured_frequency = measured_periods / total_time;

    println!("\nMeasured over {:.1} periods:", total_time / period);
    println!("  Zero crossings: {}", zero_crossings);
    println!("  Measured periods: {:.1}", measured_periods);
    println!("  Measured frequency: {:.6e} Hz", measured_frequency);
    println!(
        "  Relative error: {:.2}%",
        (measured_frequency - frequency).abs() / frequency * 100.0
    );

    let freq_error = (measured_frequency - frequency).abs() / frequency;
    assert!(
        freq_error < 0.01,
        "Frequency error too large: {:.2}%",
        freq_error * 100.0
    );

    println!("  ✓ Frequency correct");

    println!("\n✓ Frequency test PASSED");
}
