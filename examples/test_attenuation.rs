//! Test example for wave attenuation
//!
//! This example verifies that the attenuation implementation works correctly
//! by comparing numerical results with analytical solutions.

use kwavers::{physics::AttenuationCalculator, Grid, KwaversResult};
use ndarray::Array3;

fn main() -> KwaversResult<()> {
    println!("=== Wave Attenuation Test ===\n");

    // Test parameters
    let frequency = 1e6; // 1 MHz
    let wave_speed = 1500.0; // m/s (water)
    let absorption_coefficient = 0.002; // Np/m (typical for water at 1 MHz)

    // Create attenuation calculator
    let atten_calc = AttenuationCalculator::new(absorption_coefficient, frequency, wave_speed);

    // Test 1: Amplitude attenuation
    println!("Test 1: Amplitude attenuation (Beer-Lambert law)");
    let initial_amplitude = 1.0;
    let distances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]; // meters

    for &distance in &distances {
        let attenuated = atten_calc.amplitude_at_distance(initial_amplitude, distance);
        let expected = initial_amplitude * (-absorption_coefficient * distance).exp();
        let error = (attenuated - expected).abs();

        println!("  Distance: {:.2} m", distance);
        println!("    Calculated: {:.6}", attenuated);
        println!("    Expected:   {:.6}", expected);
        println!("    Error:      {:.2e}", error);
        assert!(error < 1e-10, "Amplitude attenuation error too large");
    }

    // Test 2: Intensity attenuation
    println!("\nTest 2: Intensity attenuation");
    let initial_intensity = 1.0;

    for &distance in &distances {
        let attenuated = atten_calc.intensity_at_distance(initial_intensity, distance);
        let expected = initial_intensity * (-2.0 * absorption_coefficient * distance).exp();
        let error = (attenuated - expected).abs();

        println!("  Distance: {:.2} m", distance);
        println!("    Calculated: {:.6}", attenuated);
        println!("    Expected:   {:.6}", expected);
        println!("    Error:      {:.2e}", error);
        assert!(error < 1e-10, "Intensity attenuation error too large");
    }

    // Test 3: Attenuation in dB
    println!("\nTest 3: Attenuation in dB");

    for &distance in &distances {
        let atten_db = atten_calc.attenuation_db(distance);
        let expected = 8.686 * absorption_coefficient * distance;
        let error = (atten_db - expected).abs();

        println!("  Distance: {:.2} m", distance);
        println!("    Calculated: {:.4} dB", atten_db);
        println!("    Expected:   {:.4} dB", expected);
        println!("    Error:      {:.2e}", error);
        assert!(error < 1e-10, "dB attenuation error too large");
    }

    // Test 4: Tissue absorption model
    println!("\nTest 4: Tissue absorption (frequency power law)");
    let alpha_0 = 0.5; // dB/cm/MHz
    let power_law = 1.1; // Typical for soft tissue
    let frequencies = [0.5e6, 1e6, 2e6, 5e6, 10e6]; // Hz

    for &freq in &frequencies {
        let absorption = AttenuationCalculator::tissue_absorption(freq / 1e6, alpha_0, power_law);
        let expected = alpha_0 * (freq / 1e6).powf(power_law);
        let error = (absorption - expected).abs();

        println!("  Frequency: {:.1} MHz", freq / 1e6);
        println!("    Calculated: {:.4} dB/cm", absorption);
        println!("    Expected:   {:.4} dB/cm", expected);
        println!("    Error:      {:.2e}", error);
        assert!(error < 1e-10, "Tissue absorption error too large");
    }

    // Test 5: 3D field attenuation
    println!("\nTest 5: 3D field attenuation");
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
    let mut field = Array3::from_elem((10, 10, 10), 1.0);
    let source_position = [0.005, 0.005, 0.005]; // Center of grid
    let grid_spacing = [grid.dx, grid.dy, grid.dz];

    atten_calc.apply_attenuation_field(&mut field, source_position, grid_spacing);

    // Check center point (should be unchanged)
    let center_val = field[(5, 5, 5)];
    println!("  Center value: {:.6} (should be ~1.0)", center_val);
    assert!(
        (center_val - 1.0).abs() < 0.01,
        "Center point should have minimal attenuation"
    );

    // Check corner point (should be attenuated)
    let corner_val = field[(0, 0, 0)];
    let corner_distance = ((0.0 - source_position[0]).powi(2)
        + (0.0 - source_position[1]).powi(2)
        + (0.0 - source_position[2]).powi(2))
    .sqrt();
    let expected_corner = (-absorption_coefficient * corner_distance).exp();

    println!("  Corner value: {:.6}", corner_val);
    println!("  Expected:     {:.6}", expected_corner);
    assert!(
        (corner_val - expected_corner).abs() < 0.01,
        "Corner attenuation incorrect"
    );

    // Test 6: Classical absorption
    println!("\nTest 6: Classical thermo-viscous absorption");
    let density = 1000.0; // kg/m³
    let sound_speed = 1500.0; // m/s
    let shear_viscosity = 1e-3; // Pa·s (water)
    let bulk_viscosity = 2.8e-3; // Pa·s (water)
    let thermal_conductivity = 0.6; // W/(m·K)
    let specific_heat_ratio = 1.0; // For liquids
    let specific_heat_cp = 4180.0; // J/(kg·K)

    let classical_abs = AttenuationCalculator::classical_absorption(
        frequency,
        density,
        sound_speed,
        shear_viscosity,
        bulk_viscosity,
        thermal_conductivity,
        specific_heat_ratio,
        specific_heat_cp,
    );

    println!("  Classical absorption at 1 MHz: {:.6} Np/m", classical_abs);
    println!("  (Should be small for water at this frequency)");
    assert!(
        classical_abs > 0.0 && classical_abs < 1.0,
        "Classical absorption out of range"
    );

    println!("\n✅ All attenuation tests passed!");

    Ok(())
}
