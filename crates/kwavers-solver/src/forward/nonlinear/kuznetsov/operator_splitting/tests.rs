use super::OperatorSplittingSolver;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA, TWO_PI};
use ndarray::Array3;

#[test]
fn test_harmonic_generation() -> Result<(), kwavers_core::error::KwaversError> {
    // Test that nonlinear propagation generates harmonics
    let nx = 128;
    let ny = 1;
    let nz = 1;
    let dx = 1e-4;
    let dy = dx;
    let dz = dx;

    let frequency = MHZ_TO_HZ; // 1 MHz
    let wavelength = SOUND_SPEED_WATER_SIM / frequency;
    let k = TWO_PI / wavelength;

    let solver = OperatorSplittingSolver::new(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        1000.0,                             // density
        SOUND_SPEED_WATER_SIM,              // sound speed
        3.5,                                // B/A for water
        dx / (SOUND_SPEED_WATER_SIM * 4.0), // dt from CFL
    );

    // Initialize with sinusoidal wave
    let mut pressure = Array3::zeros((nx, ny, nz));
    let mut pressure_prev = Array3::zeros((nx, ny, nz));
    let mut pressure_prev2 = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        let x = i as f64 * dx;
        let amplitude = MPA_TO_PA; // 1 MPa
        pressure[[i, 0, 0]] = amplitude * (k * x).sin();
        pressure_prev[[i, 0, 0]] = amplitude * (k * x - TWO_PI * frequency * solver.dt).sin();
    }

    // Propagate for multiple steps
    let initial_max = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let initial_energy: f64 = pressure.iter().map(|p| p * p).sum();

    for step in 0..100 {
        let pressure_next = solver.step(&pressure, &pressure_prev, &pressure_prev2);

        // Check for numerical instability and return proper error
        if pressure_next.iter().any(|p| p.is_nan() || p.abs() > 1e10) {
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::NumericalInstabilityGeneral {
                    message: format!(
                        "Numerical instability at step {}: NaN or overflow detected",
                        step
                    ),
                },
            ));
        }

        pressure_prev2.assign(&pressure_prev);
        pressure_prev.assign(&pressure);
        pressure.assign(&pressure_next);

        // Track energy conservation
        if step % 20 == 0 || step == 99 {
            let current_max = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let current_energy: f64 = pressure.iter().map(|p| p * p).sum();
            println!(
                "Step {}: max={:.2e}, energy={:.2e} (initial: max={:.2e}, energy={:.2e})",
                step, current_max, current_energy, initial_max, initial_energy
            );
        }
    }

    // Perform FFT to check for harmonics
    use kwavers_math::fft::fft_1d_array;

    let spectrum = fft_1d_array(&pressure.slice(ndarray::s![.., 0, 0]).to_owned());

    // Find fundamental and second harmonic peaks
    // For spatial FFT: wavenumber k = 2π/λ = 2πf/c
    // FFT bin = k * L / (2π) = k * nx * dx / (2π)
    let wavelength = SOUND_SPEED_WATER_SIM / frequency;
    let k_fundamental = TWO_PI / wavelength;
    let fundamental_idx = (k_fundamental * nx as f64 * dx / (TWO_PI)).round() as usize;
    let second_harmonic_idx = 2 * fundamental_idx;

    // Make sure indices are within bounds
    let fundamental_idx = fundamental_idx.min(nx / 2 - 1);
    let second_harmonic_idx = second_harmonic_idx.min(nx / 2 - 1);

    let fundamental_amp = spectrum[fundamental_idx].norm();
    let second_harmonic_amp = if second_harmonic_idx < spectrum.len() {
        spectrum[second_harmonic_idx].norm()
    } else {
        0.0
    };

    // Second harmonic should be generated (non-zero)
    assert!(
        second_harmonic_amp > fundamental_amp * 0.01,
        "Second harmonic not generated: fundamental={:.2e}, second={:.2e}",
        fundamental_amp,
        second_harmonic_amp
    );

    Ok(())
}

#[test]
fn test_shock_steepening() {
    // Test that high-amplitude waves steepen (shock formation)
    let nx = 256;
    let ny = 1;
    let nz = 1;
    let dx = 5e-5;
    let dy = dx;
    let dz = dx;

    let solver = OperatorSplittingSolver::new(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        1000.0,                             // density
        SOUND_SPEED_WATER_SIM,              // sound speed
        3.5,                                // B/A for water
        dx / (SOUND_SPEED_WATER_SIM * 4.0), // dt from CFL
    );

    // Initialize with high-amplitude sine wave
    let mut pressure = Array3::zeros((nx, ny, nz));
    let mut pressure_prev = Array3::zeros((nx, ny, nz));
    let mut pressure_prev2 = Array3::zeros((nx, ny, nz));

    let wavelength = 1.5e-3; // 1.5 mm
    let k = TWO_PI / wavelength;
    let amplitude = 5e6; // 5 MPa (high amplitude for shock formation)

    for i in 0..nx {
        let x = i as f64 * dx;
        pressure[[i, 0, 0]] = amplitude * (k * x).sin();
        pressure_prev[[i, 0, 0]] = pressure[[i, 0, 0]];
    }

    // Measure initial gradient
    let mut initial_max_gradient: f64 = 0.0;
    for i in 1..nx - 1 {
        let gradient = (pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]).abs() / (2.0 * dx);
        initial_max_gradient = initial_max_gradient.max(gradient);
    }

    // Propagate to allow shock formation
    for _ in 0..200 {
        let pressure_next = solver.step(&pressure, &pressure_prev, &pressure_prev2);
        pressure_prev2.assign(&pressure_prev);
        pressure_prev.assign(&pressure);
        pressure.assign(&pressure_next);
    }

    // Measure final gradient (should be steepened)
    let mut final_max_gradient: f64 = 0.0;
    for i in 1..nx - 1 {
        let gradient = (pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]).abs() / (2.0 * dx);
        final_max_gradient = final_max_gradient.max(gradient);
    }

    // Gradient should increase (steepening)
    assert!(
        final_max_gradient > initial_max_gradient * 1.05, // Allow 5% steepening minimum
        "No shock steepening: initial gradient={:.2e}, final={:.2e}, ratio={:.2}",
        initial_max_gradient,
        final_max_gradient,
        final_max_gradient / initial_max_gradient
    );
}
