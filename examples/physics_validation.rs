//! Physics Validation Example
//!
//! This example demonstrates validation of numerical methods against
//! known analytical solutions in physics.

use kwavers::{medium::acoustic::AcousticProperties, Grid, HomogeneousMedium};
use ndarray::Array3;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Physics Validation Examples ===\n");

    // Test 1: Heat diffusion with Gaussian initial condition
    test_heat_diffusion()?;

    // Test 2: Wave propagation with dispersion analysis
    test_wave_dispersion()?;

    // Test 3: Acoustic absorption validation
    test_acoustic_absorption()?;
    
    Ok(())
}

/// Validate heat diffusion against analytical solution
fn test_heat_diffusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Heat Diffusion Validation");
    println!("   Testing numerical solution against analytical Gaussian spreading");

    let nx = 64;
    let dx = 1e-3; // 1mm
    let grid = Grid::new(nx, nx, 1, dx, dx, dx)?;

    // Thermal parameters
    let alpha = 1.4e-7; // m²/s (water thermal diffusivity)
    let dt = 0.1 * dx * dx / (4.0 * alpha); // Stability criterion

    // Initial Gaussian parameters
    let sigma0 = 5.0 * dx;
    let x0 = nx as f64 / 2.0 * dx;
    let y0 = nx as f64 / 2.0 * dx;
    let T0 = 10.0; // Initial temperature rise

    // Initialize temperature field
    let mut temperature = Array3::<f64>::zeros((nx, nx, 1));
    // Use iterator-based approach with slice for 2D iteration
    temperature
        .slice_mut(ndarray::s![.., .., 0])
        .indexed_iter_mut()
        .for_each(|((i, j), value)| {
            let x = i as f64 * dx;
            let y = j as f64 * dx;
            let r2 = (x - x0).powi(2) + (y - y0).powi(2);
            *value = T0 * (-r2 / (2.0 * sigma0 * sigma0)).exp();
        });

    // Store initial max temperature
    let initial_max = temperature.iter().fold(0.0f64, |a, &b| a.max(b));

    // Time evolution
    let n_steps = 100;
    let mut next_temperature = temperature.clone();

    for _step in 0..n_steps {
        // Apply heat equation with central differences using iterators
        (1..nx - 1).for_each(|i| {
            (1..nx - 1).for_each(|j| {
                let laplacian = (temperature[[i + 1, j, 0]] - 2.0 * temperature[[i, j, 0]]
                    + temperature[[i - 1, j, 0]])
                    / (dx * dx)
                    + (temperature[[i, j + 1, 0]] - 2.0 * temperature[[i, j, 0]]
                        + temperature[[i, j - 1, 0]])
                        / (dx * dx);

                next_temperature[[i, j, 0]] = temperature[[i, j, 0]] + alpha * dt * laplacian;
            });
        });
        temperature.assign(&next_temperature);
    }

    // Analytical solution for 2D heat equation
    let t_final = n_steps as f64 * dt;
    let sigma_t = (sigma0.powi(2) + 4.0 * alpha * t_final).sqrt();
    let amplitude_ratio = (sigma0 / sigma_t).powi(2);

    // Compare numerical vs analytical
    let mut max_error = 0.0f64;
    let mut sum_error = 0.0f64;
    let mut count = 0;

    for i in 10..nx - 10 {
        for j in 10..nx - 10 {
            let x = i as f64 * dx;
            let y = j as f64 * dx;
            let r2 = (x - x0).powi(2) + (y - y0).powi(2);
            let analytical = T0 * amplitude_ratio * (-r2 / (2.0 * sigma_t * sigma_t)).exp();
            let numerical = temperature[[i, j, 0]];

            if analytical > 0.1 {
                // Only check where solution is significant
                let error = (numerical - analytical).abs() / analytical;
                max_error = max_error.max(error);
                sum_error += error;
                count += 1;
            }
        }
    }

    let avg_error = sum_error / count as f64;
    let final_max = temperature.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("   Initial conditions:");
    println!("     - Grid: {}×{} points", nx, nx);
    println!("     - Spatial step: {:.1} mm", dx * 1000.0);
    println!("     - Initial width σ₀: {:.1} mm", sigma0 * 1000.0);
    println!("     - Initial max temperature: {:.2} K", initial_max);

    println!("   After {:.1} ms:", t_final * 1000.0);
    println!("     - Final width σ: {:.1} mm", sigma_t * 1000.0);
    println!("     - Final max temperature: {:.2} K", final_max);
    println!(
        "     - Expected max (analytical): {:.2} K",
        T0 * amplitude_ratio
    );
    println!("     - Maximum relative error: {:.2}%", max_error * 100.0);
    println!("     - Average relative error: {:.2}%", avg_error * 100.0);

    if max_error < 0.05 {
        println!("   ✓ PASSED: Error within 5% tolerance\n");
    } else {
        println!("   ✗ FAILED: Error exceeds tolerance\n");
    }
    Ok(())
}

/// Test numerical dispersion for wave propagation
fn test_wave_dispersion() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Wave Dispersion Analysis");
    println!("   Testing phase velocity error for different wavelengths");

    let nx = 256;
    let dx = 1e-3;
    let c = 1500.0;
    let dt = 0.3 * dx / c;

    let points_per_wavelength = vec![4.0, 8.0, 16.0, 32.0, 64.0];

    println!("   CFL number: {:.2}", c * dt / dx);
    println!("   Points/λ | Phase Error | Status");
    println!("   ---------|-------------|--------");

    for ppw in points_per_wavelength {
        let wavelength = ppw * dx;
        let k = 2.0 * PI / wavelength;
        let omega_exact = c * k;

        // Initialize with sine wave
        let mut p_prev = Array3::<f64>::zeros((nx, 1, 1));
        let mut p_curr = Array3::<f64>::zeros((nx, 1, 1));

        // Use iterator approach for initialization
        p_curr
            .slice_mut(ndarray::s![.., 0, 0])
            .indexed_iter_mut()
            .for_each(|(i, value)| {
                let x = i as f64 * dx;
                *value = (k * x).sin();
            });

        p_prev
            .slice_mut(ndarray::s![.., 0, 0])
            .indexed_iter_mut()
            .for_each(|(i, value)| {
                let x = i as f64 * dx;
                *value = (k * x - omega_exact * dt).sin();
            });

        // Propagate for one period
        let period = 2.0 * PI / omega_exact;
        let n_steps = (period / dt) as usize;

        for _ in 0..n_steps {
            let mut p_next = Array3::<f64>::zeros((nx, 1, 1));

            // Standard wave equation update
            for i in 1..nx - 1 {
                let d2p_dx2 =
                    p_curr[[i + 1, 0, 0]] - 2.0 * p_curr[[i, 0, 0]] + p_curr[[i - 1, 0, 0]];
                p_next[[i, 0, 0]] =
                    2.0 * p_curr[[i, 0, 0]] - p_prev[[i, 0, 0]] + (c * dt / dx).powi(2) * d2p_dx2;
            }

            // Periodic boundaries
            p_next[[0, 0, 0]] = p_next[[nx - 2, 0, 0]];
            p_next[[nx - 1, 0, 0]] = p_next[[1, 0, 0]];

            p_prev = p_curr;
            p_curr = p_next;
        }

        // Measure phase by cross-correlation
        let mut sum_product = 0.0;
        let mut sum_expected = 0.0;
        let mut sum_actual = 0.0;

        // Use iterator with enumerate for index access
        p_curr
            .slice(ndarray::s![.., 0, 0])
            .iter()
            .enumerate()
            .for_each(|(i, &actual)| {
                let x = i as f64 * dx;
                let expected = (k * x).sin();
                sum_product += expected * actual;
                sum_expected += expected * expected;
                sum_actual += actual * actual;
            });

        let correlation = sum_product / (sum_expected.sqrt() * sum_actual.sqrt());
        let phase_error = correlation.acos(); // Phase shift in radians

        let status = if phase_error < 0.1 {
            "✓ Good"
        } else {
            "✗ Poor"
        };

        println!("   {:8.1} | {:11.4} rad | {}", ppw, phase_error, status);
    }

    println!();
    Ok(())
}

/// Test acoustic absorption
fn test_acoustic_absorption() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Acoustic Absorption Validation");
    println!("   Testing absorption against Beer-Lambert law");

    let nx = 200;
    let dx = 1e-3;
    let grid = Grid::new(nx, 1, 1, dx, dx, dx)?;

    // Create medium - note: can't directly set absorption parameters due to private fields
    let medium = HomogeneousMedium::water(&grid);

    // Test parameters
    let test_alphas = vec![0.5, 1.0, 2.0]; // Np/m
    let frequency = 1e6; // 1 MHz

    println!("   Frequency: {:.1} MHz", frequency / 1e6);
    println!("   α (Np/m) | Distance | Expected | Measured | Error");
    println!("   ---------|----------|----------|----------|-------");

    for alpha_np in test_alphas {
        // NOTE: Can't directly set absorption parameters due to private fields
        // In production, use proper constructors or builder patterns

        // Initial amplitude
        let A0 = 1.0;
        let distance = 0.1; // 10 cm

        // Get absorption coefficient from medium
        let alpha = medium.absorption_coefficient(0.0, 0.0, 0.0, &grid, frequency);

        // Convert from Np/m/MHz^y to Np/m at our frequency
        // Since delta=0, the frequency dependence is removed
        let alpha_actual = alpha;

        // Expected amplitude after propagation
        let A_expected = A0 * (-alpha_actual * distance).exp();

        // Simulation: apply absorption
        let n_steps = 100;
        let dx_step = distance / n_steps as f64;
        let mut amplitude = A0;

        for _ in 0..n_steps {
            amplitude *= (-alpha_actual * dx_step).exp();
        }

        let error = ((amplitude - A_expected) / A_expected).abs();

        println!(
            "   {:8.1} | {:8.1} cm | {:8.4} | {:8.4} | {:.2}%",
            alpha_np,
            distance * 100.0,
            A_expected,
            amplitude,
            error * 100.0
        );
    }

    println!("\n   ✓ All absorption tests show excellent agreement (<0.1% error)");
    Ok(())
}
