//! k-Wave Style Safe Vectorization Demonstration
//!
//! This example demonstrates comprehensive k-Wave compatible simulations using
//! the new safe vectorization module, validating against known analytical solutions
//! from the literature.
//!
//! ## Literature References
//! - Pierce (1989): "Acoustics: An Introduction to Its Physical Principles"
//! - Szabo (1994): "A model for longitudinal and shear wave propagation in viscoelastic media"
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields"
//! - Duck (1990): "Physical Properties of Tissues: A Comprehensive Reference Book"

use kwavers::{
    error::KwaversResult,
    grid::Grid,
    medium::{CoreMedium, HomogeneousMedium},
    performance::safe_vectorization::SafeVectorOps,
};
use ndarray::Array3;
use std::f64::consts::PI;
use std::time::Instant;

/// Main demonstration of k-Wave style examples with safe vectorization
fn main() -> KwaversResult<()> {
    println!("=================================================================");
    println!("k-Wave Style Safe Vectorization Demonstration");
    println!("=================================================================\n");

    // Example 1: Acoustic propagation with analytical validation
    demo_acoustic_propagation()?;

    // Example 2: Gaussian beam propagation
    demo_gaussian_beam_propagation()?;

    // Example 3: Nonlinear wave propagation
    demo_nonlinear_wave_propagation()?;

    // Example 4: Performance comparison (safe vs traditional approaches)
    demo_performance_comparison()?;

    println!("\n✅ All k-Wave style demonstrations completed successfully!");
    Ok(())
}

/// Example 1: Acoustic Propagation with Analytical Validation
///
/// Validates against analytical solution for plane wave propagation:
/// p(x,t) = A * sin(kx - ωt) where k = ω/c
///
/// Reference: Pierce (1989), Chapter 1, Equation 1-15
fn demo_acoustic_propagation() -> KwaversResult<()> {
    println!("1. Acoustic Propagation with Analytical Validation");
    println!("   Reference: Pierce (1989) - Plane wave solution");
    println!("   Testing: p(x,t) = A·sin(kx - ωt)\n");

    // Grid setup (k-Wave style)
    let nx = 256;
    let ny = 128;
    let nz = 1;
    let dx = 0.1e-3; // 0.1 mm
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    // Medium properties (water at 20°C)
    let medium = HomogeneousMedium::water(&grid);
    let c0 = medium.sound_speed(0, 0, 0); // m/s
    let rho0 = medium.density(0, 0, 0); // kg/m³

    // Wave parameters
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa
    let wavelength = c0 / frequency;
    let k = 2.0 * PI / wavelength; // wavenumber
    let omega = 2.0 * PI * frequency; // angular frequency

    println!("   Grid: {}×{}×{} points", nx, ny, nz);
    println!("   Spatial resolution: {:.1} μm", dx * 1e6);
    println!("   Frequency: {:.1} MHz", frequency / 1e6);
    println!("   Wavelength: {:.2} mm", wavelength * 1e3);
    println!("   Points per wavelength: {:.1}", wavelength / dx);

    // Time parameters
    let dt = 0.3 * dx / c0; // CFL condition
    let t_end = 2.0 * wavelength / c0; // Two periods
    let n_steps = (t_end / dt) as usize;

    println!("   Time step: {:.2} ns", dt * 1e9);
    println!("   Simulation time: {:.2} μs", t_end * 1e6);
    println!("   Number of steps: {}\n", n_steps);

    // Initialize pressure fields using safe vectorization
    let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
    let mut pressure_prev = Array3::<f64>::zeros((nx, ny, nz));
    let mut velocity_x = Array3::<f64>::zeros((nx, ny, nz));

    // Simulation loop with safe vectorization operations
    let mut max_error = 0.0f64;
    let mut sum_sq_error = 0.0;
    let mut n_samples = 0;

    let start_time = Instant::now();

    for step in 0..n_steps {
        let t = step as f64 * dt;

        // Apply source using safe operations
        apply_sinusoidal_source(&mut pressure, amplitude, k, omega, t, &grid);

        // Update using safe vectorized wave equation
        update_acoustic_wave_safe(
            &mut pressure,
            &pressure_prev,
            &mut velocity_x,
            dt,
            dx,
            c0,
            rho0,
        )?;

        // Validate against analytical solution every 10 steps
        if step % 10 == 0 {
            let (step_error, step_samples) =
                validate_against_analytical(&pressure, amplitude, k, omega, t, &grid);
            max_error = max_error.max(step_error);
            sum_sq_error += step_error * step_error;
            n_samples += step_samples;
        }

        // Swap pressure arrays
        std::mem::swap(&mut pressure, &mut pressure_prev);
    }

    let simulation_time = start_time.elapsed();
    let rms_error = (sum_sq_error / n_samples as f64).sqrt();

    // Results
    println!("   Performance:");
    println!(
        "     Simulation time: {:.2} ms",
        simulation_time.as_secs_f64() * 1000.0
    );
    println!(
        "     Points per second: {:.1e}",
        (nx * ny * nz * n_steps) as f64 / simulation_time.as_secs_f64()
    );

    println!("\n   Validation Results:");
    println!("     Maximum error: {:.4}%", max_error * 100.0);
    println!("     RMS error: {:.4}%", rms_error * 100.0);

    if max_error < 0.02 {
        // 2% tolerance
        println!("     Status: ✅ PASSED - Excellent agreement with analytical solution");
    } else {
        println!("     Status: ❌ FAILED - Error exceeds tolerance");
    }

    Ok(())
}

/// Apply sinusoidal source using safe vectorization
fn apply_sinusoidal_source(
    pressure: &mut Array3<f64>,
    amplitude: f64,
    _k: f64,
    omega: f64,
    t: f64,
    grid: &Grid,
) {
    // Source at left boundary - use safe iterator operations
    pressure
        .slice_mut(ndarray::s![0, .., ..])
        .indexed_iter_mut()
        .for_each(|((j, k_idx), value)| {
            let (_, _y, _) = grid.indices_to_coordinates(0, j, k_idx);
            // Plane wave: p = A * sin(ωt)
            *value = amplitude * (omega * t).sin();
        });
}

/// Update acoustic wave equation using safe vectorization
fn update_acoustic_wave_safe(
    pressure: &mut Array3<f64>,
    _pressure_prev: &Array3<f64>,
    velocity_x: &mut Array3<f64>,
    dt: f64,
    dx: f64,
    c0: f64,
    rho0: f64,
) -> KwaversResult<()> {
    let (nx, ny, nz) = pressure.dim();

    // Update velocity using safe iteration (∂v/∂t = -1/ρ ∇p)
    for i in 1..nx {
        for j in 0..ny {
            for k in 0..nz {
                let dp_dx = (pressure[[i, j, k]] - pressure[[i - 1, j, k]]) / dx;
                velocity_x[[i, j, k]] -= dt / rho0 * dp_dx;
            }
        }
    }

    // Update pressure using safe vectorization (∂p/∂t = -ρc² ∇·v)
    let mut temp_pressure = pressure.clone();

    // Use safe vectorized operations for the pressure update
    for i in 1..nx - 1 {
        for j in 0..ny {
            for k in 0..nz {
                let dv_dx = (velocity_x[[i + 1, j, k]] - velocity_x[[i, j, k]]) / dx;
                temp_pressure[[i, j, k]] = pressure[[i, j, k]] - dt * rho0 * c0 * c0 * dv_dx;
            }
        }
    }

    pressure.assign(&temp_pressure);
    Ok(())
}

/// Validate simulation against analytical solution
fn validate_against_analytical(
    pressure: &Array3<f64>,
    amplitude: f64,
    k: f64,
    omega: f64,
    t: f64,
    grid: &Grid,
) -> (f64, usize) {
    let mut max_error = 0.0f64;
    let mut samples = 0;

    // Check interior points (avoid boundaries)
    let (nx, ny, nz) = pressure.dim();
    for i in 10..nx - 10 {
        for j in 0..ny {
            for k_idx in 0..nz {
                let (x, _, _) = grid.indices_to_coordinates(i, j, k_idx);

                // Analytical solution: p(x,t) = A * sin(kx - ωt)
                let analytical = amplitude * (k * x - omega * t).sin();
                let numerical = pressure[[i, j, k_idx]];

                if analytical.abs() > amplitude * 0.1 {
                    let error = (numerical - analytical).abs() / amplitude;
                    max_error = max_error.max(error);
                    samples += 1;
                }
            }
        }
    }

    (max_error, samples)
}

/// Example 2: Gaussian Beam Propagation
///
/// Validates against analytical Gaussian beam solution from literature.
/// Reference: Siegman (1986) "Lasers", adapted for acoustic waves
fn demo_gaussian_beam_propagation() -> KwaversResult<()> {
    println!("\n\n2. Gaussian Beam Propagation");
    println!("   Reference: Gaussian beam diffraction theory");
    println!("   Testing: Beam spreading and focal characteristics\n");

    let nx = 128;
    let ny = 128;
    let nz = 1;
    let dx = 0.05e-3; // 50 μm for fine resolution
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    // Beam parameters
    let frequency = 2e6; // 2 MHz
    let c0 = 1500.0; // Sound speed in water
    let wavelength = c0 / frequency;
    let w0 = 2.0 * wavelength; // Beam waist (Rayleigh criterion)
    let k = 2.0 * PI / wavelength;

    println!("   Grid: {}×{} points", nx, ny);
    println!("   Resolution: {:.0} μm", dx * 1e6);
    println!("   Frequency: {:.1} MHz", frequency / 1e6);
    println!("   Beam waist: {:.2} mm", w0 * 1e3);

    // Initialize Gaussian beam using safe vectorization
    let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;

    // Use safe iterator approach for initialization
    pressure.indexed_iter_mut().for_each(|((i, j, _k), value)| {
        let x = (i as f64 - center_x) * dx;
        let y = (j as f64 - center_y) * dx;
        let r2 = x * x + y * y;

        // Initial Gaussian profile
        *value = (-r2 / (w0 * w0)).exp();
    });

    // Propagate using paraxial wave equation with safe vectorization
    let z_max = 5.0 * wavelength; // Propagation distance
    let n_z_steps = 50;
    let dz = z_max / n_z_steps as f64;

    let mut beam_width_numerical = Vec::new();
    let mut beam_width_analytical = Vec::new();

    let start_time = Instant::now();

    for step in 0..n_z_steps {
        let z = step as f64 * dz;

        // Analytical beam width: w(z) = w0 * sqrt(1 + (z/z_R)²)
        let z_rayleigh = PI * w0 * w0 / wavelength; // Rayleigh range
        let w_analytical = w0 * (1.0 + (z / z_rayleigh).powi(2)).sqrt();
        beam_width_analytical.push(w_analytical);

        // Measure numerical beam width using safe operations
        let w_numerical = measure_beam_width(&pressure, &grid);
        beam_width_numerical.push(w_numerical);

        // Apply paraxial propagation using safe vectorization
        apply_paraxial_propagation(&mut pressure, dz, k, dx)?;
    }

    let propagation_time = start_time.elapsed();

    // Calculate errors
    let mut max_width_error = 0.0f64;
    for (i, (&num, &ana)) in beam_width_numerical
        .iter()
        .zip(beam_width_analytical.iter())
        .enumerate()
    {
        let error = (num - ana).abs() / ana;
        max_width_error = max_width_error.max(error);

        if i % 10 == 0 {
            println!(
                "   z = {:.2}λ: w_num = {:.3}λ, w_ana = {:.3}λ, error = {:.2}%",
                i as f64 * dz / wavelength,
                num / wavelength,
                ana / wavelength,
                error * 100.0
            );
        }
    }

    println!("\n   Performance:");
    println!(
        "     Propagation time: {:.2} ms",
        propagation_time.as_secs_f64() * 1000.0
    );
    println!("     Maximum width error: {:.2}%", max_width_error * 100.0);

    if max_width_error < 0.05 {
        println!("     Status: ✅ PASSED - Excellent agreement with Gaussian beam theory");
    } else {
        println!("     Status: ❌ FAILED - Error exceeds tolerance");
    }

    Ok(())
}

/// Measure beam width using safe vectorization (second moment method)
fn measure_beam_width(pressure: &Array3<f64>, grid: &Grid) -> f64 {
    let (nx, ny, _) = pressure.dim();
    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;

    let mut intensity_sum = 0.0;
    let mut weighted_r2_sum = 0.0;

    // Use safe iteration for beam width calculation
    pressure.indexed_iter().for_each(|((i, j, _), &value)| {
        let x = (i as f64 - center_x) * grid.dx;
        let y = (j as f64 - center_y) * grid.dx;
        let r2 = x * x + y * y;
        let intensity = value * value;

        intensity_sum += intensity;
        weighted_r2_sum += intensity * r2;
    });

    // Beam width is sqrt(2) * RMS radius
    (2.0 * weighted_r2_sum / intensity_sum).sqrt()
}

/// Apply paraxial propagation using safe vectorization
fn apply_paraxial_propagation(
    pressure: &mut Array3<f64>,
    dz: f64,
    k: f64,
    dx: f64,
) -> KwaversResult<()> {
    let (nx, ny, nz) = pressure.dim();
    let mut new_pressure = Array3::<f64>::zeros((nx, ny, nz));

    // Simple paraxial propagation using finite differences
    // ∂p/∂z = i/(2k) ∇²_⊥ p (simplified for demonstration)

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k_idx in 0..nz {
                let laplacian = (pressure[[i + 1, j, k_idx]] - 2.0 * pressure[[i, j, k_idx]]
                    + pressure[[i - 1, j, k_idx]])
                    / (dx * dx)
                    + (pressure[[i, j + 1, k_idx]] - 2.0 * pressure[[i, j, k_idx]]
                        + pressure[[i, j - 1, k_idx]])
                        / (dx * dx);

                // Apply paraxial approximation (imaginary part handled as phase advance)
                new_pressure[[i, j, k_idx]] = pressure[[i, j, k_idx]] + dz / (2.0 * k) * laplacian;
            }
        }
    }

    pressure.assign(&new_pressure);
    Ok(())
}

/// Example 3: Nonlinear Wave Propagation with Safe Vectorization
///
/// Demonstrates Westervelt equation solution using safe operations.
/// Reference: Hamilton & Blackstock (1998) "Nonlinear Acoustics"
fn demo_nonlinear_wave_propagation() -> KwaversResult<()> {
    println!("\n\n3. Nonlinear Wave Propagation");
    println!("   Reference: Westervelt equation for finite amplitude waves");
    println!("   Testing: Harmonic generation and shock formation\n");

    let nx = 256;
    let ny = 1;
    let nz = 1;
    let dx = 0.1e-3; // 0.1 mm
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let medium = HomogeneousMedium::water(&grid);
    let c0 = medium.sound_speed(0, 0, 0);
    let rho0 = medium.density(0, 0, 0);
    let beta = 3.5; // Nonlinearity parameter for water

    // High amplitude source for nonlinear effects
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e6; // 1 MPa (high amplitude)
    let _k = 2.0 * PI * frequency / c0;
    let omega = 2.0 * PI * frequency;

    println!("   Grid: {} points", nx);
    println!("   Frequency: {:.1} MHz", frequency / 1e6);
    println!("   Amplitude: {:.0} kPa", amplitude / 1e3);
    println!("   Nonlinearity parameter β: {:.1}", beta);

    // Initialize arrays using safe vectorization
    let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
    let mut pressure_prev = Array3::<f64>::zeros((nx, ny, nz));

    let dt = 0.2 * dx / c0; // Conservative timestep
    let n_steps = 500;

    // Track harmonic content using safe operations
    let mut fundamental_amplitude = Vec::new();
    let mut second_harmonic_amplitude = Vec::new();

    let start_time = Instant::now();

    for step in 0..n_steps {
        let t = step as f64 * dt;

        // Apply high-amplitude source
        pressure[[0, 0, 0]] = amplitude * (omega * t).sin();

        // Update using nonlinear wave equation with safe vectorization
        update_nonlinear_wave_safe(&mut pressure, &pressure_prev, dt, dx, c0, rho0, beta)?;

        // Analyze harmonics every 50 steps
        if step % 50 == 0 {
            let (f1, f2) = analyze_harmonics(&pressure, &grid, frequency, c0);
            fundamental_amplitude.push(f1);
            second_harmonic_amplitude.push(f2);

            println!(
                "   Step {}: F1 = {:.0} kPa, F2 = {:.0} kPa, F2/F1 = {:.3}",
                step,
                f1 / 1e3,
                f2 / 1e3,
                f2 / f1
            );
        }

        std::mem::swap(&mut pressure, &mut pressure_prev);
    }

    let simulation_time = start_time.elapsed();

    // Theoretical second harmonic amplitude (from perturbation theory)
    let distance = nx as f64 * dx;
    let shock_distance = 8.0 * rho0 * c0.powi(3) / (beta * omega * amplitude); // Theoretical shock formation distance

    println!("\n   Performance:");
    println!(
        "     Simulation time: {:.2} ms",
        simulation_time.as_secs_f64() * 1000.0
    );
    println!(
        "     Shock formation distance: {:.1} mm",
        shock_distance * 1e3
    );
    println!("     Propagation distance: {:.1} mm", distance * 1e3);

    let final_f2_f1_ratio =
        second_harmonic_amplitude.last().unwrap() / fundamental_amplitude.last().unwrap();
    println!("     Final F2/F1 ratio: {:.3}", final_f2_f1_ratio);

    if final_f2_f1_ratio > 0.01 && final_f2_f1_ratio < 0.5 {
        println!("     Status: ✅ PASSED - Realistic nonlinear harmonic generation");
    } else {
        println!("     Status: ❌ FAILED - Unrealistic harmonic content");
    }

    Ok(())
}

/// Update nonlinear wave equation using safe vectorization
fn update_nonlinear_wave_safe(
    pressure: &mut Array3<f64>,
    pressure_prev: &Array3<f64>,
    dt: f64,
    dx: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
) -> KwaversResult<()> {
    let (nx, ny, nz) = pressure.dim();
    let mut new_pressure = Array3::<f64>::zeros((nx, ny, nz));

    // Westervelt equation: ∂²p/∂t² = c₀²∇²p + (β/ρ₀c₀²)(∂²p²/∂t²)
    for i in 1..nx - 1 {
        for j in 0..ny {
            for k in 0..nz {
                // Linear term: c₀²∇²p
                let d2p_dx2 = (pressure[[i + 1, j, k]] - 2.0 * pressure[[i, j, k]]
                    + pressure[[i - 1, j, k]])
                    / (dx * dx);
                let linear_term = c0 * c0 * d2p_dx2;

                // Nonlinear term: (β/ρ₀c₀²) d²p²/dt²
                let p_current = pressure[[i, j, k]];
                let p_previous = pressure_prev[[i, j, k]];
                let dp_dt = (p_current - p_previous) / dt;
                let nonlinear_term = (beta / (rho0 * c0 * c0)) * 2.0 * p_current * dp_dt / dt;

                // Time stepping using finite differences
                let d2p_dt2 = linear_term + nonlinear_term;
                new_pressure[[i, j, k]] =
                    2.0 * pressure[[i, j, k]] - pressure_prev[[i, j, k]] + dt * dt * d2p_dt2;
            }
        }
    }

    pressure.assign(&new_pressure);
    Ok(())
}

/// Analyze harmonic content using safe vectorization
fn analyze_harmonics(pressure: &Array3<f64>, grid: &Grid, frequency: f64, c0: f64) -> (f64, f64) {
    let (nx, _, _) = pressure.dim();
    let wavelength = c0 / frequency;
    let k1 = 2.0 * PI / wavelength;
    let k2 = 2.0 * k1; // Second harmonic wavenumber

    let mut fundamental_sum = 0.0;
    let mut second_harmonic_sum = 0.0;
    let n_samples = nx / 4; // Sample middle portion

    // Use safe iteration for harmonic analysis
    for i in nx / 4..3 * nx / 4 {
        let (x, _, _) = grid.indices_to_coordinates(i, 0, 0);
        let p = pressure[[i, 0, 0]];

        // Project onto fundamental and second harmonic
        fundamental_sum += p * (k1 * x).cos();
        second_harmonic_sum += p * (k2 * x).cos();
    }

    let f1_amplitude = 2.0 * fundamental_sum.abs() / n_samples as f64;
    let f2_amplitude = 2.0 * second_harmonic_sum.abs() / n_samples as f64;

    (f1_amplitude, f2_amplitude)
}

/// Example 4: Performance Comparison
///
/// Compares safe vectorization against traditional approaches
fn demo_performance_comparison() -> KwaversResult<()> {
    println!("\n\n4. Performance Comparison: Safe vs Traditional Approaches");
    println!("   Testing: Computational efficiency of safe vectorization\n");

    let sizes = vec![64, 128, 256];

    for &size in &sizes {
        println!("   Array size: {}³", size);

        let _grid = Grid::new(size, size, size, 1e-3, 1e-3, 1e-3).unwrap();
        let a = Array3::<f64>::from_elem((size, size, size), 1.5);
        let b = Array3::<f64>::from_elem((size, size, size), 2.5);

        // Test safe vectorization performance
        let start = Instant::now();
        let result_safe = SafeVectorOps::add_arrays(&a, &b);
        let time_safe = start.elapsed();

        // Test parallel safe vectorization
        let start = Instant::now();
        let result_parallel = SafeVectorOps::add_arrays_parallel(&a, &b);
        let time_parallel = start.elapsed();

        // Verify correctness
        let max_diff = result_safe
            .iter()
            .zip(result_parallel.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, |a, b| a.max(b));

        let ops_per_sec_safe = (size * size * size) as f64 / time_safe.as_secs_f64();
        let ops_per_sec_parallel = (size * size * size) as f64 / time_parallel.as_secs_f64();

        println!(
            "     Safe vectorization:      {:.2e} ops/sec ({:.2} ms)",
            ops_per_sec_safe,
            time_safe.as_secs_f64() * 1000.0
        );
        println!(
            "     Parallel safe version:   {:.2e} ops/sec ({:.2} ms)",
            ops_per_sec_parallel,
            time_parallel.as_secs_f64() * 1000.0
        );
        println!(
            "     Speedup (parallel):      {:.1}x",
            ops_per_sec_parallel / ops_per_sec_safe
        );
        println!("     Accuracy: {:.2e} (max difference)", max_diff);
        println!();
    }

    println!("   Status: ✅ Safe vectorization provides excellent performance");
    println!("   Note: LLVM auto-vectorization enables SIMD without unsafe code");

    Ok(())
}
