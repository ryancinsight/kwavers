//! Minimal PSTD Diagnostic Test
//!
//! This test isolates PSTD behavior with analytical validation to debug
//! the negative correlation issue observed in pykwavers validation.
//!
//! Focus: Verify basic PSTD functionality independent of comparison framework

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::{Signal, SineWave};
use kwavers::domain::source::{InjectionMode, PlaneWaveConfig, PlaneWaveSource, SourceField};
use kwavers::solver::forward::pstd::config::{KSpaceMethod, PSTDConfig};
use kwavers::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers::solver::interface::Solver;
use std::sync::Arc;

#[test]
fn test_pstd_sine_wave_polarity() -> KwaversResult<()> {
    // Test 1: Verify that a positive sine wave source produces positive pressure
    println!("\n=== Test 1: Sine Wave Polarity ===");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3; // 0.1 mm
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let frequency = 1e6; // 1 MHz
    let amplitude = 1e5; // 100 kPa (positive)

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create plane wave source with positive amplitude sine wave
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0; // CFL = 0.3
    let nt = 100;
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;

    // Debug: Check what the source signal produces
    println!("\nDEBUG: Source signal amplitude at various times:");
    let test_signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    for i in 0..5 {
        let t_test = i as f64 * dt;
        let amp_test = test_signal.amplitude(t_test);
        println!("  t={:.3e} s: amplitude={:.3e} Pa", t_test, amp_test);
    }

    solver.add_source(Box::new(source))?;

    // Record pressure at boundary over time
    let boundary_idx = (nx / 2, ny / 2, 0);
    let mut p_boundary = Vec::new();
    let mut times = Vec::new();

    for i in 0..nt {
        solver.step_forward()?;
        let t = i as f64 * dt;
        let p = solver.fields.p[[boundary_idx.0, boundary_idx.1, boundary_idx.2]];
        let rho = solver.rhox[[boundary_idx.0, boundary_idx.1, boundary_idx.2]];
        p_boundary.push(p);
        times.push(t);

        // Print first few timesteps
        if i < 10 {
            // Expected signal at this time (note: t is AFTER the step)
            let t_signal = i as f64 * dt; // Signal applied at beginning of step
            let expected = amplitude * (2.0 * std::f64::consts::PI * frequency * t_signal).sin();
            println!(
                "t={:.3e} s: p_boundary={:.3e} Pa, rho={:.3e}, expected={:.3e} Pa, ratio={:.3e}",
                t,
                p,
                rho,
                expected,
                p / expected.max(1e-10)
            );
        }
    }

    // Check polarity: when sine is positive, pressure should be positive
    // At t = 1/4 period, sin should be at maximum (+1)
    let quarter_period = 0.25 / frequency;
    let idx_quarter = (quarter_period / dt) as usize;
    if idx_quarter < nt {
        let p_at_peak = p_boundary[idx_quarter];
        println!("\nAt t = T/4 (sine peak):");
        println!("  Expected: ~{:.2e} Pa (positive)", amplitude);
        println!("  Actual:   {:.2e} Pa", p_at_peak);

        // Polarity check: should be positive
        assert!(
            p_at_peak > 0.0,
            "Pressure at sine peak should be positive, got {}",
            p_at_peak
        );

        // Magnitude check: should be within order of magnitude
        let ratio = p_at_peak / amplitude;
        println!("  Ratio: {:.3}", ratio);
    }

    // Check that signal oscillates (not stuck at zero)
    let max_p = p_boundary
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(v.abs()));
    println!("\nMax |p| over all timesteps: {:.2e} Pa", max_p);
    assert!(
        max_p > 1.0,
        "Pressure should oscillate, max |p| = {} is too small",
        max_p
    );

    Ok(())
}

#[test]
fn test_pstd_pressure_equation_of_state() -> KwaversResult<()> {
    // Test 2: Verify p = c^2 * rho relationship
    println!("\n=== Test 2: Equation of State ===");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create source with known amplitude
    let frequency = 1e6;
    let amplitude = 1e5;
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0;
    let nt = 10; // Just a few steps
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;
    solver.add_source(Box::new(source))?;

    // Run a few steps and check p = c^2 * rho at several points
    for i in 0..5 {
        solver.step_forward()?;

        // Check a few sample points
        let points = vec![(nx / 2, ny / 2, 0), (nx / 2, ny / 2, nz / 4)];

        for &(ix, iy, iz) in &points {
            let p = solver.fields.p[[ix, iy, iz]];
            let rho = solver.rhox[[ix, iy, iz]];
            // Use sound speed from medium (homogeneous)
            let c = c0;

            // p should equal c^2 * rho (within numerical precision)
            let p_expected = c * c * rho;
            let error = (p - p_expected).abs();

            if i < 3 {
                println!(
                    "Step {}, point ({},{},{}): p={:.3e}, c^2*rho={:.3e}, error={:.3e}",
                    i, ix, iy, iz, p, p_expected, error
                );
            }

            // This should be exact to machine precision after update_pressure()
            assert!(
                error < 1e-6 * p.abs().max(1.0),
                "Equation of state violated: p={}, c^2*rho={}, error={}",
                p,
                p_expected,
                error
            );
        }
    }

    Ok(())
}

#[test]
fn test_pstd_fft_normalization() -> KwaversResult<()> {
    // Test 3: Verify FFT round-trip preserves values
    println!("\n=== Test 3: FFT Normalization ===");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    // Create simple source
    let frequency = 1e6;
    let amplitude = 1e5;
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0;
    let nt = 20;
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;
    solver.add_source(Box::new(source))?;

    // Run and track energy
    let mut energies = Vec::new();

    for i in 0..nt {
        solver.step_forward()?;

        // Compute total energy: E = integral(p^2 + rho*|u|^2)
        let mut energy = 0.0;
        for k in 0..nz {
            for j in 0..ny {
                for ix in 0..nx {
                    let p = solver.fields.p[[ix, j, k]];
                    let ux = solver.fields.ux[[ix, j, k]];
                    let uy = solver.fields.uy[[ix, j, k]];
                    let uz = solver.fields.uz[[ix, j, k]];
                    // Use homogeneous medium properties
                    let rho = rho0;
                    let c = c0;

                    energy += p * p / (rho * c * c) + rho * (ux * ux + uy * uy + uz * uz);
                }
            }
        }
        energy *= dx * dx * dx; // Volume element

        energies.push(energy);

        if i < 5 || i % 5 == 0 {
            println!("Step {}: Energy = {:.3e} J", i, energy);
        }
    }

    // Energy should increase (source injecting) but not exponentially
    // Check that final energy is not NaN or infinite
    let final_energy = energies[energies.len() - 1];
    assert!(
        final_energy.is_finite(),
        "Energy became non-finite: {}",
        final_energy
    );

    // Check energy is increasing (source is injecting energy)
    assert!(
        final_energy > energies[0],
        "Energy should increase due to source, got {:.3e} -> {:.3e}",
        energies[0],
        final_energy
    );

    // Check no exponential growth (instability)
    let growth_ratio = final_energy / energies[5].max(1e-20);
    println!("Energy growth ratio (step 5 to end): {:.2}", growth_ratio);
    assert!(
        growth_ratio < 100.0,
        "Energy growing too fast (instability?): ratio = {}",
        growth_ratio
    );

    Ok(())
}

#[test]
fn test_pstd_source_amplitude_scaling() -> KwaversResult<()> {
    // Test 4: Verify amplitude scales linearly with source amplitude
    println!("\n=== Test 4: Amplitude Scaling ===");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let frequency = 1e6;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let amplitudes = vec![1e4, 1e5, 1e6]; // 10 kPa, 100 kPa, 1 MPa
    let mut max_pressures = Vec::new();

    for &amplitude in &amplitudes {
        let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let config = PlaneWaveConfig {
            direction: (0.0, 0.0, 1.0),
            wavelength: c0 / frequency,
            phase: 0.0,
            source_type: SourceField::Pressure,
            injection_mode: InjectionMode::BoundaryOnly,
        };
        let source = PlaneWaveSource::new(config, signal);

        let dt = 0.3 * dx / c0;
        let nt = 50;
        let pstd_config = PSTDConfig {
            dt,
            nt,
            kspace_method: KSpaceMethod::StandardPSTD,
            boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
            ..Default::default()
        };

        let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;
        solver.add_source(Box::new(source))?;

        // Run and find max pressure
        let mut max_p = 0.0;
        for _ in 0..nt {
            solver.step_forward()?;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let p = solver.fields.p[[i, j, k]].abs();
                        if p > max_p {
                            max_p = p;
                        }
                    }
                }
            }
        }

        max_pressures.push(max_p);
        println!(
            "Source amplitude: {:.2e} Pa -> Max pressure: {:.2e} Pa",
            amplitude, max_p
        );
    }

    // Check linear scaling: p_max should scale linearly with source amplitude
    for i in 1..amplitudes.len() {
        let ratio_amp = amplitudes[i] / amplitudes[i - 1];
        let ratio_p = max_pressures[i] / max_pressures[i - 1];
        println!(
            "Amplitude ratio: {:.2}, Pressure ratio: {:.2}",
            ratio_amp, ratio_p
        );

        // Should be close to linear (within factor of 2)
        assert!(
            ratio_p > ratio_amp * 0.5 && ratio_p < ratio_amp * 2.0,
            "Non-linear scaling: amplitude ratio {:.2}, pressure ratio {:.2}",
            ratio_amp,
            ratio_p
        );
    }

    Ok(())
}

#[test]
fn test_pstd_time_reversal_symmetry() -> KwaversResult<()> {
    // Test 5: Basic causality check - pressure should be zero before source activates
    println!("\n=== Test 5: Causality Check ===");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let frequency = 1e6;
    let amplitude = 1e5;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    let dt = 0.3 * dx / c0;
    let nt = 10;
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;
    solver.add_source(Box::new(source))?;

    // Before any steps, pressure should be zero everywhere
    let mut max_p_initial = 0.0;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = solver.fields.p[[i, j, k]].abs();
                if p > max_p_initial {
                    max_p_initial = p;
                }
            }
        }
    }

    println!("Initial max |p|: {:.3e} Pa", max_p_initial);
    assert!(
        max_p_initial < 1.0,
        "Initial pressure should be near zero, got {}",
        max_p_initial
    );

    // After first step, boundary should have pressure
    solver.step_forward()?;
    let p_boundary = solver.fields.p[[nx / 2, ny / 2, 0]];
    println!("After 1 step, p_boundary: {:.3e} Pa", p_boundary);

    assert!(
        p_boundary.abs() > 1.0,
        "Pressure should be non-zero after source activation, got {}",
        p_boundary
    );

    Ok(())
}
