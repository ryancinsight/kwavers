//! PSTD Heterogeneous Medium Diagnostic Test
//!
//! Replicates the Python `test_heterogeneous_medium_amplitude` in pure Rust to
//! isolate whether the amplitude=~0 failure is in the Rust core or Python binding.
//!
//! Grid: 32×32×32, dx=2mm, two-layer medium: water (x<16) | tissue (x≥16)
//! Source: GridSource point at (8, 16, 16), Hann-windowed sine burst (3 cycles, 0.5 MHz)
//! Sensor: (23, 16, 16), pml_size=6, 52 steps
//!
//! Expected: sensor amplitude >> 3.7e-10 Pa (the ~0 failure value seen in Python)

use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::heterogeneous::HeterogeneousMedium;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_source::grid_source::SourceMode;
use kwavers_source::GridSource;
use leto::{Array2, Array3};

fn build_tone_burst(nt: usize, dt: f64, f0: f64, n_cycles: usize) -> Array2<f64> {
    let mut sig = Array2::<f64>::zeros((1, nt));
    let burst_duration = n_cycles as f64 / f0;
    for t_idx in 0..nt {
        let t = t_idx as f64 * dt;
        if t < burst_duration {
            let win = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * t / burst_duration).cos());
            let s = win * (2.0 * std::f64::consts::PI * f0 * t).sin();
            sig[[0, t_idx]] = s;
        }
    }
    sig
}

fn run_pstd_two_layer(use_heterogeneous: bool) -> KwaversResult<(f64, f64)> {
    let nx = 32usize;
    let ny = 32usize;
    let nz = 32usize;
    let dx = 2e-3_f64;
    let pml_size = 6usize;
    let f0 = 0.5e6_f64;
    let n_cycles = 3usize;
    let c_water = 1500.0_f64;
    let rho_water = 1000.0_f64;
    let c_tissue = 1550.0_f64;
    let rho_tissue = 1050.0_f64;
    let c_max = if use_heterogeneous { c_tissue } else { c_water };

    let dt = 0.3 * dx / c_max;
    let t_end = 20e-6_f64;
    let nt = (t_end / dt).ceil() as usize;

    let src_ix = pml_size + 2; // = 8
    let sen_ix = nx - pml_size - 3; // = 23

    let signal = build_tone_burst(nt, dt, f0, n_cycles);

    let mut src_mask = Array3::<f64>::zeros((nx, ny, nz));
    src_mask[[src_ix, ny / 2, nz / 2]] = 1.0;

    let mut sen_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    sen_mask[[sen_ix, ny / 2, nz / 2]] = true;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;

    let grid_source = GridSource {
        p_mask: Some(src_mask),
        p_signal: Some(signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    };

    let cpml_config = CPMLConfig::with_thickness(pml_size);
    let config = PSTDConfig {
        dt,
        nt,
        sensor_mask: Some(sen_mask),
        boundary: BoundaryConfig::CPML(cpml_config),
        pml_inside: true,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..Default::default()
    };

    if use_heterogeneous {
        let mut c_arr = Array3::<f64>::from_elem((nx, ny, nz), c_water);
        let mut rho_arr = Array3::<f64>::from_elem((nx, ny, nz), rho_water);
        for i in nx / 2..nx {
            for j in 0..ny {
                for k in 0..nz {
                    c_arr[[i, j, k]] = c_tissue;
                    rho_arr[[i, j, k]] = rho_tissue;
                }
            }
        }
        let mut het = HeterogeneousMedium::new(nx, ny, nz, true);
        het.sound_speed = c_arr;
        het.density = rho_arr;

        let mut solver = PSTDSolver::new(config, grid, &het, grid_source)?;
        solver.run_orchestrated(nt)?;

        let peak_field = solver.fields.p.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let peak_sensor = solver
            .extract_pressure_data()
            .map(|d| d.iter().fold(0.0_f64, |m, &v| m.max(v.abs())))
            .unwrap_or(0.0);
        Ok((peak_sensor, peak_field))
    } else {
        let medium = HomogeneousMedium::new(rho_water, c_water, 0.0, 0.0, &grid);
        let mut solver = PSTDSolver::new(config, grid, &medium, grid_source)?;
        solver.run_orchestrated(nt)?;

        let peak_field = solver.fields.p.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let peak_sensor = solver
            .extract_pressure_data()
            .map(|d| d.iter().fold(0.0_f64, |m, &v| m.max(v.abs())))
            .unwrap_or(0.0);
        Ok((peak_sensor, peak_field))
    }
}

#[test]
fn test_pstd_homogeneous_two_layer_sensor() -> KwaversResult<()> {
    let (sensor_peak, field_peak) = run_pstd_two_layer(false)?;
    println!(
        "Homogeneous: sensor={:.4e} Pa, field_max={:.4e} Pa",
        sensor_peak, field_peak
    );
    // Field should have nonzero amplitude (wave was injected)
    assert!(
        field_peak > 1e-8,
        "Homogeneous field_max {:.4e} should be > 1e-8 Pa",
        field_peak
    );
    Ok(())
}

#[test]
fn test_pstd_heterogeneous_two_layer_sensor() -> KwaversResult<()> {
    let (sensor_peak, field_peak) = run_pstd_two_layer(true)?;
    println!(
        "Heterogeneous: sensor={:.4e} Pa, field_max={:.4e} Pa",
        sensor_peak, field_peak
    );
    // Field should have nonzero amplitude (wave was injected and propagated)
    assert!(
        field_peak > 1e-8,
        "Heterogeneous field_max {:.4e} should be > 1e-8 Pa; wave not propagating?",
        field_peak
    );
    // Sensor at x=23 should see some amplitude (wave travels ~30mm in ~20µs)
    assert!(
        sensor_peak > 1e-8,
        "Heterogeneous sensor {:.4e} should be > 1e-8 Pa; wave not reaching sensor?",
        sensor_peak
    );
    Ok(())
}
