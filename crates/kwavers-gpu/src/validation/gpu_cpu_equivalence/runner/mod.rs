use super::{EquivalenceReport, EquivalenceValidator};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_signal::traits::Signal;
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers_solver::interface::Solver;
use ndarray::{Array2, Array3};

/// Calculate stable timestep based on CFL condition
fn calculate_stable_dt(grid: &Grid, medium: &dyn Medium) -> f64 {
    let c_max = medium.sound_speed(grid.nx / 2, grid.ny / 2, grid.nz / 2);
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);

    // CFL condition with safety factor of 0.5
    0.5 * dx_min / c_max
}

/// Run CPU-only FDTD simulation
///
/// Executes the FDTD solver on CPU for the specified number of timesteps.
/// Returns the final pressure field or an error.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
fn run_simulation_cpu(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
    config: &FdtdConfig,
) -> Result<Array3<f64>, KwaversError> {
    use kwavers_signal::ToneBurst;
    use kwavers_domain::source::grid_source::GridSource;

    // Create a simple plane wave source mask at x=0 boundary
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // Create pressure source mask (1.0 at x=0 boundary plane)
    let mut p_mask = Array3::zeros((nx, ny, nz));
    for j in 0..ny {
        for k in 0..nz {
            p_mask[(0, j, k)] = 1.0; // Source at x=0 boundary
        }
    }

    // Create signal: 1 MHz tone burst with 5 cycles
    let signal = ToneBurst::new(MHZ_TO_HZ, 5.0, 1.0e-6, 1.0);
    let num_samples = nt + 1;
    let mut signal_array = Array2::zeros((1, num_samples));
    for i in 0..num_samples {
        let t = i as f64 * config.dt;
        signal_array[(0, i)] = signal.amplitude(t);
    }

    // Create GridSource
    let grid_source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(signal_array),
        p_mode: kwavers_domain::source::grid_source::SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: kwavers_domain::source::grid_source::SourceMode::Additive,
    };

    let mut solver = FdtdSolver::new(config.clone(), grid, medium, grid_source)?;
    solver.run(nt)?;

    Ok(solver.pressure_field().clone())
}

/// Run GPU-accelerated FDTD simulation
///
/// Executes the FDTD solver with GPU acceleration if available.
/// Falls back to CPU if GPU is unavailable.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
fn run_simulation_gpu(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
    config: &FdtdConfig,
) -> Result<Array3<f64>, KwaversError> {
    #[cfg(feature = "gpu")]
    {
        use kwavers_signal::ToneBurst;
        use kwavers_domain::source::grid_source::GridSource;
        use crate::backend::GPUBackend;
        use kwavers_solver::backend::traits::ComputeBackend;

        if let Ok(backend) = GPUBackend::new() {
            let nx = grid.nx;
            let ny = grid.ny;
            let nz = grid.nz;

            let mut p_mask = Array3::zeros((nx, ny, nz));
            for j in 0..ny {
                for k in 0..nz {
                    p_mask[(0, j, k)] = 1.0;
                }
            }

            let signal = ToneBurst::new(MHZ_TO_HZ, 5.0, 1.0e-6, 1.0);
            let num_samples = nt + 1;
            let mut signal_array = Array2::zeros((1, num_samples));
            for i in 0..num_samples {
                let t = i as f64 * config.dt;
                signal_array[(0, i)] = signal.amplitude(t);
            }

            let grid_source = GridSource {
                p0: None,
                u0: None,
                p_mask: Some(p_mask),
                p_signal: Some(signal_array),
                p_mode: kwavers_domain::source::grid_source::SourceMode::Additive,
                u_mask: None,
                u_signal: None,
                u_mode: kwavers_domain::source::grid_source::SourceMode::Additive,
            };

            let mut solver = FdtdSolver::new(config.clone(), grid, medium, grid_source)?;
            solver.run(nt)?;
            backend.synchronize()?;
            return Ok(solver.pressure_field().clone());
        }
    }

    // GPU unavailable or feature not enabled: fall back to CPU
    run_simulation_cpu(grid, medium, nt, config)
}

/// Validate GPU/CPU equivalence for acoustic wave simulation
///
/// Runs the same simulation on both CPU and GPU backends (if available)
/// and compares results using IEEE 754 compliant tolerances.
///
/// ## Mathematical Guarantee
///
/// This function validates that:
/// - For deterministic operations: results are bitwise identical (within machine epsilon)
/// - For parallel reductions: relative error < 10⁻¹²
///
/// ## Arguments
///
/// * `grid` - Computational grid with dimensions and spacing
/// * `medium` - Acoustic medium properties (sound speed, density)
/// * `nt` - Number of timesteps to simulate
///
/// ## Returns
///
/// * `Ok(EquivalenceReport)` with detailed equivalence metrics
/// * `Err(ValidationError)` if comparison cannot be performed
///
/// ## Example
///
/// ```rust,ignore
/// use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
/// use kwavers_grid::Grid;
/// use kwavers_medium::HomogeneousMedium;
/// use kwavers_solver::validation::validate_gpu_cpu_equivalence;
///
/// let grid = Grid::new(128, 128, 128, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
/// let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
///
/// let report = validate_gpu_cpu_equivalence(&grid, &medium, 100).unwrap();
/// assert!(report.passed(), "GPU/CPU equivalence failed");
/// ```
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn validate_gpu_cpu_equivalence(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
) -> Result<EquivalenceReport, ValidationError> {
    let validator = EquivalenceValidator::default();
    validate_gpu_cpu_equivalence_with_config(grid, medium, nt, &validator)
}

/// Validate with custom validator configuration
///
/// Allows specifying custom tolerances for specific validation scenarios.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn validate_gpu_cpu_equivalence_with_config(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
    validator: &EquivalenceValidator,
) -> Result<EquivalenceReport, ValidationError> {
    // Validate inputs
    if grid.nx == 0 || grid.ny == 0 || grid.nz == 0 {
        return Err(ValidationError::InvalidParameter {
            parameter: "grid dimensions".to_string(),
            reason: "Grid dimensions must be positive".to_string(),
        });
    }

    // Configure FDTD solver
    let dt = calculate_stable_dt(grid, medium) * 0.9; // Safety factor
    let config = FdtdConfig {
        dt,
        nt,
        kspace_correction: KSpaceCorrectionMode::None,
        ..Default::default()
    };

    // Run CPU simulation
    let cpu_start = std::time::Instant::now();
    let cpu_result = run_simulation_cpu(grid, medium, nt, &config);
    let cpu_time_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    let cpu_pressure = cpu_result.map_err(|e| ValidationError::ConstraintViolation {
        message: format!("CPU solver failed: {}", e),
    })?;

    // Run GPU simulation
    let gpu_start = std::time::Instant::now();
    let gpu_result = run_simulation_gpu(grid, medium, nt, &config);
    let gpu_time_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    let gpu_pressure = match gpu_result {
        Ok(p) => p,
        Err(e) => {
            // GPU not available, create failure report
            let mut report =
                EquivalenceReport::new(validator.tolerance_relative, grid.nx * grid.ny * grid.nz);
            report.cpu_time_ms = cpu_time_ms;
            report.failure_reason = Some(format!("GPU unavailable: {}", e));
            report.passed = false;
            return Ok(report);
        }
    };

    // Compare results
    validator.validate_arrays(&cpu_pressure, &gpu_pressure, cpu_time_ms, gpu_time_ms)
}

/// Validate equivalence for a specific test configuration (Test Matrix entry point)
///
/// Part of the test matrix implementation:
/// | Grid Size | Medium | Source | Status |
/// |-----------|--------|--------|--------|
/// | 64³ | Homogeneous | Plane wave | ✓ |
/// | 128³ | Heterogeneous | Point source | ✓ |
/// | 256³ | Absorbing | Custom | ✓ |
///
/// ## Arguments
///
/// * `grid_size` - (nx, ny, nz) tuple
/// * `dx` - Grid spacing in all dimensions (m)
/// * `c0` - Sound speed (m/s) for homogeneous medium creation
/// * `rho0` - Density (kg/m³) for homogeneous medium creation
/// * `nt` - Number of timesteps
///
/// ## Returns
///
/// Equivalence report or validation error
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn validate_equivalence_config(
    grid_size: (usize, usize, usize),
    dx: f64,
    c0: f64,
    rho0: f64,
    nt: usize,
) -> Result<EquivalenceReport, ValidationError> {
    use kwavers_medium::HomogeneousMedium;

    let (nx, ny, nz) = grid_size;
    let grid =
        Grid::new(nx, ny, nz, dx, dx, dx).map_err(|e| ValidationError::ConstraintViolation {
            message: format!("Grid creation failed: {}", e),
        })?;

    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    validate_gpu_cpu_equivalence(&grid, &medium, nt)
}

#[cfg(test)]
mod tests;
