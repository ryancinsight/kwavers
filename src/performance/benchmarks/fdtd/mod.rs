//! FDTD solver benchmarks
//!
//! Performance benchmarks for the finite-difference time-domain solver.

use crate::medium::{core::ArrayAccess, HomogeneousMedium};
use crate::performance::benchmarks::result::BenchmarkResult;
use crate::solver::fdtd::{FdtdConfig, FdtdSolver};
use crate::{Grid, KwaversResult};
use ndarray::Array3;
use std::time::Instant;

/// Benchmark FDTD solver performance
pub fn benchmark_fdtd(
    grid_size: usize,
    time_steps: usize,
    iterations: usize,
) -> KwaversResult<BenchmarkResult> {
    let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.01, 0.1, &grid);

    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.3,
        subgridding: false,
        subgrid_factor: 1,
    };

    let mut solver = FdtdSolver::new(config, &grid)?;

    // Initialize fields
    let mut pressure = Array3::zeros((grid_size, grid_size, grid_size));
    let mut vx = Array3::zeros((grid_size, grid_size, grid_size));
    let mut vy = Array3::zeros((grid_size, grid_size, grid_size));
    let mut vz = Array3::zeros((grid_size, grid_size, grid_size));

    // Add initial perturbation
    let center = grid_size / 2;
    pressure[[center, center, center]] = 1.0;

    // Get medium arrays
    let density = medium.density_array();
    let sound_speed = medium.sound_speed_array();

    let dt = 0.5e-6; // Fixed time step for benchmarking

    let mut times = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..10 {
        solver.update_pressure(&mut pressure, &vx, &vy, &vz, &density, &sound_speed, dt)?;
    }

    // Benchmark iterations
    for _ in 0..iterations {
        // Reset fields
        pressure.fill(0.0);
        pressure[[center, center, center]] = 1.0;
        vx.fill(0.0);
        vy.fill(0.0);
        vz.fill(0.0);

        let start = Instant::now();

        for _ in 0..time_steps {
            solver.update_pressure(&mut pressure, &vx, &vy, &vz, &density, &sound_speed, dt)?;

            solver.update_velocity(&mut vx, &mut vy, &mut vz, &pressure, &density, dt)?;
        }

        times.push(start.elapsed());
    }

    let mut result = BenchmarkResult::new(
        format!("FDTD_{}x{}", grid_size, grid_size),
        grid_size,
        times,
    );

    // Add memory usage estimate
    let memory_per_array = grid_size * grid_size * grid_size * 8; // f64 = 8 bytes
    let total_memory = memory_per_array * 6; // pressure + 3 velocities + 2 medium arrays
    result.memory_usage = total_memory;

    // Add performance metrics
    let points_per_step = (grid_size * grid_size * grid_size) as f64;
    let total_points = points_per_step * time_steps as f64;
    result.add_metric("total_points", total_points);
    result.add_metric("time_steps", time_steps as f64);

    Ok(result)
}
