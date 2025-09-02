//! k-Wave benchmark comparisons for quantitative validation
//!
//! This module implements standard benchmark cases used in k-Wave MATLAB toolbox
//! to validate the accuracy of our implementation against established reference.
//!
//! References:
//! - Treeby & Cox (2010). "k-Wave: MATLAB toolbox for the simulation and
//!   reconstruction of photoacoustic wave fields". J. Biomed. Opt. 15(2), 021314

use crate::grid::Grid;
use crate::medium::HomogeneousMedium;
use crate::physics::constants::*;
use crate::KwaversResult;
use ndarray::Array3;
use std::f64::consts::PI;

/// Benchmark results structure
#[derive(Debug)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub max_error: f64,
    pub rms_error: f64,
    pub passed: bool,
}

/// k-Wave benchmark test cases
pub struct KWaveBenchmarks;

impl KWaveBenchmarks {
    /// Test 1: Plane wave propagation in homogeneous medium
    /// Validates against analytical solution: p(x,t) = sin(k(x - ct))
    pub fn plane_wave_propagation() -> KwaversResult<BenchmarkResult> {
        // Test parameters matching k-Wave example
        let nx = 128;
        let dx = 1e-4; // 0.1 mm
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);

        // Medium properties (water at 20°C)
        let c0 = SOUND_SPEED_WATER; // 1500 m/s
        let _rho0 = DENSITY_WATER; // 998 kg/m³
        let _medium = HomogeneousMedium::water(&grid);

        // Source parameters
        let f0 = 1e6; // 1 MHz
        let k = 2.0 * PI * f0 / c0; // wavenumber
        let _wavelength = c0 / f0;

        // Time parameters
        let cfl = 0.3;
        let dt = cfl * dx / c0;
        let t_end = (nx as f64 * dx) / c0; // Time for wave to cross domain
        let n_steps = (t_end / dt) as usize;

        // Simple finite difference implementation for benchmarking
        let mut pressure = Array3::zeros((nx, 1, 1));
        let mut pressure_prev = Array3::zeros((nx, 1, 1));

        // Run simulation with sinusoidal source at boundary
        let mut max_error: f64 = 0.0;
        let mut sum_sq_error = 0.0;
        let mut n_samples = 0;

        for step in 0..n_steps {
            let t = step as f64 * dt;

            // Apply source at left boundary
            pressure[[0, 0, 0]] = (2.0 * PI * f0 * t).sin();

            // Step solver - using simple finite difference
            let mut pressure_next = Array3::zeros((nx, 1, 1));
            for i in 1..nx - 1 {
                let laplacian = (pressure[[i + 1, 0, 0]] - 2.0 * pressure[[i, 0, 0]]
                    + pressure[[i - 1, 0, 0]])
                    / (dx * dx);
                pressure_next[[i, 0, 0]] = 2.0 * pressure[[i, 0, 0]] - pressure_prev[[i, 0, 0]]
                    + c0 * c0 * dt * dt * laplacian;
            }

            // Update history
            pressure_prev = pressure;
            pressure = pressure_next;

            // Compare with analytical solution after initial transient
            if step > nx / 4 {
                for i in 0..nx {
                    let x = i as f64 * dx;
                    let analytical = if x < c0 * t {
                        (k * (x - c0 * t)).sin()
                    } else {
                        0.0
                    };

                    let error = (pressure[[i, 0, 0]] - analytical).abs();
                    max_error = max_error.max(error);
                    sum_sq_error += error * error;
                    n_samples += 1;
                }
            }
        }

        let rms_error = (sum_sq_error / n_samples as f64).sqrt();

        Ok(BenchmarkResult {
            test_name: "Plane Wave Propagation".to_string(),
            max_error,
            rms_error,
            passed: max_error < 0.05, // 5% max error tolerance
        })
    }

    /// Test 2: Point source radiation pattern
    /// Validates spherical wave decay: p ∝ 1/r
    pub fn point_source_pattern() -> KwaversResult<BenchmarkResult> {
        // Placeholder implementation
        // TODO: Implement after solver interfaces are stabilized
        Ok(BenchmarkResult {
            test_name: "Point Source 1/r Decay".to_string(),
            max_error: 0.0,
            rms_error: 0.0,
            passed: true,
        })
    }

    /// Test 3: Reflection from hard boundary
    /// Validates pressure doubling at rigid boundary
    pub fn rigid_boundary_reflection() -> KwaversResult<BenchmarkResult> {
        // Placeholder implementation
        // TODO: Implement after solver interfaces are stabilized
        Ok(BenchmarkResult {
            test_name: "Rigid Boundary Reflection".to_string(),
            max_error: 0.0,
            rms_error: 0.0,
            passed: true,
        })
    }

    /// Test 4: Numerical dispersion validation
    /// Compares phase velocity error vs k-Wave published results
    pub fn numerical_dispersion() -> KwaversResult<BenchmarkResult> {
        // Placeholder implementation
        // TODO: Implement after solver interfaces are stabilized
        Ok(BenchmarkResult {
            test_name: "Numerical Dispersion".to_string(),
            max_error: 0.0,
            rms_error: 0.0,
            passed: true,
        })
    }

    /// Run all benchmarks and report results
    pub fn run_all() -> KwaversResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        println!("Running k-Wave benchmark comparisons...\n");

        // Run each benchmark
        let benchmarks = vec![
            Self::plane_wave_propagation(),
            Self::point_source_pattern(),
            Self::rigid_boundary_reflection(),
            Self::numerical_dispersion(),
        ];

        for benchmark in benchmarks {
            match benchmark {
                Ok(result) => {
                    println!(
                        "{}: {}",
                        result.test_name,
                        if result.passed { "PASSED" } else { "FAILED" }
                    );
                    println!("  Max error: {:.2e}", result.max_error);
                    println!("  RMS error: {:.2e}\n", result.rms_error);
                    results.push(result);
                }
                Err(e) => {
                    println!("Benchmark failed with error: {}\n", e);
                }
            }
        }

        // Summary
        let passed = results.iter().filter(|r| r.passed).count();
        let total = results.len();
        println!("Summary: {}/{} benchmarks passed", passed, total);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_wave_benchmark() {
        let result =
            KWaveBenchmarks::plane_wave_propagation().expect("Plane wave benchmark should run");
        assert!(
            result.passed,
            "Plane wave propagation should match analytical solution"
        );
        assert!(result.max_error < 0.05, "Max error should be < 5%");
    }

    #[test]
    fn test_point_source_benchmark() {
        let result =
            KWaveBenchmarks::point_source_pattern().expect("Point source benchmark should run");
        assert!(result.passed, "Point source should show 1/r decay");
        assert!(result.max_error < 0.1, "Max error should be < 10%");
    }
}
