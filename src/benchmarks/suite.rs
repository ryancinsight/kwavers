//! Comprehensive benchmark suite for performance testing
//!
//! This module provides automated benchmarking capabilities for all major
//! components of the Kwavers simulation framework.
//!
//! ## Design Principles
//! - **Automated**: Run all benchmarks with a single command
//! - **Reproducible**: Consistent results across runs
//! - **Comprehensive**: Cover all performance-critical paths
//! - **Zero-Copy**: Efficient benchmark implementations

use crate::{KwaversResult, Grid};
use crate::performance::profiling::{PerformanceProfiler, ProfileReport};
use crate::solver::pstd::PstdSolver;
use crate::solver::fdtd::FdtdSolver;
use crate::solver::amr::AMRManager;
use crate::physics::mechanics::acoustic_wave::KuznetsovWave;
use crate::medium::{HomogeneousMedium, Medium};
use ndarray::{Array3, s};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Grid sizes to test
    pub grid_sizes: Vec<usize>,
    /// Number of time steps
    pub time_steps: usize,
    /// Number of iterations for averaging
    pub iterations: usize,
    /// Enable GPU benchmarks
    pub enable_gpu: bool,
    /// Enable AMR benchmarks
    pub enable_amr: bool,
    /// Output format
    pub output_format: OutputFormat,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            grid_sizes: vec![64, 128, 256],
            time_steps: 100,
            iterations: 5,
            enable_gpu: true,
            enable_amr: true,
            output_format: OutputFormat::Console,
        }
    }
}

/// Output format for benchmark results
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Console output
    Console,
    /// CSV file
    Csv,
    /// JSON file
    Json,
    /// Markdown table
    Markdown,
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Grid size
    pub grid_size: usize,
    /// Total runtime
    pub runtime: Duration,
    /// Grid updates per second
    pub grid_updates_per_second: f64,
    /// Memory usage (MB)
    pub memory_mb: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Benchmark suite
pub struct BenchmarkSuite {
    /// Configuration
    config: BenchmarkConfig,
    /// Results storage
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all benchmarks
    pub fn run_all(&mut self) -> KwaversResult<BenchmarkReport> {
        println!("=== Kwavers Benchmark Suite ===\n");
        
        // Run benchmarks for each grid size
        for &grid_size in &self.config.grid_sizes {
            println!("Running benchmarks for {}³ grid...", grid_size);
            
            self.benchmark_pstd(grid_size)?;
            self.benchmark_fdtd(grid_size)?;
            self.benchmark_kuznetsov(grid_size)?;
            
            if self.config.enable_amr {
                self.benchmark_amr(grid_size)?;
            }
            
            if self.config.enable_gpu {
                self.benchmark_gpu(grid_size)?;
            }
            
            println!();
        }
        
        Ok(self.generate_report())
    }

    /// Benchmark PSTD solver
    fn benchmark_pstd(&mut self, grid_size: usize) -> KwaversResult<()> {
        let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
        let profiler = PerformanceProfiler::new(&grid);
        
        // Initialize
        let solver = PstdSolver::new(&grid)?;
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0);
        let mut fields = solver.create_fields();
        
        // Initial condition
        self.initialize_gaussian_pulse(&mut fields, &grid);
        
        // Warmup
        for _ in 0..10 {
            solver.step(&mut fields, &medium, 1e-6)?;
        }
        
        // Benchmark
        let start = Instant::now();
        let mut total_memory = 0usize;
        
        for _ in 0..self.config.iterations {
            let _scope = profiler.time_scope("pstd_step");
            
            for _ in 0..self.config.time_steps {
                solver.step(&mut fields, &medium, 1e-6)?;
            }
            
            // Estimate memory usage
            let field_memory = fields.len() * std::mem::size_of::<f64>();
            total_memory = total_memory.max(field_memory);
        }
        
        let runtime = start.elapsed() / self.config.iterations as u32;
        let total_points = grid_size * grid_size * grid_size;
        let grid_updates_per_second = (total_points * self.config.time_steps) as f64 / runtime.as_secs_f64();
        
        self.results.push(BenchmarkResult {
            name: "PSTD".to_string(),
            grid_size,
            runtime,
            grid_updates_per_second,
            memory_mb: total_memory as f64 / 1e6,
            metrics: HashMap::new(),
        });
        
        Ok(())
    }

    /// Benchmark FDTD solver
    fn benchmark_fdtd(&mut self, grid_size: usize) -> KwaversResult<()> {
        let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
        
        // Initialize
        let solver = FdtdSolver::new(&grid)?;
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0);
        let mut fields = solver.create_fields();
        
        // Initial condition
        self.initialize_gaussian_pulse(&mut fields, &grid);
        
        // Warmup
        for _ in 0..10 {
            solver.step(&mut fields, &medium, 1e-6)?;
        }
        
        // Benchmark
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            for _ in 0..self.config.time_steps {
                solver.step(&mut fields, &medium, 1e-6)?;
            }
        }
        
        let runtime = start.elapsed() / self.config.iterations as u32;
        let total_points = grid_size * grid_size * grid_size;
        let grid_updates_per_second = (total_points * self.config.time_steps) as f64 / runtime.as_secs_f64();
        
        self.results.push(BenchmarkResult {
            name: "FDTD".to_string(),
            grid_size,
            runtime,
            grid_updates_per_second,
            memory_mb: fields.len() as f64 * std::mem::size_of::<f64>() as f64 / 1e6,
            metrics: HashMap::new(),
        });
        
        Ok(())
    }

    /// Benchmark Kuznetsov solver
    fn benchmark_kuznetsov(&mut self, grid_size: usize) -> KwaversResult<()> {
        use crate::physics::mechanics::KuznetsovConfig;
        
        let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
        
        // Initialize
        let config = KuznetsovConfig {
            enable_nonlinearity: true,
            enable_diffusivity: true,
            ..Default::default()
        };
        let mut solver = KuznetsovWave::new(&grid, config);
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0);
        let mut pressure = grid.zeros_array();
        
        // Initial condition
        self.initialize_gaussian_field(&mut pressure, &grid);
        
        // Warmup
        for _ in 0..10 {
            solver.update(&pressure, &medium, 1e-6)?;
        }
        
        // Benchmark
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            for _ in 0..self.config.time_steps {
                solver.update(&pressure, &medium, 1e-6)?;
            }
        }
        
        let runtime = start.elapsed() / self.config.iterations as u32;
        let total_points = grid_size * grid_size * grid_size;
        let grid_updates_per_second = (total_points * self.config.time_steps) as f64 / runtime.as_secs_f64();
        
        self.results.push(BenchmarkResult {
            name: "Kuznetsov".to_string(),
            grid_size,
            runtime,
            grid_updates_per_second,
            memory_mb: pressure.len() as f64 * std::mem::size_of::<f64>() as f64 / 1e6,
            metrics: HashMap::new(),
        });
        
        Ok(())
    }

    /// Benchmark AMR
    fn benchmark_amr(&mut self, grid_size: usize) -> KwaversResult<()> {
        use crate::solver::amr::{AMRConfig, WaveletType, InterpolationScheme};
        
        let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
        
        // Configure AMR
        let amr_config = AMRConfig {
            max_level: 3,
            min_level: 0,
            refine_threshold: 0.1,
            coarsen_threshold: 0.01,
            refinement_ratio: 2,
            buffer_cells: 2,
            wavelet_type: WaveletType::Daubechies4,
            interpolation_scheme: InterpolationScheme::Conservative,
        };
        
        let mut amr_manager = AMRManager::new(amr_config, &grid)?;
        let mut field = grid.zeros_array();
        
        // Create sharp feature for refinement
        self.initialize_sharp_feature(&mut field, &grid);
        
        // Benchmark refinement
        let start = Instant::now();
        
        for _ in 0..self.config.iterations {
            amr_manager.adapt_mesh(&field)?;
        }
        
        let runtime = start.elapsed() / self.config.iterations as u32;
        
        // Get refinement statistics
        let stats = amr_manager.get_statistics();
        let compression_ratio = stats.total_cells as f64 / (grid_size * grid_size * grid_size) as f64;
        
        let mut metrics = HashMap::new();
        metrics.insert("compression_ratio".to_string(), compression_ratio);
        metrics.insert("refined_cells".to_string(), stats.refined_cells as f64);
        
        self.results.push(BenchmarkResult {
            name: "AMR".to_string(),
            grid_size,
            runtime,
            grid_updates_per_second: 0.0, // Not applicable for AMR
            memory_mb: stats.memory_usage_mb,
            metrics,
        });
        
        Ok(())
    }

    /// Benchmark GPU operations
    fn benchmark_gpu(&mut self, grid_size: usize) -> KwaversResult<()> {
        #[cfg(feature = "gpu-acceleration")]
        {
            use crate::gpu::{GpuContext, GpuKernel};
            
            let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
            
            // Initialize GPU
            let gpu_context = GpuContext::new().await?;
            let kernel = gpu_context.create_acoustic_kernel(&grid)?;
            
            let mut pressure = grid.zeros_array();
            let mut velocity = (grid.zeros_array(), grid.zeros_array(), grid.zeros_array());
            
            // Warmup
            for _ in 0..10 {
                kernel.execute(&mut pressure, &mut velocity, 1e-6)?;
            }
            
            // Benchmark
            let start = Instant::now();
            
            for _ in 0..self.config.iterations {
                for _ in 0..self.config.time_steps {
                    kernel.execute(&mut pressure, &mut velocity, 1e-6)?;
                }
            }
            
            let runtime = start.elapsed() / self.config.iterations as u32;
            let total_points = grid_size * grid_size * grid_size;
            let grid_updates_per_second = (total_points * self.config.time_steps) as f64 / runtime.as_secs_f64();
            
            self.results.push(BenchmarkResult {
                name: "GPU".to_string(),
                grid_size,
                runtime,
                grid_updates_per_second,
                memory_mb: 0.0, // GPU memory tracked separately
                metrics: HashMap::new(),
            });
        }
        
        Ok(())
    }

    /// Initialize Gaussian pulse in field array
    fn initialize_gaussian_pulse(&self, fields: &mut ndarray::Array4<f64>, grid: &Grid) {
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        let sigma = 0.01; // 10mm
        
        fields.slice_mut(s![0, .., .., ..]).indexed_iter_mut()
            .for_each(|((i, j, k), value)| {
                let x = (i as f64 - center.0 as f64) * grid.dx;
                let y = (j as f64 - center.1 as f64) * grid.dy;
                let z = (k as f64 - center.2 as f64) * grid.dz;
                let r2 = x*x + y*y + z*z;
                *value = (-r2 / (2.0 * sigma * sigma)).exp();
            });
    }

    /// Initialize Gaussian field
    fn initialize_gaussian_field(&self, field: &mut Array3<f64>, grid: &Grid) {
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        let sigma = 0.01; // 10mm
        
        field.indexed_iter_mut()
            .for_each(|((i, j, k), value)| {
                let x = (i as f64 - center.0 as f64) * grid.dx;
                let y = (j as f64 - center.1 as f64) * grid.dy;
                let z = (k as f64 - center.2 as f64) * grid.dz;
                let r2 = x*x + y*y + z*z;
                *value = (-r2 / (2.0 * sigma * sigma)).exp();
            });
    }

    /// Initialize sharp feature for AMR testing
    fn initialize_sharp_feature(&self, field: &mut Array3<f64>, grid: &Grid) {
        let center = grid.nx / 2;
        let width = 5;
        
        field.indexed_iter_mut()
            .for_each(|((i, j, k), value)| {
                if i.abs_diff(center) < width && j.abs_diff(center) < width && k.abs_diff(center) < width {
                    *value = 1.0;
                }
            });
    }

    /// Generate benchmark report
    fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport {
            results: self.results.clone(),
            config: self.config.clone(),
        }
    }
}

/// Benchmark report
#[derive(Debug)]
pub struct BenchmarkReport {
    /// Benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Configuration used
    pub config: BenchmarkConfig,
}

impl BenchmarkReport {
    /// Print report to console
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results ===\n");
        
        match self.config.output_format {
            OutputFormat::Console => self.print_console(),
            OutputFormat::Markdown => self.print_markdown(),
            _ => self.print_console(),
        }
    }

    /// Console output
    fn print_console(&self) {
        // Group by solver
        let mut by_solver: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            by_solver.entry(result.name.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        for (solver, results) in by_solver {
            println!("{}:", solver);
            for result in results {
                println!("  {}³ grid: {:.2}M updates/s, {:.2}ms runtime, {:.1}MB memory",
                        result.grid_size,
                        result.grid_updates_per_second / 1e6,
                        result.runtime.as_secs_f64() * 1000.0,
                        result.memory_mb
                );
                
                // Print additional metrics
                for (metric, value) in &result.metrics {
                    println!("    {}: {:.2}", metric, value);
                }
            }
            println!();
        }
    }

    /// Markdown table output
    fn print_markdown(&self) {
        println!("| Solver | Grid Size | Updates/s (M) | Runtime (ms) | Memory (MB) |");
        println!("|--------|-----------|---------------|--------------|-------------|");
        
        for result in &self.results {
            println!("| {} | {}³ | {:.2} | {:.2} | {:.1} |",
                    result.name,
                    result.grid_size,
                    result.grid_updates_per_second / 1e6,
                    result.runtime.as_secs_f64() * 1000.0,
                    result.memory_mb
            );
        }
    }

    /// Export to CSV
    pub fn export_csv(&self, path: &str) -> KwaversResult<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        writeln!(file, "Solver,GridSize,UpdatesPerSecond,RuntimeMs,MemoryMB")?;
        
        for result in &self.results {
            writeln!(file, "{},{},{:.2},{:.2},{:.1}",
                    result.name,
                    result.grid_size,
                    result.grid_updates_per_second,
                    result.runtime.as_secs_f64() * 1000.0,
                    result.memory_mb
            )?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig {
            grid_sizes: vec![32, 64],
            time_steps: 10,
            iterations: 2,
            enable_gpu: false,
            enable_amr: false,
            output_format: OutputFormat::Console,
        };
        
        let suite = BenchmarkSuite::new(config);
        assert_eq!(suite.config.grid_sizes.len(), 2);
    }
}