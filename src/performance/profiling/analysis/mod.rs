//! Performance analysis tools
//!
//! Provides roofline analysis and performance bounds estimation.

use crate::grid::Grid;

/// Performance bound type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceBound {
    /// Memory bandwidth bound
    MemoryBound,
    /// Compute bound
    ComputeBound,
    /// Cache bound
    CacheBound,
    /// Balanced
    Balanced,
}

/// Roofline analysis for performance modeling
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    /// Peak computational performance (GFLOPS)
    pub peak_performance: f64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth: f64,
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f64,
    /// Achieved performance (GFLOPS)
    pub achieved_performance: f64,
    /// Performance bound
    pub bound: PerformanceBound,
}

impl RooflineAnalysis {
    /// Create a new roofline analysis
    #[must_use]
    pub fn new(peak_performance: f64, peak_bandwidth: f64) -> Self {
        Self {
            peak_performance,
            peak_bandwidth,
            arithmetic_intensity: 0.0,
            achieved_performance: 0.0,
            bound: PerformanceBound::Balanced,
        }
    }

    /// Analyze performance for given workload
    pub fn analyze(&mut self, flops: f64, bytes: f64, time_seconds: f64) {
        if bytes > 0.0 && time_seconds > 0.0 {
            self.arithmetic_intensity = flops / bytes;
            self.achieved_performance = (flops / 1e9) / time_seconds;

            // Roofline model: Performance = min(Peak_Performance, Peak_Bandwidth * AI)
            let bandwidth_limit = self.peak_bandwidth * self.arithmetic_intensity;
            let _theoretical_performance = self.peak_performance.min(bandwidth_limit);

            // Determine performance bound
            self.bound = if bandwidth_limit < self.peak_performance {
                PerformanceBound::MemoryBound
            } else {
                PerformanceBound::ComputeBound
            };
        }
    }

    /// Get efficiency (achieved / theoretical)
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        let theoretical = self.theoretical_performance();
        if theoretical > 0.0 {
            self.achieved_performance / theoretical
        } else {
            0.0
        }
    }

    /// Get theoretical performance based on roofline model
    #[must_use]
    pub fn theoretical_performance(&self) -> f64 {
        let bandwidth_limit = self.peak_bandwidth * self.arithmetic_intensity;
        self.peak_performance.min(bandwidth_limit)
    }

    /// Estimate ridge point (transition from memory to compute bound)
    #[must_use]
    pub fn ridge_point(&self) -> f64 {
        self.peak_performance / self.peak_bandwidth
    }
}

/// Estimate arithmetic intensity for FDTD operations
pub fn estimate_fdtd_intensity(grid: &Grid, spatial_order: usize) -> f64 {
    // FDTD operations per grid point
    let stencil_points = match spatial_order {
        2 => 7,  // Central point + 6 neighbors
        4 => 13, // Central point + 12 neighbors
        6 => 19, // Central point + 18 neighbors
        _ => 7,
    };

    // Account for boundary conditions reducing operations
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let total_points = (nx * ny * nz) as f64;
    let boundary_points = (2 * (nx * ny + nx * nz + ny * nz)) as f64;
    let interior_ratio = (total_points - boundary_points) / total_points;

    // Operations: multiply-add for each stencil point
    let flops_per_point = f64::from(stencil_points) * 2.0 * interior_ratio;

    // Memory: read stencil points + write result
    let bytes_per_point = f64::from(stencil_points + 1) * 8.0; // f64 = 8 bytes

    flops_per_point / bytes_per_point
}

/// Estimate arithmetic intensity for spectral operations
pub fn estimate_spectral_intensity(grid: &Grid) -> f64 {
    let n = (grid.nx * grid.ny * grid.nz) as f64;

    // FFT operations: ~5n log2(n) for 3D FFT
    let flops = 5.0 * n * n.log2();

    // Memory: read and write full grid
    let bytes = 2.0 * n * 8.0; // f64 = 8 bytes

    flops / bytes
}

/// Performance analyzer for comprehensive analysis
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    roofline: RooflineAnalysis,
    grid: Grid,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(grid: Grid) -> Self {
        // Typical values for modern CPUs
        let peak_performance = 100.0; // 100 GFLOPS
        let peak_bandwidth = 50.0; // 50 GB/s

        Self {
            roofline: RooflineAnalysis::new(peak_performance, peak_bandwidth),
            grid,
        }
    }

    /// Analyze FDTD performance
    pub fn analyze_fdtd(&mut self, time_steps: usize, elapsed: f64, spatial_order: usize) {
        let points = (self.grid.nx * self.grid.ny * self.grid.nz * time_steps) as f64;

        // Estimate FLOPS and memory
        let intensity = estimate_fdtd_intensity(&self.grid, spatial_order);
        let flops = points * intensity * 8.0; // Approximate FLOPS
        let bytes = points * 8.0; // Memory traffic

        self.roofline.analyze(flops, bytes, elapsed);
    }

    /// Get performance report
    pub fn report(&self) -> String {
        format!(
            "Performance Analysis:\n\
             Arithmetic Intensity: {:.2} FLOPS/byte\n\
             Achieved Performance: {:.2} GFLOPS\n\
             Theoretical Performance: {:.2} GFLOPS\n\
             Efficiency: {:.1}%\n\
             Performance Bound: {:?}",
            self.roofline.arithmetic_intensity,
            self.roofline.achieved_performance,
            self.roofline.theoretical_performance(),
            self.roofline.efficiency() * 100.0,
            self.roofline.bound
        )
    }
}
