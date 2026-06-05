//! Performance analysis tools
//!
//! Provides roofline analysis and performance bounds estimation.

use kwavers_core::constants::numerical::BYTES_PER_F64;
use kwavers_grid::Grid;

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
    let bytes_per_point = f64::from(stencil_points + 1) * BYTES_PER_F64;

    flops_per_point / bytes_per_point
}

/// Estimate arithmetic intensity for spectral operations
pub fn estimate_spectral_intensity(grid: &Grid) -> f64 {
    let n = (grid.nx * grid.ny * grid.nz) as f64;

    // FFT operations: ~5n log2(n) for 3D FFT
    let flops = 5.0 * n * n.log2();

    // Memory: read and write full grid
    let bytes = 2.0 * n * BYTES_PER_F64;

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
        let flops = points * intensity * BYTES_PER_F64; // Approximate FLOPS
        let bytes = points * BYTES_PER_F64; // Memory traffic

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

#[cfg(test)]
mod tests {
    use super::*;

    // ─── RooflineAnalysis exact formula tests ────────────────────────────────

    /// `ridge_point()` = peak_performance / peak_bandwidth.
    ///
    /// peak_performance=100.0 GFLOPS, peak_bandwidth=50.0 GB/s
    /// → ridge_point = 100/50 = 2.0 FLOPS/byte.
    #[test]
    fn roofline_ridge_point_exact() {
        let analysis = RooflineAnalysis::new(100.0, 50.0);
        let rp = analysis.ridge_point();
        assert!(
            (rp - 2.0).abs() < 1e-14,
            "ridge_point = {rp} (expected 2.0 = 100/50)"
        );
    }

    /// `theoretical_performance()` is memory-bound when AI < ridge_point.
    ///
    /// peak=100 GFLOPS, bw=50 GB/s, AI=1.0 FLOPS/byte.
    /// bw·AI = 50 < peak=100 → theoretical = 50.0 GFLOPS.
    #[test]
    fn roofline_theoretical_performance_memory_bound_exact() {
        let mut analysis = RooflineAnalysis::new(100.0, 50.0);
        analysis.arithmetic_intensity = 1.0;
        let tp = analysis.theoretical_performance();
        assert!(
            (tp - 50.0).abs() < 1e-14,
            "memory-bound theoretical = {tp} (expected 50.0)"
        );
        assert_eq!(
            analysis.bound,
            PerformanceBound::Balanced,
            "bound before analyze() remains Balanced"
        );
    }

    /// `theoretical_performance()` is compute-bound when AI > ridge_point.
    ///
    /// peak=100 GFLOPS, bw=50 GB/s, AI=3.0 FLOPS/byte.
    /// bw·AI = 150 > peak=100 → theoretical = 100.0 GFLOPS (clamped by peak).
    #[test]
    fn roofline_theoretical_performance_compute_bound_exact() {
        let mut analysis = RooflineAnalysis::new(100.0, 50.0);
        analysis.arithmetic_intensity = 3.0;
        let tp = analysis.theoretical_performance();
        assert!(
            (tp - 100.0).abs() < 1e-14,
            "compute-bound theoretical = {tp} (expected 100.0)"
        );
    }

    /// `efficiency()` = achieved_performance / theoretical_performance().
    ///
    /// achieved=50.0, AI=1.0, bw=50 → theoretical=50.0 → efficiency=1.0.
    #[test]
    fn roofline_efficiency_at_theoretical_limit_is_one() {
        let mut analysis = RooflineAnalysis::new(100.0, 50.0);
        analysis.arithmetic_intensity = 1.0;
        analysis.achieved_performance = 50.0;
        let eff = analysis.efficiency();
        assert!(
            (eff - 1.0).abs() < 1e-14,
            "efficiency at theoretical limit = {eff} (expected 1.0)"
        );
    }

    /// `efficiency()` returns 0.0 when theoretical_performance is 0.
    ///
    /// AI=0 → theoretical=min(100, 50·0)=0 → efficiency=0.0 (guard clause).
    #[test]
    fn roofline_efficiency_zero_theoretical_returns_zero() {
        let analysis = RooflineAnalysis::new(100.0, 50.0);
        // AI stays at 0.0 (default) → theoretical = 0.0
        let eff = analysis.efficiency();
        assert!(
            eff.abs() < 1e-14,
            "zero-theoretical efficiency = {eff} (expected 0.0)"
        );
    }

    /// `analyze()` sets MemoryBound when bandwidth_limit < peak_performance.
    ///
    /// flops=1e9, bytes=1e9, time=1.0 → AI=1.0, achieved=1.0 GFLOPS.
    /// bandwidth_limit = 50·1.0 = 50 < peak=100 → MemoryBound.
    #[test]
    fn roofline_analyze_sets_memory_bound_for_low_intensity() {
        let mut analysis = RooflineAnalysis::new(100.0, 50.0);
        analysis.analyze(1e9, 1e9, 1.0);
        assert_eq!(
            analysis.bound,
            PerformanceBound::MemoryBound,
            "AI=1 must yield MemoryBound (ridge=2.0)"
        );
        assert!(
            (analysis.arithmetic_intensity - 1.0).abs() < 1e-12,
            "AI = {} (expected 1.0)",
            analysis.arithmetic_intensity
        );
    }

    /// `analyze()` sets ComputeBound when bandwidth_limit ≥ peak_performance.
    ///
    /// AI = flops/bytes = 300e9/1e9 = 300 → bw·AI = 50·300 = 15000 > 100 → ComputeBound.
    #[test]
    fn roofline_analyze_sets_compute_bound_for_high_intensity() {
        let mut analysis = RooflineAnalysis::new(100.0, 50.0);
        analysis.analyze(300e9, 1e9, 1.0);
        assert_eq!(
            analysis.bound,
            PerformanceBound::ComputeBound,
            "AI=300 must yield ComputeBound"
        );
    }
}
