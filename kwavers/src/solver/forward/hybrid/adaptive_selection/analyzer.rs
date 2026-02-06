// adaptive_selection/analyzer.rs - Field analysis for method selection

use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3};

/// Field analyzer for adaptive selection
#[derive(Debug)]
pub struct FieldAnalyzer {
    #[allow(dead_code)]
    grid: Grid,
}

impl FieldAnalyzer {
    /// Create new analyzer
    pub fn new(grid: Grid) -> Self {
        Self { grid }
    }

    /// Analyze field smoothness
    pub fn analyze_smoothness(&self, field: ArrayView3<f64>) -> f64 {
        let (nx, ny, nz) = field.dim();
        let mut total_variation = 0.0;
        let mut count = 0;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let center = field[[i, j, k]];

                    // Total variation in all directions
                    total_variation += (field[[i + 1, j, k]] - center).abs();
                    total_variation += (field[[i, j + 1, k]] - center).abs();
                    total_variation += (field[[i, j, k + 1]] - center).abs();
                    count += 3;
                }
            }
        }

        if count > 0 {
            total_variation / f64::from(count)
        } else {
            0.0
        }
    }

    /// Detect discontinuities
    pub fn detect_discontinuities(&self, field: ArrayView3<f64>, threshold: f64) -> Array3<bool> {
        let (nx, ny, nz) = field.dim();
        let mut discontinuities = Array3::from_elem((nx, ny, nz), false);

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let center = field[[i, j, k]];

                    // Check for large jumps
                    let max_jump = [
                        (field[[i + 1, j, k]] - center).abs(),
                        (field[[i - 1, j, k]] - center).abs(),
                        (field[[i, j + 1, k]] - center).abs(),
                        (field[[i, j - 1, k]] - center).abs(),
                        (field[[i, j, k + 1]] - center).abs(),
                        (field[[i, j, k - 1]] - center).abs(),
                    ]
                    .iter()
                    .copied()
                    .fold(0.0, f64::max);

                    if max_jump > threshold * center.abs() {
                        discontinuities[[i, j, k]] = true;
                    }
                }
            }
        }

        discontinuities
    }
}
