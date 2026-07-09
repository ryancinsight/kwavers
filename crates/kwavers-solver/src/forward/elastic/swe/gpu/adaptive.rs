//! `AdaptiveResolution` for multi-level GPU SWE.

use super::types::{AdaptiveSolution, AdaptiveSolutionStep};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use log::info;
use leto::Array3;

/// Adaptive resolution techniques for large volumes
#[derive(Debug)]
pub struct AdaptiveResolution {
    base_grid: Grid,
    pub(super) resolution_levels: Vec<ResolutionLevel>,
    _quality_thresholds: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct ResolutionLevel {
    pub(super) grid: Grid,
    pub(super) scale_factor: f64,
    pub(super) _quality_metric: f64,
}

impl AdaptiveResolution {
    /// Create adaptive resolution system
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn new(base_grid: &Grid, max_levels: usize) -> Self {
        let mut resolution_levels = Vec::new();

        for level in 0..max_levels {
            let scale_factor = 2.0_f64.powi(level as i32);
            let nx = (base_grid.nx as f64 / scale_factor) as usize;
            let ny = (base_grid.ny as f64 / scale_factor) as usize;
            let nz = (base_grid.nz as f64 / scale_factor) as usize;

            let grid = Grid::new(
                nx,
                ny,
                nz,
                base_grid.dx * scale_factor,
                base_grid.dy * scale_factor,
                base_grid.dz * scale_factor,
            )
            .unwrap();

            resolution_levels.push(ResolutionLevel {
                grid,
                scale_factor,
                _quality_metric: 0.0,
            });
        }

        let quality_thresholds = (0..max_levels)
            .map(|i| (i as f64).mul_add(-0.1, 0.9))
            .collect();

        Self {
            base_grid: base_grid.clone(),
            resolution_levels,
            _quality_thresholds: quality_thresholds,
        }
    }

    /// Adaptively solve 3D SWE with resolution levels
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn adaptive_solve(
        &self,
        initial_displacement: &Array3<f64>,
        quality_requirement: f64,
    ) -> KwaversResult<AdaptiveSolution> {
        let mut solutions = Vec::new();

        for (level, resolution_level) in self.resolution_levels.iter().enumerate() {
            info!(
                "Solving at resolution level {}: {}x{}x{}",
                level, resolution_level.grid.nx, resolution_level.grid.ny, resolution_level.grid.nz
            );

            let interpolated_displacement = self.interpolate_to_resolution(
                initial_displacement,
                &self.base_grid,
                &resolution_level.grid,
            )?;

            let solution_quality =
                self.simulate_solve_quality(&interpolated_displacement, resolution_level);

            solutions.push(AdaptiveSolutionStep {
                level,
                grid: resolution_level.grid.clone(),
                quality: solution_quality,
                computation_time: 0.1 * (4.0_f64).powi(level as i32),
            });

            if solution_quality >= quality_requirement {
                break;
            }
        }

        Ok(AdaptiveSolution {
            steps: solutions.clone(),
            final_quality: solutions.last().map_or(0.0, |s| s.quality),
            total_computation_time: solutions.iter().map(|s| s.computation_time).sum(),
        })
    }

    fn interpolate_to_resolution(
        &self,
        data: &Array3<f64>,
        source_grid: &Grid,
        target_grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros((target_grid.nx, target_grid.ny, target_grid.nz));

        for k in 0..target_grid.nz {
            for j in 0..target_grid.ny {
                for i in 0..target_grid.nx {
                    let x = i as f64 * target_grid.dx;
                    let y = j as f64 * target_grid.dy;
                    let z = k as f64 * target_grid.dz;

                    let fx = x / source_grid.dx;
                    let fy = y / source_grid.dy;
                    let fz = z / source_grid.dz;

                    let x0 = fx.floor() as usize;
                    let y0 = fy.floor() as usize;
                    let z0 = fz.floor() as usize;

                    let x1 = (x0 + 1).min(source_grid.nx - 1);
                    let y1 = (y0 + 1).min(source_grid.ny - 1);
                    let z1 = (z0 + 1).min(source_grid.nz - 1);

                    let wx = fx - x0 as f64;
                    let wy = fy - y0 as f64;
                    let wz = fz - z0 as f64;

                    let c000 = data[[x0, y0, z0]];
                    let c100 = data[[x1, y0, z0]];
                    let c010 = data[[x0, y1, z0]];
                    let c110 = data[[x1, y1, z0]];
                    let c001 = data[[x0, y0, z1]];
                    let c101 = data[[x1, y0, z1]];
                    let c011 = data[[x0, y1, z1]];
                    let c111 = data[[x1, y1, z1]];

                    let c00 = c000.mul_add(1.0 - wx, c100 * wx);
                    let c01 = c001.mul_add(1.0 - wx, c101 * wx);
                    let c10 = c010.mul_add(1.0 - wx, c110 * wx);
                    let c11 = c011.mul_add(1.0 - wx, c111 * wx);

                    let c0 = c00 * (1.0 - wy) + c10 * wy;
                    let c1 = c01 * (1.0 - wy) + c11 * wy;

                    result[[i, j, k]] = c0 * (1.0 - wz) + c1 * wz;
                }
            }
        }

        Ok(result)
    }

    fn simulate_solve_quality(&self, displacement: &Array3<f64>, level: &ResolutionLevel) -> f64 {
        let base_quality = 0.7;
        let resolution_bonus = 0.1 * level.scale_factor.log2().min(1.0);

        let signal_strength = displacement
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a + b.abs())
            / displacement.len() as f64;
        let signal_bonus = (signal_strength * 1000.0).min(0.1);

        (base_quality + resolution_bonus + signal_bonus).min(1.0)
    }
}
