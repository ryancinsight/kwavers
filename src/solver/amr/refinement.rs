//! Refinement management and level control

use crate::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Refinement level information
#[derive(Debug, Clone)]
pub struct RefinementLevel {
    /// Level index (0 = coarsest)
    pub level: usize,
    /// Grid spacing at this level
    pub dx: f64,
    /// Time step at this level
    pub dt: f64,
    /// Refinement ratio to next level
    pub ratio: usize,
}

impl RefinementLevel {
    /// Create a new refinement level
    #[must_use]
    pub fn new(level: usize, dx: f64, dt: f64, ratio: usize) -> Self {
        Self {
            level,
            dx,
            dt,
            ratio,
        }
    }

    /// Get refined spacing
    #[must_use]
    pub fn refined_dx(&self) -> f64 {
        self.dx / self.ratio as f64
    }

    /// Get refined time step
    #[must_use]
    pub fn refined_dt(&self) -> f64 {
        self.dt / self.ratio as f64
    }
}

/// Manages mesh refinement decisions
#[derive(Debug)]
pub struct RefinementManager {
    /// Maximum refinement level
    #[allow(dead_code)]
    max_level: usize,
    /// Refinement levels
    levels: Vec<RefinementLevel>,
    /// Buffer zone size for refinement
    buffer_size: usize,
}

impl RefinementManager {
    /// Create a new refinement manager
    #[must_use]
    pub fn new(max_level: usize) -> Self {
        let mut levels = Vec::with_capacity(max_level + 1);

        // Initialize levels with standard 2:1 refinement
        let base_dx = 1.0;
        let base_dt = 0.1;
        let ratio = 2;

        for level in 0..=max_level {
            let factor = f64::from((ratio as u32).pow(level as u32));
            levels.push(RefinementLevel::new(
                level,
                base_dx / factor,
                base_dt / factor,
                ratio,
            ));
        }

        Self {
            max_level,
            levels,
            buffer_size: 2,
        }
    }

    /// Mark cells for refinement/coarsening
    pub fn mark_cells(&self, error: &Array3<f64>, threshold: f64) -> KwaversResult<Array3<i8>> {
        let mut markers = Array3::zeros(error.dim());

        // Mark cells based on error threshold
        Zip::from(&mut markers).and(error).for_each(|m, &e| {
            if e > threshold {
                *m = 1; // Mark for refinement
            } else if e < threshold * 0.1 {
                *m = -1; // Mark for coarsening
            }
            // else 0 = no change
        });

        // Add buffer zones around refinement regions
        self.add_buffer_zones(&mut markers)?;

        // Ensure proper nesting (2:1 balance)
        self.enforce_nesting(&mut markers)?;

        Ok(markers)
    }

    /// Add buffer zones around refinement regions
    fn add_buffer_zones(&self, markers: &mut Array3<i8>) -> KwaversResult<()> {
        let (nx, ny, nz) = markers.dim();
        let mut buffer = markers.clone();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if markers[[i, j, k]] == 1 {
                        // Add buffer around refined cell
                        for di in -(self.buffer_size as isize)..=(self.buffer_size as isize) {
                            for dj in -(self.buffer_size as isize)..=(self.buffer_size as isize) {
                                for dk in -(self.buffer_size as isize)..=(self.buffer_size as isize)
                                {
                                    let ii = (i as isize + di) as usize;
                                    let jj = (j as isize + dj) as usize;
                                    let kk = (k as isize + dk) as usize;

                                    if ii < nx && jj < ny && kk < nz && buffer[[ii, jj, kk]] == 0 {
                                        buffer[[ii, jj, kk]] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        *markers = buffer;
        Ok(())
    }

    /// Enforce proper nesting (2:1 balance)
    fn enforce_nesting(&self, markers: &mut Array3<i8>) -> KwaversResult<()> {
        // Ensure no cell differs by more than one level from neighbors
        // This is a simplified version - full implementation would be more complex

        let (nx, ny, nz) = markers.dim();
        let mut changed = true;

        while changed {
            changed = false;
            let old_markers = markers.clone();

            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        // Check neighbors
                        let neighbors = [
                            old_markers[[i - 1, j, k]],
                            old_markers[[i + 1, j, k]],
                            old_markers[[i, j - 1, k]],
                            old_markers[[i, j + 1, k]],
                            old_markers[[i, j, k - 1]],
                            old_markers[[i, j, k + 1]],
                        ];

                        // If any neighbor is refined, this cell cannot coarsen
                        if neighbors.contains(&1) && markers[[i, j, k]] == -1 {
                            markers[[i, j, k]] = 0;
                            changed = true;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get refinement level info
    #[must_use]
    pub fn get_level(&self, level: usize) -> Option<&RefinementLevel> {
        self.levels.get(level)
    }
}
