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

    /// Enforce proper nesting (2:1 balance constraint)
    ///
    /// Ensures that neighboring cells differ by at most one refinement level,
    /// which is essential for:
    /// - Smooth interpolation at level boundaries
    /// - Stable numerical schemes
    /// - Efficient data structures
    ///
    /// Algorithm (iterative relaxation):
    /// 1. For each cell marked for coarsening (-1):
    ///    - Check all 26 neighbors (6 face + 12 edge + 8 corner)
    ///    - If any neighbor is marked for refinement (+1), cancel coarsening (0)
    /// 2. For each cell marked for refinement (+1):
    ///    - Check face neighbors
    ///    - If any face neighbor is marked for coarsening, mark it no-change (0)
    /// 3. Repeat until no changes (fixed point)
    ///
    /// References:
    /// - Berger & Rigoutsos (1991): "An algorithm for point clustering and grid generation"
    /// - Khokhlov (1998): "Fully threaded tree algorithms for adaptive refinement"
    /// - Burstedde et al. (2011): "p4est: Scalable algorithms for parallel AMR"
    fn enforce_nesting(&self, markers: &mut Array3<i8>) -> KwaversResult<()> {
        let (nx, ny, nz) = markers.dim();
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        loop {
            let old_markers = markers.clone();
            let mut changed = false;
            iteration += 1;

            if iteration > MAX_ITERATIONS {
                log::warn!("AMR nesting enforcement exceeded max iterations");
                break;
            }

            // Pass 1: Handle coarsening constraints
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        if old_markers[[i, j, k]] != -1 {
                            continue; // Only process cells marked for coarsening
                        }

                        // Check all 26 neighbors (6 faces + 12 edges + 8 corners)
                        let mut has_refined_neighbor = false;

                        for di in -1..=1 {
                            for dj in -1..=1 {
                                for dk in -1..=1 {
                                    if di == 0 && dj == 0 && dk == 0 {
                                        continue; // Skip self
                                    }

                                    let ni = (i as i32 + di) as usize;
                                    let nj = (j as i32 + dj) as usize;
                                    let nk = (k as i32 + dk) as usize;

                                    if ni < nx
                                        && nj < ny
                                        && nk < nz
                                        && old_markers[[ni, nj, nk]] == 1
                                    {
                                        has_refined_neighbor = true;
                                        break;
                                    }
                                }
                                if has_refined_neighbor {
                                    break;
                                }
                            }
                            if has_refined_neighbor {
                                break;
                            }
                        }

                        // If has refined neighbor, cancel coarsening
                        if has_refined_neighbor {
                            markers[[i, j, k]] = 0;
                            changed = true;
                        }
                    }
                }
            }

            // Pass 2: Handle refinement constraints
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        if old_markers[[i, j, k]] != 1 {
                            continue; // Only process cells marked for refinement
                        }

                        // Check face neighbors (6 directions)
                        let face_neighbors = [
                            (i.wrapping_sub(1), j, k),
                            (i + 1, j, k),
                            (i, j.wrapping_sub(1), k),
                            (i, j + 1, k),
                            (i, j, k.wrapping_sub(1)),
                            (i, j, k + 1),
                        ];

                        for &(ni, nj, nk) in &face_neighbors {
                            if ni < nx && nj < ny && nk < nz {
                                // If face neighbor wants to coarsen, prevent it
                                if old_markers[[ni, nj, nk]] == -1 {
                                    markers[[ni, nj, nk]] = 0;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }

            // Check convergence
            if !changed {
                log::debug!("AMR nesting converged in {} iterations", iteration);
                break;
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
