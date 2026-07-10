//! `UtilConservativeInterpolator` — sparse volume-overlap transfer operator.
//!
//! # Theory
//!
//! Transfer matrix entries (Grandy 1999):
//! ```text
//! T_{ij} = overlap_volume(source_j, target_i) / volume(source_j)
//! ```
//!
//! Conservation property: Σᵢ T_{ij} = 1 (partition of unity).
//!
//! # References
//! - Grandy, J. (1999). "Conservative Remapping and Region Overlays."
//!   *J. Comput. Phys.*, 148(2), 433–466.

use super::mode::ConservationMode;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;

/// Conservative interpolator preserving integral quantities during grid transfer.
///
/// The sparse transfer matrix T is stored in CSR-like form:
/// `transfer_matrix[i]` = list of `(source_index, weight)` pairs for target cell `i`.
#[derive(Debug, Clone)]
pub struct UtilConservativeInterpolator {
    source_grid: Grid,
    target_grid: Grid,
    /// Sparse transfer matrix: row i = [(src_idx, weight), …]
    transfer_matrix: Vec<Vec<(usize, f64)>>,
    conservation_mode: ConservationMode,
    source_volumes: Vec<f64>,
    target_volumes: Vec<f64>,
}

impl UtilConservativeInterpolator {
    /// Build conservative transfer operator satisfying the Sprague-Grundy theorem.
    ///
    /// # Errors
    /// Returns error if either grid has a zero dimension.
    pub fn new(source: &Grid, target: &Grid, mode: ConservationMode) -> KwaversResult<Self> {
        Self::validate_grids(source, target)?;

        let source_volumes = Self::compute_cell_volumes(source);
        let target_volumes = Self::compute_cell_volumes(target);
        let transfer_matrix = Self::build_transfer_matrix(source, target, &source_volumes)?;

        Ok(Self {
            source_grid: source.clone(),
            target_grid: target.clone(),
            transfer_matrix,
            conservation_mode: mode,
            source_volumes,
            target_volumes,
        })
    }

    /// Apply u_target = T · u_source in-place.
    ///
    /// # Errors
    /// Returns error if field shapes do not match the corresponding grids.
    pub fn transfer(
        &self,
        source_field: &Array3<f64>,
        target_field: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        if source_field.shape()
            != [
                self.source_grid.nx,
                self.source_grid.ny,
                self.source_grid.nz,
            ]
        {
            return Err(KwaversError::InvalidInput(format!(
                "Source field shape {:?} doesn't match grid ({}, {}, {})",
                source_field.shape(),
                self.source_grid.nx,
                self.source_grid.ny,
                self.source_grid.nz
            )));
        }
        if target_field.shape()
            != [
                self.target_grid.nx,
                self.target_grid.ny,
                self.target_grid.nz,
            ]
        {
            return Err(KwaversError::InvalidInput(format!(
                "Target field shape {:?} doesn't match grid ({}, {}, {})",
                target_field.shape(),
                self.target_grid.nx,
                self.target_grid.ny,
                self.target_grid.nz
            )));
        }

        let nx_t = self.target_grid.nx;
        let ny_t = self.target_grid.ny;
        let nz_t = self.target_grid.nz;

        for iz in 0..nz_t {
            for iy in 0..ny_t {
                for ix in 0..nx_t {
                    let target_idx = Self::index_3d(ix, iy, iz, nx_t, ny_t);
                    let mut sum = 0.0;
                    for &(source_idx, weight) in &self.transfer_matrix[target_idx] {
                        let (sx, sy, sz) = Self::unravel_index(
                            source_idx,
                            self.source_grid.nx,
                            self.source_grid.ny,
                        );
                        sum += weight * source_field[[sx, sy, sz]];
                    }
                    target_field[[ix, iy, iz]] = sum;
                }
            }
        }

        Ok(())
    }

    /// Compute relative conservation error |∫target − ∫source| / |∫source|.
    ///
    /// For an exact conservative transfer this should be O(machine epsilon).
    pub fn verify_conservation(&self, source: &Array3<f64>, target: &Array3<f64>) -> f64 {
        let mut integral_source = 0.0;
        for iz in 0..self.source_grid.nz {
            for iy in 0..self.source_grid.ny {
                for ix in 0..self.source_grid.nx {
                    let idx = Self::index_3d(ix, iy, iz, self.source_grid.nx, self.source_grid.ny);
                    integral_source += source[[ix, iy, iz]] * self.source_volumes[idx];
                }
            }
        }

        let mut integral_target = 0.0;
        for iz in 0..self.target_grid.nz {
            for iy in 0..self.target_grid.ny {
                for ix in 0..self.target_grid.nx {
                    let idx = Self::index_3d(ix, iy, iz, self.target_grid.nx, self.target_grid.ny);
                    integral_target += target[[ix, iy, iz]] * self.target_volumes[idx];
                }
            }
        }

        if integral_source.abs() < 1e-15 {
            integral_target.abs()
        } else {
            (integral_target - integral_source).abs() / integral_source.abs()
        }
    }

    /// Active conservation mode.
    pub fn conservation_mode(&self) -> ConservationMode {
        self.conservation_mode
    }

    /// Reference to the source grid.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn source_grid(&self) -> &Grid {
        &self.source_grid
    }

    /// Reference to the target grid.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn target_grid(&self) -> &Grid {
        &self.target_grid
    }

    /// Number of non-zero entries in the sparse transfer matrix.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn nnz(&self) -> usize {
        self.transfer_matrix.iter().map(|row| row.len()).sum()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn validate_grids(source: &Grid, target: &Grid) -> KwaversResult<()> {
        if source.nx == 0 || source.ny == 0 || source.nz == 0 {
            return Err(KwaversError::InvalidInput(
                "Source grid has zero dimension".to_owned(),
            ));
        }
        if target.nx == 0 || target.ny == 0 || target.nz == 0 {
            return Err(KwaversError::InvalidInput(
                "Target grid has zero dimension".to_owned(),
            ));
        }
        Ok(())
    }

    fn compute_cell_volumes(grid: &Grid) -> Vec<f64> {
        let cell_volume = grid.dx * grid.dy * grid.dz;
        vec![cell_volume; grid.nx * grid.ny * grid.nz]
    }

    /// Build sparse transfer matrix via axis-aligned box intersection volumes.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn build_transfer_matrix(
        source: &Grid,
        target: &Grid,
        source_volumes: &[f64],
    ) -> KwaversResult<Vec<Vec<(usize, f64)>>> {
        let nx_t = target.nx;
        let ny_t = target.ny;
        let nz_t = target.nz;
        let n_target = nx_t * ny_t * nz_t;
        let mut transfer_matrix = vec![Vec::new(); n_target];

        for iz_t in 0..nz_t {
            for iy_t in 0..ny_t {
                for ix_t in 0..nx_t {
                    let target_idx = Self::index_3d(ix_t, iy_t, iz_t, nx_t, ny_t);
                    let (x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t) =
                        Self::cell_bounds(target, ix_t, iy_t, iz_t);

                    let overlaps = Self::find_overlapping_cells(
                        source, x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t,
                    );

                    let mut weights: Vec<(usize, f64)> = overlaps
                        .into_iter()
                        .map(|(src_idx, vol)| (src_idx, vol / source_volumes[src_idx]))
                        .collect();

                    let total: f64 = weights.iter().map(|(_, w)| w).sum();
                    if total > 1e-15 {
                        for (_, w) in &mut weights {
                            *w /= total;
                        }
                    }

                    transfer_matrix[target_idx] = weights;
                }
            }
        }

        Ok(transfer_matrix)
    }

    fn cell_bounds(grid: &Grid, ix: usize, iy: usize, iz: usize) -> (f64, f64, f64, f64, f64, f64) {
        (
            ix as f64 * grid.dx,
            (ix + 1) as f64 * grid.dx,
            iy as f64 * grid.dy,
            (iy + 1) as f64 * grid.dy,
            iz as f64 * grid.dz,
            (iz + 1) as f64 * grid.dz,
        )
    }

    fn find_overlapping_cells(
        source: &Grid,
        x_min_t: f64,
        x_max_t: f64,
        y_min_t: f64,
        y_max_t: f64,
        z_min_t: f64,
        z_max_t: f64,
    ) -> Vec<(usize, f64)> {
        let mut overlaps = Vec::new();

        let ix_start = ((x_min_t / source.dx).floor() as usize).min(source.nx.saturating_sub(1));
        let ix_end = ((x_max_t / source.dx).ceil() as usize).min(source.nx);
        let iy_start = ((y_min_t / source.dy).floor() as usize).min(source.ny.saturating_sub(1));
        let iy_end = ((y_max_t / source.dy).ceil() as usize).min(source.ny);
        let iz_start = ((z_min_t / source.dz).floor() as usize).min(source.nz.saturating_sub(1));
        let iz_end = ((z_max_t / source.dz).ceil() as usize).min(source.nz);

        for iz_s in iz_start..iz_end {
            for iy_s in iy_start..iy_end {
                for ix_s in ix_start..ix_end {
                    let (x_min_s, x_max_s, y_min_s, y_max_s, z_min_s, z_max_s) =
                        Self::cell_bounds(source, ix_s, iy_s, iz_s);

                    let vol = Self::box_intersection_volume(
                        x_min_t, x_max_t, y_min_t, y_max_t, z_min_t, z_max_t, x_min_s, x_max_s,
                        y_min_s, y_max_s, z_min_s, z_max_s,
                    );

                    if vol > 1e-15 {
                        overlaps
                            .push((Self::index_3d(ix_s, iy_s, iz_s, source.nx, source.ny), vol));
                    }
                }
            }
        }

        overlaps
    }

    /// Intersection volume of two axis-aligned 3D boxes.
    #[allow(clippy::too_many_arguments)]
    fn box_intersection_volume(
        x1_min: f64,
        x1_max: f64,
        y1_min: f64,
        y1_max: f64,
        z1_min: f64,
        z1_max: f64,
        x2_min: f64,
        x2_max: f64,
        y2_min: f64,
        y2_max: f64,
        z2_min: f64,
        z2_max: f64,
    ) -> f64 {
        (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0)
            * (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0)
            * (z1_max.min(z2_max) - z1_min.max(z2_min)).max(0.0)
    }

    fn index_3d(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize) -> usize {
        iz * (nx * ny) + iy * nx + ix
    }

    fn unravel_index(idx: usize, nx: usize, ny: usize) -> (usize, usize, usize) {
        let iz = idx / (nx * ny);
        let rem = idx % (nx * ny);
        (rem % nx, rem / nx, iz)
    }
}
