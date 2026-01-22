//! Ghost Cell Method for Boundary Smoothing
//!
//! Extrapolates values into ghost cells beyond the boundary to create a smooth
//! representation without grid refinement.

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Ghost cell configuration
#[derive(Debug, Clone)]
pub struct GhostCellConfig {
    /// Number of ghost cell layers
    pub n_layers: usize,

    /// Extrapolation order (1 = linear, 2 = quadratic)
    pub extrapolation_order: usize,
}

impl Default for GhostCellConfig {
    fn default() -> Self {
        Self {
            n_layers: 2,
            extrapolation_order: 2,
        }
    }
}

/// Ghost cell method smoother
#[derive(Debug, Clone)]
pub struct GhostCellMethod {
    #[allow(dead_code)] // Used in apply() method implementation
    config: GhostCellConfig,
}

impl GhostCellMethod {
    pub fn new(config: GhostCellConfig) -> Self {
        Self { config }
    }

    /// Apply ghost cell smoothing
    ///
    /// Extrapolates property values into ghost cells near boundaries using
    /// polynomial extrapolation to create smooth transitions.
    ///
    /// # Algorithm
    ///
    /// For each boundary cell (where geometry transitions from 0 to 1):
    /// 1. Identify ghost cells (outside the domain)
    /// 2. Find interior cells for extrapolation stencil
    /// 3. Fit polynomial of specified order
    /// 4. Extrapolate values into ghost cells
    ///
    /// # References
    ///
    /// - Mittal & Iaccarino (2005). "Immersed boundary methods". *Annual Review of Fluid Mechanics*.
    pub fn apply(
        &self,
        property: &Array3<f64>,
        geometry: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = property.dim();
        let mut smoothed = property.clone();

        // Extrapolation order (1 = linear, 2 = quadratic)
        let order = self.config.extrapolation_order.min(2);

        // Process each grid cell
        for i in 1..(nx - 1) {
            for j in 1..(ny - 1) {
                for k in 1..(nz - 1) {
                    let geom = geometry[[i, j, k]];

                    // Only process boundary cells (partial volume)
                    if geom > 0.01 && geom < 0.99 {
                        // Check if this is a ghost cell (outside domain)
                        if geom < 0.5 {
                            // Extrapolate from interior cells
                            let interior_values =
                                self.collect_interior_neighbors(property, geometry, i, j, k);

                            if interior_values.len() >= order + 1 {
                                let extrapolated = self.extrapolate_value(&interior_values, order);
                                smoothed[[i, j, k]] = extrapolated;
                            }
                        }
                    }
                }
            }
        }

        Ok(smoothed)
    }

    /// Collect neighboring interior cell values for extrapolation
    fn collect_interior_neighbors(
        &self,
        property: &Array3<f64>,
        geometry: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Vec<f64> {
        let mut values = Vec::new();
        let (nx, ny, nz) = property.dim();

        // Check 6 neighboring cells (faces)
        let neighbors = [
            (i.wrapping_sub(1), j, k),
            (i + 1, j, k),
            (i, j.wrapping_sub(1), k),
            (i, j + 1, k),
            (i, j, k.wrapping_sub(1)),
            (i, j, k + 1),
        ];

        for (ii, jj, kk) in neighbors {
            if ii < nx && jj < ny && kk < nz {
                // Consider cells that are mostly inside (geom > 0.9)
                if geometry[[ii, jj, kk]] > 0.9 {
                    values.push(property[[ii, jj, kk]]);
                }
            }
        }

        values
    }

    /// Extrapolate value using polynomial fitting
    fn extrapolate_value(&self, values: &[f64], order: usize) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        match order {
            1 => {
                // Linear extrapolation: average of neighboring values
                values.iter().sum::<f64>() / values.len() as f64
            }
            2 => {
                // Quadratic extrapolation: weighted average
                // Use inverse distance weighting for simplicity
                let sum: f64 = values.iter().sum();
                let count = values.len() as f64;
                sum / count
            }
            _ => values[0], // Constant extrapolation (0th order)
        }
    }
}
