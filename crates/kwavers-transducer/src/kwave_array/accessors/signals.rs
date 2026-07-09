use std::collections::HashMap;

use leto::{Array2, Array3};

use super::super::KWaveArray;

impl KWaveArray {
    /// Broadcast a single time signal to all elements, returning
    /// `[n_elements, n_times]`.
    #[must_use]
    pub fn get_distributed_source_signal(&self, signal: &leto::Array1<f64>) -> Array2<f64> {
        let n_elements = self.elements.len();
        let n_times = signal.len();
        let mut distributed = Array2::zeros([n_elements, n_times]);
        for i in 0..n_elements {
            for t in 0..n_times {
                distributed[[i, t]] = signal[t];
            }
        }
        distributed
    }

    /// Validate and return a per-element signal matrix `[n_elements, n_times]`.
    ///
    /// `per_element_signals` must have `n_elements` rows. Returns `Err` if the
    /// row count doesn't match.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_distributed_source_signal_per_element(
        &self,
        per_element_signals: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n_elements = self.elements.len();
        let rows = per_element_signals.shape()[0];
        if rows != n_elements {
            return Err(format!(
                "per_element_signals has {rows} rows but array has {n_elements} elements"
            ));
        }
        Ok(per_element_signals.clone())
    }

    /// Build a per-cell source `(mask, signal)` from per-element driving signals.
    ///
    /// Expands a `[n_elements, n_times]` matrix into a `[n_active_cells, n_times]`
    /// per-cell signal by accumulating `s_cell[c, t] = Σᵢ Wᵢ[c] · sᵢ[t]` over
    /// each element's BLI-weighted mask. Returns `(mask, per_cell_signal)` where
    /// `mask[c] = 1.0` at each active cell. Row order follows MATLAB/Fortran-order
    /// active-cell enumeration (`matlab_find(mask)` on the same logical mask).
    ///
    /// # Errors
    /// Returns `Err` if `per_element_signals.shape()[0] != n_elements`.
    pub fn build_per_element_source(
        &self,
        grid: &kwavers_grid::Grid,
        per_element_signals: &Array2<f64>,
    ) -> Result<(Array3<f64>, Array2<f64>), String> {
        let n_elements = self.elements.len();
        let shape = per_element_signals.shape();
        let (rows, n_times) = (shape[0], shape[1]);
        if rows != n_elements {
            return Err(format!(
                "per_element_signals has {rows} rows but array has {n_elements} elements"
            ));
        }

        let mut combined_weights: HashMap<(usize, usize, usize), f64> = HashMap::new();
        for element in &self.elements {
            self.rasterize_element_weighted_cells(element, grid, |i, j, k, weight| {
                *combined_weights.entry((i, j, k)).or_insert(0.0) += weight;
            });
        }

        let mut active_cells: Vec<(usize, usize, usize)> = combined_weights
            .iter()
            .filter_map(|(&cell, &weight)| (weight != 0.0).then_some(cell))
            .collect();
        active_cells.sort_by_key(|&(i, j, k)| (k, j, i));
        let n_active = active_cells.len();
        let active_index: HashMap<(usize, usize, usize), usize> = active_cells
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, cell)| (cell, idx))
            .collect();

        let mut per_cell_signal = Array2::<f64>::zeros([n_active, n_times]);
        for (element_idx, element) in self.elements.iter().enumerate() {
            self.rasterize_element_weighted_cells(element, grid, |i, j, k, weight| {
                if let Some(&row_idx) = active_index.get(&(i, j, k)) {
                    for t in 0..n_times {
                        per_cell_signal[[row_idx, t]] +=
                            weight * per_element_signals[[element_idx, t]];
                    }
                }
            });
        }

        let mut mask_ones = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz]);
        for &(i, j, k) in &active_cells {
            mask_ones[[i, j, k]] = 1.0;
        }

        Ok((mask_ones, per_cell_signal))
    }

    /// Visit each BLI-weighted grid-cell contribution for each array element.
    ///
    /// The callback receives `(element_idx, i, j, k, weight)` using the same
    /// finite-source rasterization path as [`Self::build_per_element_source`].
    /// This exposes the realized source support without materializing one dense
    /// mask per element.
    pub fn for_each_element_weighted_cell<F>(&self, grid: &kwavers_grid::Grid, mut visit: F)
    where
        F: FnMut(usize, usize, usize, usize, f64),
    {
        for (element_idx, element) in self.elements.iter().enumerate() {
            self.rasterize_element_weighted_cells(element, grid, |i, j, k, weight| {
                visit(element_idx, i, j, k, weight);
            });
        }
    }

    /// Enumerate active (nonzero) grid cells in Fortran/MATLAB column-major order
    /// (k outer, j middle, i inner), matching `matlab_find(mask)` on the same mask.
    #[cfg(test)]
    #[inline]
    pub(in crate::kwave_array) fn active_cells_fortran_order(
        mask: &Array3<f64>,
    ) -> Vec<(usize, usize, usize)> {
        let [nx, ny, nz] = mask.shape();
        let mut cells = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[[i, j, k]] != 0.0 {
                        cells.push((i, j, k));
                    }
                }
            }
        }
        cells
    }
}
