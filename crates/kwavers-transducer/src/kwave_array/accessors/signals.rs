use ndarray::{Array2, Array3};

use super::super::KWaveArray;

impl KWaveArray {
    /// Broadcast a single time signal to all elements, returning
    /// `[n_elements, n_times]`.
    #[must_use]
    pub fn get_distributed_source_signal(&self, signal: &ndarray::Array1<f64>) -> Array2<f64> {
        let n_elements = self.elements.len();
        let n_times = signal.len();
        let mut distributed = Array2::zeros((n_elements, n_times));
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
        let (rows, _) = per_element_signals.dim();
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
        let (rows, n_times) = per_element_signals.dim();
        if rows != n_elements {
            return Err(format!(
                "per_element_signals has {rows} rows but array has {n_elements} elements"
            ));
        }

        let shape = (grid.nx, grid.ny, grid.nz);
        let per_element_masks: Vec<Array3<f64>> = self
            .elements
            .iter()
            .map(|element| {
                let mut m = Array3::zeros(shape);
                self.rasterize_element_weighted(element, &mut m, grid);
                m
            })
            .collect();

        let mut combined = Array3::<f64>::zeros(shape);
        for m in &per_element_masks {
            combined += m;
        }
        let active_cells = Self::active_cells_fortran_order(&combined);
        let n_active = active_cells.len();

        let mut per_cell_signal = Array2::<f64>::zeros((n_active, n_times));
        for (idx, &(i, j, k)) in active_cells.iter().enumerate() {
            let contribs: Vec<(usize, f64)> = per_element_masks
                .iter()
                .enumerate()
                .filter_map(|(e, m)| {
                    let w = m[[i, j, k]];
                    if w != 0.0 {
                        Some((e, w))
                    } else {
                        None
                    }
                })
                .collect();
            for t in 0..n_times {
                let s: f64 = contribs
                    .iter()
                    .map(|&(e, w)| w * per_element_signals[[e, t]])
                    .sum();
                per_cell_signal[[idx, t]] = s;
            }
        }

        let mut mask_ones = Array3::<f64>::zeros(shape);
        for &(i, j, k) in &active_cells {
            mask_ones[[i, j, k]] = 1.0;
        }

        Ok((mask_ones, per_cell_signal))
    }

    /// Enumerate active (nonzero) grid cells in Fortran/MATLAB column-major order
    /// (k outer, j middle, i inner), matching `matlab_find(mask)` on the same mask.
    #[inline]
    pub(in crate::kwave_array) fn active_cells_fortran_order(
        mask: &Array3<f64>,
    ) -> Vec<(usize, usize, usize)> {
        let (nx, ny, nz) = mask.dim();
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
