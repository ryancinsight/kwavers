//! Public query methods for [`KWaveArray`]: masks, delays, apodization, and
//! distributed-source construction.

use ndarray::{Array2, Array3};

use super::{ApodizationWindow, ElementShape, KWaveArray};

impl KWaveArray {
    /// Number of elements in the array.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Centroids of all elements with the global array transform applied.
    #[must_use]
    pub fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .map(|e| {
                let local = match e {
                    ElementShape::Arc { position, .. } => *position,
                    ElementShape::Rect { position, .. } => *position,
                    ElementShape::Disc { position, .. } => *position,
                    ElementShape::Bowl { position, .. } => *position,
                    ElementShape::Annulus { position, .. } => *position,
                };
                self.apply_transform_point(local)
            })
            .collect()
    }

    /// Generate a binary mask on the computational grid.
    ///
    /// Returns a 3D boolean array where `true` indicates grid points that are
    /// part of the transducer array.
    pub fn get_array_binary_mask(&self, grid: &crate::domain::grid::Grid) -> Array3<bool> {
        let mut mask = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
        for element in &self.elements {
            match element {
                ElementShape::Arc {
                    position,
                    radius,
                    diameter,
                    start_angle,
                    end_angle,
                } => {
                    self.rasterize_arc(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *diameter,
                        *start_angle,
                        *end_angle,
                    );
                }
                ElementShape::Rect {
                    position,
                    width,
                    height,
                    length,
                    euler_xyz_deg,
                } => {
                    let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                    self.rasterize_rect(
                        &mut mask, grid, pos_eff, *width, *height, *length, euler_eff,
                    );
                }
                ElementShape::Disc {
                    position,
                    diameter,
                    focus_position,
                } => {
                    self.rasterize_disc(&mut mask, grid, *position, *diameter, *focus_position);
                }
                ElementShape::Bowl {
                    position,
                    radius,
                    diameter,
                } => {
                    self.rasterize_bowl(&mut mask, grid, *position, *radius, *diameter);
                }
                ElementShape::Annulus {
                    position,
                    radius,
                    inner_diameter,
                    outer_diameter,
                } => {
                    self.rasterize_annulus(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *inner_diameter,
                        *outer_diameter,
                    );
                }
            }
        }
        mask
    }

    /// Generate a float-weighted mask on the computational grid.
    ///
    /// Uses k-Wave-compatible surface sampling and local BLI stencils to
    /// produce per-cell weights that sum to each element's geometric measure
    /// expressed in grid cells.
    pub fn get_array_weighted_mask(&self, grid: &crate::domain::grid::Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for element in &self.elements {
            match element {
                ElementShape::Bowl {
                    position,
                    radius,
                    diameter,
                } => {
                    self.rasterize_bowl_weighted(&mut mask, grid, *position, *radius, *diameter);
                }
                ElementShape::Disc {
                    position,
                    diameter,
                    focus_position,
                } => {
                    self.rasterize_disc_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *diameter,
                        *focus_position,
                    );
                }
                ElementShape::Arc {
                    position,
                    radius,
                    diameter,
                    start_angle,
                    end_angle,
                } => {
                    self.rasterize_arc_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *diameter,
                        *start_angle,
                        *end_angle,
                    );
                }
                ElementShape::Rect {
                    position,
                    width,
                    height,
                    length,
                    euler_xyz_deg,
                } => {
                    let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                    self.rasterize_rect_weighted(
                        &mut mask, grid, pos_eff, *width, *height, *length, euler_eff,
                    );
                }
                ElementShape::Annulus {
                    position,
                    radius,
                    inner_diameter,
                    outer_diameter,
                } => {
                    self.rasterize_annulus_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *inner_diameter,
                        *outer_diameter,
                    );
                }
            }
        }
        mask
    }

    /// Compute the total geometric measure of all elements.
    ///
    /// Bowl and disc elements contribute area; arc elements contribute arc
    /// length; rect elements contribute `width × height`; annuli contribute
    /// the spherical-ring area.
    #[must_use]
    pub fn compute_total_surface_area(&self) -> f64 {
        self.elements
            .iter()
            .map(|e| match e {
                ElementShape::Bowl {
                    radius: r,
                    diameter: d,
                    ..
                } => Self::bowl_surface_area(*r, *d),
                ElementShape::Disc { diameter: d, .. } => std::f64::consts::PI * (d / 2.0).powi(2),
                ElementShape::Rect {
                    width: w,
                    height: h,
                    ..
                } => w * h,
                ElementShape::Arc {
                    radius: r,
                    start_angle: s,
                    end_angle: e,
                    ..
                } => Self::arc_line_length(*r, *s, *e),
                ElementShape::Annulus {
                    radius: r,
                    inner_diameter: di,
                    outer_diameter: d_o,
                    ..
                } => Self::annulus_surface_area(*r, *di, *d_o),
            })
            .sum()
    }

    /// Broadcast a single time signal to all elements, returning
    /// `[n_elements, n_times]`.
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

    /// Calculate focus delays `[s]` for each element to a target point
    /// (`delay = distance / c`).
    ///
    /// # Arguments
    /// * `focus_point` - Focus position `(x, y, z)` in metres
    #[must_use]
    pub fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let c = self.sound_speed;
        self.get_element_positions()
            .iter()
            .map(|(ex, ey, ez)| {
                let dist = ((ex - focus_point.0).powi(2)
                    + (ey - focus_point.1).powi(2)
                    + (ez - focus_point.2).powi(2))
                .sqrt();
                dist / c
            })
            .collect()
    }

    /// Calculate per-element time delays `[s]` for electronic focusing.
    ///
    /// # Algorithm — Time-Delay Focusing (Selfridge et al. 1980)
    ///
    /// `τᵢ = (d_max − dᵢ) / c` where `dᵢ = ‖pᵢ − f‖`. The element farthest
    /// from the focus fires first (zero delay); all others are delayed so
    /// wavefronts add coherently at the focus.
    ///
    /// All returned values are ≥ 0, and `min(τ) = 0`.
    ///
    /// Reference: Selfridge, A.R., Kino, G.S. & Khuri-Yakub, B.T. (1980).
    /// "A theory for the radiation pattern of a narrow-strip acoustic
    /// transducer." Appl. Phys. Lett. 37(1):35–36.
    #[must_use]
    pub fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let c = self.sound_speed;
        let positions = self.get_element_positions();
        let distances: Vec<f64> = positions
            .iter()
            .map(|(ex, ey, ez)| {
                ((ex - focus_point.0).powi(2)
                    + (ey - focus_point.1).powi(2)
                    + (ez - focus_point.2).powi(2))
                .sqrt()
            })
            .collect();
        let d_max = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        distances.iter().map(|&d| (d_max - d) / c).collect()
    }

    /// Compute per-element amplitude weights for beam apodization.
    ///
    /// # Algorithm — Discrete Window Functions (Harris 1978)
    ///
    /// For N elements indexed `i = 0, …, N−1`:
    /// - `Rectangular`: `wᵢ = 1.0`
    /// - `Hann`:        `wᵢ = 0.5·(1 − cos(2π·i/(N−1)))` if N > 1, else 1.0
    /// - `Hamming`:     `wᵢ = 0.54 − 0.46·cos(2π·i/(N−1))` if N > 1, else 1.0
    ///
    /// Reference: Harris, F.J. (1978). Proc. IEEE 66(1):51–83.
    #[must_use]
    pub fn get_apodization(&self, window: ApodizationWindow) -> Vec<f64> {
        let n = self.elements.len();
        if n == 0 {
            return Vec::new();
        }
        match window {
            ApodizationWindow::Rectangular => vec![1.0; n],
            ApodizationWindow::Hann => {
                if n == 1 {
                    return vec![1.0];
                }
                (0..n)
                    .map(|i| {
                        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
                    })
                    .collect()
            }
            ApodizationWindow::Hamming => {
                if n == 1 {
                    return vec![1.0];
                }
                (0..n)
                    .map(|i| {
                        0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                    })
                    .collect()
            }
        }
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
        grid: &crate::domain::grid::Grid,
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
    pub(super) fn active_cells_fortran_order(mask: &Array3<f64>) -> Vec<(usize, usize, usize)> {
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
