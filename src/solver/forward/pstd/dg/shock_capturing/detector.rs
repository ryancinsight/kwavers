//! Shock detection algorithms for spectral DG methods
//!
//! Provides various indicators to detect discontinuities and shocks in the solution.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, Array4};

use crate::domain::core::constants::numerical::EPSILON;

/// Shock detector with multiple indicators
#[derive(Debug, Clone)]
pub struct ShockDetector {
    /// Base threshold for shock detection
    threshold: f64,
    /// Use modal decay indicator
    use_modal_decay: bool,
    /// Use jump indicator
    use_jump_indicator: bool,
    /// Use entropy residual
    #[allow(dead_code)]
    use_entropy_residual: bool,
    /// Smoothness exponent for modal decay
    #[allow(dead_code)]
    smoothness_exponent: f64,
}

impl Default for ShockDetector {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            use_modal_decay: true,
            use_jump_indicator: true,
            use_entropy_residual: false,
            smoothness_exponent: 2.0,
        }
    }
}

impl ShockDetector {
    /// Create a new shock detector
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Detect shocks in the field using multiple indicators
    pub fn detect_shocks(&self, field: &Array3<f64>, grid: &Grid) -> Array3<bool> {
        let (nx, ny, nz) = field.dim();
        let mut shock_mask = Array3::from_elem((nx, ny, nz), false);

        // Modal decay indicator
        if self.use_modal_decay {
            self.apply_modal_decay_detection(field, grid, &mut shock_mask);
        }

        // Jump indicator
        if self.use_jump_indicator {
            self.apply_jump_detection(field, &mut shock_mask);
        }

        shock_mask
    }

    /// Apply modal decay shock detection
    fn apply_modal_decay_detection(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
        shock_mask: &mut Array3<bool>,
    ) {
        let (nx, ny, nz) = field.dim();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let grad_x = (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let grad_y = (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let grad_z = (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * grid.dz);

                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                    let field_mag = field[[i, j, k]].abs() + EPSILON;

                    if grad_mag / field_mag > self.threshold {
                        shock_mask[[i, j, k]] = true;
                    }
                }
            }
        }
    }

    /// Apply jump-based shock detection
    fn apply_jump_detection(&self, field: &Array3<f64>, shock_mask: &mut Array3<bool>) {
        let (nx, ny, nz) = field.dim();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Check jumps across cell interfaces
                    let jump_x = (field[[i + 1, j, k]] - field[[i, j, k]]).abs();
                    let jump_y = (field[[i, j + 1, k]] - field[[i, j, k]]).abs();
                    let jump_z = (field[[i, j, k + 1]] - field[[i, j, k]]).abs();

                    let max_jump = jump_x.max(jump_y).max(jump_z);
                    let field_scale = field[[i, j, k]].abs() + EPSILON;

                    if max_jump / field_scale > self.threshold * 10.0 {
                        shock_mask[[i, j, k]] = true;
                    }
                }
            }
        }
    }

    /// Compute entropy-based shock indicator
    pub fn compute_entropy_indicator(
        &self,
        pressure: &Array3<f64>,
        density: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));

        // Compute specific entropy s = p/ρ^γ
        let gamma = 1.4; // Adiabatic index for air

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Compute local entropy
                    let s_center = pressure[[i, j, k]] / density[[i, j, k]].powf(gamma);

                    // Check entropy in all directions
                    let mut max_entropy_jump = 0.0;

                    for (di, dj, dk) in &[
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ] {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let nk = (k as i32 + dk) as usize;

                        let s_neighbor = pressure[[ni, nj, nk]] / density[[ni, nj, nk]].powf(gamma);
                        let entropy_jump =
                            (s_center - s_neighbor).abs() / s_center.abs().max(EPSILON);
                        max_entropy_jump = f64::max(max_entropy_jump, entropy_jump);
                    }

                    // Entropy should decrease across shocks
                    indicator[[i, j, k]] = (max_entropy_jump / self.threshold).min(1.0);
                }
            }
        }

        Ok(indicator)
    }

    /// Compute pressure-based shock indicator (Ducros sensor)
    pub fn compute_pressure_indicator(
        &self,
        pressure: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));

        // Use pressure jumps and gradients
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for k in 2..nz - 2 {
                    // Compute pressure jump indicator (Ducros et al.)
                    let p_center = pressure[[i, j, k]];

                    // Second derivatives for detecting discontinuities
                    let d2p_dx2 = (pressure[[i + 1, j, k]] - 2.0 * p_center
                        + pressure[[i - 1, j, k]])
                        / (grid.dx * grid.dx);
                    let d2p_dy2 = (pressure[[i, j + 1, k]] - 2.0 * p_center
                        + pressure[[i, j - 1, k]])
                        / (grid.dy * grid.dy);
                    let d2p_dz2 = (pressure[[i, j, k + 1]] - 2.0 * p_center
                        + pressure[[i, j, k - 1]])
                        / (grid.dz * grid.dz);

                    // First derivatives
                    let dp_dx =
                        (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let dp_dy =
                        (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let dp_dz =
                        (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);

                    // Ducros sensor
                    let laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2;
                    let grad_mag = (dp_dx * dp_dx + dp_dy * dp_dy + dp_dz * dp_dz).sqrt();

                    let sensor = laplacian.abs() / (grad_mag + EPSILON * p_center.abs());
                    indicator[[i, j, k]] = (sensor / self.threshold).min(1.0);
                }
            }
        }

        Ok(indicator)
    }

    /// Compute velocity divergence indicator
    pub fn compute_divergence_indicator(
        &self,
        velocity: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (_, nx, ny, nz) = velocity.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));

        // Compute velocity divergence
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dvx_dx =
                        (velocity[[0, i + 1, j, k]] - velocity[[0, i - 1, j, k]]) / (2.0 * grid.dx);
                    let dvy_dy =
                        (velocity[[1, i, j + 1, k]] - velocity[[1, i, j - 1, k]]) / (2.0 * grid.dy);
                    let dvz_dz =
                        (velocity[[2, i, j, k + 1]] - velocity[[2, i, j, k - 1]]) / (2.0 * grid.dz);

                    let divergence = dvx_dx + dvy_dy + dvz_dz;

                    // Strong compression indicates shock
                    indicator[[i, j, k]] = (-divergence).max(0.0);
                }
            }
        }

        // Normalize
        let max_div = indicator.iter().copied().fold(0.0_f64, f64::max);
        if max_div > 0.0 {
            indicator /= max_div;
        }

        Ok(indicator)
    }
}
