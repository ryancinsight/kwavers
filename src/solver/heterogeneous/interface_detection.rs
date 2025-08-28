//! Interface detection for heterogeneous media
//!
//! Detects sharp interfaces in medium properties to enable targeted treatment

use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::Array3;

/// Interface detector for heterogeneous media
#[derive(Debug)]
pub struct InterfaceDetector {
    /// Detection threshold (relative change)
    threshold: f64,
    /// Grid reference
    grid: Grid,
}

impl InterfaceDetector {
    /// Create a new interface detector
    pub fn new(threshold: f64, grid: Grid) -> Self {
        Self { threshold, grid }
    }

    /// Detect interfaces in medium properties
    pub fn detect(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
    ) -> KwaversResult<Array3<bool>> {
        let mut mask = Array3::from_elem((self.grid.nx, self.grid.ny, self.grid.nz), false);

        // Detect based on relative change in properties
        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    let rho = density[[i, j, k]];
                    let c = sound_speed[[i, j, k]];

                    // Check neighbors for significant changes
                    let mut is_interface = false;

                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue;
                                }

                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;

                                let rho_neighbor = density[[ni, nj, nk]];
                                let c_neighbor = sound_speed[[ni, nj, nk]];

                                let rho_change = (rho - rho_neighbor).abs() / rho.max(rho_neighbor);
                                let c_change = (c - c_neighbor).abs() / c.max(c_neighbor);

                                if rho_change > self.threshold || c_change > self.threshold {
                                    is_interface = true;
                                    break;
                                }
                            }
                            if is_interface {
                                break;
                            }
                        }
                        if is_interface {
                            break;
                        }
                    }

                    mask[[i, j, k]] = is_interface;
                }
            }
        }

        Ok(mask)
    }

    /// Compute interface sharpness map
    pub fn compute_sharpness(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
    ) -> Array3<f64> {
        let mut sharpness = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    // Compute gradient magnitude as sharpness measure
                    let grad_rho = self.gradient_magnitude(density, i, j, k);
                    let grad_c = self.gradient_magnitude(sound_speed, i, j, k);

                    // Normalize by local values
                    let rho = density[[i, j, k]];
                    let c = sound_speed[[i, j, k]];

                    sharpness[[i, j, k]] = (grad_rho / rho).max(grad_c / c);
                }
            }
        }

        sharpness
    }

    /// Compute gradient magnitude at a point
    fn gradient_magnitude(&self, field: &Array3<f64>, i: usize, j: usize, k: usize) -> f64 {
        let dx = (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let dy = (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let dz = (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * self.grid.dz);

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute interface normal direction
    pub fn compute_interface_normal(
        &self,
        density: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> (f64, f64, f64) {
        let dx = (density[[i + 1, j, k]] - density[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let dy = (density[[i, j + 1, k]] - density[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let dz = (density[[i, j, k + 1]] - density[[i, j, k - 1]]) / (2.0 * self.grid.dz);

        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
        if norm > 1e-10 {
            (dx / norm, dy / norm, dz / norm)
        } else {
            (0.0, 0.0, 1.0) // Default to z-direction
        }
    }
}
