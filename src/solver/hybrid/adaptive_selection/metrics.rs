// adaptive_selection/metrics.rs - Metrics for method selection

use crate::grid::Grid;
use ndarray::ArrayView3;

/// Spectral analysis metrics for method selection
#[derive(Debug, Clone))]
pub struct SpectralMetrics {
    pub smoothness: f64,
    pub frequency_content: f64,
    pub dispersion_error: f64,
}

impl SpectralMetrics {
    /// Compute spectral metrics from field data
    pub fn compute(field: ArrayView3<f64>, grid: &Grid) -> Self {
        let smoothness = Self::compute_smoothness(field, grid);
        let frequency_content = Self::compute_frequency_content(field);
        let dispersion_error = Self::estimate_dispersion_error(field, grid);

        Self {
            smoothness,
            frequency_content,
            dispersion_error,
        }
    }

    fn compute_smoothness(field: ArrayView3<f64>, grid: &Grid) -> f64 {
        // Compute gradient magnitude as measure of smoothness
        let (nx, ny, nz) = field.dim();
        let mut total_gradient = 0.0;
        let mut count = 0;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let dx = (field[[i + 1, j, k] - field[[i - 1, j, k]) / (2.0 * grid.dx);
                    let dy = (field[[i, j + 1, k] - field[[i, j - 1, k]) / (2.0 * grid.dy);
                    let dz = (field[[i, j, k + 1] - field[[i, j, k - 1]) / (2.0 * grid.dz);

                    total_gradient += (dx * dx + dy * dy + dz * dz).sqrt();
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_gradient / count as f64
        } else {
            0.0
        }
    }

    fn compute_frequency_content(field: ArrayView3<f64>) -> f64 {
        // Simplified frequency content estimation
        // In production, would use FFT analysis
        let variance = field.var(0.0);
        variance.sqrt()
    }

    fn estimate_dispersion_error(field: ArrayView3<f64>, grid: &Grid) -> f64 {
        // Estimate numerical dispersion based on grid resolution
        let min_wavelength = 2.0 * grid.dx.min(grid.dy).min(grid.dz);
        let ppw = min_wavelength / grid.dx; // Points per wavelength

        // Dispersion error increases as ppw decreases
        if ppw > 10.0 {
            0.01 // Very low dispersion
        } else if ppw > 4.0 {
            0.1 // Moderate dispersion
        } else {
            1.0 // High dispersion
        }
    }
}

/// Material property metrics
#[derive(Debug, Clone))]
pub struct MaterialMetrics {
    pub homogeneity: f64,
    pub interface_proximity: f64,
    pub impedance_contrast: f64,
}

impl MaterialMetrics {
    /// Compute material metrics
    pub fn compute(
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        position: (usize, usize, usize),
    ) -> Self {
        let homogeneity = Self::compute_homogeneity(density, sound_speed);
        let interface_proximity = Self::compute_interface_proximity(density, position);
        let impedance_contrast = Self::compute_impedance_contrast(density, sound_speed);

        Self {
            homogeneity,
            interface_proximity,
            impedance_contrast,
        }
    }

    fn compute_homogeneity(density: ArrayView3<f64>, sound_speed: ArrayView3<f64>) -> f64 {
        // Compute coefficient of variation as measure of homogeneity
        let density_cv = density.std(0.0) / density.mean().unwrap_or(1.0);
        let speed_cv = sound_speed.std(0.0) / sound_speed.mean().unwrap_or(1.0);

        1.0 - (density_cv + speed_cv) / 2.0
    }

    fn compute_interface_proximity(
        density: ArrayView3<f64>,
        position: (usize, usize, usize),
    ) -> f64 {
        // Check for material discontinuities near position
        let (i, j, k) = position;
        let (nx, ny, nz) = density.dim();

        // Check neighbors for large changes
        let mut max_change = 0.0;
        let center = density[[i, j, k];

        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                for dk in -1i32..=1 {
                    if di == 0 && dj == 0 && dk == 0 {
                        continue;
                    }

                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;
                    let nk = (k as i32 + dk) as usize;

                    if ni < nx && nj < ny && nk < nz {
                        let change = (density[[ni, nj, nk] - center).abs() / center;
                        max_change = f64::max(max_change, change);
                    }
                }
            }
        }

        max_change
    }

    fn compute_impedance_contrast(density: ArrayView3<f64>, sound_speed: ArrayView3<f64>) -> f64 {
        // Compute acoustic impedance variation
        let impedance = &density * &sound_speed;
        impedance.std(0.0) / impedance.mean().unwrap_or(1.0)
    }
}

/// Computational efficiency metrics
#[derive(Debug, Clone))]
pub struct ComputationalMetrics {
    pub grid_resolution_quality: f64,
    pub stability_margin: f64,
    pub memory_efficiency: f64,
}

impl ComputationalMetrics {
    /// Compute efficiency metrics
    pub fn compute(grid: &Grid, dt: f64, sound_speed: f64) -> Self {
        let grid_resolution_quality = Self::compute_resolution_quality(grid, sound_speed);
        let stability_margin = Self::compute_stability_margin(grid, dt, sound_speed);
        let memory_efficiency = Self::estimate_memory_efficiency(grid);

        Self {
            grid_resolution_quality,
            stability_margin,
            memory_efficiency,
        }
    }

    fn compute_resolution_quality(grid: &Grid, sound_speed: f64) -> f64 {
        // Points per wavelength at typical frequency
        const TYPICAL_FREQUENCY: f64 = 1e6; // 1 MHz
        let wavelength = sound_speed / TYPICAL_FREQUENCY;
        let ppw = wavelength / grid.dx.min(grid.dy).min(grid.dz);

        // Quality score based on ppw
        (ppw / 10.0).min(1.0)
    }

    fn compute_stability_margin(grid: &Grid, dt: f64, sound_speed: f64) -> f64 {
        // CFL condition check
        let max_dt = 0.5 * grid.dx.min(grid.dy).min(grid.dz) / sound_speed;
        max_dt / dt
    }

    fn estimate_memory_efficiency(grid: &Grid) -> f64 {
        // Estimate based on grid size
        let total_points = grid.nx * grid.ny * grid.nz;

        if total_points < 1_000_000 {
            1.0 // Excellent
        } else if total_points < 10_000_000 {
            0.7 // Good
        } else {
            0.4 // May need optimization
        }
    }
}
