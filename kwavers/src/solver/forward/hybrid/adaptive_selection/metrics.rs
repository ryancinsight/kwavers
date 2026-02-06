// adaptive_selection/metrics.rs - Metrics for method selection

use crate::domain::grid::Grid;
use ndarray::ArrayView3;

/// Spectral analysis metrics for method selection
#[derive(Debug, Clone)]
pub struct SpectralMetrics {
    pub smoothness: f64,
    pub frequency_content: f64,
    pub spectral_centroid: f64,
    pub dispersion_error: f64,
}

impl SpectralMetrics {
    /// Compute spectral metrics from field data
    pub fn compute(field: ArrayView3<f64>, grid: &Grid) -> Self {
        let smoothness = Self::compute_smoothness(field, grid);
        let (frequency_content, spectral_centroid) = Self::compute_spectral_features(field, grid);
        let dispersion_error = Self::estimate_dispersion_error(field, grid);

        Self {
            smoothness,
            frequency_content,
            spectral_centroid,
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
                    let dx = (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let dy = (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let dz = (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * grid.dz);

                    total_gradient += (dx * dx + dy * dy + dz * dz).sqrt();
                    count += 1;
                }
            }
        }

        if count > 0 {
            // Normalize by max possible gradient estimate or just return raw
            // Ideally should be normalized for the selector
            total_gradient / f64::from(count)
        } else {
            0.0
        }
    }

    fn compute_spectral_features(field: ArrayView3<f64>, grid: &Grid) -> (f64, f64) {
        use crate::math::fft::ProcessorFft3d;
        use std::f64::consts::PI;

        let (nx, ny, nz) = field.dim();

        // 1. Perform FFT
        // Create a new processor (acceptable cost for adaptive selection)
        let fft = ProcessorFft3d::new(nx, ny, nz);

        // We need an owned array for the FFT processor
        let field_owned = field.to_owned();
        let spectrum = fft.forward(&field_owned);

        // 2. Compute Spectral Metrics
        let mut total_energy = 0.0;
        let mut weighted_k_sum = 0.0;
        let mut high_freq_energy = 0.0;

        // Nyquist limits and normalization factors
        let kx_max = PI / grid.dx;
        let ky_max = PI / grid.dy;
        let kz_max = PI / grid.dz;
        let k_max_norm = (kx_max.powi(2) + ky_max.powi(2) + kz_max.powi(2)).sqrt();
        let k_high_cutoff = k_max_norm * 0.5;

        for k_idx in 0..nz {
            let kz = if k_idx <= nz / 2 {
                2.0 * PI * k_idx as f64 / (nz as f64 * grid.dz)
            } else {
                2.0 * PI * (k_idx as f64 - nz as f64) / (nz as f64 * grid.dz)
            };

            for j in 0..ny {
                let ky = if j <= ny / 2 {
                    2.0 * PI * j as f64 / (ny as f64 * grid.dy)
                } else {
                    2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * grid.dy)
                };

                for i in 0..nx {
                    let kx = if i <= nx / 2 {
                        2.0 * PI * i as f64 / (nx as f64 * grid.dx)
                    } else {
                        2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * grid.dx)
                    };

                    let k_mag = (kx.powi(2) + ky.powi(2) + kz.powi(2)).sqrt();

                    // Use squared magnitude (energy)
                    let energy = spectrum[[i, j, k_idx]].norm_sqr();

                    // Exclude DC component
                    if k_mag > 1e-10 {
                        total_energy += energy;
                        weighted_k_sum += k_mag * energy;

                        if k_mag > k_high_cutoff {
                            high_freq_energy += energy;
                        }
                    }
                }
            }
        }

        if total_energy > 0.0 {
            let spectral_centroid = (weighted_k_sum / total_energy) / k_max_norm;
            let frequency_content = high_freq_energy / total_energy;
            (frequency_content, spectral_centroid)
        } else {
            (0.0, 0.0)
        }
    }

    fn estimate_dispersion_error(_field: ArrayView3<f64>, grid: &Grid) -> f64 {
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
#[derive(Debug, Clone)]
pub struct MaterialMetrics {
    pub homogeneity: f64,
    pub interface_proximity: f64,
    pub impedance_contrast: f64,
}

impl MaterialMetrics {
    /// Compute material metrics
    #[must_use]
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
        let center = density[[i, j, k]];

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
                        let change = (density[[ni, nj, nk]] - center).abs() / center;
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
#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::f64::consts::PI;

    #[test]
    fn test_spectral_metrics_computation() {
        let nx = 32;
        let ny = 32;
        let nz = 32;
        let dx = 0.001;
        let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

        let mut field_low = Array3::zeros((nx, ny, nz));
        let mut field_high = Array3::zeros((nx, ny, nz));

        // Low frequency: 1 period
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * dx;
                    field_low[[i, j, k]] = (2.0 * PI * x / (nx as f64 * dx)).sin();
                }
            }
        }

        // High frequency: 8 periods
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * dx;
                    field_high[[i, j, k]] = (2.0 * PI * x * 8.0 / (nx as f64 * dx)).sin();
                }
            }
        }

        let metrics_low = SpectralMetrics::compute(field_low.view(), &grid);
        let metrics_high = SpectralMetrics::compute(field_high.view(), &grid);

        println!("Low Freq Metrics: {:?}", metrics_low);
        println!("High Freq Metrics: {:?}", metrics_high);

        // High frequency content should be higher for the high frequency field
        assert!(
            metrics_high.frequency_content > metrics_low.frequency_content,
            "High frequency field should have higher frequency content score"
        );

        // Spectral centroid should be higher for high frequency field
        assert!(
            metrics_high.spectral_centroid > metrics_low.spectral_centroid,
            "High frequency field should have higher spectral centroid"
        );
    }
}
