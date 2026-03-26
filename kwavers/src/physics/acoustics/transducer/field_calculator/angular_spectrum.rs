//! Angular spectrum pressure field and harmonic field calculations
//!
//! ## Mathematical Foundation
//!
//! Angular spectrum propagation transfer function:
//! ```text
//! H(kx, ky, z) = exp(j·kz·z)  if kz² > 0 (propagating)
//!              = exp(-|kz|·z)  if kz² < 0 (evanescent)
//! ```
//!
//! ## References
//!
//! - Zeng & McGough (2008). "Evaluation of the angular spectrum approach"
//! - Christopher & Parker (1991). "New approaches to nonlinear diffractive field propagation"

use super::plugin::TransducerFieldCalculatorPlugin;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::{AcousticProperties, Medium};
use ndarray::{Array2, Array3};

impl TransducerFieldCalculatorPlugin {
    /// Calculate pressure field using angular spectrum method
    pub fn calculate_pressure_field(
        &mut self,
        frequency: f64,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        use crate::math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};

        let mut pressure_field = Array3::zeros(grid.dimensions());
        let c = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let k = 2.0 * std::f64::consts::PI * frequency / c;

        let mut source_plane = Array2::<Complex64>::zeros((grid.nx, grid.ny));

        if !self.transducer_geometries.is_empty() {
            let geometry = &self.transducer_geometries[0];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let (x, y, _) = grid.indices_to_coordinates(i, j, 0);

                    for elem_idx in 0..geometry.element_positions.nrows() {
                        let elem_pos = geometry.element_positions.row(elem_idx);
                        let elem_size = geometry.element_sizes.row(elem_idx);

                        if (x - elem_pos[0]).abs() <= elem_size[0] / 2.0
                            && (y - elem_pos[1]).abs() <= elem_size[1] / 2.0
                        {
                            let mut amplitude = Complex64::new(1.0, 0.0);

                            if let Some(delays) = &geometry.delays {
                                if elem_idx < delays.len() {
                                    let phase =
                                        2.0 * std::f64::consts::PI * frequency * delays[elem_idx];
                                    amplitude *= Complex64::from_polar(1.0, phase);
                                }
                            }

                            if let Some(apod_weights) = &geometry.apodization {
                                if elem_idx < apod_weights.len() {
                                    amplitude *= apod_weights[elem_idx];
                                }
                            }

                            source_plane[[i, j]] += amplitude;
                        }
                    }
                }
            }
        }

        let spectrum = fft_2d_complex(&source_plane);

        for k_idx in 0..grid.nz {
            let z = k_idx as f64 * grid.dz;
            let mut propagated = spectrum.clone();

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let kx = 2.0 * std::f64::consts::PI * (i as f64 - grid.nx as f64 / 2.0)
                        / (grid.nx as f64 * grid.dx);
                    let ky = 2.0 * std::f64::consts::PI * (j as f64 - grid.ny as f64 / 2.0)
                        / (grid.ny as f64 * grid.dy);

                    let kz_sq = k * k - kx * kx - ky * ky;
                    if kz_sq > 0.0 {
                        let kz = kz_sq.sqrt();
                        let phase = Complex64::from_polar(1.0, kz * z);
                        propagated[[i, j]] *= phase;
                    } else {
                        let decay = (-kz_sq.abs().sqrt() * z).exp();
                        propagated[[i, j]] *= decay;
                    }
                }
            }

            let spatial_field = ifft_2d_complex(&propagated);

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    pressure_field[[i, j, k_idx]] = spatial_field[[i, j]].re;
                }
            }
        }

        Ok(pressure_field)
    }

    /// Calculate harmonic pressure field for nonlinear propagation
    pub fn calculate_harmonic_field(
        &mut self,
        harmonic: usize,
        fundamental_freq: f64,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let fundamental_field = self.calculate_pressure_field(fundamental_freq, grid, medium)?;

        if harmonic == 1 {
            return Ok(fundamental_field);
        }

        let mut harmonic_field = Array3::zeros(grid.dimensions());
        let harmonic_freq = fundamental_freq * harmonic as f64;
        let c = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);

        let beta =
            1.0 + AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, grid) / 2.0;

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 1..grid.nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let rho = crate::domain::medium::density_at(medium, x, y, z, grid);

                    let p1 = fundamental_field[[i, j, k]];
                    let p1_prev = fundamental_field[[i, j, k.saturating_sub(1)]];
                    let _grad_z = (p1 - p1_prev) / grid.dz;

                    let source_strength = match harmonic {
                        2 => beta * p1.powi(2) / (2.0 * rho * c.powi(4)),
                        3 => beta.powi(2) * p1.powi(3) / (6.0 * rho * c.powi(4)),
                        _ => {
                            let efficiency = 1.0 / (harmonic as f64).powi(2);
                            efficiency * beta.powi(harmonic as i32 - 1) * p1.powi(harmonic as i32)
                                / (rho * c.powi(4) * harmonic as f64)
                        }
                    };

                    let alpha = AcousticProperties::absorption_coefficient(
                        medium,
                        x,
                        y,
                        z,
                        grid,
                        harmonic_freq,
                    );
                    let attenuation = (-alpha * z).exp();

                    let avg_aperture = if !self.transducer_geometries.is_empty() {
                        let geometry = &self.transducer_geometries[0];
                        geometry.element_sizes.column(0).mean().unwrap_or(0.01)
                    } else {
                        0.01
                    };

                    let k_harmonic = 2.0 * std::f64::consts::PI * harmonic_freq / c;
                    let rayleigh_distance = k_harmonic * avg_aperture.powi(2) / 4.0;
                    let divergence = (1.0 + (z / rayleigh_distance).powi(2)).sqrt();

                    harmonic_field[[i, j, k]] = source_strength * attenuation / divergence;
                }
            }
        }

        Ok(harmonic_field)
    }
}
