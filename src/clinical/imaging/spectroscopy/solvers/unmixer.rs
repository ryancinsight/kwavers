//! Core Spectral Unmixer Implementation

use crate::clinical::imaging::spectroscopy::types::{
    SpectralUnmixingConfig, UnmixingResult, VolumetricUnmixingResult,
};
use super::tikhonov::{estimate_condition_number, tikhonov_solve};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// Spectral unmixing solver
#[derive(Debug)]
pub struct SpectralUnmixer {
    /// Extinction coefficient matrix E (n_wavelengths × n_chromophores)
    extinction_matrix: Array2<f64>,
    /// Wavelengths (nm)
    wavelengths: Vec<f64>,
    /// Chromophore names
    chromophore_names: Vec<String>,
    /// Configuration
    config: SpectralUnmixingConfig,
}

impl SpectralUnmixer {
    /// Create unmixer from extinction matrix
    pub fn new(
        extinction_matrix: Array2<f64>,
        wavelengths: Vec<f64>,
        chromophore_names: Vec<String>,
        config: SpectralUnmixingConfig,
    ) -> Result<Self> {
        let (n_wavelengths, n_chromophores) = extinction_matrix.dim();

        if wavelengths.len() != n_wavelengths {
            anyhow::bail!(
                "Wavelength vector length {} does not match extinction matrix rows {}",
                wavelengths.len(),
                n_wavelengths
            );
        }

        if chromophore_names.len() != n_chromophores {
            anyhow::bail!(
                "Chromophore name count {} does not match extinction matrix columns {}",
                chromophore_names.len(),
                n_chromophores
            );
        }

        if n_wavelengths < n_chromophores {
            anyhow::bail!(
                "Underdetermined system: {} wavelengths < {} chromophores.",
                n_wavelengths,
                n_chromophores
            );
        }

        let condition = estimate_condition_number(&extinction_matrix)?;
        if condition > 1.0 / config.min_condition_number { // Note: min_condition_number is 1e-6 in config
            tracing::warn!(
                "Extinction matrix is poorly conditioned (cond ≈ {:.2e}).",
                condition
            );
        }

        Ok(Self {
            extinction_matrix,
            wavelengths,
            chromophore_names,
            config,
        })
    }

    /// Unmix single spectrum (one voxel)
    pub fn unmix_single(&self, absorption_spectrum: &Array1<f64>) -> Result<UnmixingResult> {
        let n_wavelengths = self.extinction_matrix.nrows();

        if absorption_spectrum.len() != n_wavelengths {
            anyhow::bail!(
                "Absorption spectrum length {} does not match wavelength count {}",
                absorption_spectrum.len(),
                n_wavelengths
            );
        }

        let concentrations = if self.config.non_negative {
            let c_unconstrained = tikhonov_solve(
                &self.extinction_matrix,
                absorption_spectrum,
                self.config.regularization_lambda,
            )?;
            c_unconstrained.mapv(|x| x.max(0.0))
        } else {
            tikhonov_solve(
                &self.extinction_matrix,
                absorption_spectrum,
                self.config.regularization_lambda,
            )?
        };

        // Compute residual: r = μ - EC
        let reconstruction = self.extinction_matrix.dot(&concentrations);
        let residual = absorption_spectrum - &reconstruction;
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let spectrum_norm = absorption_spectrum
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();

        let relative_residual = if spectrum_norm > 1e-12 {
            residual_norm / spectrum_norm
        } else {
            0.0
        };

        Ok(UnmixingResult {
            concentrations,
            residual_norm,
            relative_residual,
        })
    }

    /// Unmix volumetric data (entire 3D image)
    pub fn unmix_volumetric(
        &self,
        absorption_maps: &[Array3<f64>],
    ) -> Result<VolumetricUnmixingResult> {
        let n_wavelengths = self.extinction_matrix.nrows();
        let n_chromophores = self.extinction_matrix.ncols();

        if absorption_maps.len() != n_wavelengths {
            anyhow::bail!(
                "Number of absorption maps {} does not match wavelength count {}",
                absorption_maps.len(),
                n_wavelengths
            );
        }

        let (nx, ny, nz) = absorption_maps[0].dim();

        // Initialize output arrays
        let mut concentration_maps = Vec::with_capacity(n_chromophores);
        for _ in 0..n_chromophores {
            concentration_maps.push(Array3::zeros((nx, ny, nz)));
        }
        let mut residual_map = Array3::zeros((nx, ny, nz));

        // Unmix each voxel
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut spectrum = Array1::zeros(n_wavelengths);
                    for (wl_idx, abs_map) in absorption_maps.iter().enumerate() {
                        spectrum[wl_idx] = abs_map[[i, j, k]];
                    }

                    match self.unmix_single(&spectrum) {
                        Ok(result) => {
                            for (c_idx, &conc) in result.concentrations.iter().enumerate() {
                                concentration_maps[c_idx][[i, j, k]] = conc;
                            }
                            residual_map[[i, j, k]] = result.residual_norm;
                        }
                        Err(_) => {
                            residual_map[[i, j, k]] = f64::NAN;
                        }
                    }
                }
            }
        }

        Ok(VolumetricUnmixingResult {
            concentration_maps,
            residual_map,
            chromophore_names: self.chromophore_names.clone(),
        })
    }

    pub fn wavelengths(&self) -> &[f64] { &self.wavelengths }
    pub fn chromophore_names(&self) -> &[String] { &self.chromophore_names }
    pub fn extinction_matrix(&self) -> &Array2<f64> { &self.extinction_matrix }
}
