//! Core Spectral Unmixer Implementation

use super::tikhonov::{estimate_condition_number, tikhonov_solve};
use crate::signal_processing::spectroscopy::types::{
    SpectralUnmixingConfig, UnmixingResult, VolumetricUnmixingResult,
};
use anyhow::Result;
use leto::{Array1, Array2, Array3};

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
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(
        extinction_matrix: Array2<f64>,
        wavelengths: Vec<f64>,
        chromophore_names: Vec<String>,
        config: SpectralUnmixingConfig,
    ) -> Result<Self> {
        let [n_wavelengths, n_chromophores] = extinction_matrix.shape();

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
        if condition > 1.0 / config.min_condition_number {
            // Note: min_condition_number is 1e-6 in config
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
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn unmix_single(&self, absorption_spectrum: &Array1<f64>) -> Result<UnmixingResult> {
        let n_wavelengths = self.extinction_matrix.shape()[0];

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
        let mut reconstruction = Array1::<f64>::zeros(n_wavelengths);
        leto_ops::matvec(
            &self.extinction_matrix.view(),
            &concentrations.view(),
            &mut reconstruction.view_mut(),
        )
        .expect("invariant: EC reconstruction dimensions match");
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn unmix_volumetric(
        &self,
        absorption_maps: &[Array3<f64>],
    ) -> Result<VolumetricUnmixingResult> {
        let n_wavelengths = self.extinction_matrix.shape()[0];
        let n_chromophores = self.extinction_matrix.shape()[1];

        if absorption_maps.len() != n_wavelengths {
            anyhow::bail!(
                "Number of absorption maps {} does not match wavelength count {}",
                absorption_maps.len(),
                n_wavelengths
            );
        }

        let [nx, ny, nz] = absorption_maps[0].shape();

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

    #[must_use]
    pub fn wavelengths(&self) -> &[f64] {
        &self.wavelengths
    }
    #[must_use]
    pub fn chromophore_names(&self) -> &[String] {
        &self.chromophore_names
    }
    #[must_use]
    pub fn extinction_matrix(&self) -> &Array2<f64> {
        &self.extinction_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_processing::spectroscopy::types::SpectralUnmixingConfig;

    /// End-to-end blood-oxygenation recovery. Build a 2-chromophore (HbO₂, Hb)
    /// extinction matrix from the corrected Prahl/OMLC near-infrared values
    /// (per-tetramer ×4, here ÷1000 for numerical scale; `sO₂` is scale-free),
    /// forward-model `μ = E·c` for a known oxygenation, and confirm the unmixer
    /// recovers sO₂. The 700–900 nm window straddles the ~800 nm isosbestic, so
    /// the HbO₂/Hb crossover makes `E` well-conditioned and the system
    /// invertible — a regression guard on both the unmixer and the extinction
    /// spectra (the previously-incorrect HbO₂ < Hb everywhere would not invert).
    #[test]
    fn recovers_known_oxygen_saturation_from_hemoglobin_spectra() {
        // Columns: [HbO₂, Hb]; rows: 700/750/800/850/900 nm.
        let e = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.160, 7.177_12, 2.072, 5.620_96, 3.264, 3.046_88, 4.232, 2.765_28, 4.792, 3.047_36,
            ],
        )
        .unwrap();
        let config = SpectralUnmixingConfig {
            regularization_lambda: 1e-6,
            non_negative: true,
            min_condition_number: 1e-6,
        };
        let unmixer = SpectralUnmixer::new(
            e.clone(),
            vec![700.0, 750.0, 800.0, 850.0, 900.0],
            vec!["HbO2".to_string(), "Hb".to_string()],
            config,
        )
        .expect("well-conditioned unmixer");

        for &true_so2 in &[0.98_f64, 0.70, 0.50] {
            let c_true = Array1::from_vec(2, vec![true_so2, 1.0 - true_so2]).unwrap();
            // forward model μ = E·c
            let mut mu = Array1::<f64>::zeros(5);
            leto_ops::matvec(&e.view(), &c_true.view(), &mut mu.view_mut()).unwrap();
            let result = unmixer.unmix_single(&mu).expect("unmix");
            let (hbo2, hb) = (result.concentrations[0], result.concentrations[1]);
            let so2 = hbo2 / (hbo2 + hb);
            assert!(
                (so2 - true_so2).abs() < 0.01,
                "sO₂ recovery: true={true_so2}, got={so2} (HbO₂={hbo2}, Hb={hb})"
            );
            // Exact synthetic data ⇒ near-zero residual confirms consistency.
            assert!(
                result.relative_residual < 1e-3,
                "relative residual {} too large",
                result.relative_residual
            );
        }
    }
}
