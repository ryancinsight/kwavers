use kwavers_core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;
use leto::{Array1, Array3, Array4};

use crate::parallel::for_each_indexed_three_mut;

/// Parameters for sonoluminescence emission
#[derive(Debug, Clone)]
pub struct EmissionParameters {
    /// Enable blackbody radiation
    pub use_blackbody: bool,
    /// Enable bremsstrahlung radiation
    pub use_bremsstrahlung: bool,
    /// Enable Cherenkov radiation
    pub use_cherenkov: bool,
    /// Enable molecular line emission
    pub use_molecular_lines: bool,
    /// Ionization energy for gas (eV)
    pub ionization_energy: f64,
    /// Minimum temperature for light emission (K)
    pub min_temperature: f64,
    /// Opacity correction factor
    pub opacity_factor: f64,
    /// Refractive index for Cherenkov calculations
    pub cherenkov_refractive_index: f64,
    /// Cherenkov coherence enhancement factor
    pub cherenkov_coherence_factor: f64,
}

impl Default for EmissionParameters {
    fn default() -> Self {
        Self {
            use_blackbody: true,
            use_bremsstrahlung: true,
            use_cherenkov: false,       // Experimental feature
            use_molecular_lines: false, // Not implemented yet
            ionization_energy: kwavers_core::constants::chemistry::ARGON_IONIZATION_ENERGY, // eV for argon
            min_temperature: 2000.0,                                                        // K
            opacity_factor: 1.0, // Optically thin
            cherenkov_refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
            cherenkov_coherence_factor: 100.0,
        }
    }
}

/// Spectral field using Struct-of-Arrays for better performance
#[derive(Debug)]
pub struct SpectralField {
    /// Wavelength grid (shared for all spatial points)
    pub wavelengths: Array1<f64>,
    /// Spectral intensities: dimensions (nx, ny, nz, `n_wavelengths`)
    pub intensities: Array4<f64>,
    /// Peak wavelength at each point: dimensions (nx, ny, nz)
    pub peak_wavelength: Array3<f64>,
    /// Total intensity at each point: dimensions (nx, ny, nz)
    pub total_intensity: Array3<f64>,
    /// Color temperature at each point: dimensions (nx, ny, nz)
    pub color_temperature: Array3<f64>,
}

impl SpectralField {
    /// Create new spectral field
    #[must_use]
    pub fn new(grid_shape: [usize; 3], wavelengths: Array1<f64>) -> Self {
        let n_wavelengths = wavelengths.size();
        let s = grid_shape;
        let shape_4d = [s[0], s[1], s[2], n_wavelengths];

        Self {
            wavelengths,
            intensities: Array4::zeros(shape_4d),
            peak_wavelength: Array3::zeros(s),
            total_intensity: Array3::zeros(s),
            color_temperature: Array3::zeros(s),
        }
    }

    /// Update derived quantities (peak wavelength, total intensity, etc.)
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn update_derived_quantities(&mut self) {
        let SpectralField {
            wavelengths,
            intensities,
            peak_wavelength,
            total_intensity,
            color_temperature,
        } = self;

        let output_shape = total_intensity.shape();
        assert_eq!(
            peak_wavelength.shape(),
            output_shape,
            "invariant: spectral peak output shape mismatch"
        );
        assert_eq!(
            color_temperature.shape(),
            output_shape,
            "invariant: spectral color output shape mismatch"
        );
        assert_eq!(
            intensities.shape(),
            [
                output_shape[0],
                output_shape[1],
                output_shape[2],
                wavelengths.len()
            ],
            "invariant: spectral intensity lane shape mismatch"
        );

        let [_nx, ny, nz] = output_shape;
        let n_wavelengths = wavelengths.len();
        let contiguous_intensities = intensities.as_slice_memory_order();

        for_each_indexed_three_mut(
            total_intensity.view_mut(),
            peak_wavelength.view_mut(),
            color_temperature.view_mut(),
            |idx, total, peak, color_temp| {
                let quantities = if let Some(intensities) = contiguous_intensities {
                    let start = idx * n_wavelengths;
                    let end = start + n_wavelengths;
                    spectral_cell_quantities(wavelengths, intensities[start..end].iter().copied())
                } else {
                    let i = idx / (ny * nz);
                    let rem = idx % (ny * nz);
                    let j = rem / nz;
                    let k = rem % nz;
                    spectral_cell_quantities(
                        wavelengths,
                        (0..n_wavelengths).map(|l| intensities[[i, j, k, l]]),
                    )
                };
                *total = quantities.total;
                *peak = quantities.peak_wavelength;
                *color_temp = quantities.color_temperature;
            },
        );
    }

    /// Get spectrum at a specific point
    #[must_use]
    pub fn get_spectrum_at(
        &self,
        i: usize,
        j: usize,
        k: usize,
    ) -> crate::optics::sonoluminescence::spectral::EmissionSpectrum {
        let n_wavelengths = self.wavelengths.len();
        let intensities =
            Array1::from_shape_fn([n_wavelengths], |[l]| self.intensities[[i, j, k, l]]);
        crate::optics::sonoluminescence::spectral::EmissionSpectrum::new(
            self.wavelengths.clone(),
            intensities,
            0.0,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct SpectralCellQuantities {
    total: f64,
    peak_wavelength: f64,
    color_temperature: f64,
}

fn spectral_cell_quantities<I>(wavelengths: &Array1<f64>, spectrum: I) -> SpectralCellQuantities
where
    I: IntoIterator<Item = f64>,
{
    use std::cmp::Ordering;

    let mut total = 0.0_f64;
    let mut peak = None::<(usize, f64)>;

    for (idx, intensity) in spectrum.into_iter().enumerate() {
        total += intensity;
        let should_replace = peak
            .map(|(_, current)| intensity.total_cmp(&current) != Ordering::Less)
            .unwrap_or(true);
        if should_replace {
            peak = Some((idx, intensity));
        }
    }

    let peak_wavelength = peak.map_or(0.0, |(idx, _)| wavelengths[idx]);
    let color_temperature = if peak_wavelength > 0.0 {
        kwavers_core::constants::optical::WIEN_CONSTANT / peak_wavelength
    } else {
        0.0
    };

    SpectralCellQuantities {
        total,
        peak_wavelength,
        color_temperature,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::optical::WIEN_CONSTANT;

    #[test]
    fn spectral_field_derived_quantities_match_cell_spectra() {
        let wavelengths = Array1::from_vec([3], vec![400.0e-9, 500.0e-9, 600.0e-9]);
        let mut field = SpectralField::new([2, 1, 1], wavelengths.clone());
        field.intensities[[0, 0, 0, 0]] = 3.0;
        field.intensities[[0, 0, 0, 1]] = 3.0;
        field.intensities[[0, 0, 0, 2]] = 1.0;
        field.intensities[[1, 0, 0, 0]] = 1.0;
        field.intensities[[1, 0, 0, 1]] = 4.0;
        field.intensities[[1, 0, 0, 2]] = 2.0;

        field.update_derived_quantities();

        assert_eq!(field.total_intensity[[0, 0, 0]], 7.0);
        assert_eq!(field.total_intensity[[1, 0, 0]], 7.0);
        assert_eq!(field.peak_wavelength[[0, 0, 0]], wavelengths[1]);
        assert_eq!(field.peak_wavelength[[1, 0, 0]], wavelengths[1]);
        assert_eq!(
            field.color_temperature[[0, 0, 0]],
            WIEN_CONSTANT / wavelengths[1]
        );
        assert_eq!(
            field.color_temperature[[1, 0, 0]],
            WIEN_CONSTANT / wavelengths[1]
        );
    }
}
