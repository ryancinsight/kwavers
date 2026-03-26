use ndarray::{s, Array1, Array3, Array4, Zip, Axis};

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
            ionization_energy: crate::core::constants::chemistry::ARGON_IONIZATION_ENERGY, // eV for argon
            min_temperature: 2000.0,    // K
            opacity_factor: 1.0,        // Optically thin
            cherenkov_refractive_index: 1.4,
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
    pub fn new(grid_shape: (usize, usize, usize), wavelengths: Array1<f64>) -> Self {
        let n_wavelengths = wavelengths.len();
        let shape_4d = (grid_shape.0, grid_shape.1, grid_shape.2, n_wavelengths);

        Self {
            wavelengths,
            intensities: Array4::zeros(shape_4d),
            peak_wavelength: Array3::zeros(grid_shape),
            total_intensity: Array3::zeros(grid_shape),
            color_temperature: Array3::zeros(grid_shape),
        }
    }

    /// Update derived quantities (peak wavelength, total intensity, etc.)
    /// Update derived quantities (peak wavelength, total intensity, etc.)
    pub fn update_derived_quantities(&mut self) {
        let wavelengths = &self.wavelengths;

        Zip::from(&mut self.total_intensity)
            .and(&mut self.peak_wavelength)
            .and(&mut self.color_temperature)
            .and(self.intensities.lanes(Axis(3)))
            .for_each(|total, peak, color_temp, spectrum| {
                // Total intensity
                *total = spectrum.sum();

                // Peak wavelength
                if let Some(max_idx) = spectrum
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                {
                    *peak = wavelengths[max_idx];
                }

                // Wien's displacement law: λ_peak × T = b (b = 2.898×10⁻³ m·K)
                if *peak > 0.0 {
                    use crate::core::constants::optical::WIEN_CONSTANT;
                    *color_temp = WIEN_CONSTANT / *peak;
                }
            });
    }

    /// Get spectrum at a specific point
    #[must_use]
    pub fn get_spectrum_at(&self, i: usize, j: usize, k: usize) -> crate::physics::optics::sonoluminescence::spectral::EmissionSpectrum {
        let intensities = self.intensities.slice(s![i, j, k, ..]).to_owned();
        crate::physics::optics::sonoluminescence::spectral::EmissionSpectrum::new(self.wavelengths.clone(), intensities, 0.0)
    }
}
