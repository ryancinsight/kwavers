#[cfg(test)]
mod tests {
    use super::super::range::SpectralRange;
    use super::super::spectrum::EmissionSpectrum;
    use ndarray::Array1;

    #[test]
    fn test_spectral_range() {
        let range = SpectralRange::default();
        let wavelengths = range.wavelengths();

        assert_eq!(wavelengths.len(), range.n_points);
        assert_eq!(wavelengths[0], range.lambda_min);
        assert_eq!(wavelengths[wavelengths.len() - 1], range.lambda_max);
    }

    #[test]
    fn test_emission_spectrum() {
        let wavelengths = Array1::linspace(400e-9, 700e-9, 100);
        let intensities = wavelengths.mapv(|lambda| {
            // Gaussian peak at 550 nm
            let center: f64 = 550e-9;
            let sigma: f64 = 50e-9;
            (-(lambda - center).powi(2) / (2.0 * sigma.powi(2))).exp()
        });

        let spectrum = EmissionSpectrum::new(wavelengths, intensities, 0.0);

        let peak = spectrum.peak_wavelength();
        assert!((peak - 550e-9).abs() < 10e-9); // Peak should be near 550 nm

        let fwhm = spectrum.fwhm();
        assert!(fwhm > 0.0 && fwhm < 200e-9); // FWHM should be reasonable
    }

    #[test]
    fn test_wavelength_to_rgb() {
        // Test some known wavelengths
        let (r, g, b) = SpectralRange::wavelength_to_rgb(650e-9); // Deep red
        assert!(r > 0.8 && g < 0.2 && b < 0.2, "Red: ({}, {}, {})", r, g, b);

        let (r, g, b) = SpectralRange::wavelength_to_rgb(520e-9); // Green
        assert!(
            r < 0.3 && g > 0.7 && b < 0.3,
            "Green: ({}, {}, {})",
            r,
            g,
            b
        );

        let (r, g, b) = SpectralRange::wavelength_to_rgb(450e-9); // Blue
        assert!(r < 0.2 && g < 0.3 && b > 0.8, "Blue: ({}, {}, {})", r, g, b);
    }
}
