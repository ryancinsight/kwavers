//! Beamforming types: apodization window selector and array processing parameters.

/// Apodization window functions for sensor array weighting.
///
/// Tapering window applied across sensor elements reduces side-lobe energy in
/// beamformed images at the cost of slightly widened main lobe.
///
/// # References
/// - Harris, F.J. (1978): "On the use of windows for harmonic analysis with the DFT."
///   *Proc. IEEE*, 66(1), 51–83.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hanning window — smooth taper, good side-lobe suppression.
    /// w(n) = 0.5 − 0.5 cos(2πn / (N−1))
    Hanning,
    /// Hamming window — similar to Hanning with slightly higher side lobes.
    /// w(n) = 0.54 − 0.46 cos(2πn / (N−1))
    Hamming,
    /// Blackman window — excellent side-lobe suppression, wider main lobe.
    /// w(n) = 0.42 − 0.5 cos(2πn / (N−1)) + 0.08 cos(4πn / (N−1))
    Blackman,
    /// Rectangular window — uniform weighting; no apodization applied.
    Rectangular,
}

/// Physical and sampling parameters derived from a sensor array.
#[derive(Debug, Clone)]
pub struct SensorProcessingParams {
    /// Number of sensors in the array.
    pub n_sensors: usize,
    /// Sampling frequency (Hz).
    pub sampling_frequency: f64,
    /// Mean inter-element spacing (m).
    pub element_spacing: f64,
    /// Array aperture — max-x minus min-x (m).
    pub array_aperture: f64,
}

impl SensorProcessingParams {
    /// F-number = `focal_length / array_aperture`.
    ///
    /// Dimensionless depth-of-field metric (Van Trees 2002, §2.4).
    #[must_use]
    pub fn f_number(&self, focal_length: f64) -> f64 {
        focal_length / self.array_aperture
    }

    /// Spatial Nyquist limit: maximum unambiguous frequency (Hz).
    ///
    /// From the spatial sampling theorem: f_max = c / (2 Δd)
    /// where Δd = `element_spacing`.
    #[must_use]
    pub fn max_spatial_frequency(&self, sound_speed: f64) -> f64 {
        sound_speed / (2.0 * self.element_spacing)
    }
}
