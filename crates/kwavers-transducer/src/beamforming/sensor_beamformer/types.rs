//! Beamforming types: apodization window selector and array processing parameters.

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Apodization window functions for sensor array weighting.
///
/// Tapering window applied across sensor elements reduces side-lobe energy in
/// beamformed images at the cost of slightly widened main lobe.
///
/// # References
/// - Harris, F.J. (1978): "On the use of windows for harmonic analysis with the DFT."
///   *Proc. IEEE*, 66(1), 51–83.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BeamformerWindowType {
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
    pub sampling_frequency: Frequency<f64>,
    /// Mean inter-element spacing (m).
    pub element_spacing: Length<f64>,
    /// Array aperture — max-x minus min-x (m).
    pub array_aperture: Length<f64>,
}

impl SensorProcessingParams {
    /// F-number = `focal_length / array_aperture`.
    ///
    /// Dimensionless depth-of-field metric (Van Trees 2002, §2.4).
    ///
    /// # Errors
    /// Returns an error when either length is non-finite or not strictly
    /// positive, because the ratio is undefined for those inputs.
    #[must_use = "handle the validated F-number result"]
    pub fn f_number(&self, focal_length: Length<f64>) -> KwaversResult<Dimensionless<f64>> {
        let focal_length = focal_length.in_unit::<Meter>();
        let aperture = self.array_aperture.in_unit::<Meter>();
        if !focal_length.is_finite()
            || focal_length <= 0.0
            || !aperture.is_finite()
            || aperture <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "f_number requires finite positive focal length and aperture".to_owned(),
            ));
        }

        Ok(Dimensionless::from_base(focal_length / aperture))
    }

    /// Spatial Nyquist limit: maximum unambiguous frequency (Hz).
    ///
    /// From the spatial sampling theorem: f_max = c / (2 Δd)
    /// where Δd = `element_spacing`.
    ///
    /// # Errors
    /// Returns an error when sound speed or spacing is non-finite or not
    /// strictly positive, because the spatial-Nyquist ratio is undefined.
    #[must_use = "handle the validated spatial-frequency result"]
    pub fn max_spatial_frequency(
        &self,
        sound_speed: Velocity<f64>,
    ) -> KwaversResult<Frequency<f64>> {
        let sound_speed = sound_speed.in_unit::<MeterPerSecond>();
        let spacing = self.element_spacing.in_unit::<Meter>();
        if !sound_speed.is_finite() || sound_speed <= 0.0 || !spacing.is_finite() || spacing <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "max_spatial_frequency requires finite positive sound speed and spacing".to_owned(),
            ));
        }

        Ok(Frequency::from_unit::<Hertz>(sound_speed / (2.0 * spacing)))
    }
}
