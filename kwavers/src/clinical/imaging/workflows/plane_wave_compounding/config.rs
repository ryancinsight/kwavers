//! Configuration for plane wave compounding.

/// Configuration for multi-angle plane wave compounding imaging.
///
/// # References
/// - Montaldo et al. (2009): "Coherent plane-wave compounding for very high frame rate."
///   *IEEE UFFC*, 56(3), 489–506.
#[derive(Debug, Clone)]
pub struct PlaneWaveCompoundingConfig {
    /// Number of plane wave insonification angles.
    pub num_angles: usize,
    /// Half-angle sweep: angles run from −angle_range to +angle_range (degrees).
    pub angle_range: f64,
    /// Transmit frequency (Hz).
    pub frequency: f64,
    /// Speed of sound in the medium (m/s).
    pub sound_speed: f64,
    /// Aperture size (m).
    pub aperture_size: f64,
    /// Number of transducer elements.
    pub num_elements: usize,
    /// Inter-element pitch (m).
    pub element_spacing: f64,
    /// Imaging depth (m).
    pub depth: f64,
    /// Axial sampling interval (m).
    pub axial_step: f64,
    /// Lateral sampling interval (m).
    pub lateral_step: f64,
    /// Apodization window type: `"hann"`, `"hamming"`, `"blackman"`, or `"rect"`.
    pub apodization: String,
    /// Enable coherent (vs incoherent) compounding.
    pub coherent_compounding: bool,
    /// Log-compression dynamic range (dB).
    pub dynamic_range: f64,
}

impl Default for PlaneWaveCompoundingConfig {
    fn default() -> Self {
        Self {
            num_angles: 11,
            angle_range: 30.0,
            frequency: 5e6,
            sound_speed: 1540.0,
            aperture_size: 0.04,
            num_elements: 128,
            element_spacing: 0.000_312_5,
            depth: 0.1,
            axial_step: 0.0005,
            lateral_step: 0.0005,
            apodization: "hann".to_owned(),
            coherent_compounding: true,
            dynamic_range: 40.0,
        }
    }
}
