//! Configuration for plane wave compounding.

use aequitas::systems::si::quantities::{Angle, Dimensionless, Frequency, Length, Velocity};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Configuration for multi-angle plane wave compounding imaging.
///
/// # References
/// - Montaldo et al. (2009): "Coherent plane-wave compounding for very high frame rate."
///   *IEEE UFFC*, 56(3), 489–506.
#[derive(Debug, Clone)]
pub struct PlaneWaveCompoundingConfig {
    /// Number of plane wave insonification angles.
    pub num_angles: usize,
    /// Half-angle sweep: angles run from −angle_range to +angle_range.
    pub angle_range: Angle,
    /// Transmit frequency.
    pub frequency: Frequency,
    /// Speed of sound in the medium.
    pub sound_speed: Velocity,
    /// Aperture size.
    pub aperture_size: Length,
    /// Number of transducer elements.
    pub num_elements: usize,
    /// Inter-element pitch.
    pub element_spacing: Length,
    /// Imaging depth.
    pub depth: Length,
    /// Axial sampling interval.
    pub axial_step: Length,
    /// Lateral sampling interval.
    pub lateral_step: Length,
    /// Apodization window type: `"hann"`, `"hamming"`, `"blackman"`, or `"rect"`.
    pub apodization: String,
    /// Enable coherent (vs incoherent) compounding.
    pub coherent_compounding: bool,
    /// Log-compression dynamic range, expressed as a dimensionless dB ratio.
    pub dynamic_range: Dimensionless,
}

impl Default for PlaneWaveCompoundingConfig {
    fn default() -> Self {
        Self {
            num_angles: 11,
            angle_range: Angle::from_base(30.0_f64.to_radians()),
            frequency: Frequency::from_base(5.0 * MHZ_TO_HZ),
            sound_speed: Velocity::from_base(SOUND_SPEED_TISSUE),
            aperture_size: Length::from_base(0.04),
            num_elements: 128,
            element_spacing: Length::from_base(0.000_312_5),
            depth: Length::from_base(0.1),
            axial_step: Length::from_base(0.0005),
            lateral_step: Length::from_base(0.0005),
            apodization: "hann".to_owned(),
            coherent_compounding: true,
            dynamic_range: Dimensionless::from_base(40.0),
        }
    }
}
