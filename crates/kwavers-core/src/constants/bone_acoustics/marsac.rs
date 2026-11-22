//! Linear soft-tissue and skull mixtures for the Marsac CT model.

use aequitas::systems::si::quantities::{Frequency, Velocity};

use crate::constants::acoustic_parameters::DB_TO_NP;
use crate::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, SOUND_SPEED_WATER_SIM};
use crate::constants::numerical::{CM_TO_M, MHZ_TO_HZ};

use super::reference::{AttenuationSlope, SOUND_SPEED_SKULL};

/// Skull-bone attenuation upper bound in the Marsac et al. 2017 linear porosity model [Np/(m·MHz)].
///
/// The Marsac porosity-blend formula interpolates between soft tissue and this value:
/// ```text
/// α(φ) = α_soft·(1 − φ) + 70·φ   [Np/(m·MHz)]
/// ```
/// At φ = 1 (fully cortical bone), α = 70 Np/(m·MHz) ≈ 6.1 dB/(cm·MHz).
///
/// Conversion: α[dB/(cm·MHz)] = 70 × `NP_TO_DB` / 100 ≈ 6.08.
///
/// Reference: Marsac L et al. (2017). "MR-guided adaptive focusing of therapeutic
/// ultrasound beams in the human head." *Med. Phys.* 39(2), 1141–1149.
/// DOI: 10.1002/mp.12168.
pub const SKULL_MARSAC_ATTENUATION_SLOPE: AttenuationSlope =
    AttenuationSlope::from_base(70.0 / MHZ_TO_HZ);

/// Reference frequency used by the linear skull/soft-tissue CT model.
pub const MARSAC_REFERENCE_FREQUENCY: Frequency = Frequency::from_base(MHZ_TO_HZ);

/// Linear soft-tissue amplitude loss slope in the Marsac model.
pub const MARSAC_SOFT_TISSUE_ATTENUATION_SLOPE: AttenuationSlope =
    AttenuationSlope::from_base(ACOUSTIC_ABSORPTION_TISSUE * DB_TO_NP / CM_TO_M / MHZ_TO_HZ);

/// Linear mixture of water and skull reference velocities. The caller supplies
/// the bone fraction in `[0, 1]` from CT calibration.
#[inline]
#[must_use]
pub fn marsac_skull_sound_speed(bone_fraction: f64) -> Velocity {
    Velocity::from_base(SOUND_SPEED_WATER_SIM) * (1.0 - bone_fraction)
        + SOUND_SPEED_SKULL * bone_fraction
}

/// Linear mixture of the two amplitude loss slopes for a bone fraction in
/// `[0, 1]`. Multiply by frequency to obtain reciprocal length.
#[inline]
#[must_use]
pub fn marsac_skull_attenuation_slope(bone_fraction: f64) -> AttenuationSlope {
    MARSAC_SOFT_TISSUE_ATTENUATION_SLOPE * (1.0 - bone_fraction)
        + SKULL_MARSAC_ATTENUATION_SLOPE * bone_fraction
}
