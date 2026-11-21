//! Bone and skull acoustic reference quantities.

use aequitas::systems::si::quantities::{Frequency, MassDensity, ReciprocalLength, Velocity};
use std::ops::Div;

use super::acoustic_parameters::DB_TO_NP;
use super::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, SOUND_SPEED_WATER_SIM};
use super::numerical::{CM_TO_M, MHZ_TO_HZ};

/// Pressure-amplitude attenuation per unit distance per unit frequency.
/// A power law with exponent other than one requires a reference loss and an exponent.
pub type AttenuationSlope = <ReciprocalLength as Div<Frequency>>::Output;

/// Sound speed in dense cortical bone (m/s)
///
/// Value: 3500 m/s — compressional wave speed in bovine cortical bone.
///
/// Reference: Hosokawa & Otani (1997). "Ultrasonic wave propagation in bovine cancellous bone."
/// J. Acoust. Soc. Am. 101(1), 558–562. DOI: 10.1121/1.418118.
pub const BONE_SOUND_SPEED: Velocity = Velocity::from_base(3500.0);

/// Sound speed in human skull bone (m/s)
///
/// Value: 2900 m/s — effective longitudinal wave speed in the human skull,
/// which has a layered diploe structure (inner cortex + cancellous diploe +
/// outer cortex) that produces a lower effective speed than dense cortical bone.
///
/// Used in transcranial ultrasound propagation models where skull bone is the
/// primary acoustic barrier. Distinct from `BONE_SOUND_SPEED` (3500 m/s for
/// dense cortical), which applies to long bones with minimal cancellous content.
///
/// References:
/// - Fry & Barger (1978). "Acoustical properties of the human skull."
///   J. Acoust. Soc. Am. 63(5), 1576–1590. DOI: 10.1121/1.381852.
/// - Marsac et al. (2017). "MR-guided adaptive focusing of therapeutic
///   ultrasound beams in the human head." Med. Phys. 39(2), 1141–1149.
pub const SOUND_SPEED_SKULL: Velocity = Velocity::from_base(2900.0);

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

/// Bone density (kg/m³)
/// Value: 1900 kg/m³
/// Reference: Duck, F. A. (1990). "Physical properties of tissue"
pub const BONE_DENSITY: MassDensity = MassDensity::from_base(1900.0);

/// Skull (cortical bone) density (kg/m³).
///
/// Cortical bone of the human skull is slightly denser than the generic bone
/// value (BONE_DENSITY = 1900 kg/m³) due to its compact lamellar structure.
/// Value: 1920 kg/m³.
/// Reference: Duck (1990) Table 3.3.
pub const DENSITY_SKULL: MassDensity = MassDensity::from_base(1920.0);

/// Skull cortical bone longitudinal sound speed [m/s] — Pinton et al. (2012) model.
///
/// Value: 3100 m/s — measured longitudinal wave speed for dense cortical skull bone
/// at body temperature. Used in the 3-layer skull model (cortical/trabecular/cortical)
/// for transcranial focused ultrasound.
///
/// Distinct from `SOUND_SPEED_SKULL = 2900 m/s` (Marsac 2017 mean across porosity range).
///
/// Reference: Pinton G et al. (2012). "Attenuation, scattering, and absorption of
/// ultrasound in the skull bone." *Med. Phys.* 39(1):299–307. DOI: 10.1118/1.3668316.
pub const SOUND_SPEED_SKULL_CORTICAL: Velocity = Velocity::from_base(3100.0);

/// Skull cortical bone shear wave speed [m/s].
///
/// Value: 1600 m/s — measured shear wave speed in cortical skull bone.
/// Range reported in the literature: 1400–1800 m/s depending on orientation and
/// bone quality.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SHEAR_SPEED_SKULL_CORTICAL: Velocity = Velocity::from_base(1600.0);

/// Skull trabecular bone longitudinal sound speed [m/s].
///
/// Value: 2400 m/s — longitudinal speed in the porous trabecular (diploe) layer of
/// the skull, which is slower than compact cortical bone due to the higher porosity.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SOUND_SPEED_SKULL_TRABECULAR: Velocity = Velocity::from_base(2400.0);

/// Skull trabecular bone density [kg/m³].
///
/// Value: 1600 kg/m³ — lower than cortical skull (1920 kg/m³) due to trabecular
/// porosity. Used in the 3-layer skull transmission model.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const DENSITY_SKULL_TRABECULAR: MassDensity = MassDensity::from_base(1600.0);

/// Skull trabecular bone shear wave speed [m/s].
///
/// Value: 1200 m/s — lower than cortical shear speed due to trabecular porosity.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SHEAR_SPEED_SKULL_TRABECULAR: Velocity = Velocity::from_base(1200.0);

/// Skull suture (fibrous joint) longitudinal sound speed [m/s].
///
/// Cranial sutures are connective tissue joints between skull bones; their acoustic
/// speed is intermediate between soft tissue (~1540 m/s) and trabecular bone (~2400 m/s).
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307;
/// Marquet F et al. (2009). *Phys. Med. Biol.* 54(9):2895–2916.
pub const SOUND_SPEED_SKULL_SUTURE: Velocity = Velocity::from_base(1800.0);

/// Skull suture (fibrous joint) density [kg/m³].
///
/// Suture tissue is predominantly fibrous connective tissue; density is close to
/// soft tissue (~1040–1100 kg/m³) but elevated by mineralisation at suture margins.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const DENSITY_SKULL_SUTURE: MassDensity = MassDensity::from_base(1200.0);

/// Bone nonlinearity parameter (B/A)
/// Value: 8.0
/// Reference: Estimated from tissue properties
pub const BONE_NONLINEARITY: f64 = 8.0;

/// Bone attenuation coefficient [dB/(MHz·cm)]
/// Value: 20.0
/// Reference: Wear, K. A. (2000). "Measurements of phase velocity and group velocity in bone"
pub const BONE_ATTENUATION_SLOPE: AttenuationSlope =
    AttenuationSlope::from_base(20.0 * DB_TO_NP / CM_TO_M / MHZ_TO_HZ);

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
