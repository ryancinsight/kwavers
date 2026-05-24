//! CT Hounsfield-unit thresholds and CT-to-acoustic-property model parameters.
//!
//! These constants parameterise the CT-based acoustic property assignment used in
//! transcranial, abdominal, and general CT-guided ultrasound models (Aubry 2003,
//! Connor & Hynynen 2002, Mast 2000).  They are distinct from general tissue
//! acoustic properties (see [`super::tissue_acoustics`]).

// ── HU classification thresholds ─────────────────────────────────────────────

/// Hounsfield unit threshold below which tissue is classified as soft tissue / brain [HU].
///
/// Above this value, bone fraction is interpolated linearly to cortical bone.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const HU_BONE_THRESHOLD: f64 = 300.0; // Hounsfield units

/// Hounsfield unit threshold separating in-body tissue from background air in
/// transcranial CT segmentation [HU].
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const HU_BRAIN_BODY_THRESHOLD: f64 = -300.0; // Hounsfield units

/// Hounsfield unit threshold separating abdominal organ tissue from background [HU].
///
/// Lower than `HU_BRAIN_BODY_THRESHOLD` (−300 HU) to accommodate the larger
/// perinephric / retroperitoneal fat fraction in abdominal CT scans.
///
/// Reference: AAPM Report 96 (2005); clinical consensus for liver/kidney segmentation.
pub const HU_ABDOMEN_BODY_THRESHOLD: f64 = -450.0; // Hounsfield units

// ── Skull CT interpolation model ─────────────────────────────────────────────

/// Hounsfield unit range for skull bone-fraction interpolation [HU].
///
/// `bone_fraction = (HU − HU_BONE_THRESHOLD) / HU_SKULL_RANGE`, clamped to [0, 1].
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const HU_SKULL_RANGE: f64 = 1700.0; // Hounsfield units

/// Minimum skull/trabecular-bone density at bone fraction = 0 [kg/m³].
///
/// Increases linearly to `DENSITY_SKULL_MIN + DENSITY_SKULL_CORTICAL_RANGE` = 1 900 kg/m³
/// at bone fraction = 1.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93;
/// Duck (1990) Table 3.8.
pub const DENSITY_SKULL_MIN: f64 = 1200.0; // kg/m³

/// Density increment from the brain–bone boundary to pure cortical bone [kg/m³].
///
/// `density = DENSITY_SKULL_MIN + DENSITY_SKULL_CORTICAL_RANGE × bone_fraction`.
/// Maximum density (full cortical bone) = 1 900 kg/m³.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93;
/// Duck (1990) Table 3.8.
pub const DENSITY_SKULL_CORTICAL_RANGE: f64 = 700.0; // kg/m³

// ── Pinton et al. (2012) empirical skull CT model ────────────────────────────
//
// Reference: Pinton G et al. (2012). "Attenuation, scattering, and absorption
// of ultrasound in the skull bone." *Med. Phys.* 39(1), 299–307.
// DOI: 10.1118/1.3668316.
//
// Linear model above HU_BONE_THRESHOLD:
//   c(HU) = PINTON_SKULL_SPEED_BASE + PINTON_SKULL_SPEED_SLOPE × (HU − HU_BONE_THRESHOLD)
//   ρ(HU) = PINTON_SKULL_DENSITY_BASE + PINTON_SKULL_DENSITY_SLOPE × (HU − HU_BONE_THRESHOLD)
//   α(HU) = PINTON_SKULL_ALPHA_BASE + PINTON_SKULL_ALPHA_SLOPE × (HU − HU_BONE_THRESHOLD)

/// Pinton (2012) skull sound-speed model intercept at HU = HU_BONE_THRESHOLD [m/s].
pub const PINTON_SKULL_SPEED_BASE_M_S: f64 = 3000.0;

/// Pinton (2012) skull sound-speed linear slope above bone threshold [m/s per HU].
pub const PINTON_SKULL_SPEED_SLOPE_M_S_PER_HU: f64 = 2.0;

/// Pinton (2012) skull density model intercept at HU = HU_BONE_THRESHOLD [kg/m³].
pub const PINTON_SKULL_DENSITY_BASE_KG_M3: f64 = 1800.0;

/// Pinton (2012) skull density linear slope above bone threshold [kg/m³ per HU].
pub const PINTON_SKULL_DENSITY_SLOPE_KG_M3_PER_HU: f64 = 0.5;

/// Pinton (2012) skull attenuation model intercept at HU = HU_BONE_THRESHOLD [dB/(cm·MHz)].
pub const PINTON_SKULL_ALPHA_BASE_DB_CM_MHZ: f64 = 5.0;

/// Pinton (2012) skull attenuation linear slope above bone threshold [dB/(cm·MHz) per HU].
pub const PINTON_SKULL_ALPHA_SLOPE_DB_CM_MHZ_PER_HU: f64 = 0.01;

// ── CT-derived acoustic speed ceiling ────────────────────────────────────────

/// Safety ceiling for CT-based soft-tissue sound-speed interpolation (m/s).
///
/// The linear HU→speed model for brain/soft tissue extrapolates beyond the
/// calibrated HU range [−20, 120].  A hard upper ceiling of 1620 m/s prevents
/// physically implausible values from contaminating the inversion target.
///
/// The fastest normal soft tissues measured in vivo are uterine muscle and
/// liver (1580–1610 m/s); 1620 m/s provides a 10 m/s margin before
/// classifying tissue as bone.
///
/// Reference: Duck, F.A. (1990) Table 4.6; Mast, T.D. (2000) Ultrasound Med. Biol. 26(7):1085–1099.
pub const SOUND_SPEED_SOFT_TISSUE_MAX: f64 = 1620.0; // m/s
