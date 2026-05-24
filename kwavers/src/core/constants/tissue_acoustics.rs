//! Tissue- and fluid-specific acoustic properties.
//!
//! All constants are SSOT values drawn from the primary tissue-property compilations.
//! Use these in tissue-specific factory methods and acoustic-model constructors.
//!
//! Primary references:
//! - Duck FA (1990). *Physical Properties of Tissue: A Comprehensive Reference Book*.
//!   Academic Press, London. ISBN 0-12-222800-6.
//! - Goss SA, Johnston RL, Dunn F (1978). "Comprehensive compilation of empirical
//!   ultrasonic properties of mammalian tissues." J. Acoust. Soc. Am. 64(2), 423–457.
//! - Bjørnø L (2002). "Forty years of nonlinear ultrasound." Ultrasonics 40, 11–17.
//! - IT'IS Foundation (2022). Tissue Properties Database.

// ── Tissue / fluid densities ─────────────────────────────────────────────────

/// Density of human liver parenchyma at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; ICRP-23 Reference Man Table 22.
pub const DENSITY_LIVER: f64 = 1060.0;

/// Density of human renal cortex at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1.
pub const DENSITY_KIDNEY: f64 = 1050.0;

/// Density of human renal medulla at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1.
pub const DENSITY_KIDNEY_MEDULLA: f64 = 1055.0;

/// Density of human brain (mean of white + grey matter) at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; ICRP-89 (2002) Table 4.4.
pub const DENSITY_BRAIN: f64 = 1040.0;

/// Density of cerebrospinal fluid at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1.
pub const DENSITY_CSF: f64 = 1007.0;

/// Density of human adipose tissue at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; Goss et al. (1978) Table II.
pub const DENSITY_FAT: f64 = 928.0;

/// Density of whole blood at 37°C (kg/m³).
///
/// Reference: ICRP Publication 23 (1975) Table 22; Duck (1990) p. 119.
pub const DENSITY_BLOOD: f64 = 1060.0;

/// Density of breast adipose tissue at body temperature (kg/m³).
///
/// Reference: IT'IS Foundation (2018); Hasgall et al. (2022).
pub const DENSITY_BREAST_FAT: f64 = 911.0;

/// Density of skeletal muscle at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; IT'IS Foundation (2018); ICRP-89 (2002) Table 4.4.
pub const DENSITY_MUSCLE: f64 = 1090.0;

/// Density of air at 20°C (kg/m³).
///
/// Reference: NIST Standard Reference Database.
pub const DENSITY_AIR: f64 = 1.204;

/// Density of urine at body temperature (kg/m³).
///
/// Reference: Duck (1990) physical property tables.
pub const DENSITY_URINE: f64 = 1005.0;

/// Density of commercial ultrasound gel (kg/m³).
///
/// Reference: Perry & Green (2007), acoustic coupling material properties.
pub const DENSITY_ULTRASOUND_GEL: f64 = 1020.0;

/// Density of mineral oil at room/body coupling conditions (kg/m³).
///
/// Reference: Perry & Green (2007), liquid mineral-oil property tables.
pub const DENSITY_MINERAL_OIL: f64 = 870.0;

/// Effective density of an ultrasound microbubble suspension (kg/m³).
///
/// Reference: Stride & Saffari (2003), dilute contrast-agent suspensions.
pub const DENSITY_MICROBUBBLE_SUSPENSION: f64 = 1010.0;

/// Density of human skin at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; IT'IS Foundation (2022).
pub const DENSITY_SKIN: f64 = 1100.0;

/// Density of human lung parenchyma at total lung capacity (kg/m³).
///
/// Much lower than other soft tissues due to high air fraction (~70% air by volume).
///
/// Reference: Duck (1990) Table 4.1; ICRP-89 (2002).
pub const DENSITY_LUNG: f64 = 400.0;

/// Density of human breast glandular tissue at body temperature (kg/m³).
///
/// Reference: IT'IS Foundation database v4.0 (2022); Hasgall et al. (2022).
pub const DENSITY_BREAST_GLAND: f64 = 1041.0;

// ── Tissue / fluid sound speeds ───────────────────────────────────────────────

/// Small-signal sound speed in human liver parenchyma at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_LIVER: f64 = 1578.0;

/// Small-signal sound speed in human renal cortex at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_KIDNEY: f64 = 1560.0;

/// Small-signal sound speed in human renal medulla at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_KIDNEY_MEDULLA: f64 = 1565.0;

/// Small-signal sound speed in human brain (mean grey + white matter) at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goldman & Hueter (1956).
pub const SOUND_SPEED_BRAIN: f64 = 1546.0;

/// Small-signal sound speed in human brain grey matter at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_BRAIN_GRAY_MATTER: f64 = 1545.0;

/// Small-signal sound speed in cerebrospinal fluid at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_CSF: f64 = 1515.0;

/// Small-signal sound speed in human adipose tissue at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_FAT: f64 = 1450.0;

/// Small-signal sound speed in whole blood at body temperature (m/s).
///
/// Reference: Duck (1990) Physical Properties of Tissue, Table 4.6, p. 100.
pub const SOUND_SPEED_BLOOD: f64 = 1584.0;

/// Small-signal sound speed in skeletal muscle at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_MUSCLE: f64 = 1580.0;

/// Small-signal sound speed in urine at body temperature (m/s).
///
/// Reference: Duck (1990) physical property tables.
pub const SOUND_SPEED_URINE: f64 = 1541.0;

/// Small-signal sound speed in commercial ultrasound gel (m/s).
///
/// Reference: Perry & Green (2007), acoustic coupling material properties.
pub const SOUND_SPEED_ULTRASOUND_GEL: f64 = 1550.0;

/// Small-signal sound speed in mineral oil (m/s).
///
/// Reference: Perry & Green (2007), liquid mineral-oil property tables.
pub const SOUND_SPEED_MINERAL_OIL: f64 = 1450.0;

/// Effective small-signal sound speed in a nanoparticle suspension (m/s).
///
/// Reference: Stride & Saffari (2003), water-based theranostic suspensions.
pub const SOUND_SPEED_NANOPARTICLE_SUSPENSION: f64 = 1490.0;

/// Small-signal sound speed in human skin at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_SKIN: f64 = 1600.0;

/// Small-signal sound speed in lung parenchyma at body temperature (m/s).
///
/// Dramatically reduced compared to soft tissue due to gas-liquid mixture.
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_LUNG: f64 = 650.0;

/// Small-signal sound speed in breast glandular tissue at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Mast (2000) Ultrasound Med. Biol.
pub const SOUND_SPEED_BREAST_GLAND: f64 = 1510.0;

// ── B/A nonlinearity parameters ───────────────────────────────────────────────

/// Nonlinearity parameter B/A for human liver parenchyma (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_LIVER: f64 = 6.75;

/// Nonlinearity parameter B/A for human renal cortex (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_KIDNEY: f64 = 7.2;

/// Nonlinearity parameter B/A for human brain (mean grey + white matter, dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Law et al. (1985) UMB 11(2), 307–318.
pub const B_OVER_A_BRAIN: f64 = 6.55;

/// Nonlinearity parameter B/A for human adipose tissue (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_FAT: f64 = 9.6;

/// Nonlinearity parameter B/A for skeletal muscle (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_MUSCLE: f64 = 7.4;

/// Nonlinearity parameter B/A for water at 20°C (dimensionless).
///
/// Reference: Duck, F.A. (1990) Table 4.16; Beyer, R.T. (1960) J. Acoust. Soc. Am. 32(6).
pub const B_OVER_A_WATER: f64 = 5.2;

/// Nonlinearity parameter B/A for water at 37°C / body temperature (dimensionless).
///
/// Reference: Aanonsen et al. (1984) J. Acoust. Soc. Am. 75(3):749–768;
/// Duck FA (1990) Table 4.16.
pub const B_OVER_A_WATER_37C: f64 = 5.0;

/// Nonlinearity parameter B/A for whole blood at 37°C (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_BLOOD: f64 = 6.1;

/// Nonlinearity parameter B/A for cerebrospinal fluid at body temperature (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_CSF: f64 = 5.0;

/// Nonlinearity parameter B/A for cortical bone (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002).
pub const B_OVER_A_BONE: f64 = 8.0;

/// Nonlinearity parameter B/A for human skin (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_SKIN: f64 = 7.5;

/// Nonlinearity parameter B/A for lung parenchyma (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_LUNG: f64 = 8.0;

/// Nonlinearity parameter B/A for breast glandular tissue (dimensionless).
///
/// Reference: IT'IS Foundation; Duck (1990) Table 4.16.
pub const B_OVER_A_BREAST_GLAND: f64 = 7.0;

/// Nonlinearity parameter B/A for generic soft tissue (dimensionless).
///
/// Value: 6.5 — representative mean for soft tissues excluding bone and fat.
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.16.
pub const B_OVER_A_SOFT_TISSUE: f64 = 6.5;

/// Nonlinearity parameter B/A for air (ideal diatomic gas, dimensionless).
///
/// For a thermally perfect diatomic gas: B/A = γ − 1 = 1.4 − 1 = 0.4.
///
/// Reference: Hamilton MF & Blackstock DT (1998). *Nonlinear Acoustics*, Chapter 2.
pub const B_OVER_A_AIR: f64 = 0.4;

/// Acoustic nonlinearity parameter B/A for urine (37°C, dimensionless).
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.16.
pub const B_OVER_A_URINE: f64 = 5.1;

/// Acoustic nonlinearity parameter B/A for mineral oil (dimensionless).
///
/// Reference: Perry's Chemical Engineers' Handbook (2007), acoustic properties section.
pub const B_OVER_A_MINERAL_OIL: f64 = 4.5;

/// Acoustic nonlinearity parameter B/A for iron-oxide nanoparticle suspension (dimensionless).
///
/// Reference: Stride E & Saffari N (2003). *Proc. Inst. Mech. Eng. H* 217(6):429–447.
pub const B_OVER_A_NANOPARTICLE_SUSPENSION: f64 = 5.3;

// ── Acoustic absorption power-law parameters ──────────────────────────────────

/// Power-law absorption coefficient prefactor for water [dB/(cm·MHz²)].
///
/// α(f) = α₀·f^y with y = 2.0 (classical thermoviscous loss).
///
/// Reference: Duck (1990) §2; Szabo (2004) §4.2; Kinsler et al. (2000) Table B.1.
pub const WATER_ABSORPTION_ALPHA_0_DB_CM_MHZ2: f64 = 0.002; // dB/(cm·MHz²)

/// Power-law frequency exponent y for water acoustic absorption (dimensionless).
///
/// Reference: Duck (1990) §2; Szabo (2004) §4.
pub const WATER_ABSORPTION_POWER_Y: f64 = 2.0;

/// Power-law frequency exponent y for soft tissue acoustic absorption (dimensionless).
///
/// Reference: Duck (1990) Table 4.1; Szabo (2004) §4; Goss et al. (1978).
pub const SOFT_TISSUE_ABSORPTION_POWER_Y: f64 = 1.1;

// ── Tissue-specific acoustic absorption coefficients ─────────────────────────

/// Acoustic absorption coefficient of human brain tissue [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Aubry et al. (2003) J. Acoust. Soc. Am. 113(1):84–93.
pub const ACOUSTIC_ABSORPTION_BRAIN: f64 = 0.5; // dB/(cm·MHz)

/// Minimum skull bone acoustic absorption coefficient [dB/(cm·MHz)].
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93, Table I.
pub const ACOUSTIC_ABSORPTION_SKULL_MIN: f64 = 8.0; // dB/(cm·MHz)

/// Acoustic absorption range across skull bone fraction [dB/(cm·MHz)].
///
/// Full-bone absorption = ACOUSTIC_ABSORPTION_SKULL_MIN + ACOUSTIC_ABSORPTION_SKULL_RANGE = 20 dB/(cm·MHz).
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93, Table I.
pub const ACOUSTIC_ABSORPTION_SKULL_RANGE: f64 = 12.0; // dB/(cm·MHz)

/// Minimum cortical-bone acoustic absorption for CT-based HIFU planning [dB/(cm·MHz)].
///
/// Reference: Connor CW & Hynynen K (2002). *Phys. Med. Biol.* 47(12):2213–2231.
pub const ACOUSTIC_ABSORPTION_SKULL_CORTICAL_MIN: f64 = 13.0; // dB/(cm·MHz)

/// Acoustic absorption coefficient of brain white matter [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_BRAIN_WHITE: f64 = 0.6; // dB/(cm·MHz)

/// Acoustic absorption coefficient of brain gray matter [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_BRAIN_GRAY: f64 = 0.7; // dB/(cm·MHz)

/// Bulk acoustic absorption coefficient of skull bone [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_SKULL_BULK: f64 = 3.0; // dB/(cm·MHz)

/// Acoustic absorption coefficient of long-bone cortical bone [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Hosokawa A & Otani T (1997) J. Acoust. Soc. Am. 101(1):558–562.
pub const ACOUSTIC_ABSORPTION_CORTICAL_BONE: f64 = 4.0; // dB/(cm·MHz)

/// Acoustic absorption coefficient of liver tissue [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Goss et al. (1978) J. Acoust. Soc. Am. 64(2):423–457.
pub const ACOUSTIC_ABSORPTION_LIVER: f64 = 0.5; // dB/(cm·MHz)

/// Acoustic absorption coefficient of blood [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Cobbold (2007) Table 3.3.
pub const ACOUSTIC_ABSORPTION_BLOOD: f64 = 0.15; // dB/(cm·MHz)

/// Acoustic absorption coefficient of skeletal muscle, longitudinal direction [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Goss et al. (1978) J. Acoust. Soc. Am. 64(2):423–457.
pub const ACOUSTIC_ABSORPTION_MUSCLE: f64 = 0.57; // dB/(cm·MHz)

/// Acoustic absorption coefficient of skeletal muscle, transverse direction [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Hamilton & Blackstock (1998) Table 8.1.
pub const ACOUSTIC_ABSORPTION_MUSCLE_TRANSVERSE: f64 = 1.15; // dB/(cm·MHz)

/// Acoustic absorption coefficient of renal cortex [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_KIDNEY_CORTEX: f64 = 0.81; // dB/(cm·MHz)

/// Acoustic absorption coefficient of adipose (fat) tissue [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_FAT: f64 = 0.48; // dB/(cm·MHz)

/// Acoustic absorption coefficient of blood plasma [dB/(cm·MHz^y)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Gordon et al. (2009).
pub const ACOUSTIC_ABSORPTION_BLOOD_PLASMA: f64 = 0.015; // dB/(cm·MHz^y)

/// Acoustic absorption coefficient of whole blood (hematocrit 45%) [dB/(cm·MHz^y)].
///
/// Reference: Duck, F.A. (1990) Table 4.1; Gordon et al. (2009).
pub const ACOUSTIC_ABSORPTION_WHOLE_BLOOD: f64 = 0.025; // dB/(cm·MHz^y)

/// Acoustic absorption coefficient of cerebrospinal fluid [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_CSF: f64 = 0.008; // dB/(cm·MHz)

/// Acoustic absorption coefficient of urine [dB/(cm·MHz)].
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1.
pub const ACOUSTIC_ABSORPTION_URINE: f64 = 0.012; // dB/(cm·MHz)
