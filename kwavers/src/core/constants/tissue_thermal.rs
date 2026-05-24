//! Tissue- and fluid-specific thermal properties.
//!
//! All constants are SSOT values from the standard tissue-property references.
//! Use these in Pennes bioheat, thermal diffusion, and acoustic-thermal coupling models.
//!
//! Primary references:
//! - Duck FA (1990). *Physical Properties of Tissue*. Academic Press.
//! - IT'IS Foundation (2022). Tissue Properties Database.
//! - Gordon AE et al. (2009). *Phys. Med. Biol.* 54(13), 3933–3948.

// ── Specific heat capacities [J/(kg·K)] ──────────────────────────────────────

/// Specific heat capacity of generic soft tissue (J/(kg·K)).
///
/// Reference: Duck (1990) mean for mammalian soft tissue.
pub const SPECIFIC_HEAT_TISSUE: f64 = 3600.0;

/// Specific heat capacity of human brain tissue (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1; Bhattacharya & Mahajan (2003).
pub const SPECIFIC_HEAT_BRAIN: f64 = 3630.0;

/// Specific heat capacity of skeletal muscle (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_MUSCLE: f64 = 3421.0;

/// Specific heat capacity of human liver parenchyma (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_LIVER: f64 = 3540.0;

/// Specific heat capacity of human adipose tissue (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_FAT: f64 = 2348.0;

/// Specific heat capacity of whole blood at 37°C (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1; Gordon et al. (2009).
pub const SPECIFIC_HEAT_BLOOD: f64 = 3617.0;

/// Specific heat capacity of human renal cortex at body temperature (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_KIDNEY: f64 = 3763.0;

/// Specific heat capacity of cortical bone at body temperature (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1; Hasgall et al. (2022) IT'IS Foundation.
pub const SPECIFIC_HEAT_BONE: f64 = 1313.0;

/// Specific heat capacity of human skin at body temperature (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_SKIN: f64 = 3391.0;

/// Specific heat capacity of lung parenchyma (J/(kg·K)).
///
/// Reference: Duck (1990) Table 9.1; Hasgall et al. (2022) IT'IS Foundation.
pub const SPECIFIC_HEAT_LUNG: f64 = 3886.0;

/// Specific heat capacity of breast glandular tissue (J/(kg·K)).
///
/// Reference: IT'IS Foundation database v4.0 (2022).
pub const SPECIFIC_HEAT_BREAST_GLAND: f64 = 3600.0;

/// Specific heat of brain white matter at 37°C (J/(kg·K)).
///
/// Source: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_BRAIN_WHITE: f64 = 3650.0;

/// Specific heat of brain gray matter at 37°C (J/(kg·K)).
///
/// Source: Duck (1990) Table 9.1.
pub const SPECIFIC_HEAT_BRAIN_GRAY: f64 = 3680.0;

/// Specific heat of cerebrospinal fluid at 37°C (J/(kg·K)).
///
/// Source: Duck (1990).
pub const SPECIFIC_HEAT_CSF: f64 = 3900.0;

/// Specific heat of blood plasma at 37°C (J/(kg·K)).
///
/// Source: Duck (1990); Gordon et al. (2009).
pub const SPECIFIC_HEAT_BLOOD_PLASMA: f64 = 3840.0;

/// Specific heat capacity of urine at 37°C (J/(kg·K)).
///
/// Reference: Duck FA (1990). Physical Properties of Tissue, Table 9.1.
pub const SPECIFIC_HEAT_URINE: f64 = 3680.0;

/// Specific heat capacity of ultrasound coupling gel (J/(kg·K)).
///
/// Reference: Perry & Green (2007) Chemical Engineering Handbook.
pub const SPECIFIC_HEAT_ULTRASOUND_GEL: f64 = 3300.0;

/// Specific heat capacity of mineral oil (J/(kg·K)).
///
/// Reference: Perry & Green (2007).
pub const SPECIFIC_HEAT_MINERAL_OIL: f64 = 2100.0;

/// Specific heat capacity of microbubble ultrasound contrast agent [J/(kg·K)].
///
/// Reference: Stride E & Saffari N (2003). *Proc. Inst. Mech. Eng. H* 217(6):429–447.
pub const SPECIFIC_HEAT_MICROBUBBLE_SUSPENSION: f64 = 4170.0;

/// Specific heat capacity of iron-oxide nanoparticle suspension [J/(kg·K)].
///
/// Reference: Stride E & Saffari N (2003). *Proc. Inst. Mech. Eng. H* 217(6):429–447.
pub const SPECIFIC_HEAT_NANOPARTICLE_SUSPENSION: f64 = 4150.0;

// ── Thermal conductivities [W/(m·K)] ─────────────────────────────────────────

/// Thermal conductivity of tissue (W/(m·K)).
pub const THERMAL_CONDUCTIVITY_TISSUE: f64 = 0.5;

/// Thermal conductivity of whole blood at 37°C and hematocrit 45% (W/(m·K)).
///
/// Reference: Gordon et al. (2009). Phys. Med. Biol. 54(13), 3933–3948;
/// Duck, F.A. (1990) Table 9.1.
pub const THERMAL_CONDUCTIVITY_BLOOD: f64 = 0.52;

/// Thermal conductivity of blood plasma at 37°C [W/(m·K)].
///
/// Reference: Duck, F.A. (1990) Table 9.1; Cobbold (2007) Table 3.3.
pub const THERMAL_CONDUCTIVITY_BLOOD_PLASMA: f64 = 0.55;

/// Thermal conductivity of microbubble ultrasound contrast agent [W/(m·K)].
///
/// Reference: Stride E & Saffari N (2003). *Proc. Inst. Mech. Eng. H* 217(6):429–447.
pub const THERMAL_CONDUCTIVITY_MICROBUBBLE_SUSPENSION: f64 = 0.60;

/// Thermal conductivity of iron-oxide nanoparticle suspension [W/(m·K)].
///
/// Reference: Stride E & Saffari N (2003). *Proc. Inst. Mech. Eng. H* 217(6):429–447.
pub const THERMAL_CONDUCTIVITY_NANOPARTICLE_SUSPENSION: f64 = 0.59;

/// Thermal conductivity of brain white matter at 37°C (W/(m·K)).
///
/// Source: Duck (1990) Table 9.1; IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_BRAIN: f64 = 0.50;

/// Thermal conductivity of brain gray matter at 37°C (W/(m·K)).
///
/// Source: Duck (1990) Table 9.1; IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_BRAIN_GRAY: f64 = 0.52;

/// Thermal conductivity of cortical skull bone at 37°C (W/(m·K)).
///
/// Source: Duck (1990) Table 9.1.
pub const THERMAL_CONDUCTIVITY_SKULL: f64 = 0.40;

/// Thermal conductivity of liver at 37°C (W/(m·K)).
///
/// Source: Duck (1990) Table 9.1; IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_LIVER: f64 = 0.56;

/// Thermal conductivity of kidney at 37°C (W/(m·K)).
///
/// Source: IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_KIDNEY: f64 = 0.50;

/// Thermal conductivity of skeletal muscle at 37°C (W/(m·K)).
///
/// Source: IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_MUSCLE: f64 = 0.49;

/// Thermal conductivity of adipose (fat) tissue at 37°C (W/(m·K)).
///
/// Source: IT'IS Foundation v4.1.
pub const THERMAL_CONDUCTIVITY_FAT: f64 = 0.21;

/// Thermal conductivity of cerebrospinal fluid at 37°C (W/(m·K)).
///
/// Source: Duck (1990).
pub const THERMAL_CONDUCTIVITY_CSF: f64 = 0.60;

/// Thermal conductivity of urine at 37°C (W/(m·K)).
///
/// Reference: Duck FA (1990). Physical Properties of Tissue.
pub const THERMAL_CONDUCTIVITY_URINE: f64 = 0.61;

/// Thermal conductivity of ultrasound coupling gel (W/(m·K)).
///
/// Reference: Perry & Green (2007).
pub const THERMAL_CONDUCTIVITY_ULTRASOUND_GEL: f64 = 0.15;

/// Thermal conductivity of mineral oil (W/(m·K)).
///
/// Reference: Perry & Green (2007).
pub const THERMAL_CONDUCTIVITY_MINERAL_OIL: f64 = 0.14;

// ── Thermal diffusivities [m²/s] ──────────────────────────────────────────────

/// Thermal diffusivity of water (m²/s).
pub const THERMAL_DIFFUSIVITY_WATER: f64 = 1.43e-7;

/// Thermal diffusivity of tissue (m²/s).
///
/// Reference: Duck, F.A. (1990). Physical Properties of Tissue, Table 9.1.
pub const THERMAL_DIFFUSIVITY_TISSUE: f64 = 1.36e-7;

/// Thermal diffusivity of whole blood at 37°C and hematocrit 45% (m²/s).
///
/// Directly measured value. Reference: Duck, F.A. (1990) Table 9.1.
pub const THERMAL_DIFFUSIVITY_BLOOD: f64 = 1.35e-7;

// ── Tissue temperature-coupling coefficients ──────────────────────────────────

/// Temperature coefficient of sound speed in soft tissue near 37°C (m/s per °C).
///
/// Measured range: 1.0–2.5 m/s/°C (Bamber & Hill 1979, Lynch 1988).
pub const DC_DT_SOFT_TISSUE: f64 = 2.0;

/// Temperature coefficient of density in soft tissue near 37°C (kg/m³ per °C).
///
/// Reference: NIST thermophysical data; Duck (1990) p. 119.
pub const DRHO_DT_SOFT_TISSUE: f64 = -0.2;

/// Volumetric heat capacity of soft tissue at 37°C (J per m³ per °C).
///
/// `ρ c_p = DENSITY_TISSUE(1050) × SPECIFIC_HEAT_TISSUE(3600) = 3 780 000 J/(m³·°C)`.
///
/// **WARNING — common dimensional error**: use this constant (not `SPECIFIC_HEAT_TISSUE` alone)
/// as the bioheat volumetric coefficient in the Pennes equation.
///
/// Reference: Pennes (1948) J. Appl. Physiol. 1(2):93–122; Duck (1990) pp. 147–151.
pub const RHO_C_SOFT_TISSUE: f64 = 3_780_000.0;

/// Isobaric thermal expansion coefficient of generic soft tissue at 37°C (K⁻¹).
///
/// Reference: Duck FA (1990). Physical Properties of Tissue. Academic Press.
pub const THERMAL_EXPANSION_SOFT_TISSUE: f64 = 3.0e-4;
