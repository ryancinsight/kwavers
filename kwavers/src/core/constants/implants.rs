//! Physical constants for biomedical implant materials
//!
//! Provides thermal, acoustic, and mechanical properties for engineering
//! materials used in surgical implants. These are material-specific constants
//! (SSOT), distinct from biological tissue properties in `thermodynamic.rs`.
//!
//! # Sources
//!
//! - ISO 5832 — Metallic materials for surgical implants
//! - ASTM F136 — Ti-6Al-4V for surgical implants
//! - ASTM F139 — Stainless steel 316L
//! - ASTM F216 — Platinum and platinum alloys
//! - ASTM F451 — Acrylic bone cement (PMMA)
//! - ASTM F648 — UHMWPE for joint replacement
//! - ASTM F381 — Silicone rubber
//! - ASTM F1634 — Polyurethane
//! - ASTM F603 — Alumina (Al₂O₃) ceramics
//! - ASTM F1873 — Zirconia (ZrO₂) ceramics
//! - ASTM F1185 — Hydroxyapatite (HA)
//! - ASTM E2748 — Carbon fibre reinforced polymer (CFRP)
//! - Perry & Green (2007) — Chemical Engineering Handbook (polymer data)

// ============================================================================
// Acoustic Absorption Coefficients [dB/(cm·MHz^y)]
// ============================================================================
//
// Sources: Szabo TL (2004) *Diagnostic Ultrasound Imaging*. Elsevier, Table A2.
//          Duck FA (1990) *Physical Properties of Tissue*. Academic Press.
//          ASTM material handbooks cited per material.

/// Titanium Grade 5 acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; Ti alloy bulk attenuation ≈ 0.1.
pub const ACOUSTIC_ABSORPTION_TITANIUM: f64 = 0.1;

/// Stainless steel 316L acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; austenitic steel ≈ 0.15.
pub const ACOUSTIC_ABSORPTION_STAINLESS_STEEL_316L: f64 = 0.15;

/// Platinum acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; dense noble metal ≈ 0.5.
pub const ACOUSTIC_ABSORPTION_PLATINUM: f64 = 0.5;

/// PMMA (acrylic bone cement) acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Duck (1990) p. 92; Szabo (2004) Table A2.
pub const ACOUSTIC_ABSORPTION_PMMA: f64 = 0.08;

/// UHMWPE acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; low-loss semi-crystalline polymer.
pub const ACOUSTIC_ABSORPTION_UHMWPE: f64 = 0.05;

/// Silicone rubber acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; elastomer attenuation ≈ 0.12.
pub const ACOUSTIC_ABSORPTION_SILICONE_RUBBER: f64 = 0.12;

/// Polyurethane acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; flexible elastomer ≈ 0.10.
pub const ACOUSTIC_ABSORPTION_POLYURETHANE: f64 = 0.10;

/// Alumina (Al₂O₃) ceramic acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; low-loss oxide ceramic.
pub const ACOUSTIC_ABSORPTION_ALUMINA: f64 = 0.05;

/// Zirconia (ZrO₂) ceramic acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; zirconia ≈ 0.08.
pub const ACOUSTIC_ABSORPTION_ZIRCONIA: f64 = 0.08;

/// Carbon fiber reinforced polymer (CFRP) acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; anisotropic composite attenuation ≈ 0.15.
pub const ACOUSTIC_ABSORPTION_CFRP: f64 = 0.15;

/// Hydroxyapatite ceramic acoustic absorption coefficient [dB/(cm·MHz)].
/// Source: Szabo (2004) Table A2; bone-mimetic mineral ≈ 0.2.
pub const ACOUSTIC_ABSORPTION_HYDROXYAPATITE: f64 = 0.2;

// ============================================================================
// Effective Acoustic Nonlinearity Parameters B/A [-]
// ============================================================================
//
// These constants preserve the implant material model's prior per-material
// B/A values while moving them into the material SSOT. Literature values for
// solids are less standardized than biological-fluid B/A; callers should treat
// them as effective model parameters for the existing nonlinear acoustic
// material law, not as universal metrology constants.

/// Effective B/A for Titanium Grade 5 in the implant acoustic model.
pub const NONLINEARITY_TITANIUM: f64 = 1.5;

/// Effective B/A for stainless steel 316L in the implant acoustic model.
pub const NONLINEARITY_STAINLESS_STEEL_316L: f64 = 1.8;

/// Effective B/A for platinum in the implant acoustic model.
pub const NONLINEARITY_PLATINUM: f64 = 2.0;

/// Effective B/A for PMMA in the implant acoustic model.
pub const NONLINEARITY_PMMA: f64 = 3.0;

/// Effective B/A for UHMWPE in the implant acoustic model.
pub const NONLINEARITY_UHMWPE: f64 = 2.8;

/// Effective B/A for silicone rubber in the implant acoustic model.
pub const NONLINEARITY_SILICONE_RUBBER: f64 = 4.0;

/// Effective B/A for polyurethane in the implant acoustic model.
pub const NONLINEARITY_POLYURETHANE: f64 = 3.5;

/// Effective B/A for alumina in the implant acoustic model.
pub const NONLINEARITY_ALUMINA: f64 = 1.2;

/// Effective B/A for zirconia in the implant acoustic model.
pub const NONLINEARITY_ZIRCONIA: f64 = 1.4;

/// Effective B/A for carbon-fibre reinforced polymer in the implant acoustic model.
pub const NONLINEARITY_CFRP: f64 = 2.5;

/// Effective B/A for hydroxyapatite in the implant acoustic model.
pub const NONLINEARITY_HYDROXYAPATITE: f64 = 1.8;

// ============================================================================
// Metallic Implants
// ============================================================================

// --- Titanium Grade 5 (Ti-6Al-4V) ---
/// Titanium Grade 5 density (kg/m³) — ASTM F136, ISO 5832-3
pub const DENSITY_TITANIUM_GRADE5: f64 = 4430.0;
/// Titanium Grade 5 compressional sound speed (m/s)
pub const SOUND_SPEED_TITANIUM_GRADE5: f64 = 6070.0;
/// Titanium Grade 5 specific heat capacity (J/(kg·K)) — ASTM F136
pub const SPECIFIC_HEAT_TITANIUM: f64 = 560.0;
/// Titanium Grade 5 thermal conductivity (W/(m·K)) — ASTM F136
pub const THERMAL_CONDUCTIVITY_TITANIUM: f64 = 7.4;
/// Titanium Grade 5 thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 7.4 / (4430 × 560) = 2.98e-6 m²/s
pub const THERMAL_DIFFUSIVITY_TITANIUM: f64 = 2.98e-6;

// --- Stainless Steel 316L ---
/// Stainless Steel 316L density (kg/m³) — ISO 5832-1, ASTM F139
pub const DENSITY_STAINLESS_STEEL_316L: f64 = 8000.0;
/// Stainless Steel 316L compressional sound speed (m/s)
pub const SOUND_SPEED_STAINLESS_STEEL_316L: f64 = 5960.0;
/// Stainless Steel 316L specific heat capacity (J/(kg·K)) — ISO 5832-1
pub const SPECIFIC_HEAT_STAINLESS_STEEL_316L: f64 = 500.0;
/// Stainless Steel 316L thermal conductivity (W/(m·K)) — ISO 5832-1
pub const THERMAL_CONDUCTIVITY_STAINLESS_STEEL_316L: f64 = 16.0;
/// Stainless Steel 316L thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 16.0 / (8000 × 500) = 4.0e-6 m²/s
pub const THERMAL_DIFFUSIVITY_STAINLESS_STEEL_316L: f64 = 4.0e-6;

// --- Platinum ---
/// Platinum density (kg/m³) — ASTM F216
pub const DENSITY_PLATINUM: f64 = 21_450.0;
/// Platinum compressional sound speed (m/s)
pub const SOUND_SPEED_PLATINUM: f64 = 3960.0;
/// Platinum specific heat capacity (J/(kg·K)) — ASTM F216
pub const SPECIFIC_HEAT_PLATINUM: f64 = 135.0;
/// Platinum thermal conductivity (W/(m·K)) — ASTM F216
pub const THERMAL_CONDUCTIVITY_PLATINUM: f64 = 71.6;
/// Platinum thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 71.6 / (21450 × 135) = 2.47e-5 m²/s
pub const THERMAL_DIFFUSIVITY_PLATINUM: f64 = 2.47e-5;

// ============================================================================
// Polymeric Implants
// ============================================================================

// --- PMMA (Polymethyl methacrylate / bone cement) ---
/// PMMA density (kg/m³) — ASTM F451
pub const DENSITY_PMMA: f64 = 1190.0;
/// PMMA compressional sound speed (m/s)
pub const SOUND_SPEED_PMMA: f64 = 2670.0;
/// PMMA specific heat capacity (J/(kg·K)) — ASTM F451
pub const SPECIFIC_HEAT_PMMA: f64 = 1470.0;
/// PMMA thermal conductivity (W/(m·K)) — ASTM F451
pub const THERMAL_CONDUCTIVITY_PMMA: f64 = 0.19;
/// PMMA thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 0.19 / (1190 × 1470) = 1.086e-7 m²/s
pub const THERMAL_DIFFUSIVITY_PMMA: f64 = 1.086e-7;

// --- UHMWPE (Ultra-high molecular weight polyethylene) ---
/// UHMWPE density (kg/m³) — ASTM F648
pub const DENSITY_UHMWPE: f64 = 935.0;
/// UHMWPE compressional sound speed (m/s)
pub const SOUND_SPEED_UHMWPE: f64 = 2380.0;
/// UHMWPE specific heat capacity (J/(kg·K)) — ASTM F648
pub const SPECIFIC_HEAT_UHMWPE: f64 = 2300.0;
/// UHMWPE thermal conductivity (W/(m·K)) — ASTM F648
pub const THERMAL_CONDUCTIVITY_UHMWPE: f64 = 0.42;
/// UHMWPE thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 0.42 / (935 × 2300) = 1.951e-7 m²/s
pub const THERMAL_DIFFUSIVITY_UHMWPE: f64 = 1.951e-7;

// --- Silicone rubber ---
/// Silicone rubber density (kg/m³) — ASTM F381
pub const DENSITY_SILICONE_RUBBER: f64 = 970.0;
/// Silicone rubber compressional sound speed (m/s)
pub const SOUND_SPEED_SILICONE_RUBBER: f64 = 1050.0;
/// Silicone rubber specific heat capacity (J/(kg·K)) — ASTM F381
pub const SPECIFIC_HEAT_SILICONE_RUBBER: f64 = 1500.0;
/// Silicone rubber thermal conductivity (W/(m·K)) — ASTM F381
pub const THERMAL_CONDUCTIVITY_SILICONE_RUBBER: f64 = 0.25;
/// Silicone rubber thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 0.25 / (970 × 1500) = 1.717e-7 m²/s
pub const THERMAL_DIFFUSIVITY_SILICONE_RUBBER: f64 = 1.717e-7;

// --- Polyurethane ---
/// Polyurethane specific heat capacity (J/(kg·K)) — ASTM F1634
pub const SPECIFIC_HEAT_POLYURETHANE: f64 = 1800.0;
/// Polyurethane thermal conductivity (W/(m·K)) — ASTM F1634
pub const THERMAL_CONDUCTIVITY_POLYURETHANE: f64 = 0.24;
/// Polyurethane sound speed (m/s)
pub const SOUND_SPEED_POLYURETHANE: f64 = 1890.0;

// ============================================================================
// Ceramic Implants
// ============================================================================

// --- Alumina (Al₂O₃) ---
/// Alumina density (kg/m³) — ASTM F603
pub const DENSITY_ALUMINA: f64 = 3970.0;
/// Alumina compressional sound speed (m/s)
pub const SOUND_SPEED_ALUMINA: f64 = 11_100.0;
/// Alumina specific heat capacity (J/(kg·K)) — ASTM F603
pub const SPECIFIC_HEAT_ALUMINA: f64 = 880.0;
/// Alumina thermal conductivity (W/(m·K)) — ASTM F603
pub const THERMAL_CONDUCTIVITY_ALUMINA: f64 = 30.0;
/// Alumina thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 30.0 / (3970 × 880) = 8.582e-6 m²/s
pub const THERMAL_DIFFUSIVITY_ALUMINA: f64 = 8.582e-6;

// --- Zirconia (ZrO₂) ---
/// Zirconia density (kg/m³) — ASTM F1873
pub const DENSITY_ZIRCONIA: f64 = 6050.0;
/// Zirconia compressional sound speed (m/s)
pub const SOUND_SPEED_ZIRCONIA: f64 = 6000.0;
/// Zirconia specific heat capacity (J/(kg·K)) — ASTM F1873
pub const SPECIFIC_HEAT_ZIRCONIA: f64 = 500.0;
/// Zirconia thermal conductivity (W/(m·K)) — ASTM F1873
pub const THERMAL_CONDUCTIVITY_ZIRCONIA: f64 = 2.0;
/// Zirconia thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 2.0 / (6050 × 500) = 6.612e-7 m²/s
pub const THERMAL_DIFFUSIVITY_ZIRCONIA: f64 = 6.612e-7;

// ============================================================================
// Composite Implants
// ============================================================================

// --- Carbon Fibre Reinforced Polymer (CFRP) ---
/// CFRP density (kg/m³) — ASTM E2748
pub const DENSITY_CFRP: f64 = 1600.0;
/// CFRP compressional sound speed (m/s)
pub const SOUND_SPEED_CFRP: f64 = 3100.0;
/// CFRP specific heat capacity (J/(kg·K)) — ASTM E2748
pub const SPECIFIC_HEAT_CFRP: f64 = 900.0;
/// CFRP thermal conductivity (W/(m·K)) — ASTM E2748 (through-thickness)
pub const THERMAL_CONDUCTIVITY_CFRP: f64 = 5.0;
/// CFRP thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 5.0 / (1600 × 900) = 3.472e-6 m²/s
pub const THERMAL_DIFFUSIVITY_CFRP: f64 = 3.472e-6;

// --- Hydroxyapatite (HA) ---
/// Hydroxyapatite density (kg/m³) — ASTM F1185
pub const DENSITY_HYDROXYAPATITE: f64 = 3220.0;
/// Hydroxyapatite compressional sound speed (m/s)
pub const SOUND_SPEED_HYDROXYAPATITE: f64 = 3640.0;
/// Hydroxyapatite specific heat capacity (J/(kg·K)) — ASTM F1185
pub const SPECIFIC_HEAT_HYDROXYAPATITE: f64 = 880.0;
/// Hydroxyapatite thermal conductivity (W/(m·K)) — ASTM F1185
pub const THERMAL_CONDUCTIVITY_HYDROXYAPATITE: f64 = 1.2;
/// Hydroxyapatite thermal diffusivity α = k/(ρ·c_p) (m²/s)
/// = 1.2 / (3220 × 880) = 4.231e-7 m²/s
pub const THERMAL_DIFFUSIVITY_HYDROXYAPATITE: f64 = 4.231e-7;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify α = k / (ρ · c_p) for each implant material within 1 % tolerance.
    #[test]
    fn thermal_diffusivities_consistent() {
        let materials: &[(&str, f64, f64, f64, f64)] = &[
            (
                "Titanium",
                THERMAL_CONDUCTIVITY_TITANIUM,
                DENSITY_TITANIUM_GRADE5,
                SPECIFIC_HEAT_TITANIUM,
                THERMAL_DIFFUSIVITY_TITANIUM,
            ),
            (
                "Steel316L",
                THERMAL_CONDUCTIVITY_STAINLESS_STEEL_316L,
                DENSITY_STAINLESS_STEEL_316L,
                SPECIFIC_HEAT_STAINLESS_STEEL_316L,
                THERMAL_DIFFUSIVITY_STAINLESS_STEEL_316L,
            ),
            (
                "Platinum",
                THERMAL_CONDUCTIVITY_PLATINUM,
                DENSITY_PLATINUM,
                SPECIFIC_HEAT_PLATINUM,
                THERMAL_DIFFUSIVITY_PLATINUM,
            ),
            (
                "PMMA",
                THERMAL_CONDUCTIVITY_PMMA,
                DENSITY_PMMA,
                SPECIFIC_HEAT_PMMA,
                THERMAL_DIFFUSIVITY_PMMA,
            ),
            (
                "UHMWPE",
                THERMAL_CONDUCTIVITY_UHMWPE,
                DENSITY_UHMWPE,
                SPECIFIC_HEAT_UHMWPE,
                THERMAL_DIFFUSIVITY_UHMWPE,
            ),
            (
                "Silicone",
                THERMAL_CONDUCTIVITY_SILICONE_RUBBER,
                DENSITY_SILICONE_RUBBER,
                SPECIFIC_HEAT_SILICONE_RUBBER,
                THERMAL_DIFFUSIVITY_SILICONE_RUBBER,
            ),
            (
                "Alumina",
                THERMAL_CONDUCTIVITY_ALUMINA,
                DENSITY_ALUMINA,
                SPECIFIC_HEAT_ALUMINA,
                THERMAL_DIFFUSIVITY_ALUMINA,
            ),
            (
                "Zirconia",
                THERMAL_CONDUCTIVITY_ZIRCONIA,
                DENSITY_ZIRCONIA,
                SPECIFIC_HEAT_ZIRCONIA,
                THERMAL_DIFFUSIVITY_ZIRCONIA,
            ),
            (
                "CFRP",
                THERMAL_CONDUCTIVITY_CFRP,
                DENSITY_CFRP,
                SPECIFIC_HEAT_CFRP,
                THERMAL_DIFFUSIVITY_CFRP,
            ),
            (
                "Hydroxyapatite",
                THERMAL_CONDUCTIVITY_HYDROXYAPATITE,
                DENSITY_HYDROXYAPATITE,
                SPECIFIC_HEAT_HYDROXYAPATITE,
                THERMAL_DIFFUSIVITY_HYDROXYAPATITE,
            ),
        ];

        for &(name, k, rho, cp, alpha_stored) in materials {
            let alpha_derived = k / (rho * cp);
            let rel_err = (alpha_stored - alpha_derived).abs() / alpha_derived;
            assert!(
                rel_err < 0.01,
                "{name}: stored α={alpha_stored:.4e} vs derived {alpha_derived:.4e} (rel err {rel_err:.4})"
            );
        }
    }

    #[test]
    fn all_implant_constants_positive() {
        let values = [
            DENSITY_TITANIUM_GRADE5,
            SOUND_SPEED_TITANIUM_GRADE5,
            SPECIFIC_HEAT_TITANIUM,
            THERMAL_CONDUCTIVITY_TITANIUM,
            DENSITY_STAINLESS_STEEL_316L,
            SPECIFIC_HEAT_STAINLESS_STEEL_316L,
            DENSITY_PLATINUM,
            SPECIFIC_HEAT_PLATINUM,
            DENSITY_PMMA,
            SPECIFIC_HEAT_PMMA,
            DENSITY_UHMWPE,
            SPECIFIC_HEAT_UHMWPE,
            DENSITY_SILICONE_RUBBER,
            SPECIFIC_HEAT_SILICONE_RUBBER,
            SPECIFIC_HEAT_POLYURETHANE,
            DENSITY_ALUMINA,
            SPECIFIC_HEAT_ALUMINA,
            DENSITY_ZIRCONIA,
            SPECIFIC_HEAT_ZIRCONIA,
            DENSITY_CFRP,
            SPECIFIC_HEAT_CFRP,
            DENSITY_HYDROXYAPATITE,
            SPECIFIC_HEAT_HYDROXYAPATITE,
            NONLINEARITY_TITANIUM,
            NONLINEARITY_STAINLESS_STEEL_316L,
            NONLINEARITY_PLATINUM,
            NONLINEARITY_PMMA,
            NONLINEARITY_UHMWPE,
            NONLINEARITY_SILICONE_RUBBER,
            NONLINEARITY_POLYURETHANE,
            NONLINEARITY_ALUMINA,
            NONLINEARITY_ZIRCONIA,
            NONLINEARITY_CFRP,
            NONLINEARITY_HYDROXYAPATITE,
        ];
        for &v in &values {
            assert!(v > 0.0, "Implant constant must be positive, got {v}");
        }
    }
}
