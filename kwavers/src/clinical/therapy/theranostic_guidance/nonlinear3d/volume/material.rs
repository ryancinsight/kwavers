//! CT-derived sound-speed, density, beta, and attenuation maps.

use ndarray::Array3;

use super::super::super::AnatomyKind;
use super::attenuation::{attenuation_np_per_m_mhz_from_hu, attenuation_power_law_y_from_hu};
use crate::core::constants::acoustic_parameters::SOUND_SPEED_SKULL;
use crate::core::constants::fundamental::{
    B_OVER_A_BONE, B_OVER_A_SOFT_TISSUE, B_OVER_A_WATER_37C, DENSITY_AIR,
    DENSITY_WATER_NOMINAL, HU_BONE_THRESHOLD, SOUND_SPEED_AIR, SOUND_SPEED_KIDNEY,
    SOUND_SPEED_LIVER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER, SOUND_SPEED_WATER_SIM,
};

const COUPLING_SOUND_SPEED_M_S: f64 = SOUND_SPEED_WATER;
const COUPLING_DENSITY_KG_M3: f64 = DENSITY_WATER_NOMINAL;
/// β = 1 + B/(2A) for water-equivalent coupling gel at body temperature (37°C).
///
/// Uses `B_OVER_A_WATER_37C = 5.0` (Aanonsen et al. 1984; Duck 1990 Table 4.16)
/// rather than the 20°C value (5.2), giving β = 3.5.  Coupling gels are
/// applied at skin temperature (≈ 37°C), so the body-temperature parameter
/// is the correct physical choice here.
const COUPLING_BETA: f64 = 1.0 + B_OVER_A_WATER_37C / 2.0;
const AIR_DENSITY_KG_M3: f64 = DENSITY_AIR;
/// Nonlinearity parameter for air (B/A_air ≈ 0.4, β = 1.2).
const AIR_BETA: f64 = 1.2;
const INTERNAL_GAS_HU_THRESHOLD: f64 = -700.0;

/// CT-to-density linear slope [kg/m³ per HU].
///
/// Schneider W, Bortfeld T, Schlegel W (1996) Correlation between CT numbers and tissue
/// parameters needed for Monte Carlo simulations of clinical dose distributions.
/// Phys Med Biol 41(1):111–124, Table 2 soft-tissue fit.
const CT_DENSITY_SLOPE_KG_M3_PER_HU: f64 = 0.45;

/// Minimum plausible tissue density in the CT-to-density map [kg/m³].
///
/// Corresponds to the fat density floor; values below this indicate air pockets or
/// implant artefacts outside the soft-tissue model domain.
const CT_DENSITY_MIN_KG_M3: f64 = 930.0;

/// Maximum plausible tissue density in the CT-to-density map [kg/m³].
///
/// Corresponds to the dense cortical bone ceiling: ρ_skull_min (1200) +
/// ρ_skull_cortical_range (700) = 1900 kg/m³.
const CT_DENSITY_MAX_KG_M3: f64 = 1900.0;

/// HU at which bone fraction saturates to 1 in the simplified speed model [HU].
///
/// Dense cortical bone exhibits Hounsfield values up to ~1000 HU; above this the
/// voxel is treated as fully bone-equivalent for sound-speed interpolation.
const HU_SPEED_BONE_MAX: f64 = 1000.0;

/// Lower HU clamp for the CT-to-density conversion [HU].
///
/// At this bound: ρ = `DENSITY_WATER_NOMINAL` + `CT_DENSITY_SLOPE_KG_M3_PER_HU` × (−160)
/// = 1000 − 72 = 928 kg/m³ < `CT_DENSITY_MIN_KG_M3` (930 kg/m³), so the outer lower
/// clamp activates and is reachable.  Values more negative than −160 HU inside the body
/// outline are classified as internal gas by `is_internal_gas` (threshold −700 HU) or
/// remain in the subcutaneous-fat range, which is well-captured by this bound.
const CT_HU_DENSITY_LOWER: f64 = -160.0;

/// Upper HU clamp for the CT-to-density conversion [HU].
///
/// HU values above 1800 correspond to dense metallic implant artefacts; clamping prevents
/// unphysical super-bone densities while leaving all physiological bone values (≤ ~1000 HU)
/// unaffected.
const CT_HU_DENSITY_UPPER: f64 = 1800.0;

/// Lower HU clamp for the labelled-tissue sound-speed correction [HU].
///
/// Prevents the speed-of-sound from falling below the organ baseline for voxels with
/// anomalously low CT values within an organ label.
const CT_HU_SPEED_LABELED_LOWER: f64 = -100.0;

/// Upper HU clamp for the labelled-tissue sound-speed correction [HU].
///
/// Above 200 HU a labelled soft-tissue voxel is likely a partial-volume artefact or
/// early calcification; capping here limits over-correction.
const CT_HU_SPEED_LABELED_UPPER: f64 = 200.0;

/// Lower HU clamp for the background (unlabelled) tissue sound-speed correction [HU].
const CT_HU_SPEED_BACKGROUND_LOWER: f64 = -150.0;

/// Upper HU clamp for the background (unlabelled) tissue sound-speed correction [HU].
const CT_HU_SPEED_BACKGROUND_UPPER: f64 = 250.0;

/// HU-to-speed slope for organ-labelled soft tissue [m/s per HU].
///
/// Empirical calibration for labelled tissue voxels from CT imaging.
const CT_SPEED_SLOPE_LABELED_M_S_PER_HU: f64 = 0.10;

/// HU-to-speed slope for background (unlabelled) soft tissue [m/s per HU].
///
/// Empirical calibration for background tissue voxels without organ label.
const CT_SPEED_SLOPE_BACKGROUND_M_S_PER_HU: f64 = 0.18;

/// Nonlinearity parameter β for generic soft tissue (dimensionless).
///
/// Derived from `B_OVER_A_SOFT_TISSUE = 6.5` (Duck 1990 Table 4.16 median):
/// β = 1 + B/(2A) = 1 + 6.5/2 = 4.25.
/// Applied to non-bone, non-air voxels inside the body outline.
const SOFT_TISSUE_BETA: f64 = 1.0 + B_OVER_A_SOFT_TISSUE / 2.0;

/// Nonlinearity parameter β for skull/dense cortical bone (dimensionless).
///
/// Derived from `B_OVER_A_BONE = 8.0` (Duck 1990 Table 4.16; measured range 6–9
/// for cortical bone): β = 1 + B/(2A) = 1 + 8.0/2 = 5.0.
/// Applied when HU ≥ HU_BONE_THRESHOLD.
const BETA_BONE: f64 = 1.0 + B_OVER_A_BONE / 2.0;

pub(super) fn material_maps(
    anatomy: AnatomyKind,
    ct: &Array3<f64>,
    label: &Array3<i16>,
    body: &Array3<bool>,
) -> (
    Array3<f64>,
    Array3<f64>,
    Array3<f64>,
    Array3<f64>,
    Array3<f64>,
) {
    let speed = Array3::from_shape_fn(ct.dim(), |idx| {
        if body[idx] {
            speed_from_hu(anatomy, ct[idx], label[idx])
        } else {
            COUPLING_SOUND_SPEED_M_S
        }
    });
    let density = Array3::from_shape_fn(ct.dim(), |idx| {
        if !body[idx] {
            COUPLING_DENSITY_KG_M3
        } else if is_internal_gas(ct[idx], label[idx]) {
            AIR_DENSITY_KG_M3
        } else {
            (DENSITY_WATER_NOMINAL
                + CT_DENSITY_SLOPE_KG_M3_PER_HU
                    * ct[idx].clamp(CT_HU_DENSITY_LOWER, CT_HU_DENSITY_UPPER))
            .clamp(CT_DENSITY_MIN_KG_M3, CT_DENSITY_MAX_KG_M3)
        }
    });
    let beta = Array3::from_shape_fn(ct.dim(), |idx| {
        if !body[idx] {
            COUPLING_BETA
        } else if is_internal_gas(ct[idx], label[idx]) {
            AIR_BETA
        } else if ct[idx] >= HU_BONE_THRESHOLD {
            BETA_BONE
        } else {
            SOFT_TISSUE_BETA
        }
    });
    let attenuation = Array3::from_shape_fn(ct.dim(), |idx| {
        attenuation_np_per_m_mhz_from_hu(ct[idx], label[idx], body[idx])
    });
    let power_law_y = Array3::from_shape_fn(ct.dim(), |idx| {
        attenuation_power_law_y_from_hu(ct[idx], label[idx], body[idx])
    });
    (speed, density, beta, attenuation, power_law_y)
}

fn speed_from_hu(anatomy: AnatomyKind, hu: f64, label: i16) -> f64 {
    if is_internal_gas(hu, label) {
        return SOUND_SPEED_AIR;
    }
    if hu >= HU_BONE_THRESHOLD {
        let phi = (hu / HU_SPEED_BONE_MAX).clamp(0.0, 1.0);
        return SOUND_SPEED_WATER_SIM * (1.0 - phi) + SOUND_SPEED_SKULL * phi;
    }
    let organ_speed = match anatomy {
        AnatomyKind::Brain => SOUND_SPEED_TISSUE,
        AnatomyKind::Liver => SOUND_SPEED_LIVER,
        AnatomyKind::Kidney => SOUND_SPEED_KIDNEY,
    };
    if label > 0 {
        organ_speed
            + CT_SPEED_SLOPE_LABELED_M_S_PER_HU
                * hu.clamp(CT_HU_SPEED_LABELED_LOWER, CT_HU_SPEED_LABELED_UPPER)
    } else {
        SOUND_SPEED_WATER
            + CT_SPEED_SLOPE_BACKGROUND_M_S_PER_HU
                * hu.clamp(CT_HU_SPEED_BACKGROUND_LOWER, CT_HU_SPEED_BACKGROUND_UPPER)
    }
}

fn is_internal_gas(hu: f64, label: i16) -> bool {
    hu < INTERNAL_GAS_HU_THRESHOLD && label == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn exterior_ct_air_maps_to_coupling_fluid() {
        let ct = Array3::from_elem((1, 1, 1), -1000.0);
        let label = Array3::from_elem((1, 1, 1), 0);
        let body = Array3::from_elem((1, 1, 1), false);
        let (speed, density, beta, attenuation, power_law_y) =
            material_maps(AnatomyKind::Brain, &ct, &label, &body);
        assert_close(speed[[0, 0, 0]], COUPLING_SOUND_SPEED_M_S);
        assert_close(density[[0, 0, 0]], COUPLING_DENSITY_KG_M3);
        assert_close(beta[[0, 0, 0]], COUPLING_BETA);
        assert_close(attenuation[[0, 0, 0]], 0.0);
        assert_close(power_law_y[[0, 0, 0]], 1.0);
    }

    #[test]
    fn internal_gas_pocket_preserves_gas_material_properties() {
        let ct = Array3::from_elem((1, 1, 1), -900.0);
        let label = Array3::from_elem((1, 1, 1), 0);
        let body = Array3::from_elem((1, 1, 1), true);
        let (speed, density, beta, attenuation, power_law_y) =
            material_maps(AnatomyKind::Liver, &ct, &label, &body);
        assert_close(speed[[0, 0, 0]], SOUND_SPEED_AIR);
        assert_close(density[[0, 0, 0]], AIR_DENSITY_KG_M3);
        assert_close(beta[[0, 0, 0]], AIR_BETA);
        assert_close(attenuation[[0, 0, 0]], 1000.0);
        assert_close(power_law_y[[0, 0, 0]], 1.0);
    }

    /// β for a normal soft-tissue voxel must equal 1 + B_OVER_A_SOFT_TISSUE / 2 = 4.25.
    ///
    /// Previously this path incorrectly returned COUPLING_BETA (3.5 or 3.6 depending on
    /// the B/A temperature used for water); correct soft-tissue β is higher than the
    /// water value and must be sourced from B_OVER_A_SOFT_TISSUE.
    #[test]
    fn soft_tissue_voxel_uses_soft_tissue_beta() {
        // HU = 0 (water-equivalent soft tissue), label = 1 (organ-labelled), inside body.
        let ct = Array3::from_elem((1, 1, 1), 0.0_f64);
        let label = Array3::from_elem((1, 1, 1), 1_i16);
        let body = Array3::from_elem((1, 1, 1), true);
        let (_, _, beta, _, _) = material_maps(AnatomyKind::Brain, &ct, &label, &body);
        // SOFT_TISSUE_BETA = 1.0 + 6.5 / 2.0 = 4.25 exactly.
        assert_close(beta[[0, 0, 0]], 4.25);
    }

    /// β for a bone voxel must equal 1 + B_OVER_A_BONE / 2 = 5.0.
    ///
    /// The prior hard-coded value of 6.0 implied B/A = 10, which exceeds all published
    /// bone measurements (Duck 1990 range 6–9).  After fix: BETA_BONE = 1 + 8/2 = 5.0.
    #[test]
    fn bone_voxel_uses_derived_beta_bone() {
        // HU = 500 (≥ HU_BONE_THRESHOLD = 300), label = 0, inside body.
        let ct = Array3::from_elem((1, 1, 1), 500.0_f64);
        let label = Array3::from_elem((1, 1, 1), 0_i16);
        let body = Array3::from_elem((1, 1, 1), true);
        let (_, _, beta, _, _) = material_maps(AnatomyKind::Brain, &ct, &label, &body);
        // BETA_BONE = 1.0 + B_OVER_A_BONE / 2.0 = 1.0 + 8.0 / 2.0 = 5.0 exactly.
        assert_close(beta[[0, 0, 0]], 5.0);
    }

    /// The outer lower density clamp (CT_DENSITY_MIN_KG_M3 = 930) must be reachable.
    ///
    /// With CT_HU_DENSITY_LOWER = -160: pre-clamp density = 1000 + 0.45 × (−160) = 928,
    /// which is below 930, so the outer clamp activates and the result is exactly 930.
    /// This verifies that the outer floor is not dead code.
    #[test]
    fn fat_range_hu_activates_density_lower_clamp() {
        let ct = Array3::from_elem((1, 1, 1), CT_HU_DENSITY_LOWER);
        let label = Array3::from_elem((1, 1, 1), 0_i16);
        let body = Array3::from_elem((1, 1, 1), true);
        let (_, density, _, _, _) = material_maps(AnatomyKind::Liver, &ct, &label, &body);
        // pre-clamp: 1000 + 0.45 × (-160) = 928 < 930 → outer clamp returns 930.
        assert_close(density[[0, 0, 0]], CT_DENSITY_MIN_KG_M3);
    }

    /// COUPLING_BETA must equal 3.5 (body-temperature water, B/A = 5.0, not 5.2).
    ///
    /// Verifies that the constant uses B_OVER_A_WATER_37C and not the 20°C value,
    /// preventing a silent 2% physics error in the coupling-fluid nonlinearity.
    #[test]
    fn coupling_beta_is_body_temperature_value() {
        // Exact: 1.0 + 5.0 / 2.0 = 3.5.
        assert_close(COUPLING_BETA, 3.5);
    }
}
