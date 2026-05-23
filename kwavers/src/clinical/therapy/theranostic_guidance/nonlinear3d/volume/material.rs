//! CT-derived sound-speed, density, beta, and attenuation maps.

use ndarray::Array3;

use super::super::super::AnatomyKind;
use super::attenuation::{attenuation_np_per_m_mhz_from_hu, attenuation_power_law_y_from_hu};
use crate::core::constants::acoustic_parameters::SOUND_SPEED_SKULL;
use crate::core::constants::fundamental::{
    B_OVER_A_WATER, DENSITY_AIR, DENSITY_WATER_NOMINAL, HU_BONE_THRESHOLD, SOUND_SPEED_AIR,
    SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
    SOUND_SPEED_WATER_SIM,
};

const COUPLING_SOUND_SPEED_M_S: f64 = SOUND_SPEED_WATER;
const COUPLING_DENSITY_KG_M3: f64 = DENSITY_WATER_NOMINAL;
/// β = 1 + B/(2A) for water-equivalent coupling fluid.
const COUPLING_BETA: f64 = 1.0 + B_OVER_A_WATER / 2.0;
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

/// HU-to-speed slope for organ-labelled soft tissue [m/s per HU].
///
/// Empirical calibration for labelled tissue voxels from CT imaging.
const CT_SPEED_SLOPE_LABELED_M_S_PER_HU: f64 = 0.10;

/// HU-to-speed slope for background (unlabelled) soft tissue [m/s per HU].
///
/// Empirical calibration for background tissue voxels without organ label.
const CT_SPEED_SLOPE_BACKGROUND_M_S_PER_HU: f64 = 0.18;

/// Nonlinearity parameter β for skull/dense bone tissue.
///
/// Approximated from B/A literature for mineralized bone; used when ct ≥ HU_BONE_THRESHOLD.
const BETA_BONE: f64 = 6.0;

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
                + CT_DENSITY_SLOPE_KG_M3_PER_HU * ct[idx].clamp(-100.0, 1800.0))
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
            COUPLING_BETA
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
        organ_speed + CT_SPEED_SLOPE_LABELED_M_S_PER_HU * hu.clamp(-100.0, 200.0)
    } else {
        SOUND_SPEED_WATER + CT_SPEED_SLOPE_BACKGROUND_M_S_PER_HU * hu.clamp(-150.0, 250.0)
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
}
