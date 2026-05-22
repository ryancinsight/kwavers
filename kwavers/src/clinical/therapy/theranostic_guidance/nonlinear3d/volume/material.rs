//! CT-derived sound-speed, density, beta, and attenuation maps.

use ndarray::Array3;

use crate::core::constants::fundamental::{SOUND_SPEED_AIR, SOUND_SPEED_WATER_SIM};
use super::super::super::AnatomyKind;
use super::attenuation::{attenuation_np_per_m_mhz_from_hu, attenuation_power_law_y_from_hu};

const COUPLING_SOUND_SPEED_M_S: f64 = 1480.0;
const COUPLING_DENSITY_KG_M3: f64 = 1000.0;
const COUPLING_BETA: f64 = 3.5;
const AIR_DENSITY_KG_M3: f64 = 1.225;
const AIR_BETA: f64 = 1.2;
const INTERNAL_GAS_HU_THRESHOLD: f64 = -700.0;

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
            (1000.0 + 0.45 * ct[idx].clamp(-100.0, 1800.0)).clamp(930.0, 1900.0)
        }
    });
    let beta = Array3::from_shape_fn(ct.dim(), |idx| {
        if !body[idx] {
            COUPLING_BETA
        } else if is_internal_gas(ct[idx], label[idx]) {
            AIR_BETA
        } else if ct[idx] >= 300.0 {
            6.0
        } else {
            3.5
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
    if hu >= 300.0 {
        let phi = (hu / 1000.0).clamp(0.0, 1.0);
        return SOUND_SPEED_WATER_SIM * (1.0 - phi) + 2900.0 * phi;
    }
    let organ_speed = match anatomy {
        AnatomyKind::Brain => 1540.0,
        AnatomyKind::Liver => 1595.0,
        AnatomyKind::Kidney => 1567.0,
    };
    if label > 0 {
        organ_speed + 0.10 * hu.clamp(-100.0, 200.0)
    } else {
        1480.0 + 0.18 * hu.clamp(-150.0, 250.0)
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
