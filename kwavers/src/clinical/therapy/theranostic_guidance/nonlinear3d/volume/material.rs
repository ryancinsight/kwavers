//! CT-derived sound-speed, density, beta, and attenuation maps.

use ndarray::Array3;

use super::super::super::AnatomyKind;
use super::attenuation::{attenuation_np_per_m_mhz_from_hu, attenuation_power_law_y_from_hu};

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
    let speed = Array3::from_shape_fn(ct.dim(), |idx| speed_from_hu(anatomy, ct[idx], label[idx]));
    let density = Array3::from_shape_fn(ct.dim(), |idx| {
        if body[idx] {
            (1000.0 + 0.45 * ct[idx].clamp(-100.0, 1800.0)).clamp(930.0, 1900.0)
        } else {
            1.2
        }
    });
    let beta = Array3::from_shape_fn(ct.dim(), |idx| {
        if !body[idx] {
            1.2
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
    if hu < -700.0 && label == 0 {
        return 343.0;
    }
    if hu >= 300.0 {
        let phi = (hu / 1000.0).clamp(0.0, 1.0);
        return 1500.0 * (1.0 - phi) + 2900.0 * phi;
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
