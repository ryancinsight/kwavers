//! Van Cittert-Zernike spatial coherence helpers for passive cavitation maps.

/// Van Cittert-Zernike coherence for an incoherent planar source.
///
/// Returns `mu(delta_x) = sinc(L_src * delta_x / (lambda * z))`, where
/// `sinc(u) = sin(pi * u) / (pi * u)`. This is the far-field paraxial
/// coherence law used by Chapter 23's passive-acoustic mapping figure.
pub fn van_cittert_zernike_coherence(
    delta_x_m: &[f64],
    source_extent_m: f64,
    depth_m: f64,
    wavelength_m: f64,
) -> Result<Vec<f64>, String> {
    if !source_extent_m.is_finite() || source_extent_m <= 0.0 {
        return Err("source_extent_m must be positive and finite".to_owned());
    }
    if !depth_m.is_finite() || depth_m <= 0.0 {
        return Err("depth_m must be positive and finite".to_owned());
    }
    if !wavelength_m.is_finite() || wavelength_m <= 0.0 {
        return Err("wavelength_m must be positive and finite".to_owned());
    }
    if delta_x_m.iter().any(|value| !value.is_finite()) {
        return Err("delta_x_m must contain only finite values".to_owned());
    }

    let scale = source_extent_m / (wavelength_m * depth_m);
    Ok(delta_x_m
        .iter()
        .map(|&delta_x| normalized_sinc(scale * delta_x))
        .collect())
}

fn normalized_sinc(value: f64) -> f64 {
    let x = std::f64::consts::PI * value;
    if x.abs() <= f64::EPSILON {
        1.0
    } else {
        x.sin() / x
    }
}
