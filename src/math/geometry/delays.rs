use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array1;
use std::f64::consts::PI;

pub fn focus_phase_delays(
    element_positions: &[[f64; 3]],
    focal_point: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    if element_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Element positions array is empty".into(),
        ));
    }

    if !focal_point.iter().all(|&x| x.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "Focal point contains non-finite values".into(),
        ));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    let wavenumber = 2.0 * PI * frequency / sound_speed;

    let mut distances = Vec::with_capacity(element_positions.len());
    for (i, pos) in element_positions.iter().enumerate() {
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Element position {} contains non-finite values",
                i
            )));
        }

        let dx = focal_point[0] - pos[0];
        let dy = focal_point[1] - pos[1];
        let dz = focal_point[2] - pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if !distance.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Computed distance for element {} is non-finite",
                i
            )));
        }

        distances.push(distance);
    }

    let max_distance = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_distance.is_finite() || max_distance <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "Maximum distance is non-positive or non-finite".into(),
        ));
    }

    let mut phase_delays = Array1::<f64>::zeros(element_positions.len());
    for (i, &distance) in distances.iter().enumerate() {
        phase_delays[i] = wavenumber * (max_distance - distance);
        debug_assert!(phase_delays[i] >= 0.0, "Phase delay must be non-negative");
    }

    Ok(phase_delays)
}

pub fn plane_wave_phase_delays(
    element_positions: &[[f64; 3]],
    direction: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    if element_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Element positions array is empty".into(),
        ));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    let dir_norm_sq = direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2);

    if !dir_norm_sq.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Direction vector contains non-finite values".into(),
        ));
    }

    if (dir_norm_sq - 1.0).abs() > 1e-6 {
        return Err(KwaversError::InvalidInput(format!(
            "Direction must be unit vector, got norm² = {} (norm = {})",
            dir_norm_sq,
            dir_norm_sq.sqrt()
        )));
    }

    let wavenumber = 2.0 * PI * frequency / sound_speed;

    let mut phase_delays = Array1::<f64>::zeros(element_positions.len());

    for (i, pos) in element_positions.iter().enumerate() {
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Element position {} contains non-finite values",
                i
            )));
        }

        let dot_product = pos[0] * direction[0] + pos[1] * direction[1] + pos[2] * direction[2];

        if !dot_product.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Dot product for element {} is non-finite",
                i
            )));
        }

        phase_delays[i] = -wavenumber * dot_product;
    }

    Ok(phase_delays)
}

pub fn spherical_steering_phase_delays(
    element_positions: &[[f64; 3]],
    theta: f64,
    phi: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    if !theta.is_finite() || !phi.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Steering angles must be finite".into(),
        ));
    }

    let direction = [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ];

    plane_wave_phase_delays(element_positions, direction, frequency, sound_speed)
}

pub fn calculate_beam_width(
    aperture_size: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<f64> {
    if !aperture_size.is_finite() || aperture_size <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Aperture size must be positive and finite, got {}",
            aperture_size
        )));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    let wavelength = sound_speed / frequency;
    Ok(1.22 * wavelength / aperture_size)
}

pub fn calculate_focal_zone(
    aperture_size: f64,
    focal_distance: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<f64> {
    if !aperture_size.is_finite() || aperture_size <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Aperture size must be positive and finite, got {}",
            aperture_size
        )));
    }

    if !focal_distance.is_finite() || focal_distance <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Focal distance must be positive and finite, got {}",
            focal_distance
        )));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    let wavelength = sound_speed / frequency;
    let f_number = focal_distance / aperture_size;
    Ok(7.0 * wavelength * f_number.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn focus_phase_delays_are_non_negative_and_symmetric() {
        let positions = vec![
            [-0.75e-3, 0.0, 0.0],
            [-0.25e-3, 0.0, 0.0],
            [0.25e-3, 0.0, 0.0],
            [0.75e-3, 0.0, 0.0],
        ];

        let delays = focus_phase_delays(&positions, [0.0, 0.0, 0.05], 1e6, 1540.0).unwrap();

        for &delay in delays.iter() {
            assert!(delay >= 0.0);
        }

        assert_relative_eq!(delays[0], delays[3], epsilon = 1e-10);
        assert_relative_eq!(delays[1], delays[2], epsilon = 1e-10);
    }

    #[test]
    fn plane_wave_broadside_has_zero_phase_delays_for_x_array() {
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];
        let delays = plane_wave_phase_delays(&positions, [0.0, 0.0, 1.0], 1e6, 1540.0).unwrap();
        for &delay in delays.iter() {
            assert_relative_eq!(delay, 0.0, epsilon = 1e-10);
        }
    }
}
