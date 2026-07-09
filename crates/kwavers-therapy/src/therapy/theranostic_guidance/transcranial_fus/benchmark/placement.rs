use leto::{
    Array1,
    Array2,
};

use kwavers_core::error::{KwaversError, KwaversResult};

use super::types::{SkullAdaptiveBenchmarkConfig, SkullAwareTransducerPlacement};

pub(super) fn select_skull_aware_placement(
    element_positions_m: Array2<f64>,
    skull_lengths_m: &Array1<f64>,
    amplitude_weights: &Array1<f64>,
    config: &SkullAdaptiveBenchmarkConfig,
) -> KwaversResult<SkullAwareTransducerPlacement> {
    let n = element_positions_m.nrows();
    if n == 0 || element_positions_m.ncols() != 3 {
        return Err(KwaversError::InvalidInput(
            "benchmark placement requires nonempty element_positions_m with 3 columns".to_owned(),
        ));
    }
    if skull_lengths_m.len() != n || amplitude_weights.len() != n {
        return Err(KwaversError::InvalidInput(
            "placement vectors must match element count".to_owned(),
        ));
    }
    let radius = config.fus.radius_m;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "focused-bowl radius must be positive and finite".to_owned(),
        ));
    }
    if !config.aperture_diameter_m.is_finite()
        || config.aperture_diameter_m <= 0.0
        || config.aperture_diameter_m > 2.0 * radius
    {
        return Err(KwaversError::InvalidInput(
            "aperture_diameter_m must be in (0, 2 * radius_m]".to_owned(),
        ));
    }

    let anchor = best_transmissive_anchor(skull_lengths_m, amplitude_weights)?;
    let half_angle = (0.5 * config.aperture_diameter_m / radius)
        .clamp(0.0, 1.0)
        .asin();
    let anchor_unit = unit_row(&element_positions_m, anchor)?;
    let mut active = Array1::<bool>::from_elem(n, false);
    for idx in 0..n {
        let unit = unit_row(&element_positions_m, idx)?;
        let dot = (anchor_unit[0] * unit[0] + anchor_unit[1] * unit[1] + anchor_unit[2] * unit[2])
            .clamp(-1.0, 1.0);
        let angle = dot.acos();
        active[idx] = angle <= half_angle
            && amplitude_weights[idx].is_finite()
            && amplitude_weights[idx] > 0.0;
    }
    let active_count = active.iter().filter(|value| **value).count();
    if active_count < config.minimum_active_elements.max(1) {
        return Err(KwaversError::InvalidInput(format!(
            "skull-aware aperture selected {active_count} active elements, below minimum {}",
            config.minimum_active_elements.max(1)
        )));
    }

    let mut sum_skull = 0.0_f64;
    let mut sum_amp = 0.0_f64;
    let mut min_amp = f64::INFINITY;
    let mut max_amp = f64::NEG_INFINITY;
    for idx in 0..n {
        if active[idx] {
            sum_skull += skull_lengths_m[idx];
            let amp = amplitude_weights[idx];
            sum_amp += amp;
            min_amp = min_amp.min(amp);
            max_amp = max_amp.max(amp);
        }
    }

    Ok(SkullAwareTransducerPlacement {
        element_positions_m,
        active_elements: active,
        aperture_anchor_index: anchor,
        active_element_count: active_count,
        aperture_diameter_m: config.aperture_diameter_m,
        radius_of_curvature_m: radius,
        focal_length_m: radius,
        mean_skull_length_m: sum_skull / active_count as f64,
        mean_amplitude_weight: sum_amp / active_count as f64,
        min_amplitude_weight: min_amp,
        max_amplitude_weight: max_amp,
    })
}

fn best_transmissive_anchor(
    skull_lengths_m: &Array1<f64>,
    amplitude_weights: &Array1<f64>,
) -> KwaversResult<usize> {
    let mut best: Option<(usize, f64)> = None;
    for idx in 0..amplitude_weights.len() {
        let skull_length = skull_lengths_m[idx];
        let amplitude = amplitude_weights[idx];
        if skull_length > 0.0 && amplitude.is_finite() && amplitude > 0.0 {
            best = match best {
                Some((_, score)) if score >= amplitude => best,
                _ => Some((idx, amplitude)),
            };
        }
    }
    if let Some((idx, _)) = best {
        return Ok(idx);
    }
    amplitude_weights
        .iter()
        .enumerate()
        .filter(|(_, amplitude)| amplitude.is_finite() && **amplitude > 0.0)
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .ok_or_else(|| {
            KwaversError::InvalidInput("no finite positive element amplitude".to_owned())
        })
}

fn unit_row(element_positions_m: &Array2<f64>, row: usize) -> KwaversResult<[f64; 3]> {
    let x = element_positions_m[[row, 0]];
    let y = element_positions_m[[row, 1]];
    let z = element_positions_m[[row, 2]];
    let norm = (x * x + y * y + z * z).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "element position has non-positive norm".to_owned(),
        ));
    }
    Ok([x / norm, y / norm, z / norm])
}
