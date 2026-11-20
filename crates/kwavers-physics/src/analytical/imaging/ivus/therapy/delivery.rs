/// IVUS microbubble delivery fraction from acoustic radiation force.
///
/// The model computes `F = 2 * alpha * I / c`, applies a Gaussian radial band
/// and wall/target weights, normalizes by the maximum weighted force, then maps
/// to delivered fraction with `1 - exp(-3 * normalized)`.
///
/// # Errors
///
/// Returns an error when array lengths differ, when any sample is non-finite,
/// or when `sound_speed_m_s`, `radial_center_m`, or `radial_width_m` are not
/// finite positive values.
pub struct IvusMicrobubbleDeliveryInput<'a> {
    /// Radial distance from the catheter wall for each sample.
    pub range_m: &'a [f64],
    /// Acoustic attenuation coefficient for each sample.
    pub attenuation_np_m: &'a [f64],
    /// Local acoustic intensity for each sample.
    pub intensity_w_m2: &'a [f64],
    /// Wall membership mask.
    pub wall_mask: &'a [bool],
    /// Target membership mask.
    pub target_mask: &'a [bool],
    /// Tissue sound speed used for acoustic radiation-force scaling.
    pub sound_speed_m_s: f64,
    /// Radial center of the delivery band.
    pub radial_center_m: f64,
    /// Radial width of the delivery band.
    pub radial_width_m: f64,
}

/// # Errors
///
/// Returns a descriptive error when input arrays have different lengths or a physical parameter
/// is non-finite or outside its valid domain.
pub fn ivus_microbubble_delivery_fraction(
    input: IvusMicrobubbleDeliveryInput<'_>,
) -> Result<Vec<f64>, String> {
    let IvusMicrobubbleDeliveryInput {
        range_m,
        attenuation_np_m,
        intensity_w_m2,
        wall_mask,
        target_mask,
        sound_speed_m_s,
        radial_center_m,
        radial_width_m,
    } = input;
    let len = range_m.len();
    if attenuation_np_m.len() != len
        || intensity_w_m2.len() != len
        || wall_mask.len() != len
        || target_mask.len() != len
    {
        return Err(format!(
            "all arrays must have length {len}; got attenuation {}, intensity {}, wall {}, target {}",
            attenuation_np_m.len(),
            intensity_w_m2.len(),
            wall_mask.len(),
            target_mask.len()
        ));
    }
    if !(sound_speed_m_s.is_finite() && sound_speed_m_s > 0.0) {
        return Err("sound_speed_m_s must be finite and positive".to_owned());
    }
    if !(radial_center_m.is_finite() && radial_center_m >= 0.0) {
        return Err("radial_center_m must be finite and non-negative".to_owned());
    }
    if !(radial_width_m.is_finite() && radial_width_m > 0.0) {
        return Err("radial_width_m must be finite and positive".to_owned());
    }

    let mut weighted_force = Vec::with_capacity(len);
    let mut max_force = 0.0_f64;
    for index in 0..len {
        let range = range_m[index];
        let attenuation = attenuation_np_m[index];
        let intensity = intensity_w_m2[index];
        if !range.is_finite() {
            return Err(format!("range_m[{index}] must be finite, got {range}"));
        }
        if !(attenuation.is_finite() && attenuation >= 0.0) {
            return Err(format!(
                "attenuation_np_m[{index}] must be finite and non-negative, got {attenuation}"
            ));
        }
        if !(intensity.is_finite() && intensity >= 0.0) {
            return Err(format!(
                "intensity_w_m2[{index}] must be finite and non-negative, got {intensity}"
            ));
        }

        let radial = (-((range - radial_center_m) / radial_width_m).powi(2)).exp();
        let wall_weight = if wall_mask[index] { 0.20 } else { 0.0 };
        let target_weight = if target_mask[index] { 0.80 } else { 0.0 };
        let tissue_weight = wall_weight + target_weight;
        let force = 2.0 * attenuation * intensity / sound_speed_m_s;
        let weighted = force * radial * tissue_weight;
        max_force = max_force.max(weighted);
        weighted_force.push(weighted);
    }

    let denominator = max_force.max(1.0e-12);
    Ok(weighted_force
        .iter()
        .map(|force| 1.0 - (-3.0 * (force / denominator)).exp())
        .collect())
}

#[cfg(test)]
mod tests;
