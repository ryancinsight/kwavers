use crate::analytical::imaging::ivus::therapy::delivery::ivus_microbubble_delivery_fraction;
use crate::analytical::imaging::ivus::therapy::delivery::IvusMicrobubbleDeliveryInput;

/// IVUS therapy response fields and scalar safety/targeting metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct IvusTherapyResponse {
    /// Time-averaged intensity [W/m²].
    pub intensity_w_m2: Vec<f64>,
    /// Adiabatic temperature rise `K`.
    pub temperature_rise_k: Vec<f64>,
    /// Microbubble delivery/deposition fraction [-].
    pub deposition: Vec<f64>,
    /// Mechanical index from the peak pressure sample [-].
    pub mechanical_index: f64,
    /// Target mean deposition divided by off-target deposition [-].
    pub target_to_offtarget_ratio: f64,
    /// Peak adiabatic temperature rise `K`.
    pub peak_delta_t_k: f64,
}

/// IVUS therapy response from pressure, tissue attenuation, and anatomy masks.
///
/// This helper owns the Chapter 30 therapy-response field algebra:
/// `I = p²/(2ρc)`, `Q = 2 α_eff I duty`, `ΔT = Q τ/(ρ c_p)`, microbubble
/// delivery through [`ivus_microbubble_delivery_fraction`], peak-pressure
/// mechanical index, and target/off-target deposition ratio.
///
/// The five vessel-wall tissue masks, one flag per sample.
///
/// Factored out because [`ivus_therapy_response`] and
/// [`ivus_therapy_fields`] take **all five** — 14 of 17 and 14 of 21 of their
/// parameters respectively. `ivus_chapter_metrics` takes three and is
/// deliberately left with named fields: a type it would only half-use is
/// speculative generality, not consolidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvusTissueMasks<'a> {
    /// External elastic lamina boundary mask.
    pub eel_mask: &'a [bool],
    /// Lumen mask.
    pub lumen_mask: &'a [bool],
    /// Fibrous-cap mask.
    pub fibrous_cap_mask: &'a [bool],
    /// Lipid-core mask.
    pub lipid_mask: &'a [bool],
    /// Plaque mask.
    pub plaque_mask: &'a [bool],
}

/// The therapy-dose scalars shared by the two IVUS therapy kernels.
///
/// Every field here is validated inside the kernel that consumes it, not here,
/// so that each function's `Err` messages stay attached to the function that
/// produces them — the tests assert on those strings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusTherapyDose {
    /// Radius of the catheter itself `m`.
    pub catheter_radius_m: f64,
    /// Therapy frequency `[Hz]`.
    pub therapy_frequency_hz: f64,
    /// Duty cycle in the interval `[0, 1]`.
    pub therapy_duty_cycle: f64,
    /// Sonication duration `s`.
    pub therapy_sonication_s: f64,
    /// Tissue density `[kg/m^3]`.
    pub density_kg_m3: f64,
    /// Tissue sound speed `[m/s]`.
    pub sound_speed_m_s: f64,
    /// Tissue specific heat `[J/(kg K)]`.
    pub specific_heat_j_kg_k: f64,
}

/// Inputs for [`ivus_therapy_response`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusTherapyResponseInput<'a> {
    /// Acoustic pressure `[Pa]`, one per sample.
    pub pressure_pa: &'a [f64],
    /// Radius from catheter center `m`, same length as [`Self::pressure_pa`].
    pub radius_m: &'a [f64],
    /// Attenuation `[dB/(cm MHz)]`, same length as [`Self::pressure_pa`].
    pub attenuation_db_cm_mhz: &'a [f64],
    /// The five vessel-wall tissue masks.
    pub masks: IvusTissueMasks<'a>,
    /// Therapy-dose scalars.
    pub dose: IvusTherapyDose,
    /// Radial centre of the delivery band `m`.
    pub delivery_radial_center_m: f64,
    /// Radial width of the delivery band `m`.
    pub delivery_radial_width_m: f64,
}

/// # Errors
///
/// Returns an error when array lengths differ, physical scalars are invalid, or
/// any sample is non-finite.
pub fn ivus_therapy_response(
    input: IvusTherapyResponseInput<'_>,
) -> Result<IvusTherapyResponse, String> {
    let IvusTherapyResponseInput {
        pressure_pa,
        radius_m,
        attenuation_db_cm_mhz,
        masks:
            IvusTissueMasks {
                eel_mask,
                lumen_mask,
                fibrous_cap_mask,
                lipid_mask,
                plaque_mask,
            },
        dose:
            IvusTherapyDose {
                catheter_radius_m,
                therapy_frequency_hz,
                therapy_duty_cycle,
                therapy_sonication_s,
                density_kg_m3,
                sound_speed_m_s,
                specific_heat_j_kg_k,
            },
        delivery_radial_center_m,
        delivery_radial_width_m,
    } = input;
    let len = pressure_pa.len();
    if radius_m.len() != len
        || attenuation_db_cm_mhz.len() != len
        || eel_mask.len() != len
        || lumen_mask.len() != len
        || fibrous_cap_mask.len() != len
        || lipid_mask.len() != len
        || plaque_mask.len() != len
    {
        return Err(format!(
            "all arrays must have length {len}; got radius {}, attenuation {}, eel {}, lumen {}, cap {}, lipid {}, plaque {}",
            radius_m.len(),
            attenuation_db_cm_mhz.len(),
            eel_mask.len(),
            lumen_mask.len(),
            fibrous_cap_mask.len(),
            lipid_mask.len(),
            plaque_mask.len()
        ));
    }
    if !(catheter_radius_m.is_finite() && catheter_radius_m >= 0.0) {
        return Err("catheter_radius_m must be finite and non-negative".to_owned());
    }
    if !(therapy_frequency_hz.is_finite() && therapy_frequency_hz > 0.0) {
        return Err("therapy_frequency_hz must be finite and positive".to_owned());
    }
    if !(therapy_duty_cycle.is_finite() && therapy_duty_cycle >= 0.0) {
        return Err("therapy_duty_cycle must be finite and non-negative".to_owned());
    }
    if !(therapy_sonication_s.is_finite() && therapy_sonication_s >= 0.0) {
        return Err("therapy_sonication_s must be finite and non-negative".to_owned());
    }
    if !(density_kg_m3.is_finite() && density_kg_m3 > 0.0) {
        return Err("density_kg_m3 must be finite and positive".to_owned());
    }
    if !(sound_speed_m_s.is_finite() && sound_speed_m_s > 0.0) {
        return Err("sound_speed_m_s must be finite and positive".to_owned());
    }
    if !(specific_heat_j_kg_k.is_finite() && specific_heat_j_kg_k > 0.0) {
        return Err("specific_heat_j_kg_k must be finite and positive".to_owned());
    }

    let intensity_w_m2 = crate::analytical::thermal::acoustic_intensity_from_amplitude(
        pressure_pa,
        density_kg_m3,
        sound_speed_m_s,
    );
    let frequency_mhz = therapy_frequency_hz / 1.0e6;
    let mut range_m = Vec::with_capacity(len);
    let mut alpha_eff_np_m = Vec::with_capacity(len);
    let mut absorbed_power_w_m3 = Vec::with_capacity(len);
    let mut wall_mask = Vec::with_capacity(len);
    let mut target_mask = Vec::with_capacity(len);
    let mut peak_pressure_pa = 0.0_f64;

    for index in 0..len {
        let pressure = pressure_pa[index];
        let radius = radius_m[index];
        let attenuation = attenuation_db_cm_mhz[index];
        if !pressure.is_finite() {
            return Err(format!(
                "pressure_pa[{index}] must be finite, got {pressure}"
            ));
        }
        if !radius.is_finite() {
            return Err(format!("radius_m[{index}] must be finite, got {radius}"));
        }
        if !(attenuation.is_finite() && attenuation >= 0.0) {
            return Err(format!(
                "attenuation_db_cm_mhz[{index}] must be finite and non-negative, got {attenuation}"
            ));
        }
        let alpha = attenuation * 100.0 / 8.686 * frequency_mhz;
        let range = (radius - catheter_radius_m).max(0.0);
        let wall = eel_mask[index] && !lumen_mask[index];
        let target = fibrous_cap_mask[index] || lipid_mask[index];

        peak_pressure_pa = peak_pressure_pa.max(pressure.abs());
        range_m.push(range);
        alpha_eff_np_m.push(alpha);
        absorbed_power_w_m3.push(2.0 * alpha * intensity_w_m2[index] * therapy_duty_cycle);
        wall_mask.push(wall);
        target_mask.push(target);
    }

    let tau_s = vec![therapy_sonication_s; len];
    let temperature_rise_k = crate::analytical::thermal::adiabatic_temperature_rise_kelvin(
        &absorbed_power_w_m3,
        &tau_s,
        density_kg_m3,
        specific_heat_j_kg_k,
    );
    let deposition = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
        range_m: &range_m,
        attenuation_np_m: &alpha_eff_np_m,
        intensity_w_m2: &intensity_w_m2,
        wall_mask: &wall_mask,
        target_mask: &target_mask,
        sound_speed_m_s,
        radial_center_m: delivery_radial_center_m,
        radial_width_m: delivery_radial_width_m,
    })?;
    let target_to_offtarget_ratio =
        deposition_target_to_offtarget_ratio(&deposition, &target_mask, plaque_mask)?;
    let peak_delta_t_k = temperature_rise_k.iter().copied().fold(0.0_f64, f64::max);

    Ok(IvusTherapyResponse {
        intensity_w_m2,
        temperature_rise_k,
        deposition,
        mechanical_index: crate::analytical::safety::mechanical_index(
            peak_pressure_pa,
            therapy_frequency_hz,
        ),
        target_to_offtarget_ratio,
        peak_delta_t_k,
    })
}

fn deposition_target_to_offtarget_ratio(
    deposition: &[f64],
    target_mask: &[bool],
    plaque_mask: &[bool],
) -> Result<f64, String> {
    let mut target_sum = 0.0;
    let mut target_count = 0_usize;
    let mut off_sum = 0.0;
    let mut off_count = 0_usize;

    for (index, ((&value, &target), &plaque)) in deposition
        .iter()
        .zip(target_mask)
        .zip(plaque_mask)
        .enumerate()
    {
        if !value.is_finite() {
            return Err(format!("deposition[{index}] must be finite, got {value}"));
        }
        if target {
            target_sum += value;
            target_count += 1;
        }
        if !plaque && value > 0.0 {
            off_sum += value;
            off_count += 1;
        }
    }

    if target_count == 0 {
        return Err("target mask must select at least one sample".to_owned());
    }
    let target_mean = target_sum / target_count as f64;
    let off_mean = if off_count == 0 {
        1.0e-12
    } else {
        (off_sum / off_count as f64).max(1.0e-12)
    };
    Ok(target_mean / off_mean)
}

#[cfg(test)]
mod tests;
