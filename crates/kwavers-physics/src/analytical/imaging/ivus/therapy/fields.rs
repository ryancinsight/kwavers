use crate::analytical::imaging::ivus::therapy::pressure::ivus_therapy_pressure_field;
use crate::analytical::imaging::ivus::therapy::pressure::IvusTherapyPressureFieldInput;
use crate::analytical::imaging::ivus::therapy::response::ivus_therapy_response;
use crate::analytical::imaging::ivus::therapy::response::IvusTherapyDose;
use crate::analytical::imaging::ivus::therapy::response::IvusTherapyResponseInput;
use crate::analytical::imaging::ivus::therapy::response::IvusTissueMasks;

/// Complete IVUS therapy field payload for Chapter 30.
#[derive(Debug, Clone, PartialEq)]
pub struct IvusTherapyFields {
    /// Sector-focused peak pressure `Pa`.
    pub pressure_pa: Vec<f64>,
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

/// Complete IVUS therapy pressure and response fields for Chapter 30.
///
/// This helper owns the book therapy field orchestration: sector pressure,
/// intensity, absorption-weighted temperature rise, delivery fraction, and
/// scalar safety/targeting metrics.
///
/// Inputs for [`ivus_therapy_fields`].
///
/// The five masks and the seven dose scalars are the same
/// [`IvusTissueMasks`] and [`IvusTherapyDose`] the response kernel takes; what
/// stays here is what is specific to this one — a polar grid, the sector
/// geometry, and the delivery window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusTherapyFieldsInput<'a> {
    /// Radius from catheter center `m`, one per sample.
    pub radius_m: &'a [f64],
    /// Polar angle `rad`, same length as [`Self::radius_m`].
    pub theta_rad: &'a [f64],
    /// Attenuation `[dB/(cm MHz)]`, same length as [`Self::radius_m`].
    pub attenuation_db_cm_mhz: &'a [f64],
    /// The five vessel-wall tissue masks.
    pub masks: IvusTissueMasks<'a>,
    /// Therapy-dose scalars.
    pub dose: IvusTherapyDose,
    /// Sector-focused peak pressure `[Pa]`.
    pub therapy_pressure_pa: f64,
    /// Central azimuth of the therapy sector `rad`.
    pub therapy_azimuth_rad: f64,
    /// Angular width of the therapy sector `rad`.
    pub therapy_sector_width_rad: f64,
    /// Radial attenuation length of the pressure field `m`.
    pub pressure_attenuation_length_m: f64,
    /// Radial centre of the delivery band `m`.
    pub delivery_radial_center_m: f64,
    /// Radial width of the delivery band `m`.
    pub delivery_radial_width_m: f64,
}

/// # Errors
///
/// Returns an error when pressure-field or response validation fails.
pub fn ivus_therapy_fields(input: IvusTherapyFieldsInput<'_>) -> Result<IvusTherapyFields, String> {
    let IvusTherapyFieldsInput {
        radius_m,
        theta_rad,
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
        therapy_pressure_pa,
        therapy_azimuth_rad,
        therapy_sector_width_rad,
        pressure_attenuation_length_m,
        delivery_radial_center_m,
        delivery_radial_width_m,
    } = input;
    let pressure_pa = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m,
        theta_rad,
        catheter_radius_m,
        peak_pressure_pa: therapy_pressure_pa,
        therapy_azimuth_rad,
        sector_width_rad: therapy_sector_width_rad,
        attenuation_length_m: pressure_attenuation_length_m,
    })?;
    let masks = IvusTissueMasks {
        eel_mask,
        lumen_mask,
        fibrous_cap_mask,
        lipid_mask,
        plaque_mask,
    };
    let dose = IvusTherapyDose {
        catheter_radius_m,
        therapy_frequency_hz,
        therapy_duty_cycle,
        therapy_sonication_s,
        density_kg_m3,
        sound_speed_m_s,
        specific_heat_j_kg_k,
    };
    let response = ivus_therapy_response(IvusTherapyResponseInput {
        pressure_pa: &pressure_pa,
        radius_m,
        attenuation_db_cm_mhz,
        masks,
        dose,
        delivery_radial_center_m,
        delivery_radial_width_m,
    })?;

    Ok(IvusTherapyFields {
        pressure_pa,
        intensity_w_m2: response.intensity_w_m2,
        temperature_rise_k: response.temperature_rise_k,
        deposition: response.deposition,
        mechanical_index: response.mechanical_index,
        target_to_offtarget_ratio: response.target_to_offtarget_ratio,
        peak_delta_t_k: response.peak_delta_t_k,
    })
}

#[cfg(test)]
mod tests;
