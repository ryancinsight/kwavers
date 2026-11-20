use crate::analytical::imaging::ivus::geometry::angle_difference;

/// Inputs for [`ivus_therapy_pressure_field`].
///
/// Borrows its arrays rather than owning them, like every other `*Input` in
/// this module, so a caller sampling on the same grid pays one allocation in
/// total instead of one per call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusTherapyPressureFieldInput<'a> {
    /// Radius from catheter center `m`, one per sample.
    pub radius_m: &'a [f64],
    /// Polar angle `rad`, same length as [`Self::radius_m`].
    pub theta_rad: &'a [f64],
    /// Radius of the catheter itself `m`; samples inside it score zero.
    pub catheter_radius_m: f64,
    /// Sector-focused peak pressure `Pa`.
    pub peak_pressure_pa: f64,
    /// Central azimuth of the therapy sector `rad`.
    pub therapy_azimuth_rad: f64,
    /// Angular width of the therapy sector `rad`; must be positive.
    pub sector_width_rad: f64,
    /// Radial attenuation length `m`; must be positive.
    pub attenuation_length_m: f64,
}

/// IVUS therapy pressure field for a sector-focused intravascular source.
///
/// The model applies a Gaussian angular aperture and exponential radial decay:
///
/// `p(r, theta) = p0 * exp(-0.5 * (wrap(theta - theta0) / sigma_theta)^2)
///                * exp(-max(r - r_catheter, 0) / decay_length)`.
///
/// Samples inside the catheter radius are set to zero.
///
/// # Errors
///
/// Returns an error when arrays differ in length or when any scalar/sample is
/// non-finite. `sector_width_rad` and `attenuation_length_m` must be positive;
/// `catheter_radius_m` and `peak_pressure_pa` must be non-negative.
pub fn ivus_therapy_pressure_field(
    input: IvusTherapyPressureFieldInput<'_>,
) -> Result<Vec<f64>, String> {
    let IvusTherapyPressureFieldInput {
        radius_m,
        theta_rad,
        catheter_radius_m,
        peak_pressure_pa,
        therapy_azimuth_rad,
        sector_width_rad,
        attenuation_length_m,
    } = input;
    if radius_m.len() != theta_rad.len() {
        return Err(format!(
            "radius_m length {} must match theta_rad length {}",
            radius_m.len(),
            theta_rad.len()
        ));
    }
    if !(catheter_radius_m.is_finite() && catheter_radius_m >= 0.0) {
        return Err("catheter_radius_m must be finite and non-negative".to_owned());
    }
    if !(peak_pressure_pa.is_finite() && peak_pressure_pa >= 0.0) {
        return Err("peak_pressure_pa must be finite and non-negative".to_owned());
    }
    if !therapy_azimuth_rad.is_finite() {
        return Err("therapy_azimuth_rad must be finite".to_owned());
    }
    if !(sector_width_rad.is_finite() && sector_width_rad > 0.0) {
        return Err("sector_width_rad must be finite and positive".to_owned());
    }
    if !(attenuation_length_m.is_finite() && attenuation_length_m > 0.0) {
        return Err("attenuation_length_m must be finite and positive".to_owned());
    }

    radius_m
        .iter()
        .zip(theta_rad)
        .enumerate()
        .map(|(index, (&radius, &theta))| {
            if !radius.is_finite() {
                return Err(format!("radius_m[{index}] must be finite, got {radius}"));
            }
            if !theta.is_finite() {
                return Err(format!("theta_rad[{index}] must be finite, got {theta}"));
            }
            if radius <= catheter_radius_m {
                return Ok(0.0);
            }

            let angular = angle_difference(theta, therapy_azimuth_rad) / sector_width_rad;
            let range_m = (radius - catheter_radius_m).max(0.0);
            Ok(peak_pressure_pa
                * (-0.5 * angular * angular).exp()
                * (-range_m / attenuation_length_m).exp())
        })
        .collect()
}

#[cfg(test)]
mod tests;
