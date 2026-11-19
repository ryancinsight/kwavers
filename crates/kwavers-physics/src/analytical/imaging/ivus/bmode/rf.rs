use crate::analytical::imaging::ivus::geometry::square_grid_len;

/// Inputs for [`ivus_polar_bmode_rf`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusPolarBmodeRfInput<'a> {
    /// Cartesian phantom x coordinate `m`.
    pub x_m: &'a [f64],
    /// Cartesian phantom y coordinate `m`, same length as [`Self::x_m`].
    pub y_m: &'a [f64],
    /// Normalized backscatter amplitude per phantom sample.
    pub backscatter: &'a [f64],
    /// Attenuation `[dB/(cm MHz)]` per phantom sample.
    pub attenuation_db_cm_mhz: &'a [f64],
    /// Radial axis of the polar output `m`.
    pub r_axis_m: &'a [f64],
    /// Angular axis of the polar output `rad`.
    pub theta_axis_rad: &'a [f64],
    /// Radius of the catheter itself `m`.
    pub catheter_radius_m: f64,
    /// Imaging frequency `Hz`; must be positive.
    pub frequency_hz: f64,
    /// Amplitude of the ring-down artefact, non-negative.
    pub ring_amplitude: f64,
    /// Radial width of the ring-down artefact `m`; must be positive.
    pub ring_width_m: f64,
}

/// Polar IVUS RF fixture from phantom backscatter and attenuation fields.
///
/// The function samples a Cartesian phantom on a polar catheter grid, applies
/// two-way amplitude attenuation, and adds the deterministic catheter-ring echo
/// used by the Chapter 30 IVUS B-mode panel:
///
/// `rf(r, theta) = backscatter(x,y) * exp(-2 alpha(x,y) f_MHz range)
///                 + A_ring exp(-(range / w_ring)^2)`, where
/// `range = max(r-r_catheter, 0)`.
///
/// # Errors
///
/// Returns an error when phantom arrays do not form the same square grid, when
/// axes/scalars are invalid, or when any numeric sample is non-finite. Backscatter
/// and attenuation samples must also be non-negative.
pub fn ivus_polar_bmode_rf(input: IvusPolarBmodeRfInput<'_>) -> Result<Vec<f64>, String> {
    let IvusPolarBmodeRfInput {
        x_m,
        y_m,
        backscatter,
        attenuation_db_cm_mhz,
        r_axis_m,
        theta_axis_rad,
        catheter_radius_m,
        frequency_hz,
        ring_amplitude,
        ring_width_m,
    } = input;
    let n = square_grid_len(x_m.len())?;
    if y_m.len() != x_m.len()
        || backscatter.len() != x_m.len()
        || attenuation_db_cm_mhz.len() != x_m.len()
    {
        return Err(format!(
            "phantom arrays must all have length {}; got y {}, backscatter {}, attenuation {}",
            x_m.len(),
            y_m.len(),
            backscatter.len(),
            attenuation_db_cm_mhz.len()
        ));
    }
    if r_axis_m.is_empty() || theta_axis_rad.is_empty() {
        return Err("r_axis_m and theta_axis_rad must not be empty".to_owned());
    }
    if !(catheter_radius_m.is_finite() && catheter_radius_m >= 0.0) {
        return Err("catheter_radius_m must be finite and non-negative".to_owned());
    }
    if !(frequency_hz.is_finite() && frequency_hz > 0.0) {
        return Err("frequency_hz must be finite and positive".to_owned());
    }
    if !(ring_amplitude.is_finite() && ring_amplitude >= 0.0) {
        return Err("ring_amplitude must be finite and non-negative".to_owned());
    }
    if !(ring_width_m.is_finite() && ring_width_m > 0.0) {
        return Err("ring_width_m must be finite and positive".to_owned());
    }

    let x0 = x_m[0];
    let y0 = y_m[0];
    let dx = x_m[n] - x_m[0];
    let dy = y_m[1] - y_m[0];
    if !(x0.is_finite()
        && y0.is_finite()
        && dx.is_finite()
        && dx > 0.0
        && dy.is_finite()
        && dy > 0.0)
    {
        return Err("phantom grid coordinates must be finite, monotone, and square".to_owned());
    }

    for (index, (((&x, &y), &back), &atten)) in x_m
        .iter()
        .zip(y_m)
        .zip(backscatter)
        .zip(attenuation_db_cm_mhz)
        .enumerate()
    {
        if !(x.is_finite() && y.is_finite()) {
            return Err(format!("phantom coordinate {index} must be finite"));
        }
        if !(back.is_finite() && back >= 0.0) {
            return Err(format!(
                "backscatter[{index}] must be finite and non-negative, got {back}"
            ));
        }
        if !(atten.is_finite() && atten >= 0.0) {
            return Err(format!(
                "attenuation_db_cm_mhz[{index}] must be finite and non-negative, got {atten}"
            ));
        }
    }
    for (index, &radius) in r_axis_m.iter().enumerate() {
        if !(radius.is_finite() && radius >= 0.0) {
            return Err(format!(
                "r_axis_m[{index}] must be finite and non-negative, got {radius}"
            ));
        }
    }
    for (index, &theta) in theta_axis_rad.iter().enumerate() {
        if !theta.is_finite() {
            return Err(format!(
                "theta_axis_rad[{index}] must be finite, got {theta}"
            ));
        }
    }

    let frequency_mhz = frequency_hz / 1.0e6;
    let mut rf = Vec::with_capacity(r_axis_m.len() * theta_axis_rad.len());
    for &radius in r_axis_m {
        let range = (radius - catheter_radius_m).max(0.0);
        let ring = ring_amplitude * (-(range / ring_width_m).powi(2)).exp();
        for &theta in theta_axis_rad {
            let x = radius * theta.cos();
            let y = radius * theta.sin();
            let row = nearest_grid_index(x, x0, dx, n);
            let col = nearest_grid_index(y, y0, dy, n);
            let idx = row * n + col;
            let alpha_np_m_mhz = attenuation_db_cm_mhz[idx] * 100.0 / 8.686;
            let attenuation = (-2.0 * alpha_np_m_mhz * frequency_mhz * range).exp();
            rf.push(backscatter[idx] * attenuation + ring);
        }
    }
    Ok(rf)
}

fn nearest_grid_index(value: f64, origin: f64, spacing: f64, n: usize) -> usize {
    ((value - origin) / spacing)
        .round()
        .clamp(0.0, (n - 1) as f64) as usize
}

#[cfg(test)]
mod tests;
