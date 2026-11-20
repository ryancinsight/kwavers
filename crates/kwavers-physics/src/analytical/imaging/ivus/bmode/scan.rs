/// Nearest-neighbour IVUS polar-to-Cartesian scan conversion.
///
/// `polar` is row-major `(r_axis_m.len(), theta_axis_rad.len())`. Each phantom
/// sample is mapped to its nearest polar radius/angle bin. Samples outside the
/// radial axis are set to zero; angular samples wrap periodically.
///
/// # Errors
///
/// Returns an error when axes are too short, array lengths are inconsistent,
/// axes are not finite/increasing, or any sample is non-finite.
///
/// # Panics
///
/// Panics only if the checked radial axis is empty when extracting its final
/// sample, which would violate the preceding length validation.
pub fn ivus_scan_convert(
    polar: &[f64],
    r_axis_m: &[f64],
    theta_axis_rad: &[f64],
    radius_m: &[f64],
    theta_rad: &[f64],
) -> Result<Vec<f64>, String> {
    if r_axis_m.len() < 2 || theta_axis_rad.len() < 2 {
        return Err(
            "r_axis_m and theta_axis_rad must each contain at least two samples".to_owned(),
        );
    }
    if radius_m.len() != theta_rad.len() {
        return Err(format!(
            "radius_m length {} must match theta_rad length {}",
            radius_m.len(),
            theta_rad.len()
        ));
    }
    let expected_polar_len = r_axis_m
        .len()
        .checked_mul(theta_axis_rad.len())
        .ok_or_else(|| "polar grid length overflow".to_owned())?;
    if polar.len() != expected_polar_len {
        return Err(format!(
            "polar length {} must equal r_axis_m.len() * theta_axis_rad.len() ({expected_polar_len})",
            polar.len()
        ));
    }

    let r0 = r_axis_m[0];
    let r_last = *r_axis_m.last().expect("invariant: length checked above");
    let dr = r_axis_m[1] - r_axis_m[0];
    let theta0 = theta_axis_rad[0];
    let dtheta = theta_axis_rad[1] - theta_axis_rad[0];
    if !(r0.is_finite() && r_last.is_finite() && dr.is_finite() && dr > 0.0) {
        return Err("r_axis_m must be finite and strictly increasing".to_owned());
    }
    if !(theta0.is_finite() && dtheta.is_finite() && dtheta > 0.0) {
        return Err("theta_axis_rad must be finite and strictly increasing".to_owned());
    }
    for (index, &radius) in r_axis_m.iter().enumerate() {
        if !radius.is_finite() || (index > 0 && radius <= r_axis_m[index - 1]) {
            return Err(format!(
                "r_axis_m[{index}] must be finite and strictly increasing"
            ));
        }
    }
    for (index, &theta) in theta_axis_rad.iter().enumerate() {
        if !theta.is_finite() || (index > 0 && theta <= theta_axis_rad[index - 1]) {
            return Err(format!(
                "theta_axis_rad[{index}] must be finite and strictly increasing"
            ));
        }
    }
    for (index, &value) in polar.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("polar[{index}] must be finite, got {value}"));
        }
    }

    let n_theta = theta_axis_rad.len() as isize;
    let mut image = Vec::with_capacity(radius_m.len());
    for (index, (&radius, &theta)) in radius_m.iter().zip(theta_rad).enumerate() {
        if !radius.is_finite() {
            return Err(format!("radius_m[{index}] must be finite, got {radius}"));
        }
        if !theta.is_finite() {
            return Err(format!("theta_rad[{index}] must be finite, got {theta}"));
        }
        if radius < r0 || radius > r_last {
            image.push(0.0);
            continue;
        }

        let ri = ((radius - r0) / dr)
            .round()
            .clamp(0.0, (r_axis_m.len() - 1) as f64) as usize;
        let ti = (((theta - theta0) / dtheta).round() as isize).rem_euclid(n_theta) as usize;
        image.push(polar[ri * theta_axis_rad.len() + ti]);
    }
    Ok(image)
}

#[cfg(test)]
mod tests;
