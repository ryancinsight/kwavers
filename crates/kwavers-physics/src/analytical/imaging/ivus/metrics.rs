use crate::analytical::imaging::ivus::geometry::square_grid_len;

/// Chapter 30 IVUS scalar metrics computed from Rust-owned fields.
#[derive(Debug, Clone, PartialEq)]
pub struct IvusChapterMetrics {
    /// Imaging wavelength `µM`.
    pub imaging_wavelength_um: f64,
    /// Therapy wavelength `mm`.
    pub therapy_wavelength_mm: f64,
    /// Lumen mask area `mm²`.
    pub lumen_area_mm2: f64,
    /// Plaque mask area `mm²`.
    pub plaque_area_mm2: f64,
    /// B-mode display dynamic range `dB`.
    pub bmode_dynamic_range_db: f64,
    /// Mean B-mode intensity inside the lumen mask.
    pub bmode_mean_lumen_intensity: f64,
    /// Mean B-mode intensity inside the vessel wall mask.
    pub bmode_mean_wall_intensity: f64,
    /// Mechanical index [-].
    pub therapy_mechanical_index: f64,
    /// Peak temperature rise [°C].
    pub therapy_peak_delta_t_c: f64,
    /// Target/off-target deposition ratio [-].
    pub therapy_target_to_offtarget_deposition_ratio: f64,
}

/// Inputs for [`ivus_chapter_metrics`].
///
/// The three masks the metrics are computed over are kept as separate fields
/// rather than a shared mask type: the therapy functions that *do* share all
/// five masks get one, and a two-of-five overlap is not a reason to invent a
/// type this function would only half-use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusChapterMetricsInput<'a> {
    /// Cartesian phantom x coordinate `m`.
    pub x_m: &'a [f64],
    /// Cartesian phantom y coordinate `m`, same length as [`Self::x_m`].
    pub y_m: &'a [f64],
    /// Lumen mask, one flag per phantom sample.
    pub lumen_mask: &'a [bool],
    /// External elastic lamina mask, same length as [`Self::lumen_mask`].
    pub eel_mask: &'a [bool],
    /// Plaque mask, same length as [`Self::lumen_mask`].
    pub plaque_mask: &'a [bool],
    /// Cartesian B-mode image the display metrics are read from.
    pub bmode_cartesian: &'a [f64],
    /// Tissue sound speed `[m/s]`; must be positive.
    pub sound_speed_m_s: f64,
    /// Imaging frequency `[Hz]`; must be positive.
    pub imaging_frequency_hz: f64,
    /// Therapy frequency `[Hz]`; must be positive.
    pub therapy_frequency_hz: f64,
    /// B-mode display dynamic range `[dB]`; must be positive.
    pub bmode_dynamic_range_db: f64,
    /// Mechanical index of the therapy pulse `[-]`.
    pub therapy_mechanical_index: f64,
    /// Peak adiabatic temperature rise `[K]`.
    pub therapy_peak_delta_t_c: f64,
    /// Target to off-target deposition ratio `[-]`.
    pub therapy_target_to_offtarget_deposition_ratio: f64,
}

/// Compute Chapter 30 IVUS scalar metrics from Rust-owned fields.
///
/// # Errors
///
/// Returns an error when grid arrays are inconsistent, masks are empty, scalar
/// frequencies/sound speed are invalid, or any B-mode sample is non-finite.
pub fn ivus_chapter_metrics(
    input: IvusChapterMetricsInput<'_>,
) -> Result<IvusChapterMetrics, String> {
    let IvusChapterMetricsInput {
        x_m,
        y_m,
        lumen_mask,
        eel_mask,
        plaque_mask,
        bmode_cartesian,
        sound_speed_m_s,
        imaging_frequency_hz,
        therapy_frequency_hz,
        bmode_dynamic_range_db,
        therapy_mechanical_index,
        therapy_peak_delta_t_c,
        therapy_target_to_offtarget_deposition_ratio,
    } = input;
    let len = x_m.len();
    if len < 4 {
        return Err("phantom grid must contain at least four samples".to_owned());
    }
    if y_m.len() != len
        || lumen_mask.len() != len
        || eel_mask.len() != len
        || plaque_mask.len() != len
        || bmode_cartesian.len() != len
    {
        return Err(format!(
            "all arrays must have length {len}; got y {}, lumen {}, eel {}, plaque {}, bmode {}",
            y_m.len(),
            lumen_mask.len(),
            eel_mask.len(),
            plaque_mask.len(),
            bmode_cartesian.len()
        ));
    }
    if !(sound_speed_m_s.is_finite() && sound_speed_m_s > 0.0) {
        return Err("sound_speed_m_s must be finite and positive".to_owned());
    }
    if !(imaging_frequency_hz.is_finite() && imaging_frequency_hz > 0.0) {
        return Err("imaging_frequency_hz must be finite and positive".to_owned());
    }
    if !(therapy_frequency_hz.is_finite() && therapy_frequency_hz > 0.0) {
        return Err("therapy_frequency_hz must be finite and positive".to_owned());
    }
    if !(bmode_dynamic_range_db.is_finite() && bmode_dynamic_range_db > 0.0) {
        return Err("bmode_dynamic_range_db must be finite and positive".to_owned());
    }
    if !(therapy_mechanical_index.is_finite()
        && therapy_peak_delta_t_c.is_finite()
        && therapy_target_to_offtarget_deposition_ratio.is_finite())
    {
        return Err("therapy scalar metrics must be finite".to_owned());
    }

    let n = square_grid_len(len)?;
    let dx = x_m[n] - x_m[0];
    let dy = y_m[1] - y_m[0];
    if !(dx.is_finite() && dx > 0.0 && dy.is_finite() && dy > 0.0) {
        return Err("phantom grid spacing must be finite and positive".to_owned());
    }
    for (index, ((&x, &y), &bmode)) in x_m.iter().zip(y_m).zip(bmode_cartesian).enumerate() {
        if !(x.is_finite() && y.is_finite()) {
            return Err(format!("phantom coordinate {index} must be finite"));
        }
        if !bmode.is_finite() {
            return Err(format!(
                "bmode_cartesian[{index}] must be finite, got {bmode}"
            ));
        }
    }

    let pixel_area_mm2 = dx * dy * 1.0e6;
    let lumen_count = lumen_mask.iter().filter(|&&value| value).count();
    let plaque_count = plaque_mask.iter().filter(|&&value| value).count();
    if lumen_count == 0 || plaque_count == 0 {
        return Err("lumen and plaque masks must each select at least one sample".to_owned());
    }
    let wall_mask: Vec<bool> = eel_mask
        .iter()
        .zip(lumen_mask)
        .map(|(&eel, &lumen)| eel && !lumen)
        .collect();
    let bmode_mean_lumen_intensity = finite_masked_mean(bmode_cartesian, lumen_mask, "lumen")?;
    let bmode_mean_wall_intensity = finite_masked_mean(bmode_cartesian, &wall_mask, "wall")?;

    Ok(IvusChapterMetrics {
        imaging_wavelength_um: sound_speed_m_s / imaging_frequency_hz * 1.0e6,
        therapy_wavelength_mm: sound_speed_m_s / therapy_frequency_hz * 1.0e3,
        lumen_area_mm2: lumen_count as f64 * pixel_area_mm2,
        plaque_area_mm2: plaque_count as f64 * pixel_area_mm2,
        bmode_dynamic_range_db,
        bmode_mean_lumen_intensity,
        bmode_mean_wall_intensity,
        therapy_mechanical_index,
        therapy_peak_delta_t_c,
        therapy_target_to_offtarget_deposition_ratio,
    })
}

fn finite_masked_mean(values: &[f64], mask: &[bool], name: &str) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (index, (&value, &selected)) in values.iter().zip(mask).enumerate() {
        if !value.is_finite() {
            return Err(format!("{name} value {index} must be finite, got {value}"));
        }
        if selected {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return Err(format!("{name} mask must select at least one sample"));
    }
    Ok(sum / count as f64)
}

#[cfg(test)]
mod tests;
