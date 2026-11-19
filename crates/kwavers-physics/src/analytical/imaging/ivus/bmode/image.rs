use crate::analytical::imaging::ivus::bmode::rf::ivus_polar_bmode_rf;
use crate::analytical::imaging::ivus::bmode::rf::IvusPolarBmodeRfInput;
use crate::analytical::imaging::ivus::bmode::scan::ivus_scan_convert;
use leto::Array1;

/// Complete IVUS B-mode image payload for Chapter 30.
#[derive(Debug, Clone, PartialEq)]
pub struct IvusBmodeImage {
    /// Polar RF fixture, row-major `(r_axis, theta_axis)`.
    pub rf: Vec<f64>,
    /// Hilbert-envelope magnitude, row-major `(r_axis, theta_axis)`.
    pub envelope: Vec<f64>,
    /// Log-compressed decibel image, row-major `(r_axis, theta_axis)`.
    pub db: Vec<f64>,
    /// Normalized polar B-mode image in `[0, 1]`, row-major `(r_axis, theta_axis)`.
    pub polar: Vec<f64>,
    /// Cartesian phantom-grid B-mode image, row-major matching `radius_m`.
    pub cartesian: Vec<f64>,
}

/// Inputs for [`ivus_bmode_image`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvusBmodeImageInput<'a> {
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
    /// Radius of the scan-conversion source, `m` per Cartesian sample.
    pub radius_m: &'a [f64],
    /// Angle of the scan-conversion source, `rad` per Cartesian sample.
    pub theta_m_rad: &'a [f64],
    /// Radius of the catheter itself `m`.
    pub catheter_radius_m: f64,
    /// Imaging frequency `Hz`.
    pub frequency_hz: f64,
    /// Log-compression floor `dB`; must be finite and negative.
    pub floor_db: f64,
    /// Amplitude of the ring-down artefact, non-negative.
    pub ring_amplitude: f64,
    /// Radial width of the ring-down artefact `m`; must be positive.
    pub ring_width_m: f64,
}

/// Complete IVUS B-mode RF-to-display fixture for Chapter 30.
///
/// The function builds the polar RF field, computes Hilbert envelopes per
/// angular line, applies fixed-reference log compression, normalizes the polar
/// image to `[0, 1]`, and scan-converts the result onto the Cartesian phantom
/// grid.
///
/// # Errors
///
/// Returns an error when any subordinate IVUS RF or scan-conversion validation
/// fails, when the log-compression floor is not finite and negative, or when
/// the polar grid is empty.
pub fn ivus_bmode_image(input: IvusBmodeImageInput<'_>) -> Result<IvusBmodeImage, String> {
    let IvusBmodeImageInput {
        x_m,
        y_m,
        backscatter,
        attenuation_db_cm_mhz,
        r_axis_m,
        theta_axis_rad,
        radius_m,
        theta_m_rad,
        catheter_radius_m,
        frequency_hz,
        floor_db,
        ring_amplitude,
        ring_width_m,
    } = input;
    if !(floor_db.is_finite() && floor_db < 0.0) {
        return Err("floor_db must be finite and negative".to_owned());
    }
    let n_r = r_axis_m.len();
    let n_theta = theta_axis_rad.len();
    if n_r == 0 || n_theta == 0 {
        return Err("r_axis_m and theta_axis_rad must not be empty".to_owned());
    }

    let rf = ivus_polar_bmode_rf(IvusPolarBmodeRfInput {
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
    })?;
    let mut envelope = vec![0.0; rf.len()];
    for col in 0..n_theta {
        let line = Array1::from_shape_fn([n_r], |[row]| rf[row * n_theta + col]);
        let analytic = kwavers_math::fft::analytic_signal_1d(&line);
        for (row, sample) in analytic.iter().enumerate() {
            envelope[row * n_theta + col] = sample.norm().max(1.0e-9);
        }
    }

    let reference = envelope.iter().copied().fold(0.0_f64, f64::max);
    if !(reference.is_finite() && reference > 0.0) {
        return Err("B-mode envelope reference must be finite and positive".to_owned());
    }
    let db =
        crate::analytical::pulse_echo::bmode_db_fixed_reference(&envelope, reference, floor_db);
    let dynamic_range = -floor_db;
    let polar: Vec<f64> = db
        .iter()
        .map(|&value| ((value - floor_db) / dynamic_range).clamp(0.0, 1.0))
        .collect();
    let cartesian = ivus_scan_convert(&polar, r_axis_m, theta_axis_rad, radius_m, theta_m_rad)?;

    Ok(IvusBmodeImage {
        rf,
        envelope,
        db,
        polar,
        cartesian,
    })
}

#[cfg(test)]
mod tests;
