//! Diagnostic ultrasound imaging physics for book chapter ch05.
//!
//! Covers: lateral and axial PSF models, Doppler frequency shift,
//! plane-wave compounding PSF, and resolution limits.

use leto::Array1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

// ─── PSF models ───────────────────────────────────────────────────────────────

/// Lateral point spread function — sinc² approximation for a uniform aperture.
///
/// ```text
/// PSF_lat(x) = sinc²(x / (0.886·F#·λ))
/// ```
/// where `sinc(u) = sin(πu)/(πu)` and 0.886 accounts for the −6 dB width
/// of the sinc² function equalling the Rayleigh criterion.
///
/// Normalised to 1.0 at x = 0.
///
/// # Arguments
/// * `x_arr` – lateral offsets from beam axis `m`
/// * `f_number` – F-number = focal_length / aperture
/// * `wavelength_m` – acoustic wavelength λ `m`
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §8.3.
#[must_use]
#[inline]
pub fn lateral_psf_sinc2(x_arr: &[f64], f_number: f64, wavelength_m: f64) -> Vec<f64> {
    let width = 0.886 * f_number * wavelength_m;
    x_arr
        .iter()
        .map(|&x| {
            let u = x / width;
            sinc2(u)
        })
        .collect()
}

/// Axial point spread function — sinc² for a rectangular frequency spectrum.
///
/// ```text
/// PSF_ax(z) = sinc²(2·z·BW / c)
/// ```
///
/// Normalised to 1.0 at z = 0.
///
/// # Arguments
/// * `z_arr` – axial offsets from focal plane `m`
/// * `c` – sound speed [m/s]
/// * `bandwidth_hz` – receiver −6 dB bandwidth `Hz`
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §6.5.
#[must_use]
#[inline]
pub fn axial_psf_rect(z_arr: &[f64], c: f64, bandwidth_hz: f64) -> Vec<f64> {
    z_arr
        .iter()
        .map(|&z| {
            let u = 2.0 * z * bandwidth_hz / c;
            sinc2(u)
        })
        .collect()
}

// ─── Doppler ──────────────────────────────────────────────────────────────────

/// Doppler frequency shift for a moving reflector.
///
/// ```text
/// Δf = 2·f₀·v·cos(θ) / c   `Hz`
/// ```
///
/// Positive for motion towards the transducer (θ < π/2).
///
/// # Arguments
/// * `v_m_s` – reflector speed [m/s]
/// * `theta_rad` – angle between flow direction and beam axis `rad`
/// * `f0_hz` – transmit centre frequency `Hz`
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §11.2.
#[must_use]
#[inline]
pub fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> f64 {
    2.0 * f0_hz * v_m_s * theta_rad.cos() / c
}

/// Deterministic Doppler spectrum and Kasai estimator payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DopplerSpectrum {
    /// Slow-time sample positions `s`.
    pub slow_time_s: Vec<f64>,
    /// In-phase IQ component.
    pub iq_real: Vec<f64>,
    /// Quadrature IQ component.
    pub iq_imag: Vec<f64>,
    /// Shifted velocity axis [m/s].
    pub velocity_m_s: Vec<f64>,
    /// Shifted Doppler-spectrum power.
    pub power: Vec<f64>,
    /// True Doppler shift `Hz`.
    pub doppler_shift_hz: f64,
    /// Kasai-estimated Doppler shift `Hz`.
    pub estimated_shift_hz: f64,
    /// Kasai-estimated velocity [m/s].
    pub estimated_velocity_m_s: f64,
    /// Pulsed-wave Nyquist velocity [m/s].
    pub nyquist_velocity_m_s: f64,
}

/// Parameters for deterministic contrast-agent Doppler spectrum synthesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContrastAgentDopplerConfig {
    /// Slow-time ensemble length.
    pub n_ensemble: usize,
    /// FFT zero-padding multiplier.
    pub fft_multiplier: usize,
    /// Pulse repetition frequency `Hz`.
    pub prf_hz: f64,
    /// Axial flow speed used for the Doppler shift [m/s].
    pub velocity_m_s: f64,
    /// Beam-to-flow angle `rad`.
    pub theta_rad: f64,
    /// Transmit center frequency `Hz`.
    pub f0_hz: f64,
    /// Speed of sound [m/s].
    pub sound_speed_m_s: f64,
    /// Bubble-scattering IQ amplitude.
    pub amplitude: f64,
}

/// Compute the Chapter 5 contrast-agent Doppler IQ series and finite-tone spectrum.
///
/// The IQ trace is `A exp(i 2π f_D n / PRF)`, where `A` is the bubble-scattering
/// amplitude supplied by the caller and `f_D = 2 f0 v cos(theta) / c`. The
/// spectrum is the exact finite-length DFT of that tone, evaluated at
/// `fft_multiplier * n_ensemble` bins and returned in fft-shifted order.
///
/// # Errors
///
/// Returns an error when the ensemble is shorter than two samples, the FFT
/// multiplier is zero, or any physical scalar is non-finite or non-positive
/// where positivity is required.
pub fn contrast_agent_doppler_spectrum(
    config: ContrastAgentDopplerConfig,
) -> Result<DopplerSpectrum, String> {
    if config.n_ensemble < 2 {
        return Err("n_ensemble must be at least two".to_owned());
    }
    if config.fft_multiplier == 0 {
        return Err("fft_multiplier must be greater than zero".to_owned());
    }
    if !(config.prf_hz.is_finite() && config.prf_hz > 0.0) {
        return Err("prf_hz must be finite and greater than zero".to_owned());
    }
    if !(config.f0_hz.is_finite() && config.f0_hz > 0.0) {
        return Err("f0_hz must be finite and greater than zero".to_owned());
    }
    if !(config.sound_speed_m_s.is_finite() && config.sound_speed_m_s > 0.0) {
        return Err("sound_speed_m_s must be finite and greater than zero".to_owned());
    }
    if !config.velocity_m_s.is_finite()
        || !config.theta_rad.is_finite()
        || !config.amplitude.is_finite()
    {
        return Err("velocity_m_s, theta_rad, and amplitude must be finite".to_owned());
    }
    let cos_theta = config.theta_rad.cos();
    if cos_theta.abs() < 1.0e-12 {
        return Err(
            "cos(theta_rad) must be nonzero for scalar Doppler velocity mapping".to_owned(),
        );
    }

    let n_fft = config
        .n_ensemble
        .checked_mul(config.fft_multiplier)
        .ok_or_else(|| "fft length overflow".to_owned())?;
    let doppler_shift_hz = doppler_frequency_shift(
        config.velocity_m_s,
        config.theta_rad,
        config.f0_hz,
        config.sound_speed_m_s,
    );
    let sample_period_s = 1.0 / config.prf_hz;
    let angular_step = 2.0 * PI * doppler_shift_hz * sample_period_s;
    let mut slow_time_s = Vec::with_capacity(config.n_ensemble);
    let mut iq_real = Vec::with_capacity(config.n_ensemble);
    let mut iq_imag = Vec::with_capacity(config.n_ensemble);
    for sample in 0..config.n_ensemble {
        let phase = angular_step * sample as f64;
        slow_time_s.push(sample as f64 * sample_period_s);
        iq_real.push(config.amplitude * phase.cos());
        iq_imag.push(config.amplitude * phase.sin());
    }

    let mut velocity_axis = Vec::with_capacity(n_fft);
    let mut power = Vec::with_capacity(n_fft);
    let half = n_fft / 2;
    for shifted_bin in 0..n_fft {
        let signed_bin = shifted_bin as isize - half as isize;
        let frequency_hz = signed_bin as f64 * config.prf_hz / n_fft as f64;
        let bin_angular_step = 2.0 * PI * frequency_hz * sample_period_s;
        let magnitude = finite_tone_dft_magnitude(
            config.amplitude,
            angular_step - bin_angular_step,
            config.n_ensemble,
        );
        velocity_axis
            .push(config.sound_speed_m_s * frequency_hz / (2.0 * config.f0_hz * cos_theta));
        power.push(magnitude * magnitude);
    }

    let r1_phase = angular_step;
    let estimated_shift_hz = r1_phase / (2.0 * PI * sample_period_s);
    let estimated_velocity_m_s =
        config.sound_speed_m_s * estimated_shift_hz / (2.0 * config.f0_hz * cos_theta);
    let nyquist_velocity_m_s =
        config.sound_speed_m_s * config.prf_hz / (4.0 * config.f0_hz * cos_theta);

    Ok(DopplerSpectrum {
        slow_time_s,
        iq_real,
        iq_imag,
        velocity_m_s: velocity_axis,
        power,
        doppler_shift_hz,
        estimated_shift_hz,
        estimated_velocity_m_s,
        nyquist_velocity_m_s,
    })
}

fn finite_tone_dft_magnitude(amplitude: f64, phase_step: f64, n_samples: usize) -> f64 {
    let denominator = (0.5 * phase_step).sin();
    if denominator.abs() < 1.0e-12 {
        return amplitude.abs() * n_samples as f64;
    }
    amplitude.abs() * ((0.5 * n_samples as f64 * phase_step).sin() / denominator).abs()
}

// ─── Plane-wave compounding ───────────────────────────────────────────────────

/// Effective lateral PSF for coherent plane-wave compounding.
///
/// Each compounding angle contributes a sinc² PSF shifted in angle; coherent
/// averaging reduces the effective FWHM by roughly 1/√N_angles. The
/// resulting PSF is approximated as:
/// ```text
/// PSF_comp(x) = sinc²(x / (0.886·F#·λ / √N_angles))
/// ```
///
/// # Arguments
/// * `x_arr` – lateral offsets `m`
/// * `n_angles` – number of compounding angles
/// * `f_number` – effective F-number for a single angle
/// * `wavelength_m` – acoustic wavelength `m`
///
/// # Reference
/// Montaldo et al. (2009), *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*
/// 56, 489.
#[must_use]
pub fn pw_compounding_lateral_psf(
    x_arr: &[f64],
    n_angles: usize,
    f_number: f64,
    wavelength_m: f64,
) -> Vec<f64> {
    let eff_width = 0.886 * f_number * wavelength_m / (n_angles as f64).sqrt();
    x_arr.iter().map(|&x| sinc2(x / eff_width)).collect()
}

// ─── Resolution limit ─────────────────────────────────────────────────────────

/// −6 dB lateral resolution (Rayleigh criterion).
///
/// ```text
/// δx = 0.886 · F# · λ   `m`
/// ```
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §8.3.
#[must_use]
#[inline]
pub fn lateral_resolution_m(f_number: f64, wavelength_m: f64) -> f64 {
    0.886 * f_number * wavelength_m
}

// ─── IVUS synthetic phantom ──────────────────────────────────────────────────

/// Rust-owned deterministic IVUS vessel phantom used by the Chapter 30 book path.
///
/// The arrays are row-major `(n, n)` grids over a square field of view. Labels:
/// `0` background, `1` catheter, `2` lumen, `3` vessel wall, `4` plaque,
/// `5` fibrous cap, `6` lipid core, `7` calcium.
#[derive(Debug, Clone)]
pub struct IvusVesselPhantom {
    /// Cartesian x coordinate `m`.
    pub x_m: Vec<f64>,
    /// Cartesian y coordinate `m`.
    pub y_m: Vec<f64>,
    /// Radius from catheter center `m`.
    pub radius_m: Vec<f64>,
    /// Polar angle `rad`.
    pub theta_rad: Vec<f64>,
    /// Tissue labels.
    pub labels: Vec<u8>,
    /// Sound speed [m/s].
    pub sound_speed_m_s: Vec<f64>,
    /// Density [kg/m^3].
    pub density_kg_m3: Vec<f64>,
    /// Attenuation [dB/(cm MHz)].
    pub attenuation_db_cm_mhz: Vec<f64>,
    /// Normalized reflectivity/backscatter amplitude.
    pub backscatter: Vec<f64>,
    /// Lumen mask.
    pub lumen_mask: Vec<bool>,
    /// External elastic lamina mask.
    pub eel_mask: Vec<bool>,
    /// Plaque mask.
    pub plaque_mask: Vec<bool>,
    /// Fibrous-cap mask.
    pub fibrous_cap_mask: Vec<bool>,
    /// Lipid-core mask.
    pub lipid_mask: Vec<bool>,
    /// Calcium mask.
    pub calcium_mask: Vec<bool>,
}

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

/// Generate a deterministic synthetic IVUS vessel phantom.
///
/// # Errors
/// Returns an error when `n < 2`, `fov_m <= 0`, or `catheter_radius_m <= 0`.
pub fn ivus_vessel_phantom(
    n: usize,
    fov_m: f64,
    catheter_radius_m: f64,
    therapy_azimuth_rad: f64,
    seed: u64,
) -> Result<IvusVesselPhantom, String> {
    if n < 2 {
        return Err("IVUS phantom grid size must be at least 2".to_owned());
    }
    if fov_m <= 0.0 {
        return Err("IVUS phantom field of view must be positive".to_owned());
    }
    if catheter_radius_m <= 0.0 {
        return Err("IVUS catheter radius must be positive".to_owned());
    }

    let len = n * n;
    let mut x_m = Vec::with_capacity(len);
    let mut y_m = Vec::with_capacity(len);
    let mut radius_m = Vec::with_capacity(len);
    let mut theta_rad = Vec::with_capacity(len);
    let mut labels = vec![0_u8; len];
    let mut sound_speed_m_s = vec![343.0; len];
    let mut density_kg_m3 = vec![1.2; len];
    let mut attenuation_db_cm_mhz = vec![0.02; len];
    let mut lumen_mask = vec![false; len];
    let mut eel_mask = vec![false; len];
    let mut plaque_mask = vec![false; len];
    let mut fibrous_cap_mask = vec![false; len];
    let mut lipid_mask = vec![false; len];
    let mut calcium_mask = vec![false; len];

    let spacing = fov_m / (n - 1) as f64;
    for row in 0..n {
        let x = -0.5 * fov_m + row as f64 * spacing;
        for col in 0..n {
            let idx = row * n + col;
            let y = -0.5 * fov_m + col as f64 * spacing;
            let radius = x.hypot(y);
            let theta = y.atan2(x);
            let plaque_angle = angle_difference(theta, therapy_azimuth_rad);
            let plaque_weight = (-0.5 * (plaque_angle / 0.60).powi(2)).exp();
            let lumen_boundary =
                1.70e-3 - 0.45e-3 * plaque_weight + 0.08e-3 * (3.0 * theta + 0.30).cos();
            let eel_boundary =
                3.20e-3 + 0.22e-3 * plaque_weight + 0.10e-3 * (2.0 * theta - 0.40).sin();

            let catheter = radius <= catheter_radius_m;
            let lumen = radius > catheter_radius_m && radius <= lumen_boundary;
            let wall = radius > lumen_boundary && radius <= eel_boundary;
            let plaque = wall && plaque_weight > 0.28;
            let cap = plaque && radius - lumen_boundary < 0.24e-3;
            let lipid = plaque && radius - lumen_boundary > 0.45e-3 && plaque_weight > 0.62;
            let calcium_angle = angle_difference(theta, 1.35);
            let calcium = wall && calcium_angle.abs() < 0.20 && radius > eel_boundary - 0.42e-3;
            let eel = radius <= eel_boundary;

            x_m.push(x);
            y_m.push(y);
            radius_m.push(radius);
            theta_rad.push(theta);
            lumen_mask[idx] = lumen;
            eel_mask[idx] = eel;
            plaque_mask[idx] = plaque;
            fibrous_cap_mask[idx] = cap;
            lipid_mask[idx] = lipid;
            calcium_mask[idx] = calcium;

            if catheter {
                labels[idx] = 1;
            }
            if lumen {
                labels[idx] = 2;
                sound_speed_m_s[idx] = 1570.0;
                density_kg_m3[idx] = 1060.0;
                attenuation_db_cm_mhz[idx] = 0.12;
            }
            if wall {
                labels[idx] = 3;
                sound_speed_m_s[idx] = 1585.0;
                density_kg_m3[idx] = 1080.0;
                attenuation_db_cm_mhz[idx] = 0.65;
            }
            if plaque {
                labels[idx] = 4;
                sound_speed_m_s[idx] = 1520.0;
                density_kg_m3[idx] = 1040.0;
                attenuation_db_cm_mhz[idx] = 0.95;
            }
            if cap {
                labels[idx] = 5;
                sound_speed_m_s[idx] = 1630.0;
                density_kg_m3[idx] = 1120.0;
                attenuation_db_cm_mhz[idx] = 0.70;
            }
            if lipid {
                labels[idx] = 6;
                sound_speed_m_s[idx] = 1450.0;
                density_kg_m3[idx] = 980.0;
                attenuation_db_cm_mhz[idx] = 1.15;
            }
            if calcium {
                labels[idx] = 7;
                sound_speed_m_s[idx] = 2900.0;
                density_kg_m3[idx] = 1850.0;
                attenuation_db_cm_mhz[idx] = 4.0;
            }
        }
    }

    let interface_echo = normalized_impedance_gradient(n, &sound_speed_m_s, &density_kg_m3);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut backscatter = vec![0.0; len];
    for idx in 0..len {
        let tissue_weight = if lumen_mask[idx] {
            0.15
        } else if lipid_mask[idx] {
            0.55
        } else if eel_mask[idx] && !lumen_mask[idx] {
            0.80
        } else {
            0.0
        };
        let u = rng.gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let speckle = 0.18 * (-2.0 * u.ln()).sqrt();
        backscatter[idx] = interface_echo[idx] + speckle * tissue_weight;
        if labels[idx] == 1 {
            backscatter[idx] = 0.0;
        }
    }
    let max_backscatter = backscatter.iter().copied().fold(1.0_f64, f64::max);
    for value in &mut backscatter {
        *value /= max_backscatter;
    }

    Ok(IvusVesselPhantom {
        x_m,
        y_m,
        radius_m,
        theta_rad,
        labels,
        sound_speed_m_s,
        density_kg_m3,
        attenuation_db_cm_mhz,
        backscatter,
        lumen_mask,
        eel_mask,
        plaque_mask,
        fibrous_cap_mask,
        lipid_mask,
        calcium_mask,
    })
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
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_pressure_field(
    radius_m: &[f64],
    theta_rad: &[f64],
    catheter_radius_m: f64,
    peak_pressure_pa: f64,
    therapy_azimuth_rad: f64,
    sector_width_rad: f64,
    attenuation_length_m: f64,
) -> Result<Vec<f64>, String> {
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

/// IVUS therapy response from pressure, tissue attenuation, and anatomy masks.
///
/// This helper owns the Chapter 30 therapy-response field algebra:
/// `I = p²/(2ρc)`, `Q = 2 α_eff I duty`, `ΔT = Q τ/(ρ c_p)`, microbubble
/// delivery through [`ivus_microbubble_delivery_fraction`], peak-pressure
/// mechanical index, and target/off-target deposition ratio.
///
/// # Errors
///
/// Returns an error when array lengths differ, physical scalars are invalid, or
/// any sample is non-finite.
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_response(
    pressure_pa: &[f64],
    radius_m: &[f64],
    attenuation_db_cm_mhz: &[f64],
    eel_mask: &[bool],
    lumen_mask: &[bool],
    fibrous_cap_mask: &[bool],
    lipid_mask: &[bool],
    plaque_mask: &[bool],
    catheter_radius_m: f64,
    therapy_frequency_hz: f64,
    therapy_duty_cycle: f64,
    therapy_sonication_s: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    specific_heat_j_kg_k: f64,
    delivery_radial_center_m: f64,
    delivery_radial_width_m: f64,
) -> Result<IvusTherapyResponse, String> {
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

/// Complete IVUS therapy pressure and response fields for Chapter 30.
///
/// This helper owns the book therapy field orchestration: sector pressure,
/// intensity, absorption-weighted temperature rise, delivery fraction, and
/// scalar safety/targeting metrics.
///
/// # Errors
///
/// Returns an error when pressure-field or response validation fails.
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_fields(
    radius_m: &[f64],
    theta_rad: &[f64],
    attenuation_db_cm_mhz: &[f64],
    eel_mask: &[bool],
    lumen_mask: &[bool],
    fibrous_cap_mask: &[bool],
    lipid_mask: &[bool],
    plaque_mask: &[bool],
    catheter_radius_m: f64,
    therapy_pressure_pa: f64,
    therapy_azimuth_rad: f64,
    therapy_sector_width_rad: f64,
    pressure_attenuation_length_m: f64,
    therapy_frequency_hz: f64,
    therapy_duty_cycle: f64,
    therapy_sonication_s: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    specific_heat_j_kg_k: f64,
    delivery_radial_center_m: f64,
    delivery_radial_width_m: f64,
) -> Result<IvusTherapyFields, String> {
    let pressure_pa = ivus_therapy_pressure_field(
        radius_m,
        theta_rad,
        catheter_radius_m,
        therapy_pressure_pa,
        therapy_azimuth_rad,
        therapy_sector_width_rad,
        pressure_attenuation_length_m,
    )?;
    let response = ivus_therapy_response(
        &pressure_pa,
        radius_m,
        attenuation_db_cm_mhz,
        eel_mask,
        lumen_mask,
        fibrous_cap_mask,
        lipid_mask,
        plaque_mask,
        catheter_radius_m,
        therapy_frequency_hz,
        therapy_duty_cycle,
        therapy_sonication_s,
        density_kg_m3,
        sound_speed_m_s,
        specific_heat_j_kg_k,
        delivery_radial_center_m,
        delivery_radial_width_m,
    )?;

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

/// Polar IVUS RF fixture from phantom backscatter and attenuation fields.
///
/// The function samples a Cartesian phantom on a polar catheter grid, applies
/// two-way amplitude attenuation, and adds the deterministic catheter-ring echo
/// used by the Chapter 30 IVUS B-mode panel:
///
/// `rf(r, theta) = backscatter(x,y) * exp(-2 alpha(x,y) f_MHz (r-r_catheter))
///                 + A_ring exp(-((r-r_catheter) / w_ring)^2)`.
///
/// # Errors
///
/// Returns an error when phantom arrays do not form the same square grid, when
/// axes/scalars are invalid, or when any numeric sample is non-finite.
#[allow(clippy::too_many_arguments)]
pub fn ivus_polar_bmode_rf(
    x_m: &[f64],
    y_m: &[f64],
    backscatter: &[f64],
    attenuation_db_cm_mhz: &[f64],
    r_axis_m: &[f64],
    theta_axis_rad: &[f64],
    catheter_radius_m: f64,
    frequency_hz: f64,
    ring_amplitude: f64,
    ring_width_m: f64,
) -> Result<Vec<f64>, String> {
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
#[allow(clippy::too_many_arguments)]
pub fn ivus_bmode_image(
    x_m: &[f64],
    y_m: &[f64],
    backscatter: &[f64],
    attenuation_db_cm_mhz: &[f64],
    r_axis_m: &[f64],
    theta_axis_rad: &[f64],
    radius_m: &[f64],
    theta_m_rad: &[f64],
    catheter_radius_m: f64,
    frequency_hz: f64,
    floor_db: f64,
    ring_amplitude: f64,
    ring_width_m: f64,
) -> Result<IvusBmodeImage, String> {
    if !(floor_db.is_finite() && floor_db < 0.0) {
        return Err("floor_db must be finite and negative".to_owned());
    }
    let n_r = r_axis_m.len();
    let n_theta = theta_axis_rad.len();
    if n_r == 0 || n_theta == 0 {
        return Err("r_axis_m and theta_axis_rad must not be empty".to_owned());
    }

    let rf = ivus_polar_bmode_rf(
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
    )?;
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

/// Compute Chapter 30 IVUS scalar metrics from Rust-owned fields.
///
/// # Errors
///
/// Returns an error when grid arrays are inconsistent, masks are empty, scalar
/// frequencies/sound speed are invalid, or any B-mode sample is non-finite.
#[allow(clippy::too_many_arguments)]
pub fn ivus_chapter_metrics(
    x_m: &[f64],
    y_m: &[f64],
    lumen_mask: &[bool],
    eel_mask: &[bool],
    plaque_mask: &[bool],
    bmode_cartesian: &[f64],
    sound_speed_m_s: f64,
    imaging_frequency_hz: f64,
    therapy_frequency_hz: f64,
    bmode_dynamic_range_db: f64,
    therapy_mechanical_index: f64,
    therapy_peak_delta_t_c: f64,
    therapy_target_to_offtarget_deposition_ratio: f64,
) -> Result<IvusChapterMetrics, String> {
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

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Normalised sinc-squared: sinc²(u) = (sin(πu)/(πu))²
#[inline]
fn sinc2(u: f64) -> f64 {
    if u.abs() < 1e-12 {
        1.0
    } else {
        let s = (PI * u).sin() / (PI * u);
        s * s
    }
}

fn angle_difference(a: f64, b: f64) -> f64 {
    (a - b).sin().atan2((a - b).cos())
}

fn square_grid_len(len: usize) -> Result<usize, String> {
    let n = (len as f64).sqrt() as usize;
    if n < 2 || n * n != len {
        return Err(format!(
            "phantom arrays must form a square grid, got length {len}"
        ));
    }
    Ok(n)
}

fn nearest_grid_index(value: f64, origin: f64, spacing: f64, n: usize) -> usize {
    ((value - origin) / spacing)
        .round()
        .clamp(0.0, (n - 1) as f64) as usize
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

fn normalized_impedance_gradient(n: usize, sound_speed: &[f64], density: &[f64]) -> Vec<f64> {
    let impedance: Vec<f64> = sound_speed
        .iter()
        .zip(density)
        .map(|(&c, &rho)| c * rho)
        .collect();
    let mut gradient = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            let gx = if row == 0 {
                impedance[(row + 1) * n + col] - impedance[idx]
            } else if row + 1 == n {
                impedance[idx] - impedance[(row - 1) * n + col]
            } else {
                0.5 * (impedance[(row + 1) * n + col] - impedance[(row - 1) * n + col])
            };
            let gy = if col == 0 {
                impedance[row * n + col + 1] - impedance[idx]
            } else if col + 1 == n {
                impedance[idx] - impedance[row * n + col - 1]
            } else {
                0.5 * (impedance[row * n + col + 1] - impedance[row * n + col - 1])
            };
            gradient[idx] = gx.hypot(gy);
        }
    }
    let max_gradient = gradient.iter().copied().fold(1.0_f64, f64::max);
    for value in &mut gradient {
        *value /= max_gradient;
    }
    gradient
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn lateral_psf_peak_at_zero() {
        let psf = lateral_psf_sinc2(&[0.0], 2.0, 1e-3);
        assert!((psf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn axial_psf_peak_at_zero() {
        let psf = axial_psf_rect(&[0.0], SOUND_SPEED_WATER_SIM, 5.0 * MHZ_TO_HZ);
        assert!((psf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lateral_psf_decreases_away_from_axis() {
        let lam = SOUND_SPEED_WATER_SIM / (2.0 * MHZ_TO_HZ);
        let psf = lateral_psf_sinc2(&[0.0, 0.5e-3, 1e-3], 2.0, lam);
        assert!(psf[0] > psf[1] && psf[1] > psf[2]);
    }

    #[test]
    fn doppler_towards_transducer() {
        // θ = 0 → maximum shift
        let df = doppler_frequency_shift(1.0, 0.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!((df - 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM).abs() < 1e-6);
    }

    #[test]
    fn doppler_perpendicular_is_zero() {
        let df = doppler_frequency_shift(1.0, PI / 2.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!(df.abs() < 1e-10);
    }

    #[test]
    fn contrast_agent_doppler_spectrum_recovers_velocity() {
        let spectrum = contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
            n_ensemble: 128,
            fft_multiplier: 4,
            prf_hz: 10_000.0,
            velocity_m_s: 0.3,
            theta_rad: 60_f64.to_radians(),
            f0_hz: 5.0 * MHZ_TO_HZ,
            sound_speed_m_s: 1540.0,
            amplitude: 0.02,
        })
        .unwrap();

        assert_eq!(spectrum.slow_time_s.len(), 128);
        assert_eq!(spectrum.iq_real.len(), 128);
        assert_eq!(spectrum.iq_imag.len(), 128);
        assert_eq!(spectrum.velocity_m_s.len(), 512);
        assert_eq!(spectrum.power.len(), 512);
        assert!((spectrum.doppler_shift_hz - 974.025_974_025_974).abs() < 1.0e-9);
        assert!((spectrum.estimated_velocity_m_s - 0.3).abs() < 1.0e-12);
        assert!((spectrum.nyquist_velocity_m_s - 1.54).abs() < 1.0e-12);

        let peak_idx = spectrum
            .power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap();
        assert!((spectrum.velocity_m_s[peak_idx] - 0.300_781_25).abs() < 1.0e-12);
    }

    #[test]
    fn contrast_agent_doppler_spectrum_rejects_invalid_inputs() {
        assert!(contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
            n_ensemble: 1,
            fft_multiplier: 4,
            prf_hz: 10_000.0,
            velocity_m_s: 0.3,
            theta_rad: 0.0,
            f0_hz: 5.0 * MHZ_TO_HZ,
            sound_speed_m_s: 1540.0,
            amplitude: 0.02,
        })
        .unwrap_err()
        .contains("n_ensemble"));
        assert!(contrast_agent_doppler_spectrum(ContrastAgentDopplerConfig {
            n_ensemble: 128,
            fft_multiplier: 4,
            prf_hz: 10_000.0,
            velocity_m_s: 0.3,
            theta_rad: PI / 2.0,
            f0_hz: 5.0 * MHZ_TO_HZ,
            sound_speed_m_s: 1540.0,
            amplitude: 0.02,
        })
        .unwrap_err()
        .contains("cos(theta_rad)"));
    }

    #[test]
    fn compounding_narrower_than_single() {
        let lam = SOUND_SPEED_WATER_SIM / (2.0 * MHZ_TO_HZ);
        let psf1 = lateral_psf_sinc2(&[0.5e-3], 2.0, lam);
        let psf4 = pw_compounding_lateral_psf(&[0.5e-3], 4, 2.0, lam);
        // 4-angle compounding narrows the PSF (FWHM radius 1.33mm → 0.665mm).
        // At x=0.5mm the compound PSF is past its −6 dB point while the
        // single-angle PSF is still within its mainlobe → psf4 ≪ psf1.
        // sinc²: u1≈0.376 → 0.613, u4≈0.752 → 0.088.
        assert!(psf4[0] < psf1[0], "psf4={} psf1={}", psf4[0], psf1[0]);
    }

    #[test]
    fn lateral_resolution_positive() {
        let lam = SOUND_SPEED_WATER_SIM / (3.5 * MHZ_TO_HZ);
        let dx = lateral_resolution_m(2.0, lam);
        assert!(dx > 0.0 && dx < 1e-3);
    }

    #[test]
    fn ivus_vessel_phantom_preserves_nested_masks_and_properties() {
        let phantom = ivus_vessel_phantom(128, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();

        assert_eq!(phantom.labels.len(), 128 * 128);
        assert!(phantom.lumen_mask.iter().filter(|&&v| v).count() > 500);
        assert!(phantom
            .lumen_mask
            .iter()
            .zip(&phantom.eel_mask)
            .all(|(&lumen, &eel)| !lumen || eel));
        assert!(phantom
            .fibrous_cap_mask
            .iter()
            .zip(&phantom.plaque_mask)
            .all(|(&cap, &plaque)| !cap || plaque));
        let calcium_mean = masked_mean(&phantom.sound_speed_m_s, &phantom.calcium_mask);
        let lipid_mean = masked_mean(&phantom.sound_speed_m_s, &phantom.lipid_mask);
        assert!(calcium_mean > 2500.0, "calcium_mean={calcium_mean}");
        assert!(lipid_mean < 1500.0, "lipid_mean={lipid_mean}");
    }

    #[test]
    fn ivus_vessel_phantom_is_seed_deterministic() {
        let a = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();
        let b = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();
        let c = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 31).unwrap();

        assert_eq!(a.labels, b.labels);
        assert_eq!(a.backscatter, b.backscatter);
        assert_ne!(a.backscatter, c.backscatter);
    }

    #[test]
    fn ivus_therapy_pressure_field_matches_sector_decay() {
        let catheter = 0.55e-3;
        let peak = 300.0e3;
        let azimuth = -0.72;
        let width = 0.50;
        let decay = 3.2e-3;
        let radius = [catheter, catheter + decay, catheter + decay];
        let theta = [azimuth, azimuth, azimuth + width];

        let pressure =
            ivus_therapy_pressure_field(&radius, &theta, catheter, peak, azimuth, width, decay)
                .unwrap();

        assert_eq!(pressure[0], 0.0);
        assert!((pressure[1] - peak / std::f64::consts::E).abs() < 1.0e-9);
        assert!((pressure[2] - peak * (-1.5_f64).exp()).abs() < 1.0e-9);
    }

    #[test]
    fn ivus_therapy_pressure_field_rejects_invalid_inputs() {
        let err = ivus_therapy_pressure_field(&[1.0], &[], 0.0, 1.0, 0.0, 1.0, 1.0).unwrap_err();
        assert!(err.contains("radius_m length"));

        let err =
            ivus_therapy_pressure_field(&[f64::NAN], &[0.0], 0.0, 1.0, 0.0, 1.0, 1.0).unwrap_err();
        assert!(err.contains("radius_m[0] must be finite"));

        let err = ivus_therapy_pressure_field(&[1.0], &[0.0], 0.0, 1.0, 0.0, 0.0, 1.0).unwrap_err();
        assert!(err.contains("sector_width_rad"));
    }

    #[test]
    fn ivus_microbubble_delivery_fraction_matches_weighted_force_model() {
        let range = [0.0, 1.75e-3, 1.75e-3];
        let attenuation = [10.0, 10.0, 10.0];
        let intensity = [1.0, 1.0, 1.0];
        let wall = [false, true, true];
        let target = [false, false, true];

        let delivered = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
            range_m: &range,
            attenuation_np_m: &attenuation,
            intensity_w_m2: &intensity,
            wall_mask: &wall,
            target_mask: &target,
            sound_speed_m_s: 1540.0,
            radial_center_m: 1.75e-3,
            radial_width_m: 1.2e-3,
        })
        .unwrap();

        assert_eq!(delivered[0], 0.0);
        let expected_wall = 1.0 - (-3.0_f64 * 0.2).exp();
        let expected_target = 1.0 - (-3.0_f64).exp();
        assert!((delivered[1] - expected_wall).abs() < 1.0e-12);
        assert!((delivered[2] - expected_target).abs() < 1.0e-12);
    }

    #[test]
    fn ivus_microbubble_delivery_fraction_rejects_invalid_inputs() {
        let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
            range_m: &[0.0],
            attenuation_np_m: &[],
            intensity_w_m2: &[1.0],
            wall_mask: &[true],
            target_mask: &[false],
            sound_speed_m_s: 1540.0,
            radial_center_m: 1.75e-3,
            radial_width_m: 1.2e-3,
        })
        .unwrap_err();
        assert!(err.contains("all arrays must have length"));

        let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
            range_m: &[0.0],
            attenuation_np_m: &[1.0],
            intensity_w_m2: &[f64::NAN],
            wall_mask: &[true],
            target_mask: &[false],
            sound_speed_m_s: 1540.0,
            radial_center_m: 1.75e-3,
            radial_width_m: 1.2e-3,
        })
        .unwrap_err();
        assert!(err.contains("intensity_w_m2[0]"));

        let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
            range_m: &[0.0],
            attenuation_np_m: &[1.0],
            intensity_w_m2: &[1.0],
            wall_mask: &[true],
            target_mask: &[false],
            sound_speed_m_s: 0.0,
            radial_center_m: 1.75e-3,
            radial_width_m: 1.2e-3,
        })
        .unwrap_err();
        assert!(err.contains("sound_speed_m_s"));
    }

    #[test]
    fn ivus_polar_bmode_rf_matches_two_way_attenuation_and_ring() {
        let x = vec![-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
        let y = vec![-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
        let backscatter = vec![1.0, 2.0, 3.0, 4.0];
        let attenuation = vec![0.0, 1.0, 2.0, 3.0];
        let radius = [1.0e-3, 1.55e-3];
        let theta = [0.0];

        let rf = ivus_polar_bmode_rf(
            &x,
            &y,
            &backscatter,
            &attenuation,
            &radius,
            &theta,
            1.0e-3,
            20.0e6,
            0.10,
            0.22e-3,
        )
        .unwrap();

        assert_eq!(rf[0], 4.0 + 0.10);
        let alpha = 3.0 * 100.0 / 8.686;
        let expected = 4.0 * (-2.0_f64 * alpha * 20.0 * 0.55e-3).exp()
            + 0.10 * (-(0.55e-3_f64 / 0.22e-3_f64).powi(2)).exp();
        assert!((rf[1] - expected).abs() < 1.0e-12);
    }

    #[test]
    fn ivus_polar_bmode_rf_rejects_invalid_inputs() {
        let err = ivus_polar_bmode_rf(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[1.0],
            &[0.0],
            0.0,
            1.0,
            0.0,
            1.0,
        )
        .unwrap_err();
        assert!(err.contains("square grid"));

        let err = ivus_polar_bmode_rf(
            &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0, 0.0, 1.0],
            &[1.0, f64::NAN, 1.0, 1.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0],
            &[0.0],
            0.0,
            1.0,
            0.0,
            1.0,
        )
        .unwrap_err();
        assert!(err.contains("backscatter[1]"));
    }

    #[test]
    fn ivus_scan_convert_maps_nearest_bins_and_radial_bounds() {
        let polar = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r_axis = [1.0e-3, 2.0e-3];
        let theta_axis = [-PI, -0.5 * PI, 0.0, 0.5 * PI];
        let radius = [0.5e-3, 1.0e-3, 2.0e-3, 2.0e-3, 2.5e-3];
        let theta = [-PI, -PI, 0.0, PI, 0.0];

        let image = ivus_scan_convert(&polar, &r_axis, &theta_axis, &radius, &theta).unwrap();

        assert_eq!(image, vec![0.0, 1.0, 7.0, 5.0, 0.0]);
    }

    #[test]
    fn ivus_scan_convert_rejects_invalid_inputs() {
        let err = ivus_scan_convert(&[1.0], &[1.0], &[-PI, 0.0], &[1.0], &[0.0]).unwrap_err();
        assert!(err.contains("at least two"));

        let err = ivus_scan_convert(
            &[1.0, f64::NAN, 3.0, 4.0],
            &[1.0, 2.0],
            &[-PI, 0.0],
            &[1.0],
            &[0.0],
        )
        .unwrap_err();
        assert!(err.contains("polar[1]"));
    }

    #[test]
    fn ivus_bmode_image_returns_consistent_polar_and_cartesian_images() {
        let x = vec![-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
        let y = vec![-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
        let backscatter = vec![1.0, 2.0, 3.0, 4.0];
        let attenuation = vec![0.0, 1.0, 2.0, 3.0];
        let r_axis = [1.0e-3, 1.55e-3, 2.0e-3, 2.45e-3];
        let theta_axis = [-PI, -0.5 * PI, 0.0, 0.5 * PI];
        let radius = [0.5e-3, 1.0e-3, 2.0e-3, 2.5e-3];
        let theta = [-PI, -PI, 0.0, 0.0];

        let image = ivus_bmode_image(
            &x,
            &y,
            &backscatter,
            &attenuation,
            &r_axis,
            &theta_axis,
            &radius,
            &theta,
            1.0e-3,
            20.0e6,
            -60.0,
            0.10,
            0.22e-3,
        )
        .unwrap();

        assert_eq!(image.rf.len(), r_axis.len() * theta_axis.len());
        assert_eq!(image.envelope.len(), image.rf.len());
        assert_eq!(image.db.len(), image.rf.len());
        assert_eq!(image.polar.len(), image.rf.len());
        assert_eq!(image.cartesian.len(), radius.len());
        assert!(image
            .polar
            .iter()
            .all(|&value| (0.0..=1.0).contains(&value)));
        assert!(image.db.iter().all(|&value| (-60.0..=0.0).contains(&value)));
        let expected_cartesian =
            ivus_scan_convert(&image.polar, &r_axis, &theta_axis, &radius, &theta).unwrap();
        assert_eq!(image.cartesian, expected_cartesian);
    }

    #[test]
    fn ivus_bmode_image_rejects_invalid_floor() {
        let err = ivus_bmode_image(
            &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0, 0.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 2.0],
            &[0.0, 1.0],
            &[1.0],
            &[0.0],
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
        )
        .unwrap_err();
        assert!(err.contains("floor_db"));
    }

    #[test]
    fn ivus_chapter_metrics_match_masked_means_and_areas() {
        let x = [-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
        let y = [-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
        let lumen = [true, false, false, false];
        let eel = [true, true, true, false];
        let plaque = [false, true, false, false];
        let bmode = [0.20, 0.70, 0.50, 0.10];

        let metrics = ivus_chapter_metrics(
            &x, &y, &lumen, &eel, &plaque, &bmode, 1540.0, 20.0e6, 1.5e6, 60.0, 0.25, 0.04, 101.0,
        )
        .unwrap();

        assert!((metrics.imaging_wavelength_um - 77.0).abs() < 1.0e-12);
        assert!((metrics.therapy_wavelength_mm - 1540.0 / 1.5e6 * 1.0e3).abs() < 1.0e-12);
        assert!((metrics.lumen_area_mm2 - 4.0).abs() < 1.0e-12);
        assert!((metrics.plaque_area_mm2 - 4.0).abs() < 1.0e-12);
        assert_eq!(metrics.bmode_dynamic_range_db, 60.0);
        assert_eq!(metrics.bmode_mean_lumen_intensity, 0.20);
        assert!((metrics.bmode_mean_wall_intensity - 0.60).abs() < 1.0e-12);
        assert_eq!(metrics.therapy_mechanical_index, 0.25);
        assert_eq!(metrics.therapy_peak_delta_t_c, 0.04);
        assert_eq!(metrics.therapy_target_to_offtarget_deposition_ratio, 101.0);
    }

    #[test]
    fn ivus_chapter_metrics_rejects_empty_masks() {
        let err = ivus_chapter_metrics(
            &[-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3],
            &[-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3],
            &[false, false, false, false],
            &[true, true, true, true],
            &[true, false, false, false],
            &[0.0, 0.0, 0.0, 0.0],
            1540.0,
            20.0e6,
            1.5e6,
            60.0,
            0.25,
            0.04,
            101.0,
        )
        .unwrap_err();
        assert!(err.contains("lumen and plaque masks"));
    }

    #[test]
    fn ivus_therapy_response_matches_closed_forms() {
        let pressure = [1.0e6, 1.0e6];
        let radius = [2.30e-3, 2.30e-3];
        let attenuation = [1.0, 1.0];
        let eel = [true, true];
        let lumen = [false, false];
        let cap = [false, true];
        let lipid = [false, false];
        let plaque = [false, true];

        let response = ivus_therapy_response(
            &pressure,
            &radius,
            &attenuation,
            &eel,
            &lumen,
            &cap,
            &lipid,
            &plaque,
            0.55e-3,
            2.0e6,
            0.25,
            0.50,
            1000.0,
            1500.0,
            4000.0,
            1.75e-3,
            1.2e-3,
        )
        .unwrap();

        let intensity = 1.0e12 / (2.0 * 1000.0 * 1500.0);
        let alpha_eff = 1.0 * 100.0 / 8.686 * 2.0;
        let expected_delta_t = 2.0 * alpha_eff * intensity * 0.25 * 0.50 / (1000.0 * 4000.0);
        let off_deposition = 1.0 - (-0.6_f64).exp();
        let target_deposition = 1.0 - (-3.0_f64).exp();

        assert!((response.intensity_w_m2[0] - intensity).abs() < 1.0e-9);
        assert!((response.temperature_rise_k[0] - expected_delta_t).abs() < 1.0e-12);
        assert!((response.peak_delta_t_k - expected_delta_t).abs() < 1.0e-12);
        assert!((response.deposition[0] - off_deposition).abs() < 1.0e-12);
        assert!((response.deposition[1] - target_deposition).abs() < 1.0e-12);
        assert!((response.mechanical_index - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-12);
        assert!(
            (response.target_to_offtarget_ratio - target_deposition / off_deposition).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn ivus_therapy_response_rejects_missing_target() {
        let err = ivus_therapy_response(
            &[1.0],
            &[1.0e-3],
            &[1.0],
            &[true],
            &[false],
            &[false],
            &[false],
            &[false],
            0.5e-3,
            2.0e6,
            0.25,
            0.5,
            1000.0,
            1500.0,
            4000.0,
            1.0e-3,
            1.0e-3,
        )
        .unwrap_err();
        assert!(err.contains("target mask"));
    }

    #[test]
    fn ivus_therapy_fields_matches_pressure_and_response_helpers() {
        let radius = [0.55e-3, 2.30e-3, 2.30e-3];
        let theta = [-0.72, -0.72, -0.72];
        let attenuation = [1.0, 1.0, 1.0];
        let eel = [false, true, true];
        let lumen = [false, false, false];
        let cap = [false, false, true];
        let lipid = [false, false, false];
        let plaque = [false, false, true];

        let fields = ivus_therapy_fields(
            &radius,
            &theta,
            &attenuation,
            &eel,
            &lumen,
            &cap,
            &lipid,
            &plaque,
            0.55e-3,
            1.0e6,
            -0.72,
            0.50,
            3.2e-3,
            2.0e6,
            0.25,
            0.50,
            1000.0,
            1500.0,
            4000.0,
            1.75e-3,
            1.2e-3,
        )
        .unwrap();

        let pressure =
            ivus_therapy_pressure_field(&radius, &theta, 0.55e-3, 1.0e6, -0.72, 0.50, 3.2e-3)
                .unwrap();
        let response = ivus_therapy_response(
            &pressure,
            &radius,
            &attenuation,
            &eel,
            &lumen,
            &cap,
            &lipid,
            &plaque,
            0.55e-3,
            2.0e6,
            0.25,
            0.50,
            1000.0,
            1500.0,
            4000.0,
            1.75e-3,
            1.2e-3,
        )
        .unwrap();

        assert_eq!(fields.pressure_pa, pressure);
        assert_eq!(fields.intensity_w_m2, response.intensity_w_m2);
        assert_eq!(fields.temperature_rise_k, response.temperature_rise_k);
        assert_eq!(fields.deposition, response.deposition);
        assert_eq!(fields.mechanical_index, response.mechanical_index);
        assert_eq!(
            fields.target_to_offtarget_ratio,
            response.target_to_offtarget_ratio
        );
        assert_eq!(fields.peak_delta_t_k, response.peak_delta_t_k);
    }

    #[test]
    fn ivus_therapy_fields_rejects_missing_target() {
        let err = ivus_therapy_fields(
            &[1.0e-3],
            &[0.0],
            &[1.0],
            &[true],
            &[false],
            &[false],
            &[false],
            &[false],
            0.5e-3,
            1.0e6,
            0.0,
            0.5,
            1.0e-3,
            2.0e6,
            0.25,
            0.5,
            1000.0,
            1500.0,
            4000.0,
            1.0e-3,
            1.0e-3,
        )
        .unwrap_err();
        assert!(err.contains("target mask"));
    }

    fn masked_mean(values: &[f64], mask: &[bool]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0_usize;
        for (&value, &selected) in values.iter().zip(mask) {
            if selected {
                sum += value;
                count += 1;
            }
        }
        sum / count as f64
    }
}
