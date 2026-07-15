//! Active-imaging Delay-and-Sum (DAS) reconstruction.
//!
//! Reconstructs a pressure-like image from recorded sensor data by computing the
//! one-way time-of-flight from each grid pixel to each sensor element and
//! coherently summing the linearly-interpolated samples. This is the
//! reconstruction primitive published as
//! `KWave.jl/reconstruction/beamform.jl::beamform_delay_and_sum` and is distinct
//! from the *passive-acoustic-mapping* DAS in
//! [`crate::signal_processing::pam`], which returns a windowed energy
//! map instead of a per-pixel coherent sum.
//!
//! # Mathematical contract
//!
//! For each pixel `p` and sensor `s` at position `r_s`,
//!
//! ```text
//! τ_{s,p} = τ_tx(p) + ‖r_p − r_s‖ / c  (active-imaging time of flight)
//! k_{s,p} = τ_{s,p} · f_s              (fractional sample index)
//! image[p] = (1 / N_active(p)) · Σ_s w_s · linterp(sensor_data[s, ·], k_{s,p})
//! ```
//!
//! Sensors whose interpolated sample index falls outside `[0, n_samples − 1]`
//! are excluded from both the sum and `N_active(p)`. When `N_active(p) = 0` the
//! pixel value is zero.

use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, ArrayView1, ArrayView2};

/// Apodization windows supported by the imaging-DAS primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ImagingDasApodization {
    #[default]
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
}

/// Configuration for [`beamform_image_das`].
#[derive(Debug, Clone, Copy)]
pub struct ImagingDasConfig {
    pub sound_speed: f64,
    pub sampling_frequency: f64,
    pub apodization: ImagingDasApodization,
}

impl ImagingDasConfig {
    /// Construct a new configuration validating physical positivity.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `sound_speed` or
    /// `sampling_frequency` is not finite or not strictly positive.
    pub fn new(
        sound_speed: f64,
        sampling_frequency: f64,
        apodization: ImagingDasApodization,
    ) -> KwaversResult<Self> {
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "imaging_das: sound_speed must be finite and > 0; got {sound_speed}"
            )));
        }
        if !sampling_frequency.is_finite() || sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "imaging_das: sampling_frequency must be finite and > 0; got {sampling_frequency}"
            )));
        }
        Ok(Self {
            sound_speed,
            sampling_frequency,
            apodization,
        })
    }
}

/// Active-imaging DAS reconstruction.
///
/// `sensor_data` shape `(n_sensors, n_samples)`; `sensor_positions` and
/// `grid_points` shape `(n, 3)` in `[x, y, z]` order. Returns a flat vector of
/// length `grid_points.shape()[0]` matching grid-point row order.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] on any of:
/// - shape mismatch between `sensor_data` rows and `sensor_positions` rows;
/// - `sensor_positions` or `grid_points` lacking 3 columns;
/// - empty sensor set, empty time axis, or empty grid;
/// - non-finite entries in any input.
pub fn beamform_image_das(
    sensor_data: ArrayView2<'_, f64>,
    sensor_positions: ArrayView2<'_, f64>,
    grid_points: ArrayView2<'_, f64>,
    config: &ImagingDasConfig,
) -> KwaversResult<Array1<f64>> {
    let transmit_delays_s = zero_transmit_delays(grid_points.shape()[0])?;
    beamform_image_das_with_transmit_delays(
        sensor_data,
        sensor_positions,
        grid_points,
        transmit_delays_s.view(),
        config,
    )
}

/// Active-imaging DAS reconstruction with a shared transmit arrival per pixel.
///
/// `transmit_delays_s[p]` is the non-negative transmit travel time for
/// `grid_points[p]`. It is deliberately a per-pixel array rather than a
/// wavefront enum so synthetic, measured, and refracting transmit models can
/// reuse this receive-DAS kernel without duplicating interpolation or
/// apodization. The total sample index is
/// `fs · (transmit_delays_s[p] + ‖pixel - sensor‖ / c)`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] for the [`beamform_image_das`]
/// validation failures, a transmit-delay length mismatch, or a non-finite or
/// negative transmit delay.
pub fn beamform_image_das_with_transmit_delays(
    sensor_data: ArrayView2<'_, f64>,
    sensor_positions: ArrayView2<'_, f64>,
    grid_points: ArrayView2<'_, f64>,
    transmit_delays_s: ArrayView1<'_, f64>,
    config: &ImagingDasConfig,
) -> KwaversResult<Array1<f64>> {
    beamform_with_transmit_delays(
        sensor_data,
        sensor_positions,
        grid_points,
        transmit_delays_s,
        config,
        0.0,
    )
}

/// Complex active-imaging DAS reconstruction.
///
/// This is the complex-I/Q counterpart of [`beamform_image_das`]. It preserves
/// coherent phase through fractional interpolation and carrier rephasing. The
/// `center_frequency_hz` must be the frequency removed when the channel data
/// became baseband I/Q; DAS restores `exp(j 2π f₀ τ)` for each physical total
/// delay `τ` before summation.
///
/// # Errors
///
/// Returns the same validation errors as [`beamform_image_das`], including
/// non-finite real or imaginary channel samples, plus an invalid or
/// Nyquist-or-higher baseband carrier frequency.
pub fn beamform_complex_image_das(
    sensor_data: ArrayView2<'_, Complex64>,
    sensor_positions: ArrayView2<'_, f64>,
    grid_points: ArrayView2<'_, f64>,
    config: &ImagingDasConfig,
    center_frequency_hz: f64,
) -> KwaversResult<Array1<Complex64>> {
    let transmit_delays_s = zero_transmit_delays(grid_points.shape()[0])?;
    beamform_complex_image_das_with_transmit_delays(
        sensor_data,
        sensor_positions,
        grid_points,
        transmit_delays_s.view(),
        config,
        center_frequency_hz,
    )
}

/// Complex active-imaging DAS with a shared transmit arrival per pixel.
///
/// `transmit_delays_s[p]` has the identical physical interpretation as in
/// [`beamform_image_das_with_transmit_delays`]. Complex linear interpolation
/// occurs before carrier rephasing and coherent summation, so a demodulated
/// I/Q phase ramp remains coherent at fractional receive delays. The carrier
/// frequency must match the prior analytic demodulation.
///
/// # Errors
///
/// Returns the same validation errors as
/// [`beamform_image_das_with_transmit_delays`], including a non-finite real or
/// imaginary channel component and an invalid baseband carrier frequency.
pub fn beamform_complex_image_das_with_transmit_delays(
    sensor_data: ArrayView2<'_, Complex64>,
    sensor_positions: ArrayView2<'_, f64>,
    grid_points: ArrayView2<'_, f64>,
    transmit_delays_s: ArrayView1<'_, f64>,
    config: &ImagingDasConfig,
    center_frequency_hz: f64,
) -> KwaversResult<Array1<Complex64>> {
    validate_baseband_carrier_frequency(center_frequency_hz, config.sampling_frequency)?;
    beamform_with_transmit_delays(
        sensor_data,
        sensor_positions,
        grid_points,
        transmit_delays_s,
        config,
        center_frequency_hz,
    )
}

trait DasSample: Copy {
    fn is_finite(self) -> bool;
    fn zero() -> Self;
    fn interpolate_weighted(self, next: Self, fraction: f64, weight: f64) -> Self;
    fn rephase_baseband(self, phase_radians: f64) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn normalize(self, active_count: usize) -> Self;
}

impl DasSample for f64 {
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    fn zero() -> Self {
        0.0
    }

    fn interpolate_weighted(self, next: Self, fraction: f64, weight: f64) -> Self {
        weight * ((1.0 - fraction) * self + fraction * next)
    }

    fn rephase_baseband(self, _phase_radians: f64) -> Self {
        self
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn normalize(self, active_count: usize) -> Self {
        self / active_count as f64
    }
}

impl DasSample for Complex64 {
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    fn interpolate_weighted(self, next: Self, fraction: f64, weight: f64) -> Self {
        (self * (1.0 - fraction) + next * fraction) * weight
    }

    fn rephase_baseband(self, phase_radians: f64) -> Self {
        self * Complex64::cis(phase_radians)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn normalize(self, active_count: usize) -> Self {
        self / active_count as f64
    }
}

fn beamform_with_transmit_delays<S: DasSample>(
    sensor_data: ArrayView2<'_, S>,
    sensor_positions: ArrayView2<'_, f64>,
    grid_points: ArrayView2<'_, f64>,
    transmit_delays_s: ArrayView1<'_, f64>,
    config: &ImagingDasConfig,
    center_frequency_hz: f64,
) -> KwaversResult<Array1<S>> {
    if !config.sound_speed.is_finite() || config.sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sound_speed must be finite and > 0; got {}",
            config.sound_speed
        )));
    }
    if !config.sampling_frequency.is_finite() || config.sampling_frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sampling_frequency must be finite and > 0; got {}",
            config.sampling_frequency
        )));
    }
    let [n_sensors, n_samples] = sensor_data.shape();
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sensor_data must have n_sensors > 0 and n_samples > 0; got ({n_sensors}, {n_samples})"
        )));
    }
    if sensor_positions.shape()[1] != 3 || sensor_positions.shape()[0] != n_sensors {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sensor_positions must have shape ({n_sensors}, 3); got {:?}",
            sensor_positions.shape()
        )));
    }
    if grid_points.shape()[1] != 3 || grid_points.shape()[0] == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: grid_points must have shape (n_pixels >= 1, 3); got {:?}",
            grid_points.shape()
        )));
    }
    if !sensor_data.iter().all(|sample| sample.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "imaging_das: sensor_data must contain only finite values".to_owned(),
        ));
    }
    if !sensor_positions.iter().all(|v| v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "imaging_das: sensor_positions must contain only finite coordinates".to_owned(),
        ));
    }
    if !grid_points.iter().all(|v| v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "imaging_das: grid_points must contain only finite coordinates".to_owned(),
        ));
    }
    let transmit_delay_count = transmit_delays_s.shape()[0];
    if transmit_delay_count != grid_points.shape()[0] {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: transmit_delays_s must have one delay per grid point; got {} delays for {} points",
            transmit_delay_count,
            grid_points.shape()[0]
        )));
    }
    if !transmit_delays_s
        .iter()
        .all(|delay| delay.is_finite() && *delay >= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "imaging_das: transmit_delays_s must contain only finite non-negative values"
                .to_owned(),
        ));
    }

    let weights = apodization_weights(n_sensors, config.apodization);
    let inv_c = 1.0 / config.sound_speed;
    let fs = config.sampling_frequency;
    let last_sample_f = (n_samples - 1) as f64;

    let n_pixels = grid_points.shape()[0];
    let mut image = Vec::new();
    image.try_reserve_exact(n_pixels).map_err(|_| {
        KwaversError::InvalidInput("imaging_das: output allocation failed".to_owned())
    })?;

    for pix in 0..n_pixels {
        let px = grid_points[[pix, 0]];
        let py = grid_points[[pix, 1]];
        let pz = grid_points[[pix, 2]];

        let mut acc = S::zero();
        let mut n_active = 0_usize;

        for s in 0..n_sensors {
            let sx = sensor_positions[[s, 0]];
            let sy = sensor_positions[[s, 1]];
            let sz = sensor_positions[[s, 2]];

            let dx = px - sx;
            let dy = py - sy;
            let dz = pz - sz;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let total_delay_s = transmit_delays_s[pix] + dist * inv_c;
            let sample_idx = total_delay_s * fs;

            if !(0.0..=last_sample_f).contains(&sample_idx) {
                continue;
            }

            let lo = sample_idx.floor() as usize;
            let hi = (lo + 1).min(n_samples - 1);
            let frac = sample_idx - lo as f64;
            let v_lo = sensor_data[[s, lo]];
            let v_hi = sensor_data[[s, hi]];
            let interp = v_lo
                .interpolate_weighted(v_hi, frac, weights[s])
                .rephase_baseband(TWO_PI * center_frequency_hz * total_delay_s);
            acc = acc.add(interp);
            n_active += 1;
        }

        if n_active > 0 {
            image.push(acc.normalize(n_active));
        } else {
            image.push(S::zero());
        }
    }

    Array1::from_shape_vec([n_pixels], image)
        .map_err(|error| KwaversError::InvalidInput(format!("imaging_das: output shape: {error}")))
}

fn validate_baseband_carrier_frequency(
    center_frequency_hz: f64,
    sampling_frequency_hz: f64,
) -> KwaversResult<()> {
    if !center_frequency_hz.is_finite() || center_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: center_frequency_hz must be finite and > 0; got {center_frequency_hz}"
        )));
    }
    if center_frequency_hz >= 0.5 * sampling_frequency_hz {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: center_frequency_hz must be below Nyquist; got {center_frequency_hz} for sampling frequency {sampling_frequency_hz}"
        )));
    }
    Ok(())
}

fn zero_transmit_delays(pixel_count: usize) -> KwaversResult<Array1<f64>> {
    let mut delays = Vec::new();
    delays.try_reserve_exact(pixel_count).map_err(|_| {
        KwaversError::InvalidInput("imaging_das: transmit-delay allocation failed".to_owned())
    })?;
    delays.resize(pixel_count, 0.0);
    Array1::from_shape_vec([pixel_count], delays).map_err(|error| {
        KwaversError::InvalidInput(format!("imaging_das: transmit-delay shape: {error}"))
    })
}

fn apodization_weights(n: usize, kind: ImagingDasApodization) -> Vec<f64> {
    use kwavers_math::signal::window::{blackman, hamming, hann};
    if n == 0 {
        return Vec::new();
    }
    // Normalized symmetric position x = i/(n-1) ∈ [0, 1]; n=1 → x=0 (single element).
    // Window coefficients delegate to the kwavers-math SSOT.
    let denom = n.saturating_sub(1).max(1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 / denom;
            match kind {
                ImagingDasApodization::Rectangular => 1.0,
                ImagingDasApodization::Hamming => hamming(x),
                ImagingDasApodization::Hanning => hann(x),
                ImagingDasApodization::Blackman => blackman(x),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::Complex64;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use leto::{Array1, Array2};

    /// Analytical sanity check: a single Gaussian pulse emitted from a point
    /// scatterer arrives at every sensor with a TOF equal to the source–sensor
    /// distance divided by `c`. When the image is beamformed at the true source
    /// location, every sensor's contribution is sampled at its pulse peak, so
    /// the pixel value at the source location must exceed every other pixel in
    /// the imaging grid.
    #[test]
    fn point_source_localizes_at_true_position() {
        let c = SOUND_SPEED_WATER_SIM;
        let fs = 5.0 * MHZ_TO_HZ;
        let dt = 1.0 / fs;

        // Single point source at (depth=10mm, lateral=0).
        let source = [10.0e-3_f64, 0.0_f64, 0.0_f64];

        // 8 sensors on a line at depth=0, lateral=-3.5..3.5 mm at 1 mm pitch.
        let mut sensor_positions = Array2::<f64>::zeros((8, 3));
        for s in 0..8 {
            sensor_positions[[s, 0]] = 0.0;
            sensor_positions[[s, 1]] = (s as f64 - 3.5) * 1.0e-3;
            sensor_positions[[s, 2]] = 0.0;
        }

        // Synthesize a narrow Gaussian pulse at each sensor centred at its TOF.
        let n_samples: usize = 256;
        let pulse_sigma = 2.0 * dt;
        let mut sensor_data = Array2::<f64>::zeros((8, n_samples));
        for s in 0..8 {
            let dx = source[0] - sensor_positions[[s, 0]];
            let dy = source[1] - sensor_positions[[s, 1]];
            let dz = source[2] - sensor_positions[[s, 2]];
            let tof = (dx * dx + dy * dy + dz * dz).sqrt() / c;
            for t in 0..n_samples {
                let dtime = t as f64 * dt - tof;
                sensor_data[[s, t]] = (-(dtime * dtime) / (2.0 * pulse_sigma * pulse_sigma)).exp();
            }
        }

        // Grid: 31 lateral × 21 depth around the source.
        let mut grid = Vec::with_capacity(31 * 21);
        for ix in 0..21 {
            let depth = 5.0e-3 + (ix as f64) * 0.5e-3;
            for iy in 0..31 {
                let lateral = -3.0e-3 + (iy as f64) * 0.2e-3;
                grid.push([depth, lateral, 0.0]);
            }
        }
        let grid_arr =
            Array2::from_shape_vec((grid.len(), 3), grid.into_iter().flatten().collect()).unwrap();

        let config = ImagingDasConfig::new(c, fs, ImagingDasApodization::Rectangular).unwrap();
        let image = beamform_image_das(
            sensor_data.view(),
            sensor_positions.view(),
            grid_arr.view(),
            &config,
        )
        .unwrap();

        // Peak pixel
        let (peak_idx, peak_val) = image
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let peak_depth_idx = peak_idx / 31;
        let peak_lat_idx = peak_idx % 31;
        let peak_depth = 5.0e-3 + (peak_depth_idx as f64) * 0.5e-3;
        let peak_lat = -3.0e-3 + (peak_lat_idx as f64) * 0.2e-3;

        // Source at (10 mm, 0). Grid step: 0.5 mm depth, 0.2 mm lateral. Peak
        // must land within one cell.
        assert!(
            (peak_depth - source[0]).abs() <= 0.5e-3 + 1e-9,
            "peak depth {peak_depth} not within 0.5 mm of source depth {}",
            source[0]
        );
        assert!(
            (peak_lat - source[1]).abs() <= 0.2e-3 + 1e-9,
            "peak lateral {peak_lat} not within 0.2 mm of source lateral {}",
            source[1]
        );
        // For a Gaussian of sigma=2dt the largest fractional-sample offset
        // (≤0.5 dt) attenuates the linearly-interpolated peak to ≈0.97. With 8
        // sensors averaged this floors at ~0.95.
        assert!(
            *peak_val > 0.95,
            "peak image value {peak_val} should be ≈ 1.0 (sum of 8 pulse peaks / 8)"
        );
    }

    #[test]
    fn transmit_delay_localizes_an_active_plane_wave_echo() {
        let sound_speed = 1500.0;
        let sampling_frequency = 1.5e6;
        let mut sensor_data = Array2::<f64>::zeros((2, 32));
        // A +z plane event and a 10 mm receive path each consume ten samples.
        sensor_data[[0, 20]] = 1.0;
        sensor_data[[1, 20]] = 1.0;
        let sensors = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let grid = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 10e-3, 0.0, 0.0, 11e-3]).unwrap();
        let transmit_delays_s =
            Array1::from_vec([2], vec![10e-3 / sound_speed, 11e-3 / sound_speed]).unwrap();
        let config = ImagingDasConfig::new(
            sound_speed,
            sampling_frequency,
            ImagingDasApodization::Rectangular,
        )
        .unwrap();

        let image = beamform_image_das_with_transmit_delays(
            sensor_data.view(),
            sensors.view(),
            grid.view(),
            transmit_delays_s.view(),
            &config,
        )
        .unwrap();

        assert_eq!(image[[0]], 1.0);
        assert_eq!(image[[1]], 0.0);
    }

    #[test]
    fn complex_transmit_das_interpolates_and_rephases_baseband() {
        let sensor_data = Array2::from_shape_vec(
            [1, 3],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(5.0, -2.0),
            ],
        )
        .unwrap();
        let positions = Array2::from_shape_vec([1, 3], vec![0.0, 0.0, 0.0]).unwrap();
        let grid = Array2::from_shape_vec([1, 3], vec![0.0, 0.0, 0.0]).unwrap();
        let delays = Array1::from_vec([1], vec![0.375]).unwrap();
        let config =
            ImagingDasConfig::new(1500.0, 4.0, ImagingDasApodization::Rectangular).unwrap();

        let image = beamform_complex_image_das_with_transmit_delays(
            sensor_data.view(),
            positions.view(),
            grid.view(),
            delays.view(),
            &config,
            1.0,
        )
        .unwrap();

        // Linear interpolation produces 3 + 0j at 1.5 samples. Baseband
        // rephasing restores exp(j 2π × 1 Hz × 0.375 s) = exp(j 3π/4).
        let expected = Complex64::new(-3.0 / 2.0_f64.sqrt(), 3.0 / 2.0_f64.sqrt());
        // Two weighted products, one sum, and a complex phasor product give a
        // conservative gamma_64 roundoff bound for this fixed fixture.
        let scaled_epsilon = 64.0 * f64::EPSILON;
        let bound = scaled_epsilon / (1.0 - scaled_epsilon);
        assert!((image[[0]] - expected).norm() <= bound * expected.norm());
        let non_finite =
            Array2::from_shape_vec([1, 1], vec![Complex64::new(f64::NAN, 0.0)]).unwrap();
        let error = beamform_complex_image_das(
            non_finite.view(),
            positions.view(),
            grid.view(),
            &config,
            1.0,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Invalid input: imaging_das: sensor_data must contain only finite values"
        );
        let carrier_error = beamform_complex_image_das(
            sensor_data.view(),
            positions.view(),
            grid.view(),
            &config,
            2.0,
        )
        .unwrap_err();
        assert_eq!(
            carrier_error.to_string(),
            "Invalid input: imaging_das: center_frequency_hz must be below Nyquist; got 2 for sampling frequency 4"
        );
    }

    #[test]
    fn rejects_invalid_transmit_delays() {
        let sensor_data = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let positions = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let grid = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let delays = Array1::from_vec([1], vec![-1.0e-6]).unwrap();
        let config = ImagingDasConfig::new(
            SOUND_SPEED_WATER_SIM,
            MHZ_TO_HZ,
            ImagingDasApodization::Rectangular,
        )
        .unwrap();

        let error = beamform_image_das_with_transmit_delays(
            sensor_data.view(),
            positions.view(),
            grid.view(),
            delays.view(),
            &config,
        )
        .unwrap_err();
        match error {
            KwaversError::InvalidInput(message) => assert_eq!(
                message,
                "imaging_das: transmit_delays_s must contain only finite non-negative values"
            ),
            other => panic!("expected invalid transmit delay error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_shape_mismatch() {
        let sensor_data =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 0.0, 0.0, 0.5, 0.0]).unwrap();
        let sensor_positions = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let cfg = ImagingDasConfig::new(
            SOUND_SPEED_WATER_SIM,
            MHZ_TO_HZ,
            ImagingDasApodization::Rectangular,
        )
        .unwrap();
        let err = beamform_image_das(
            sensor_data.view(),
            sensor_positions.view(),
            grid_points.view(),
            &cfg,
        )
        .unwrap_err();
        match err {
            KwaversError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_positive_sound_speed() {
        assert!(ImagingDasConfig::new(0.0, MHZ_TO_HZ, ImagingDasApodization::Rectangular).is_err());
        assert!(
            ImagingDasConfig::new(-1.0, MHZ_TO_HZ, ImagingDasApodization::Rectangular).is_err()
        );
        assert!(
            ImagingDasConfig::new(f64::NAN, MHZ_TO_HZ, ImagingDasApodization::Rectangular).is_err()
        );
    }

    #[test]
    fn apodization_windows_have_expected_endpoints() {
        let n = 16;
        let hann = apodization_weights(n, ImagingDasApodization::Hanning);
        assert!(hann[0].abs() < 1e-12);
        assert!(hann[n - 1].abs() < 1e-12);
        let hamm = apodization_weights(n, ImagingDasApodization::Hamming);
        assert!((hamm[0] - 0.08).abs() < 1e-12);
        assert!((hamm[n - 1] - 0.08).abs() < 1e-12);
    }
}
