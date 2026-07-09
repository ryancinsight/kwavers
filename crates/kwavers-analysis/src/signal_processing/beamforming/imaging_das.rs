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
//! τ_{s,p} = ‖r_p − r_s‖ / c            (one-way time of flight)
//! k_{s,p} = τ_{s,p} · f_s              (fractional sample index)
//! image[p] = (1 / N_active(p)) · Σ_s w_s · linterp(sensor_data[s, ·], k_{s,p})
//! ```
//!
//! Sensors whose interpolated sample index falls outside `[0, n_samples − 1]`
//! are excluded from both the sum and `N_active(p)`. When `N_active(p) = 0` the
//! pixel value is zero.

use leto::{
    Array1,
    ArrayView2,
};

use kwavers_core::error::{KwaversError, KwaversResult};

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
/// length `grid_points.nrows()` matching grid-point row order.
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
    let (n_sensors, n_samples) = sensor_data.dim();
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sensor_data must have n_sensors > 0 and n_samples > 0; got ({n_sensors}, {n_samples})"
        )));
    }
    if sensor_positions.ncols() != 3 || sensor_positions.nrows() != n_sensors {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: sensor_positions must have shape ({n_sensors}, 3); got {:?}",
            sensor_positions.dim()
        )));
    }
    if grid_points.ncols() != 3 || grid_points.nrows() == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "imaging_das: grid_points must have shape (n_pixels >= 1, 3); got {:?}",
            grid_points.dim()
        )));
    }
    if !sensor_data.iter().all(|v| v.is_finite()) {
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

    let weights = apodization_weights(n_sensors, config.apodization);
    let inv_c = 1.0 / config.sound_speed;
    let fs = config.sampling_frequency;
    let last_sample_f = (n_samples - 1) as f64;

    let n_pixels = grid_points.nrows();
    let mut image = Array1::<f64>::zeros(n_pixels);

    for (pix, point) in grid_points.outer_iter().enumerate() {
        let px = point[0];
        let py = point[1];
        let pz = point[2];

        let mut acc = 0.0_f64;
        let mut n_active = 0_usize;

        for s in 0..n_sensors {
            let sx = sensor_positions[(s, 0)];
            let sy = sensor_positions[(s, 1)];
            let sz = sensor_positions[(s, 2)];

            let dx = px - sx;
            let dy = py - sy;
            let dz = pz - sz;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let sample_idx = dist * inv_c * fs;

            if !(0.0..=last_sample_f).contains(&sample_idx) {
                continue;
            }

            let lo = sample_idx.floor() as usize;
            let hi = (lo + 1).min(n_samples - 1);
            let frac = sample_idx - lo as f64;
            let v_lo = sensor_data[(s, lo)];
            let v_hi = sensor_data[(s, hi)];
            let interp = (1.0 - frac) * v_lo + frac * v_hi;

            acc += weights[s] * interp;
            n_active += 1;
        }

        if n_active > 0 {
            image[pix] = acc / n_active as f64;
        }
    }

    Ok(image)
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
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use leto::{
    /* array -- no leto equivalent */,
    Array2,
};

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
        for (s, mut row) in sensor_positions.outer_iter_mut().enumerate() {
            row[0] = 0.0;
            row[1] = (s as f64 - 3.5) * 1.0e-3;
            row[2] = 0.0;
        }

        // Synthesize a narrow Gaussian pulse at each sensor centred at its TOF.
        let n_samples: usize = 256;
        let pulse_sigma = 2.0 * dt;
        let mut sensor_data = Array2::<f64>::zeros((8, n_samples));
        for s in 0..8 {
            let dx = source[0] - sensor_positions[(s, 0)];
            let dy = source[1] - sensor_positions[(s, 1)];
            let dz = source[2] - sensor_positions[(s, 2)];
            let tof = (dx * dx + dy * dy + dz * dz).sqrt() / c;
            for t in 0..n_samples {
                let dtime = t as f64 * dt - tof;
                sensor_data[(s, t)] = (-(dtime * dtime) / (2.0 * pulse_sigma * pulse_sigma)).exp();
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
    fn rejects_shape_mismatch() {
        let sensor_data = array![[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]];
        let sensor_positions = array![[0.0, 0.0, 0.0]];
        let grid_points = array![[0.0, 0.0, 0.0]];
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
