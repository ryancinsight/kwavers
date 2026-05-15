//! k-space line reconstruction for 2-D photoacoustic line sensors.
//!
//! # Theory: k-Space Line Reconstruction
//!
//! The vendored k-Wave line-sensor example mirrors the recorded pressure in
//! time, computes a 2-D FFT over `(t, y)`, maps the spectrum from
//! `(ω, k_y)` to `(k_x, k_y)` using the homogeneous dispersion relation
//! `ω = c·sqrt(k_x² + k_y²)`, and then performs an inverse FFT. The result is
//! a reconstruction of the initial pressure field on the line-sensor grid.
//!
//! In the discrete setting used here, the mirrored time axis has length
//! `2Nt - 1`. The target `k_y` coordinates lie exactly on the source
//! `k_y` grid, so the 2-D regular-grid interpolation degenerates to a
//! 1-D interpolation along the frequency axis for each detector column.
//! This is the same transform as the vendored reference, but expressed with
//! a tighter memory footprint and explicit SSOT helpers for the FFT shifts.
//!
//! # Theorem
//!
//! Let `p(t, y)` be a uniformly sampled pressure trace recorded on a line
//! detector in a homogeneous medium with constant sound speed `c`. If the
//! discrete spectrum is mirrored about `t = 0`, shifted to centered FFT
//! ordering, multiplied by the k-space line-reconstruction scaling factor,
//! and interpolated onto the dispersion manifold `ω = c·|k|`, then the
//! inverse transform produces the same discrete reconstruction as the
//! k-Wave line-reconstruction algorithm for the same interpolation mode.
//!
//! # Proof sketch
//!
//! The mirrored trace is an even extension in time, so the FFT represents the
//! cosine-series content of the measured field. The centered FFT ordering used
//! by `fft_shift_2d` and `ifft_shift_2d` matches the k-Wave spectral grid
//! convention. The target interpolation points share the source `k_y` axis
//! exactly, so only the frequency axis requires interpolation. The remaining
//! scaling factor is the same `2·2/c` compensation used by the reference
//! implementation to account for the one-sided detector line.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::math::fft::utils::{fft_shift_2d, ifft_shift_2d};
use crate::math::fft::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::f64::consts::PI;

/// Order of the sensor data axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineReconDataOrder {
    /// `(time, sensor)`
    Ty,
    /// `(sensor, time)`
    Yt,
}

/// Interpolation mode used on the k-space frequency axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineReconInterpolation {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Linear interpolation.
    Linear,
}

/// Reconstruct an initial pressure estimate from a 2-D line sensor.
///
/// # Inputs
/// * `sensor_data` - pressure trace matrix in either `(time, sensor)` or
///   `(sensor, time)` order
/// * `dy` - detector spacing in meters
/// * `dt` - sampling interval in seconds
/// * `c` - homogeneous sound speed in m/s
/// * `data_order` - input axis order
/// * `interp` - interpolation mode on the k-space frequency axis
/// * `pos_cond` - if true, clamp negative output values to zero
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn kspace_line_recon(
    sensor_data: ArrayView2<'_, f64>,
    dy: f64,
    dt: f64,
    c: f64,
    data_order: LineReconDataOrder,
    interp: LineReconInterpolation,
    pos_cond: bool,
) -> KwaversResult<Array2<f64>> {
    validate_scalar("dy", dy)?;
    validate_scalar("dt", dt)?;
    validate_scalar("c", c)?;

    let data = match data_order {
        LineReconDataOrder::Ty => sensor_data.to_owned(),
        LineReconDataOrder::Yt => sensor_data.t().to_owned(),
    };

    if data.nrows() < 2 || data.ncols() == 0 {
        return Err(KwaversError::Validation(ValidationError::FieldValidation {
            field: "sensor_data".to_owned(),
            value: format!("{:?}", data.dim()),
            constraint: "expected at least two time samples and one detector column".to_owned(),
        }));
    }

    let mirrored = mirror_time_axis(&data);
    let (nt, ny) = mirrored.dim();
    let x_spacing = dt * c;
    let kx_axis = centered_wavenumber_vector(nt, x_spacing);
    let ky_axis = centered_wavenumber_vector(ny, dy);
    let w_axis: Vec<f64> = kx_axis.iter().map(|&kx| c * kx).collect();

    let mut spectrum = mirrored.mapv(|v| Complex64::new(v, 0.0));
    ifft_shift_2d(&mut spectrum);
    fft_2d_complex_inplace(&mut spectrum);
    fft_shift_2d(&mut spectrum);

    let mut scaled = Array2::<Complex64>::zeros((nt, ny));
    for j in 0..ny {
        let ky = ky_axis[j];
        for i in 0..nt {
            let w = w_axis[i];
            if w.abs() < (c * ky).abs() {
                continue;
            }

            let scale = if w == 0.0 && ky == 0.0 {
                Complex64::new(c / 2.0, 0.0)
            } else {
                let arg = ky.mul_add(-ky, (w / c).powi(2));
                let root = Complex64::new(arg, 0.0).sqrt();
                c * c * root / (2.0 * w)
            };

            scaled[[i, j]] = scale * spectrum[[i, j]];
        }
    }

    let mut interpolated = Array2::<Complex64>::zeros((nt, ny));
    for j in 0..ny {
        let column = scaled.index_axis(Axis(1), j);
        let ky = ky_axis[j];
        for i in 0..nt {
            let target_w = c * kx_axis[i].hypot(ky);
            interpolated[[i, j]] = interpolate_on_axis(&w_axis, column, target_w, interp);
        }
    }

    ifft_shift_2d(&mut interpolated);
    ifft_2d_complex_inplace(&mut interpolated);
    fft_shift_2d(&mut interpolated);

    let trim_start = nt / 2;
    let mut recon = interpolated
        .slice(s![trim_start.., ..])
        .mapv(|value| value.re * (4.0 / c));

    if pos_cond {
        recon.par_mapv_inplace(|value| value.max(0.0));
    }

    Ok(recon)
}

fn validate_scalar(name: &str, value: f64) -> KwaversResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::FieldValidation {
            field: name.to_owned(),
            value: value.to_string(),
            constraint: "expected a finite, strictly positive scalar".to_owned(),
        }));
    }
    Ok(())
}

fn centered_wavenumber_vector(n: usize, spacing: f64) -> Array1<f64> {
    let scale = 2.0 * PI / (n as f64 * spacing);
    Array1::from_shape_fn(n, |idx| (idx as isize - (n as isize / 2)) as f64 * scale)
}

fn mirror_time_axis(input: &Array2<f64>) -> Array2<f64> {
    let (nt, ny) = input.dim();
    Array2::from_shape_fn((2 * nt - 1, ny), |(i, j)| {
        if i < nt {
            input[[nt - 1 - i, j]]
        } else {
            input[[i - nt + 1, j]]
        }
    })
}

fn interpolate_on_axis(
    axis: &[f64],
    values: ArrayView1<'_, Complex64>,
    target: f64,
    interp: LineReconInterpolation,
) -> Complex64 {
    if axis.is_empty() || !target.is_finite() {
        return Complex64::new(0.0, 0.0);
    }
    let first = axis[0];
    let last = axis[axis.len() - 1];
    if target < first || target > last {
        return Complex64::new(0.0, 0.0);
    }

    let upper = axis.partition_point(|&value| value <= target);
    if upper == 0 {
        return values[0];
    }
    if upper >= axis.len() {
        return values[axis.len() - 1];
    }

    let lower = upper - 1;
    match interp {
        LineReconInterpolation::Nearest => {
            let lower_dist = (target - axis[lower]).abs();
            let upper_dist = (axis[upper] - target).abs();
            if upper_dist < lower_dist {
                values[upper]
            } else {
                values[lower]
            }
        }
        LineReconInterpolation::Linear => {
            let span = axis[upper] - axis[lower];
            if span.abs() <= f64::EPSILON {
                values[lower]
            } else {
                let alpha = (target - axis[lower]) / span;
                values[lower] * (1.0 - alpha) + values[upper] * alpha
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use ndarray::array;

    #[test]
    fn line_reconstruction_zero_input_stays_zero() {
        let sensor = Array2::<f64>::zeros((5, 4));
        let recon = kspace_line_recon(
            sensor.view(),
            0.1e-3,
            1.0e-8,
            SOUND_SPEED_WATER_SIM,
            LineReconDataOrder::Ty,
            LineReconInterpolation::Linear,
            true,
        )
        .expect("zero input must reconstruct");
        assert_eq!(recon.shape(), &[5, 4]);
        assert!(recon.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn line_reconstruction_data_order_is_equivalent_under_transpose() {
        let sensor_ty = array![
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ];
        let recon_ty = kspace_line_recon(
            sensor_ty.view(),
            0.1e-3,
            1.0e-8,
            SOUND_SPEED_WATER_SIM,
            LineReconDataOrder::Ty,
            LineReconInterpolation::Nearest,
            false,
        )
        .expect("line reconstruction must succeed");
        let recon_yt = kspace_line_recon(
            sensor_ty.t(),
            0.1e-3,
            1.0e-8,
            SOUND_SPEED_WATER_SIM,
            LineReconDataOrder::Yt,
            LineReconInterpolation::Nearest,
            false,
        )
        .expect("transposed reconstruction must succeed");
        assert_eq!(recon_ty, recon_yt);
    }

    #[test]
    fn interpolation_rejects_out_of_range_target_values() {
        let axis = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let values = array![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let interpolated =
            interpolate_on_axis(&axis, values.view(), 3.0, LineReconInterpolation::Linear);
        assert_eq!(interpolated, Complex64::new(0.0, 0.0));
    }
}
