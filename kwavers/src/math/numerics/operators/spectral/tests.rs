use super::derivative::PseudospectralDerivative;
use super::filter::{FilterType, SpectralFilter};
use super::trait_def::SpectralOperator;
use crate::math::fft::Complex64;
use approx::assert_abs_diff_eq;
use ndarray::Array3;
use std::f64::consts::PI;

#[test]
fn test_wavenumber_vector() {
    let n = 8;
    let d = 0.1;
    let k = PseudospectralDerivative::wavenumber_vector(n, d);

    assert_eq!(k.len(), n);
    assert_abs_diff_eq!(k[0], 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(k[1], -k[n - 1], epsilon = 1e-10);
}

#[test]
fn test_pseudospectral_creation() {
    let op = PseudospectralDerivative::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();

    let (kx, ky, kz) = op.wavenumber_grid();
    assert_eq!(kx.len(), 64);
    assert_eq!(ky.len(), 64);
    assert_eq!(kz.len(), 64);
}

#[test]
fn test_nyquist_wavenumber() {
    let dx = 0.001;
    let op = PseudospectralDerivative::new(100, 100, 100, dx, dx, dx).unwrap();

    let (kx_nyq, ky_nyq, kz_nyq) = op.nyquist_wavenumber();
    let expected = PI / dx;
    assert_abs_diff_eq!(kx_nyq, expected, epsilon = 1e-10);
    assert_abs_diff_eq!(ky_nyq, expected, epsilon = 1e-10);
    assert_abs_diff_eq!(kz_nyq, expected, epsilon = 1e-10);
}

#[test]
fn test_spectral_filter_sharp_cutoff() {
    let filter = SpectralFilter::new(0.67, FilterType::SharpCutoff);
    let k_nyquist = 1000.0;

    assert_abs_diff_eq!(
        filter.transfer_function(0.5 * k_nyquist, k_nyquist),
        1.0,
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        filter.transfer_function(0.8 * k_nyquist, k_nyquist),
        0.0,
        epsilon = 1e-10
    );
}

#[test]
fn test_spectral_filter_smooth() {
    let filter = SpectralFilter::new(0.67, FilterType::Smooth);
    let k_nyquist = 1000.0;

    let h_cutoff = filter.transfer_function(0.67 * k_nyquist, k_nyquist);
    assert_abs_diff_eq!(h_cutoff, 1.0, epsilon = 1e-10);

    let h_mid = filter.transfer_function(0.8 * k_nyquist, k_nyquist);
    assert!(h_mid > 0.0 && h_mid < 1.0);

    let h_nyq = filter.transfer_function(k_nyquist, k_nyquist);
    assert!(h_nyq < 0.1);
}

#[test]
fn spectral_filter_preserves_constant_field() {
    let filter = SpectralFilter::new(0.5, FilterType::SharpCutoff);
    let field = Array3::from_elem((8, 4, 2), 3.25);

    let filtered = filter.apply(field.view()).unwrap();

    for value in filtered {
        assert_abs_diff_eq!(value, 3.25, epsilon = 1e-12);
    }
}

#[test]
fn spectral_filter_removes_rejected_nyquist_mode_and_preserves_low_mode() {
    let nx = 16;
    let dx = 0.1;
    let low_k = 2.0 * PI / (nx as f64 * dx);
    let filter = SpectralFilter::new(0.5, FilterType::SharpCutoff);
    let mut field = Array3::zeros((nx, 1, 1));

    for i in 0..nx {
        let x = i as f64 * dx;
        let low_mode = (low_k * x).sin();
        let nyquist_mode = if i.is_multiple_of(2) { 0.4 } else { -0.4 };
        field[[i, 0, 0]] = low_mode + nyquist_mode;
    }

    let filtered = filter.apply(field.view()).unwrap();

    for i in 0..nx {
        let expected = (low_k * i as f64 * dx).sin();
        assert_abs_diff_eq!(filtered[[i, 0, 0]], expected, epsilon = 1e-12);
    }
}

#[test]
fn spectral_filter_apply_into_reuses_workspaces_and_matches_apply() {
    let nx = 16;
    let ny = 4;
    let nz = 2;
    let filter = SpectralFilter::new(0.5, FilterType::SharpCutoff);
    let mut field = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        let low_mode = (2.0 * PI * i as f64 / nx as f64).sin();
        let rejected_mode = if i.is_multiple_of(2) { 0.25 } else { -0.25 };
        for j in 0..ny {
            for k in 0..nz {
                field[[i, j, k]] = low_mode + rejected_mode + 0.125 * j as f64;
            }
        }
    }

    let expected = filter.apply(field.view()).unwrap();
    let mut spectrum = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut output = Array3::<f64>::zeros((nx, ny, nz));
    let spectrum_ptr = spectrum.as_ptr();
    let output_ptr = output.as_ptr();

    filter
        .apply_into(field.view(), &mut spectrum, &mut output)
        .unwrap();

    assert_eq!(spectrum.as_ptr(), spectrum_ptr);
    assert_eq!(output.as_ptr(), output_ptr);
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-12);
    }

    field.mapv_inplace(|value| 0.5 * value);
    let expected_second = filter.apply(field.view()).unwrap();
    filter
        .apply_into(field.view(), &mut spectrum, &mut output)
        .unwrap();

    assert_eq!(spectrum.as_ptr(), spectrum_ptr);
    assert_eq!(output.as_ptr(), output_ptr);
    for (actual, expected) in output.iter().zip(expected_second.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-12);
    }
}

#[test]
fn spectral_filter_apply_into_rejects_mismatched_workspaces() {
    let filter = SpectralFilter::new(0.5, FilterType::SharpCutoff);
    let field = Array3::<f64>::zeros((4, 4, 4));
    let mut spectrum = Array3::<Complex64>::zeros((4, 4, 3));
    let mut output = Array3::<f64>::zeros((4, 4, 4));

    let error = filter
        .apply_into(field.view(), &mut spectrum, &mut output)
        .unwrap_err();

    assert!(format!("{error}").contains("spectrum workspace shape"));

    spectrum = Array3::<Complex64>::zeros((4, 4, 4));
    output = Array3::<f64>::zeros((4, 3, 4));
    let error = filter
        .apply_into(field.view(), &mut spectrum, &mut output)
        .unwrap_err();

    assert!(format!("{error}").contains("output workspace shape"));
}

#[test]
fn spectral_operator_antialias_filter_uses_real_filter() {
    let nx = 16;
    let op = PseudospectralDerivative::new(nx, 1, 1, 0.1, 0.1, 0.1).unwrap();
    let mut field = Array3::zeros((nx, 1, 1));
    for i in 0..nx {
        field[[i, 0, 0]] = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
    }

    let filtered = op.apply_antialias_filter(field.view()).unwrap();

    for value in filtered {
        assert_abs_diff_eq!(value, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn spectral_filter_rejects_invalid_cutoff_on_apply() {
    let filter = SpectralFilter::new(f64::NAN, FilterType::SharpCutoff);
    let field = Array3::zeros((2, 2, 2));

    let error = filter.apply(field.view()).unwrap_err();

    assert!(format!("{error}").contains("cutoff must be finite"));
}

#[test]
fn test_invalid_grid_spacing() {
    assert!(PseudospectralDerivative::new(10, 10, 10, -0.1, 0.1, 0.1).is_err());
}

#[test]
fn test_derivative_x_sine_wave() {
    let nx = 64;
    let ny = 4;
    let nz = 4;
    let dx = 0.1;
    let dy = 0.1;
    let dz = 0.1;

    let op = PseudospectralDerivative::new(nx, ny, nz, dx, dy, dz).unwrap();
    let k = 2.0 * PI / (nx as f64 * dx);

    let mut field = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        let val = (k * i as f64 * dx).sin();
        for j in 0..ny {
            for l in 0..nz {
                field[[i, j, l]] = val;
            }
        }
    }

    let deriv = op.derivative_x(field.view()).unwrap();

    for i in 0..nx {
        let expected = k * (k * i as f64 * dx).cos();
        assert_abs_diff_eq!(deriv[[i, 0, 0]], expected, epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_y_sine_wave() {
    let nx = 4;
    let ny = 64;
    let nz = 4;
    let dx = 0.1;
    let dy = 0.1;
    let dz = 0.1;

    let op = PseudospectralDerivative::new(nx, ny, nz, dx, dy, dz).unwrap();
    let k = 2.0 * PI / (ny as f64 * dy);

    let mut field = Array3::zeros((nx, ny, nz));
    for j in 0..ny {
        let val = (k * j as f64 * dy).sin();
        for i in 0..nx {
            for l in 0..nz {
                field[[i, j, l]] = val;
            }
        }
    }

    let deriv = op.derivative_y(field.view()).unwrap();

    for j in 0..ny {
        let expected = k * (k * j as f64 * dy).cos();
        assert_abs_diff_eq!(deriv[[0, j, 0]], expected, epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_z_sine_wave() {
    let nx = 4;
    let ny = 4;
    let nz = 64;
    let dx = 0.1;
    let dy = 0.1;
    let dz = 0.1;

    let op = PseudospectralDerivative::new(nx, ny, nz, dx, dy, dz).unwrap();
    let k = 2.0 * PI / (nz as f64 * dz);

    let mut field = Array3::zeros((nx, ny, nz));
    for l in 0..nz {
        let val = (k * l as f64 * dz).sin();
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j, l]] = val;
            }
        }
    }

    let deriv = op.derivative_z(field.view()).unwrap();

    for l in 0..nz {
        let expected = k * (k * l as f64 * dz).cos();
        assert_abs_diff_eq!(deriv[[0, 0, l]], expected, epsilon = 1e-10);
    }
}

#[test]
fn spectral_derivative_into_reuses_workspace_and_matches_allocating() {
    let nx = 16;
    let ny = 8;
    let nz = 4;
    let op = PseudospectralDerivative::new(nx, ny, nz, 0.1, 0.2, 0.3).unwrap();
    let mut field = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field[[i, j, k]] = (2.0 * PI * i as f64 / nx as f64).sin()
                    + 0.25 * (2.0 * PI * j as f64 / ny as f64).cos()
                    + 0.125 * (2.0 * PI * k as f64 / nz as f64).sin();
            }
        }
    }

    let expected_x = op.derivative_x(field.view()).unwrap();
    let mut line_x = ndarray::Array1::<Complex64>::zeros(nx);
    let mut out_x = Array3::<f64>::zeros((nx, ny, nz));
    let line_x_ptr = line_x.as_ptr();
    let out_x_ptr = out_x.as_ptr();
    op.derivative_x_into(field.view(), &mut line_x, &mut out_x)
        .unwrap();
    assert_eq!(line_x.as_ptr(), line_x_ptr);
    assert_eq!(out_x.as_ptr(), out_x_ptr);
    assert_arrays_close(&out_x, &expected_x, 1e-12);

    let expected_y = op.derivative_y(field.view()).unwrap();
    let mut line_y = ndarray::Array1::<Complex64>::zeros(ny);
    let mut out_y = Array3::<f64>::zeros((nx, ny, nz));
    let line_y_ptr = line_y.as_ptr();
    let out_y_ptr = out_y.as_ptr();
    op.derivative_y_into(field.view(), &mut line_y, &mut out_y)
        .unwrap();
    assert_eq!(line_y.as_ptr(), line_y_ptr);
    assert_eq!(out_y.as_ptr(), out_y_ptr);
    assert_arrays_close(&out_y, &expected_y, 1e-12);

    let expected_z = op.derivative_z(field.view()).unwrap();
    let mut line_z = ndarray::Array1::<Complex64>::zeros(nz);
    let mut out_z = Array3::<f64>::zeros((nx, ny, nz));
    let line_z_ptr = line_z.as_ptr();
    let out_z_ptr = out_z.as_ptr();
    op.derivative_z_into(field.view(), &mut line_z, &mut out_z)
        .unwrap();
    assert_eq!(line_z.as_ptr(), line_z_ptr);
    assert_eq!(out_z.as_ptr(), out_z_ptr);
    assert_arrays_close(&out_z, &expected_z, 1e-12);
}

#[test]
fn spectral_derivative_into_rejects_mismatched_workspaces() {
    let op = PseudospectralDerivative::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
    let field = Array3::<f64>::zeros((4, 4, 4));
    let mut line = ndarray::Array1::<Complex64>::zeros(3);
    let mut output = Array3::<f64>::zeros((4, 4, 4));

    let error = op
        .derivative_x_into(field.view(), &mut line, &mut output)
        .unwrap_err();
    assert!(format!("{error}").contains("line workspace length"));

    line = ndarray::Array1::<Complex64>::zeros(4);
    output = Array3::<f64>::zeros((4, 4, 3));
    let error = op
        .derivative_x_into(field.view(), &mut line, &mut output)
        .unwrap_err();
    assert!(format!("{error}").contains("output shape"));
}

#[test]
fn test_derivative_of_constant_is_zero() {
    let nx = 32;
    let ny = 32;
    let nz = 32;

    let op = PseudospectralDerivative::new(nx, ny, nz, 0.1, 0.1, 0.1).unwrap();
    let field = Array3::from_elem((nx, ny, nz), 5.0);

    let deriv_x = op.derivative_x(field.view()).unwrap();
    let deriv_y = op.derivative_y(field.view()).unwrap();
    let deriv_z = op.derivative_z(field.view()).unwrap();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                assert_abs_diff_eq!(deriv_x[[i, j, k]], 0.0, epsilon = 1e-12);
                assert_abs_diff_eq!(deriv_y[[i, j, k]], 0.0, epsilon = 1e-12);
                assert_abs_diff_eq!(deriv_z[[i, j, k]], 0.0, epsilon = 1e-12);
            }
        }
    }
}

fn assert_arrays_close(actual: &Array3<f64>, expected: &Array3<f64>, epsilon: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = epsilon);
    }
}

#[test]
fn test_spectral_accuracy_exponential() {
    let nx = 32;
    let ny = 4;
    let nz = 4;
    let dx = 0.05;
    let dy = 0.1;
    let dz = 0.1;

    let op = PseudospectralDerivative::new(nx, ny, nz, dx, dy, dz).unwrap();

    let k1 = 2.0 * PI / (nx as f64 * dx);
    let k2 = 4.0 * PI / (nx as f64 * dx);

    let mut field = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        let x = i as f64 * dx;
        let val = (k1 * x).sin() + 0.5 * (k2 * x).cos();
        for j in 0..ny {
            for l in 0..nz {
                field[[i, j, l]] = val;
            }
        }
    }

    let deriv = op.derivative_x(field.view()).unwrap();

    let mut max_error: f64 = 0.0;
    for i in 0..nx {
        let x = i as f64 * dx;
        let expected = k1 * (k1 * x).cos() - 0.5 * k2 * (k2 * x).sin();
        max_error = max_error.max((deriv[[i, 0, 0]] - expected).abs());
    }

    assert!(
        max_error < 1e-11,
        "Max error {} exceeds spectral accuracy threshold",
        max_error
    );
}
