use crate::math::fft::Complex64;
use crate::math::numerics::operators::spectral::derivative::PseudospectralDerivative;
use crate::math::numerics::operators::spectral::filter::{FilterType, SpectralFilter};
use crate::math::numerics::operators::spectral::trait_def::SpectralOperator;
use approx::assert_abs_diff_eq;
use ndarray::Array3;
use std::f64::consts::PI;

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
