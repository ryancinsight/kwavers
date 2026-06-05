use super::super::phase_correction::TranscranialAberrationCorrection;
use kwavers_grid::Grid;
use kwavers_math::numerics::operators::interpolation::trilinear_index_space;
use ndarray::Array3;

fn make_correction() -> TranscranialAberrationCorrection {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    TranscranialAberrationCorrection::new(&grid).unwrap()
}

/// 3D Gaussian intensity field centred at fractional grid coordinates (cx, cy, cz).
fn gaussian_field(
    nx: usize,
    ny: usize,
    nz: usize,
    cx: f64,
    cy: f64,
    cz: f64,
    sigma: f64,
) -> Array3<f64> {
    let mut f = Array3::zeros((nx, ny, nz));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let r2 =
                    (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2);
                f[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    f
}

#[test]
fn test_trilinear_at_grid_node() {
    let field = gaussian_field(8, 8, 8, 4.0, 4.0, 4.0, 1.5);
    let result = trilinear_index_space(&field, 3.0, 3.0, 3.0);
    assert!((result - field[[3, 3, 3]]).abs() < 1e-12);
}

#[test]
fn test_trilinear_midpoint_is_average() {
    let mut field = Array3::zeros((4, 4, 4));
    field[[0, 0, 0]] = 2.0;
    field[[1, 0, 0]] = 4.0;
    let result = trilinear_index_space(&field, 0.5, 0.0, 0.0);
    assert!(
        (result - 3.0).abs() < 1e-12,
        "midpoint should be 3.0, got {result}"
    );
}

#[test]
fn test_focal_intensity_positive() {
    let correction = make_correction();
    let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, 3.0);
    let target = [16e-3_f64, 16e-3, 16e-3];
    let intensity = correction.calculate_focal_intensity(&field, &target);
    assert!(
        intensity > 0.0,
        "focal intensity must be positive for non-zero field"
    );
}

#[test]
fn test_focal_intensity_zero_field() {
    let correction = make_correction();
    let field = Array3::<f64>::zeros((32, 32, 32));
    let target = [16e-3_f64, 16e-3, 16e-3];
    assert_eq!(correction.calculate_focal_intensity(&field, &target), 0.0);
}

#[test]
fn test_sidelobe_zero_for_uniform_field() {
    let correction = make_correction();
    let field = Array3::from_elem((32, 32, 32), 1.0);
    let target = [16e-3_f64, 16e-3, 16e-3];
    let ratio = correction.calculate_sidelobe_level(&field, &target);
    assert_eq!(
        ratio, 0.0,
        "uniform field has no sidelobe outside bounding box"
    );
}

#[test]
fn test_sidelobe_less_than_main_lobe_for_gaussian() {
    let correction = make_correction();
    let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, 2.0);
    let target = [16e-3_f64, 16e-3, 16e-3];
    let ratio = correction.calculate_sidelobe_level(&field, &target);
    assert!(
        ratio < 1.0,
        "sidelobe ratio must be < 1 for Gaussian field; got {ratio}"
    );
}

#[test]
fn test_focal_spot_size_matches_gaussian_fwhm() {
    let correction = make_correction();
    let sigma_cells = 3.0_f64;
    let dx = 1e-3_f64;
    let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, sigma_cells);
    let target = [16e-3_f64, 16e-3, 16e-3];
    let spot = correction.calculate_focal_spot_size(&field, &target);
    let expected_fwhm = 2.0 * (2.0_f64 * std::f64::consts::LN_2).sqrt() * sigma_cells * dx;
    let tol = 2.0 * dx;
    assert!(
        (spot - expected_fwhm).abs() <= tol,
        "FWHM {spot:.4e} m, expected ≈ {expected_fwhm:.4e} m (±{tol:.1e})"
    );
}

#[test]
fn test_focal_spot_size_zero_field() {
    let correction = make_correction();
    let field = Array3::<f64>::zeros((32, 32, 32));
    let target = [0.0_f64, 0.0, 0.0];
    assert_eq!(correction.calculate_focal_spot_size(&field, &target), 0.0);
}
