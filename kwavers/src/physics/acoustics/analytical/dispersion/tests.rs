//! Tests for dispersion analysis and correction

use super::*;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

#[test]
fn test_fdtd_dispersion_3d_axis_aligned_low_dispersion() {
    let freq = 1e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;

    let dx = wavelength / 20.0;
    let dt = 0.4 * dx / (c * 3.0_f64.sqrt());

    let dispersion_3d = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c);

    assert!(
        dispersion_3d.abs() < 0.01,
        "Dispersion should be < 1% at 20 PPW: got {}",
        dispersion_3d
    );
    assert!(dispersion_3d < 0.0, "FDTD typically has negative dispersion");
}

#[test]
fn test_fdtd_dispersion_3d_oblique_propagation() {
    let freq = 1e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k_mag = 2.0 * PI / wavelength;
    let kx = k_mag / 2.0_f64.sqrt();
    let ky = k_mag / 2.0_f64.sqrt();

    let dx = wavelength / 20.0;
    let dt = 0.4 * dx / (c * 3.0_f64.sqrt());

    let dispersion = DispersionAnalysis::fdtd_dispersion_3d(kx, ky, 0.0, dx, dx, dx, dt, c);
    assert!(dispersion.abs() < 0.02);
}

#[test]
fn test_fdtd_dispersion_3d_anisotropic_grid() {
    let freq = 1e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;

    let dx = wavelength / 20.0;
    let dy = wavelength / 15.0;
    let dz = wavelength / 25.0;
    let dt = 0.3 * dx.min(dy).min(dz) / (c * 3.0_f64.sqrt());

    let dispersion_x = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dy, dz, dt, c);
    let dispersion_y = DispersionAnalysis::fdtd_dispersion_3d(0.0, k, 0.0, dx, dy, dz, dt, c);
    let dispersion_z = DispersionAnalysis::fdtd_dispersion_3d(0.0, 0.0, k, dx, dy, dz, dt, c);

    assert!((dispersion_x - dispersion_y).abs() > 1e-6);
    assert!((dispersion_x - dispersion_z).abs() > 1e-6);
}

#[test]
fn test_fdtd_dispersion_3d_cfl_stability() {
    let freq = 1e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;
    let dx = wavelength / 10.0;
    let dt_stable = 0.5 * dx / (c * 3.0_f64.sqrt());

    let dispersion_stable =
        DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt_stable, c);

    assert!(dispersion_stable.is_finite());
    assert!(dispersion_stable.abs() < 0.1);
}

#[test]
fn test_pstd_dispersion_3d_isotropic() {
    let freq = 2e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;
    let dx = wavelength / 10.0;
    let dt = 0.25 * dx / c;

    let k_comp = k / 3.0_f64.sqrt();
    let dispersion =
        DispersionAnalysis::pstd_dispersion_3d(k_comp, k_comp, k_comp, dx, dx, dx, dt, c, 2);

    assert!(dispersion.abs() < 0.01);
}

#[test]
fn test_pstd_dispersion_3d_fourth_order() {
    let freq = 2e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;
    let dx = wavelength / 8.0;
    let dt = 0.2 * dx / c;

    let dispersion_2nd =
        DispersionAnalysis::pstd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c, 2);
    let dispersion_4th =
        DispersionAnalysis::pstd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c, 4);

    assert!(dispersion_4th.abs() < dispersion_2nd.abs());
}

#[test]
fn test_apply_correction_3d() {
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).expect("Failed to create grid");
    let mut field = Array3::from_elem((32, 32, 32), 1.0);

    let freq = 1e6;
    let c = 1500.0;
    let k = 2.0 * PI * freq / c;
    let dt = 5e-8;

    let method = DispersionMethod::FDTD3D { dt };

    DispersionAnalysis::apply_correction_3d(&mut field, &grid, k, 0.0, 0.0, c, method);

    assert!((field[[0, 0, 0]] - 1.0).abs() > 1e-6);
    assert!(field.iter().all(|&v| v.is_finite() && v > 0.0));
}

#[test]
fn test_dispersion_zero_wavenumber() {
    let dispersion =
        DispersionAnalysis::fdtd_dispersion_3d(0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4, 1e-7, 1500.0);
    assert_eq!(dispersion, 0.0);

    let dispersion_pstd = DispersionAnalysis::pstd_dispersion_3d(
        0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4, 1e-7, 1500.0, 2,
    );
    assert_eq!(dispersion_pstd, 0.0);
}

#[test]
fn test_dispersion_symmetry() {
    let freq = 1e6;
    let c = 1500.0;
    let wavelength = c / freq;
    let k = 2.0 * PI / wavelength;
    let dx = wavelength / 15.0;
    let dt = 0.3 * dx / (c * 3.0_f64.sqrt());

    let disp_pos = DispersionAnalysis::fdtd_dispersion_3d(k, 0.0, 0.0, dx, dx, dx, dt, c);
    let disp_neg = DispersionAnalysis::fdtd_dispersion_3d(-k, 0.0, 0.0, dx, dx, dx, dt, c);

    assert!((disp_pos - disp_neg).abs() < 1e-10);
}

#[test]
fn test_dispersion_method_enum_variants() {
    let _fdtd = DispersionMethod::FDTD(1e-7);
    let _pstd = DispersionMethod::PSTD(2);
    let _fdtd_3d = DispersionMethod::FDTD3D { dt: 1e-7 };
    let _pstd_3d = DispersionMethod::PSTD3D { dt: 1e-7, order: 2 };
    let _none = DispersionMethod::None;

    let method = DispersionMethod::FDTD3D { dt: 1e-7 };
    let _copy = method;
    let _another = method;
}
