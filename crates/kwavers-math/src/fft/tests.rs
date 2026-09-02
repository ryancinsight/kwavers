//! Half-spectrum r2c/c2r emulation against the full-spectrum plan.

use super::{fft_3d_complex_inplace, get_fft_for_grid, Fft3dInOutExt};
use eunomia::Complex64;
use leto::{Array3, Layout, VecStorage};

fn check_shape(nx: usize, ny: usize, nz: usize) {
    let nz_c = nz / 2 + 1;
    let fft = get_fft_for_grid(nx, ny, nz);
    let real = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        let x = ((i * 131 + j * 17 + k * 7) % 101) as f64 / 101.0 - 0.5;
        (x * std::f64::consts::TAU).sin() + 0.3 * x + 0.1
    });

    // (1) forward_r2c is bit-identical to a full c2c + truncation.
    let mut half_new = Array3::zeros([nx, ny, nz_c]);
    fft.forward_r2c_into(&real, &mut half_new);
    let mut full = real.mapv(|v| Complex64::new(v, 0.0));
    fft_3d_complex_inplace(&mut full);
    let ref_half = full.slice(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)]).unwrap();
    let fwd_err = half_new
        .iter()
        .zip(ref_half.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        fwd_err < 1e-9,
        "forward_r2c({nx},{ny},{nz}) vs full-c2c reference: {fwd_err:.2e}"
    );

    // (2) inverse_c2r recovers the real field (round-trip).
    let mut out = Array3::zeros([nx, ny, nz]);
    let mut scratch = Array3::zeros([nx, ny, nz_c]);
    fft.inverse_c2r_into(&half_new, &mut out, &mut scratch);
    let rt_err = out
        .iter()
        .zip(real.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        rt_err < 1e-9,
        "r2c→c2r round-trip ({nx},{ny},{nz}): {rt_err:.2e}"
    );
}

#[test]
fn optimized_r2c_c2r_matches_reference_and_roundtrips() {
    check_shape(8, 6, 10); // even nz
    check_shape(7, 5, 9); // odd nz
    check_shape(16, 16, 16); // power-of-two cube
    check_shape(12, 1, 8); // degenerate y (2-D-like)
}

#[test]
fn strided_half_spectrum_matches_contiguous_paths() {
    const NX: usize = 3;
    const NY: usize = 3;
    const NZ: usize = 4;
    const NZ_C: usize = NZ / 2 + 1;

    let fft = get_fft_for_grid(NX, NY, NZ);
    let real = Array3::from_shape_fn([NX, NY, NZ], |[i, j, k]| {
        (i * 37 + j * 11 + k * 5) as f64 / 17.0 - 2.0
    });
    let mut contiguous_half = Array3::zeros([NX, NY, NZ_C]);
    fft.forward_r2c_into(&real, &mut contiguous_half);

    // Equal first and second extents let this transposed owned layout retain
    // the required shape while forcing the general strided fallback.
    let row_stride = isize::try_from(NZ_C).expect("test dimensions fit isize");
    let plane_stride = isize::try_from(NY * NZ_C).expect("test dimensions fit isize");
    let strided_layout = Layout::try_new([NX, NY, NZ_C], [row_stride, plane_stride, 1], 0)
        .expect("invariant: test strided layout is valid");
    let mut strided_half = Array3::new(
        strided_layout,
        VecStorage::fill(NX * NY * NZ_C, Complex64::default()),
    )
    .expect("invariant: transposed test layout fits its owned storage");
    assert!(strided_half.as_slice().is_none());

    fft.forward_r2c_into(&real, &mut strided_half);
    for (actual, expected) in strided_half.iter().zip(contiguous_half.iter()) {
        assert_eq!(actual, expected);
    }

    let mut contiguous_real = Array3::zeros([NX, NY, NZ]);
    let mut contiguous_scratch = Array3::zeros([NX, NY, NZ_C]);
    fft.inverse_c2r_into(
        &contiguous_half,
        &mut contiguous_real,
        &mut contiguous_scratch,
    );
    let mut strided_real = Array3::zeros([NX, NY, NZ]);
    let mut strided_scratch = Array3::zeros([NX, NY, NZ_C]);
    fft.inverse_c2r_into(&strided_half, &mut strided_real, &mut strided_scratch);
    for (actual, expected) in strided_real.iter().zip(contiguous_real.iter()) {
        assert_eq!(actual, expected);
    }
}
