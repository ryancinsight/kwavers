use super::shift_gpu::KspaceShiftGpu;
use approx::assert_abs_diff_eq;
use ndarray::Array3;
use std::f64::consts::PI;

/// Zero shift must be the identity: output equals input.
#[test]
fn test_zero_shift_is_identity() {
    let (nx, ny, nz) = (4, 4, 4);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j * 2 + k * 3) as f64);
    let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * 3 + j + k * 2) as f64);
    let kx = vec![0.0; nx];
    let ky = vec![0.0; ny];
    let kz = vec![0.0; nz];

    let (re_out, im_out) = gpu
        .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, [0.0, 0.0, 0.0])
        .unwrap();

    for idx in real_in.indexed_iter() {
        let (i, &v) = idx;
        assert_abs_diff_eq!(re_out[i], v, epsilon = 1e-13);
    }
    for idx in imag_in.indexed_iter() {
        let (i, &v) = idx;
        assert_abs_diff_eq!(im_out[i], v, epsilon = 1e-13);
    }
}

/// Phase shift by 2π must be the identity.
#[test]
fn test_full_cycle_shift_is_identity() {
    let (nx, ny, nz) = (4, 1, 1);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let kx: Vec<f64> = (0..nx).map(|i| i as f64).collect();
    let ky = vec![0.0];
    let kz = vec![0.0];
    let real_in = Array3::from_elem((nx, 1, 1), 1.0_f64);
    let imag_in = Array3::zeros((nx, 1, 1));
    let shift = [2.0 * PI, 0.0, 0.0];

    let (re_out, im_out) = gpu
        .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
        .unwrap();

    for i in 0..nx {
        assert_abs_diff_eq!(re_out[[i, 0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(im_out[[i, 0, 0]], 0.0, epsilon = 1e-12);
    }
}

/// Quarter-cycle (π/2) shift on real-only spectrum gives pure imaginary output.
///
/// exp(−i·(k·π/2)) for k = 1 rad/m, shift = π/2 m: phase = -π/2
#[test]
fn test_quarter_cycle_shift() {
    let (nx, ny, nz) = (1, 1, 1);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let kx = vec![1.0_f64];
    let ky = vec![0.0];
    let kz = vec![0.0];
    let real_in = Array3::from_elem((1, 1, 1), 2.0_f64);
    let imag_in = Array3::zeros((1, 1, 1));
    let shift = [PI / 2.0, 0.0, 0.0];

    let (re_out, im_out) = gpu
        .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
        .unwrap();

    // cos(-π/2) = 0, sin(-π/2) = -1 → Re' = 0, Im' = -2
    assert_abs_diff_eq!(re_out[[0, 0, 0]], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(im_out[[0, 0, 0]], -2.0, epsilon = 1e-12);
}

/// apply_shift and apply_shift_into produce identical output.
#[test]
fn test_into_matches_alloc() {
    let (nx, ny, nz) = (4, 3, 3);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j + k) as f64 * 0.1);
    let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * j + k) as f64 * 0.05);
    let kx: Vec<f64> = (0..nx).map(|i| i as f64 * 100.0).collect();
    let ky = vec![0.0; ny];
    let kz = vec![0.0; nz];
    let shift = [0.5e-3, 0.0, 0.0];

    let (re_a, im_a) = gpu
        .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
        .unwrap();
    let mut re_b = Array3::zeros((nx, ny, nz));
    let mut im_b = Array3::zeros((nx, ny, nz));
    gpu.apply_shift_into(
        &real_in, &imag_in, &kx, &ky, &kz, shift, &mut re_b, &mut im_b,
    )
    .unwrap();

    for idx in re_a.indexed_iter() {
        let (i, &v) = idx;
        assert_abs_diff_eq!(re_b[i], v, epsilon = 1e-15);
        assert_abs_diff_eq!(im_b[i], im_a[i], epsilon = 1e-15);
    }
}

/// Magnitude is preserved by a unitary phase rotation.
#[test]
fn test_magnitude_preserved() {
    let (nx, ny, nz) = (3, 3, 3);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j + k) as f64);
    let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * 2 + j + k) as f64);
    let kx: Vec<f64> = (0..nx).map(|i| i as f64 * 200.0).collect();
    let ky: Vec<f64> = (0..ny).map(|j| j as f64 * 150.0).collect();
    let kz: Vec<f64> = (0..nz).map(|k| k as f64 * 100.0).collect();
    let shift = [0.25e-3, 0.33e-3, 0.15e-3];

    let (re_out, im_out) = gpu
        .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
        .unwrap();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let mag_in = (real_in[[i, j, k]].powi(2) + imag_in[[i, j, k]].powi(2)).sqrt();
                let mag_out = (re_out[[i, j, k]].powi(2) + im_out[[i, j, k]].powi(2)).sqrt();
                assert_abs_diff_eq!(mag_out, mag_in, epsilon = 1e-10);
            }
        }
    }
}

/// Zero grid dimension returns an error.
#[test]
fn test_zero_dimension_error() {
    assert!(KspaceShiftGpu::new(0, 4, 4).is_err());
}

/// The `kspace_shift.wgsl` shader must declare the correct entry point and bindings.
#[test]
fn test_kspace_shift_wgsl_shader_content() {
    let src = include_str!("../shaders/kspace_shift.wgsl");
    assert!(
        src.contains("fn kspace_shift"),
        "kspace_shift.wgsl must declare entry point 'kspace_shift'"
    );
    assert!(
        src.contains("@group(0) @binding(0)"),
        "kspace_shift.wgsl must declare binding(0) for spec_real"
    );
    assert!(
        src.contains("@group(0) @binding(1)"),
        "kspace_shift.wgsl must declare binding(1) for spec_imag"
    );
    assert!(
        src.contains("@group(1) @binding(0)"),
        "kspace_shift.wgsl must declare group(1) binding(0) for ShiftParams uniform"
    );
    assert!(
        src.contains("@workgroup_size(8, 8, 4)"),
        "kspace_shift.wgsl must use workgroup_size(8, 8, 4)"
    );
}

/// `ShiftParams` pod layout: 6 fields × 4 bytes = 24 bytes, no padding.
#[test]
fn test_kspace_shift_params_pod_layout() {
    #[repr(C)]
    struct ShiftParams {
        nx: u32,
        ny: u32,
        nz: u32,
        sx: f32,
        sy: f32,
        sz: f32,
    }
    assert_eq!(std::mem::size_of::<ShiftParams>(), 24);
    assert_eq!(std::mem::size_of::<u32>(), 4);
    assert_eq!(std::mem::size_of::<f32>(), 4);
}

/// Wavenumber vector length mismatch returns an error.
#[test]
fn test_kv_length_mismatch_error() {
    let (nx, ny, nz) = (4, 4, 4);
    let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
    let real_in = Array3::zeros((nx, ny, nz));
    let imag_in = Array3::zeros((nx, ny, nz));
    let kx_bad = vec![0.0; nx + 1];
    let ky = vec![0.0; ny];
    let kz = vec![0.0; nz];
    assert!(gpu
        .apply_shift(&real_in, &imag_in, &kx_bad, &ky, &kz, [0.0; 3])
        .is_err());
}
