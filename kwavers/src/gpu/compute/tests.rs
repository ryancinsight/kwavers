use super::*;
use approx::assert_abs_diff_eq;

// ── FdtdGpuShaderDispatcher tests ─────────────────────────────────────────

/// `PressureParams` must be `Pod` + `Zeroable` (compile-time assertion).
/// This guarantees safe byte-level copy to the GPU uniform buffer.
///
/// ## Reference
/// bytemuck crate documentation: Pod requires repr(C) + no padding + no uninit bytes.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_pressure_params_pod_layout() {
    use std::mem;
    let p = PressureParams {
        nx: 16,
        ny: 8,
        nz: 4,
        coeff: 0.25,
    };
    let bytes = bytemuck::bytes_of(&p);
    assert_eq!(bytes.len(), 16);
    assert_eq!(mem::size_of::<PressureParams>(), 16);
}

/// The `fdtd_pressure.wgsl` shader source must declare the correct entry point
/// and all required bindings.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fdtd_wgsl_shader_content() {
    let src = include_str!("../shaders/fdtd_pressure.wgsl");
    assert!(
        src.contains("fn fdtd_pressure_update"),
        "fdtd_pressure.wgsl must declare entry point 'fdtd_pressure_update'"
    );
    assert!(
        src.contains("@group(0) @binding(0)"),
        "fdtd_pressure.wgsl must declare group(0) binding(0) for pressure_curr"
    );
    assert!(
        src.contains("@group(0) @binding(2)"),
        "fdtd_pressure.wgsl must declare group(0) binding(2) for pressure_new"
    );
    assert!(
        src.contains("@group(1) @binding(0)"),
        "fdtd_pressure.wgsl must declare group(1) binding(0) for PressureParams uniform"
    );
    assert!(
        src.contains("@workgroup_size(8, 8, 4)"),
        "fdtd_pressure.wgsl must use workgroup_size(8, 8, 4)"
    );
}

// ── FdtdGpuDispatcher tests (CPU fallback) ────────────────────────────────

/// Uniform pressure field: Laplacian = 0, so p_new = 2*p_curr - p_prev.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_gpu_uniform_field() {
    let (nx, ny, nz) = (8, 8, 8);
    let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
    let p_curr = ndarray::Array3::from_elem((nx, ny, nz), 1.0_f64);
    let p_prev = ndarray::Array3::from_elem((nx, ny, nz), 0.5_f64);
    let coeff = 0.25_f64;
    let p_new = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert_abs_diff_eq!(p_new[[i, j, k]], 1.5, epsilon = 1e-12);
            }
        }
    }
}

/// Linear ramp p(i,j,k) = i+j+k: Laplacian is zero (exact polynomial).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_gpu_linear_ramp_zero_laplacian() {
    let (nx, ny, nz) = (8, 8, 8);
    let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
    let mut p_curr = ndarray::Array3::zeros((nx, ny, nz));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                p_curr[[i, j, k]] = (i + j + k) as f64;
            }
        }
    }
    let p_prev = p_curr.clone();
    let coeff = 0.5;
    let p_new = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert_abs_diff_eq!(p_new[[i, j, k]], p_curr[[i, j, k]], epsilon = 1e-12);
            }
        }
    }
}

/// Dimension mismatch returns an error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_gpu_dimension_mismatch_error() {
    let mut disp = FdtdGpuDispatcher::new(8, 8, 8).unwrap();
    let p_wrong = ndarray::Array3::zeros((4, 4, 4));
    let p_curr = ndarray::Array3::zeros((8, 8, 8));
    let mut output = ndarray::Array3::zeros((8, 8, 8));
    assert!(disp
        .update_pressure_into(&p_wrong, &p_curr, 0.25, &mut output)
        .is_err());
}

/// Grid too small (< 3 in any axis) returns an error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_gpu_minimum_dimension_check() {
    assert!(FdtdGpuDispatcher::new(2, 8, 8).is_err());
    assert!(FdtdGpuDispatcher::new(8, 2, 8).is_err());
    assert!(FdtdGpuDispatcher::new(8, 8, 2).is_err());
    let _disp = FdtdGpuDispatcher::new(3, 3, 3).unwrap();
}

/// update_pressure_into and update_pressure return identical results.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_gpu_into_matches_alloc() {
    let (nx, ny, nz) = (6, 6, 6);
    let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
    let p_curr = ndarray::Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        ((i * 3 + j * 5 + k * 7) as f64) * 0.01
    });
    let p_prev = ndarray::Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        ((i * 2 + j * 4 + k * 6) as f64) * 0.005
    });
    let coeff = 0.3;

    let p_alloc = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
    let mut p_into = ndarray::Array3::from_elem((nx, ny, nz), 123.456_f64);
    disp.update_pressure_into(&p_curr, &p_prev, coeff, &mut p_into)
        .unwrap();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                assert_abs_diff_eq!(p_alloc[[i, j, k]], p_into[[i, j, k]], epsilon = 1e-15);
            }
        }
    }
}
