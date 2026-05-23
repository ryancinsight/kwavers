//! Integration tests for the field-surrogate cache. Tests use small
//! synthetic kernels (no real PSTD runs needed) to exercise resampling,
//! placement, and `(f0, pnp)` blending invariants.

use ndarray::Array3;

use crate::core::constants::numerical::MHZ_TO_HZ;
use super::{
    helmholtz_residual_field, helmholtz_residual_kernel, helmholtz_residual_stats,
    place_kernel_at_focus, resample_trilinear, FocalKernel, KernelCube, HELMHOLTZ_C0_WATER,
};

/// Build a synthetic Gaussian focal kernel for testing.
fn synthetic_gaussian_kernel(
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    f0: f64,
    pnp: f64,
) -> FocalKernel {
    let focus = (nx / 2, ny / 2, nz / 2);
    let mut field = Array3::<f64>::zeros((nx, ny, nz));
    // Gaussian half-widths set to a few voxels so the peak is clearly
    // localised and resampling effects are testable.
    let sx = 3.0; // axial half-width in voxels
    let sy = 1.5; // lateral
    let sz = 1.5;
    for ((i, j, k), v) in field.indexed_iter_mut() {
        let di = (i as f64) - (focus.0 as f64);
        let dj = (j as f64) - (focus.1 as f64);
        let dk = (k as f64) - (focus.2 as f64);
        let r2 = (di / sx).powi(2) + (dj / sy).powi(2) + (dk / sz).powi(2);
        *v = pnp * (-0.5 * r2).exp();
    }
    FocalKernel::new(field, dx_m, focus, f0, pnp, 1.0e6, 2.0e-3, 10.0e-3)
}

#[test]
fn test_focal_pressure_matches_pnp() {
    let k = synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, 1.0e6, 30.0e6);
    assert!((k.focal_pressure() - 30.0e6).abs() < 1e-9);
    assert_eq!(k.shape(), (40, 30, 30));
}

#[test]
fn test_scale_in_place_is_linear() {
    let mut k = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 0.5e6, 15.0e6);
    let original_focal = k.focal_pressure();
    k.scale_in_place(0.6);
    assert!((k.focal_pressure() - original_focal * 0.6).abs() < 1e-9);
    assert!((k.pnp_realised - 15.0e6 * 0.6).abs() < 1e-9);
}

#[test]
fn test_resample_identity_when_dx_unchanged() {
    let k = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 0.5e6, 15.0e6);
    let r = resample_trilinear(&k, 1.0e-3);
    assert_eq!(r.shape(), k.shape());
    let max_abs_diff = k
        .field
        .iter()
        .zip(r.field.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs_diff < 1e-9, "identity resample should be exact");
}

#[test]
fn test_resample_changes_shape_proportionally() {
    let k = synthetic_gaussian_kernel(40, 20, 20, 0.5e-3, 1.0e6, 30.0e6);
    let r = resample_trilinear(&k, 1.0e-3); // 2× downsample
    assert_eq!(r.shape(), (20, 10, 10));
    assert!((r.dx_m - 1.0e-3).abs() < 1e-12);
}

#[test]
fn test_resample_preserves_focal_position() {
    let k = synthetic_gaussian_kernel(40, 20, 20, 0.5e-3, 1.0e6, 30.0e6);
    let r = resample_trilinear(&k, 1.0e-3);
    // input focus at (20, 10, 10) at 0.5mm dx -> physical (10mm, 5mm, 5mm)
    // output focus at 1.0mm dx should be (10, 5, 5)
    assert_eq!(r.focus_idx, (10, 5, 5));
}

#[test]
fn test_place_kernel_at_focus_aligns_voxel() {
    let k = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 1.0e6, 30.0e6);
    let target_shape = (60, 40, 40);
    let target_focus = (45, 20, 20);
    let placed = place_kernel_at_focus(&k, target_shape, target_focus);
    assert_eq!(placed.dim(), target_shape);
    let placed_at_focus = placed[target_focus];
    let kernel_focal = k.field[k.focus_idx];
    assert!(
        (placed_at_focus - kernel_focal).abs() < 1e-9,
        "kernel focal voxel must land at target_focus"
    );
}

#[test]
fn test_place_kernel_zero_fills_outside_footprint() {
    let k = synthetic_gaussian_kernel(10, 10, 10, 1.0e-3, 1.0e6, 30.0e6);
    let target_shape = (50, 30, 30);
    let target_focus = (25, 15, 15);
    let placed = place_kernel_at_focus(&k, target_shape, target_focus);
    // Far corner should be zero (kernel extent only ±5 voxels from focus)
    assert_eq!(placed[[0, 0, 0]], 0.0);
    assert_eq!(placed[[49, 29, 29]], 0.0);
}

#[test]
fn test_cube_construction_validates_cartesian_completeness() {
    let k1 = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 0.5e6, 15.0e6);
    let k2 = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 0.5e6, 30.0e6);
    let k3 = synthetic_gaussian_kernel(40, 20, 20, 0.5e-3, 1.0e6, 15.0e6);
    let k4 = synthetic_gaussian_kernel(40, 20, 20, 0.5e-3, 1.0e6, 30.0e6);
    let cube = KernelCube::new(vec![k1, k2, k3, k4]).expect("complete cube");
    assert_eq!(cube.len(), 4);
    assert_eq!(cube.f0_axis().len(), 2);
    assert_eq!(cube.pnp_axis().len(), 2);
}

#[test]
fn test_cube_construction_rejects_missing_corner() {
    let k1 = synthetic_gaussian_kernel(20, 20, 20, 1.0e-3, 0.5e6, 15.0e6);
    let k4 = synthetic_gaussian_kernel(40, 20, 20, 0.5e-3, 1.0e6, 30.0e6);
    let result = KernelCube::new(vec![k1, k4]);
    assert!(
        result.is_err(),
        "missing-corner cube must fail construction"
    );
}

#[test]
fn test_cube_query_corner_returns_normalized_envelope() {
    let kernels: Vec<FocalKernel> = [
        (0.5e6, 15.0e6),
        (0.5e6, 30.0e6),
        (1.0e6, 15.0e6),
        (1.0e6, 30.0e6),
    ]
    .iter()
    .map(|&(f0, pnp)| synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, f0, pnp))
    .collect();
    let cube = KernelCube::new(kernels).unwrap();
    let env = cube.query(1.0e6, 30.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    let max = env.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (max - 1.0).abs() < 1e-9,
        "corner query must produce env.max() == 1, got {max}"
    );
}

#[test]
fn test_cube_query_clamps_outside_sweep() {
    let kernels: Vec<FocalKernel> = [
        (0.5e6, 15.0e6),
        (0.5e6, 30.0e6),
        (1.0e6, 15.0e6),
        (1.0e6, 30.0e6),
    ]
    .iter()
    .map(|&(f0, pnp)| synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, f0, pnp))
    .collect();
    let cube = KernelCube::new(kernels).unwrap();
    let target_shape = (60, 40, 40);
    let target_focus = (45, 20, 20);
    let env_below = cube.query(0.1e6, 20.0e6, target_shape, target_focus, 0.5e-3);
    let env_at_lo = cube.query(0.5e6, 20.0e6, target_shape, target_focus, 0.5e-3);
    let max_diff = env_below
        .iter()
        .zip(env_at_lo.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-9,
        "below-sweep query must clamp to lowest corner, max_diff = {max_diff}"
    );
}

#[test]
fn test_cube_query_pnp_amplitude_invariant() {
    let kernels: Vec<FocalKernel> = [
        (0.5e6, 15.0e6),
        (0.5e6, 30.0e6),
        (1.0e6, 15.0e6),
        (1.0e6, 30.0e6),
    ]
    .iter()
    .map(|&(f0, pnp)| synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, f0, pnp))
    .collect();
    let cube = KernelCube::new(kernels).unwrap();
    let env_15 = cube.query(1.0e6, 15.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    let env_30 = cube.query(1.0e6, 30.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    let max_diff = env_15
        .iter()
        .zip(env_30.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-9,
        "pnp must not change envelope shape (linear water)"
    );
}

#[test]
fn test_resample_large_kernel_completes() {
    // Performance smoke: a 200×100×100 input → 167×83×83 output runs
    // in well under a second on the parallel resample. If this ever
    // regresses to single-thread it'll show as a multi-second test.
    let nx = 200usize;
    let ny = 100usize;
    let nz = 100usize;
    let dx_in = 0.5e-3;
    let dx_out = 0.6e-3;
    let k = synthetic_gaussian_kernel(nx, ny, nz, dx_in, 1.0e6, 30.0e6);
    let t0 = std::time::Instant::now();
    let r = resample_trilinear(&k, dx_out);
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_millis() < 1500,
        "resample took {}ms — parallelisation likely broken",
        elapsed.as_millis()
    );
    // Field-integral conservation across resample (trilinear approx)
    let int_in: f64 = k.field.iter().sum::<f64>() * dx_in.powi(3);
    let int_out: f64 = r.field.iter().sum::<f64>() * dx_out.powi(3);
    assert!(
        (int_in - int_out).abs() / int_in < 0.05,
        "integral changed by {:.1}% during resample",
        (int_in - int_out).abs() / int_in * 100.0
    );
}

#[test]
fn test_cube_blend_in_place_zero_extra_allocation() {
    // Functional test that the in-place blend produces the same result
    // as the previous out-of-place implementation. Exercises the new
    // Zip-based blend path inside KernelCube::query.
    let kernels: Vec<FocalKernel> = [
        (0.5e6, 15.0e6),
        (0.5e6, 30.0e6),
        (1.0e6, 15.0e6),
        (1.0e6, 30.0e6),
    ]
    .iter()
    .map(|&(f0, pnp)| synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, f0, pnp))
    .collect();
    let cube = KernelCube::new(kernels).unwrap();
    let env_a = cube.query(0.6e6, 20.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    let env_b = cube.query(0.6e6, 20.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    assert_eq!(env_a, env_b, "deterministic");
    let max_a = env_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((max_a - 1.0).abs() < 1e-9);
}

#[test]
fn test_helmholtz_residual_zero_on_plane_wave() {
    // A discrete plane wave `p[i] = cos(k * x[i])` satisfies the
    // Helmholtz equation exactly in the continuum limit. With finite
    // differences the residual scales as O(k² · dx² · |p|) and must
    // be small relative to k²·|p_max|.
    let f0 = MHZ_TO_HZ;
    let c0 = HELMHOLTZ_C0_WATER;
    let k = 2.0 * std::f64::consts::PI * f0 / c0; // ~4189 m^-1
    let lam = 2.0 * std::f64::consts::PI / k; // ~1.5 mm
    let dx = lam / 16.0; // 16 PPW for low FD error
    let n = 32usize;
    let mut p = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let x = (i as f64) * dx;
        let val = (k * x).cos();
        for j in 0..n {
            for kk in 0..n {
                p[[i, j, kk]] = val;
            }
        }
    }
    let r = helmholtz_residual_field(&p, dx, f0, c0);
    let stats = helmholtz_residual_stats(&r, &p, f0, c0);
    // 16 PPW central differences should give residual_ratio ≪ 1 %.
    assert!(
        stats.normalised_ratio < 0.01,
        "plane-wave residual ratio {} too large (FD error)",
        stats.normalised_ratio
    );
}

#[test]
fn test_helmholtz_residual_nonzero_on_constant_field() {
    // For a uniform field p ≡ c, ∇²p = 0 and R = k²c ≠ 0. Confirms
    // the residual formulation actually includes the k² term and
    // isn't accidentally returning the bare Laplacian.
    let f0 = MHZ_TO_HZ;
    let c0 = HELMHOLTZ_C0_WATER;
    let dx = 1.0e-4;
    let n = 8usize;
    let p = Array3::<f64>::from_elem((n, n, n), 5.0e6);
    let r = helmholtz_residual_field(&p, dx, f0, c0);
    let k = 2.0 * std::f64::consts::PI * f0 / c0;
    let expected = k * k * 5.0e6;
    let interior_value = r[[n / 2, n / 2, n / 2]];
    assert!(
        (interior_value - expected).abs() / expected < 1e-6,
        "expected {expected:.3e}, got {interior_value:.3e}"
    );
}

#[test]
fn test_helmholtz_residual_boundary_shell_zero() {
    let f0 = MHZ_TO_HZ;
    let c0 = HELMHOLTZ_C0_WATER;
    let dx = 1.0e-4;
    let n = 8usize;
    let p = Array3::<f64>::from_elem((n, n, n), 5.0e6);
    let r = helmholtz_residual_field(&p, dx, f0, c0);
    // Every face of the cube should be zero (1-cell shell).
    for j in 0..n {
        for kk in 0..n {
            assert_eq!(r[[0, j, kk]], 0.0);
            assert_eq!(r[[n - 1, j, kk]], 0.0);
        }
    }
    for i in 0..n {
        for kk in 0..n {
            assert_eq!(r[[i, 0, kk]], 0.0);
            assert_eq!(r[[i, n - 1, kk]], 0.0);
        }
    }
}

#[test]
fn test_helmholtz_residual_kernel_wrapper_matches_field() {
    let n = 16usize;
    let dx = 1.0e-3;
    let f0 = MHZ_TO_HZ;
    let kernel = synthetic_gaussian_kernel(n, n, n, dx, f0, 30.0e6);
    let r_kernel = helmholtz_residual_kernel(&kernel, HELMHOLTZ_C0_WATER);
    let r_field = helmholtz_residual_field(&kernel.field, dx, f0, HELMHOLTZ_C0_WATER);
    let max_diff = r_kernel
        .iter()
        .zip(r_field.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(max_diff, 0.0);
}

#[test]
fn test_cube_query_midpoint_blends_and_renormalizes() {
    let kernels: Vec<FocalKernel> = [
        (0.5e6, 15.0e6),
        (0.5e6, 30.0e6),
        (1.0e6, 15.0e6),
        (1.0e6, 30.0e6),
    ]
    .iter()
    .map(|&(f0, pnp)| synthetic_gaussian_kernel(40, 30, 30, 0.5e-3, f0, pnp))
    .collect();
    let cube = KernelCube::new(kernels).unwrap();
    let env_mid = cube.query(0.75e6, 20.0e6, (60, 40, 40), (45, 20, 20), 0.5e-3);
    let max = env_mid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (max - 1.0).abs() < 1e-9,
        "midpoint query must re-normalize, got max = {max}"
    );
    let min = env_mid.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min >= -1e-9, "envelope must remain non-negative");
}
