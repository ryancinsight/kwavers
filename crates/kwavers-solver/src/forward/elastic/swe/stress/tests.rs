use super::super::{scratch::ElasticStepScratch, types::ElasticWaveField};
use super::*;
use kwavers_grid::Grid;
use leto::Array3;

fn from_shape_fn_fortran<F>(shape: [usize; 3], mut f: F) -> Array3<f64>
where
    F: FnMut([usize; 3]) -> f64,
{
    let layout = leto::Layout::f_contiguous(shape).expect("f-contiguous layout");
    let [d0, d1, d2] = shape;
    let mut data = vec![0.0; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                data[i + j * d0 + k * d0 * d1] = f([i, j, k]);
            }
        }
    }
    leto::Array::new(layout, leto::VecStorage::new(data)).expect("valid f-contiguous array")
}

/// fd1_x on a linear field f = A*x gives the constant A at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_x_linear_field() {
    let nx = 10;
    let dx = 0.001;
    let a = 3.0_f64;
    let mut f = Array3::<f64>::zeros((nx, 1, 1));
    for i in 0..nx {
        f[[i, 0, 0]] = a * (i as f64) * dx;
    }
    // Interior points: derivative should equal A.
    for i in 2..nx - 2 {
        let d = fd1_x(f.view(), i, 0, 0, nx, dx);
        assert!(
            (d - a).abs() < 1e-10,
            "fd1_x at i={i}: got {d}, expected {a}"
        );
    }
}

/// fd1_y on a linear field f = B*y gives the constant B at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_y_linear_field() {
    let ny = 10;
    let dy = 0.001;
    let b = -2.5_f64;
    let mut f = Array3::<f64>::zeros((1, ny, 1));
    for j in 0..ny {
        f[[0, j, 0]] = b * (j as f64) * dy;
    }
    for j in 2..ny - 2 {
        let d = fd1_y(f.view(), 0, j, 0, ny, dy);
        assert!(
            (d - b).abs() < 1e-10,
            "fd1_y at j={j}: got {d}, expected {b}"
        );
    }
}

/// fd1_z on a linear field f = C*z gives the constant C at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_z_linear_field() {
    let nz = 10;
    let dz = 0.001;
    let c = 1.7_f64;
    let mut f = Array3::<f64>::zeros((1, 1, nz));
    for k in 0..nz {
        f[[0, 0, k]] = c * (k as f64) * dz;
    }
    for k in 2..nz - 2 {
        let d = fd1_z(f.view(), 0, 0, k, nz, dz);
        assert!(
            (d - c).abs() < 1e-10,
            "fd1_z at k={k}: got {d}, expected {c}"
        );
    }
}

/// Degenerate axis (size=1) must return 0.0 without panic.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_degenerate_axes() {
    let f = Array3::<f64>::ones((1, 1, 1));
    assert_eq!(fd1_x(f.view(), 0, 0, 0, 1, 0.001), 0.0);
    assert_eq!(fd1_y(f.view(), 0, 0, 0, 1, 0.001), 0.0);
    assert_eq!(fd1_z(f.view(), 0, 0, 0, 1, 0.001), 0.0);
}

/// Uniform displacement ux = A (constant) → zero stress divergence.
///
/// ## Numerical note
///
/// Displacement values must be exactly representable in f64 (binary fractions such as
/// 0.5 = 2⁻¹, 0.25 = 2⁻², 0.125 = 2⁻³).  Non-binary values (e.g. 0.3, 0.1) are not
/// representable exactly; the 4th-order interior stencil and the 1st/2nd-order boundary
/// stencils then produce *different* ULP-level rounding errors, so the stress arrays
/// are not numerically constant across j/k even though they should be physically zero.
/// The resulting non-constant stress produces a spurious FD-of-stress value ≈ 0.024
/// that far exceeds any physically meaningful tolerance.  Exact binary fractions cancel
/// identically in all stencil variants, giving div = 0.0 to floating-point precision.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_uniform_displacement() {
    let n = 10;
    let dx = 0.001;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let lambda = Array3::from_elem([n, n, n], 1e9_f64);
    let mu = Array3::from_elem([n, n, n], 5e8_f64);
    let mut field = ElasticWaveField::new(n, n, n);
    // Use exact binary fractions: 0.5=2⁻¹, 0.25=2⁻², 0.125=2⁻³.
    // Non-binary values produce stencil-dependent ULP rounding that makes
    // stress spatially non-uniform at the 1e-5 level when multiplied by λ,μ∼1e9.
    field.ux.fill(0.5);
    field.uy.fill(0.25);
    field.uz.fill(0.125);
    let (dx_arr, dy_arr, dz_arr) = stress_divergence(&grid, &lambda, &mu, &field);
    for k in 2..n - 2 {
        for j in 2..n - 2 {
            for i in 2..n - 2 {
                // With exact binary fractions, all FD stencils cancel exactly → 0.0.
                // The tolerance 1e-10 guards against any unexpected ULP drift.
                assert!(
                    dx_arr[[i, j, k]].abs() < 1e-10,
                    "div_x at ({i},{j},{k}) = {}",
                    dx_arr[[i, j, k]]
                );
                assert!(
                    dy_arr[[i, j, k]].abs() < 1e-10,
                    "div_y at ({i},{j},{k}) = {}",
                    dy_arr[[i, j, k]]
                );
                assert!(
                    dz_arr[[i, j, k]].abs() < 1e-10,
                    "div_z at ({i},{j},{k}) = {}",
                    dz_arr[[i, j, k]]
                );
            }
        }
    }
}

/// Linear ux = A·x in a homogeneous medium with μ=0 (fluid).
///
/// εxx = A (constant) → σxx = (λ+2μ)·A (constant) → ∂σxx/∂x = 0.
/// Acceleration must be zero everywhere in the interior.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_linear_ux_fluid() {
    let n = 12;
    let dx = 1e-3;
    let grid = Grid::new(n, 1, 1, dx, dx, dx).unwrap();
    let la_val = 2.25e9_f64; // water-like λ
    let lambda = Array3::from_elem([n, 1, 1], la_val);
    let mu = Array3::zeros((n, 1, 1)); // fluid: μ=0
    let mut field = ElasticWaveField::new(n, 1, 1);
    // Linear displacement ux = A·x → constant strain → constant σxx → zero divergence
    let a = 0.01_f64;
    for i in 0..n {
        field.ux[[i, 0, 0]] = a * (i as f64) * dx;
    }
    let (div_x, div_y, div_z) = stress_divergence(&grid, &lambda, &mu, &field);
    for i in 2..n - 2 {
        assert!(
            div_x[[i, 0, 0]].abs() < 1e-3,
            "div_x at i={i} = {}",
            div_x[[i, 0, 0]]
        );
        assert_eq!(div_y[[i, 0, 0]], 0.0);
        assert_eq!(div_z[[i, 0, 0]], 0.0);
    }
}

/// Quadratic ux = A·x² in a homogeneous fluid.
///
/// εxx = 2A·x → σxx = (λ)·2A·x → ∂σxx/∂x = 2λA (constant).
/// The interior acceleration a_x = 2λA/ρ must match within FD error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_quadratic_ux_fluid() {
    let n = 12;
    let dx = 1e-3;
    let grid = Grid::new(n, 1, 1, dx, dx, dx).unwrap();
    let la_val = 2.25e9_f64;
    let lambda = Array3::from_elem([n, 1, 1], la_val);
    let mu = Array3::zeros((n, 1, 1));
    let mut field = ElasticWaveField::new(n, 1, 1);
    let a = 10.0_f64;
    for i in 0..n {
        let x = (i as f64) * dx;
        field.ux[[i, 0, 0]] = a * x * x;
    }
    let expected = 2.0 * la_val * a; // ∂σxx/∂x = 2λA
    let (div_x, _, _) = stress_divergence(&grid, &lambda, &mu, &field);
    for i in 3..n - 3 {
        let got = div_x[[i, 0, 0]];
        let rel_err = (got - expected).abs() / expected.abs();
        assert!(
            rel_err < 1e-4,
            "div_x at i={i}: got {got:.6e}, expected {expected:.6e}, rel_err={rel_err:.2e}"
        );
    }
}

#[test]
fn plane_strain_divergence_matches_spatial_operator_exactly() {
    let (nx, ny) = (11, 9);
    let grid = Grid::new(nx, ny, 1, 0.7e-3, 1.3e-3, 2.0e-3).expect("grid");
    let lambda = Array3::from_shape_fn((nx, ny, 1), |[i, j, _]| 2.0e6 + (i * 37 + j * 11) as f64);
    let mu = Array3::from_shape_fn((nx, ny, 1), |[i, j, _]| 0.8e6 + (i * 17 + j * 29) as f64);
    let mut field = ElasticWaveField::new(nx, ny, 1);
    field.ux = Array3::from_shape_fn((nx, ny, 1), |[i, j, _]| {
        ((i * 13 + j * 7) as f64 * 0.037).sin()
    });
    field.uy = Array3::from_shape_fn((nx, ny, 1), |[i, j, _]| {
        ((i * 5 + j * 19) as f64 * 0.041).cos()
    });
    let mut spatial = ElasticStepScratch::new(nx, ny, 1);
    let mut plane = ElasticStepScratch::new(nx, ny, 1);

    stress_divergence_into(&grid, &lambda, &mu, &field, &mut spatial);
    stress_divergence_plane_strain_into(&grid, &lambda, &mu, &field, &mut plane);

    assert_eq!(plane.div_x, spatial.div_x);
    assert_eq!(plane.div_y, spatial.div_y);
    assert_eq!(plane.div_z, spatial.div_z);
}

fn sequential_stress_divergence(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut sxx = Array3::zeros((nx, ny, nz));
    let mut sxy = Array3::zeros((nx, ny, nz));
    let mut sxz = Array3::zeros((nx, ny, nz));
    let mut syy = Array3::zeros((nx, ny, nz));
    let mut syz = Array3::zeros((nx, ny, nz));
    let mut szz = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let exx = fd1_x(field.ux.view(), i, j, k, nx, grid.dx);
                let eyy = fd1_y(field.uy.view(), i, j, k, ny, grid.dy);
                let ezz = fd1_z(field.uz.view(), i, j, k, nz, grid.dz);
                let exy_2 = fd1_y(field.ux.view(), i, j, k, ny, grid.dy)
                    + fd1_x(field.uy.view(), i, j, k, nx, grid.dx);
                let exz_2 = fd1_z(field.ux.view(), i, j, k, nz, grid.dz)
                    + fd1_x(field.uz.view(), i, j, k, nx, grid.dx);
                let eyz_2 = fd1_z(field.uy.view(), i, j, k, nz, grid.dz)
                    + fd1_y(field.uz.view(), i, j, k, ny, grid.dy);
                let la = lambda[[i, j, k]];
                let mv = mu[[i, j, k]];
                let la2mu = 2.0f64.mul_add(mv, la);
                sxx[[i, j, k]] = la2mu.mul_add(exx, la * (eyy + ezz));
                syy[[i, j, k]] = la2mu.mul_add(eyy, la * (exx + ezz));
                szz[[i, j, k]] = la2mu.mul_add(ezz, la * (exx + eyy));
                sxy[[i, j, k]] = mv * exy_2;
                sxz[[i, j, k]] = mv * exz_2;
                syz[[i, j, k]] = mv * eyz_2;
            }
        }
    }

    let mut div_x = Array3::zeros((nx, ny, nz));
    let mut div_y = Array3::zeros((nx, ny, nz));
    let mut div_z = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                div_x[[i, j, k]] = fd1_x(sxx.view(), i, j, k, nx, grid.dx)
                    + fd1_y(sxy.view(), i, j, k, ny, grid.dy)
                    + fd1_z(sxz.view(), i, j, k, nz, grid.dz);
                div_y[[i, j, k]] = fd1_x(sxy.view(), i, j, k, nx, grid.dx)
                    + fd1_y(syy.view(), i, j, k, ny, grid.dy)
                    + fd1_z(syz.view(), i, j, k, nz, grid.dz);
                div_z[[i, j, k]] = fd1_x(sxz.view(), i, j, k, nx, grid.dx)
                    + fd1_y(syz.view(), i, j, k, ny, grid.dy)
                    + fd1_z(szz.view(), i, j, k, nz, grid.dz);
            }
        }
    }
    (div_x, div_y, div_z)
}

#[test]
fn fused_stress_traversal_matches_sequential_reference_exactly() {
    let (nx, ny, nz) = (7, 6, 5);
    let grid = Grid::new(nx, ny, nz, 0.7e-3, 1.1e-3, 1.3e-3).expect("grid");
    let lambda = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        2.0e6 + (i * 37 + j * 11 + k * 5) as f64
    });
    let mu = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        0.8e6 + (i * 17 + j * 29 + k * 13) as f64
    });
    let mut field = ElasticWaveField::new(nx, ny, nz);
    field.ux = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 13 + j * 7 + k * 3) as f64 * 0.037).sin()
    });
    field.uy = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 5 + j * 19 + k * 11) as f64 * 0.041).cos()
    });
    field.uz = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 23 + j * 2 + k * 17) as f64 * 0.029).sin()
    });

    let expected = sequential_stress_divergence(&grid, &lambda, &mu, &field);
    let mut fused = ElasticStepScratch::new(nx, ny, nz);
    stress_divergence_into(&grid, &lambda, &mu, &field, &mut fused);

    assert_eq!(fused.div_x, expected.0);
    assert_eq!(fused.div_y, expected.1);
    assert_eq!(fused.div_z, expected.2);
}

#[test]
fn strided_stress_inputs_and_outputs_match_standard_layout_exactly() {
    let shape = [7, 6, 5];
    let [nx, ny, nz] = shape;
    let grid = Grid::new(nx, ny, nz, 0.7e-3, 1.1e-3, 1.3e-3).expect("grid");
    let lambda_value = |[i, j, k]: [usize; 3]| 2.0e6 + (i * 37 + j * 11 + k * 5) as f64;
    let mu_value = |[i, j, k]: [usize; 3]| 0.8e6 + (i * 17 + j * 29 + k * 13) as f64;
    let ux_value = |[i, j, k]: [usize; 3]| ((i * 13 + j * 7 + k * 3) as f64 * 0.037).sin();
    let uy_value = |[i, j, k]: [usize; 3]| ((i * 5 + j * 19 + k * 11) as f64 * 0.041).cos();
    let uz_value = |[i, j, k]: [usize; 3]| ((i * 23 + j * 2 + k * 17) as f64 * 0.029).sin();

    let lambda = Array3::from_shape_fn(shape, lambda_value);
    let mu = Array3::from_shape_fn(shape, mu_value);
    let mut field = ElasticWaveField::new(nx, ny, nz);
    field.ux = Array3::from_shape_fn(shape, ux_value);
    field.uy = Array3::from_shape_fn(shape, uy_value);
    field.uz = Array3::from_shape_fn(shape, uz_value);
    let mut expected = ElasticStepScratch::new(nx, ny, nz);
    stress_divergence_into(&grid, &lambda, &mu, &field, &mut expected);

    let strided_lambda = from_shape_fn_fortran(shape, lambda_value);
    let strided_mu = from_shape_fn_fortran(shape, mu_value);
    let mut strided_field = ElasticWaveField::new(nx, ny, nz);
    strided_field.ux = from_shape_fn_fortran(shape, ux_value);
    strided_field.uy = from_shape_fn_fortran(shape, uy_value);
    strided_field.uz = from_shape_fn_fortran(shape, uz_value);
    let mut actual = ElasticStepScratch::new(nx, ny, nz);
    actual.sxx = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.sxy = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.sxz = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.syy = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.syz = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.szz = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.div_x = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.div_y = from_shape_fn_fortran(shape, |_| f64::NAN);
    actual.div_z = from_shape_fn_fortran(shape, |_| f64::NAN);
    assert!(strided_field.ux.as_slice().is_none());
    assert!(actual.sxx.as_slice().is_none());

    stress_divergence_into(
        &grid,
        &strided_lambda,
        &strided_mu,
        &strided_field,
        &mut actual,
    );

    for (actual, expected) in [
        (&actual.sxx, &expected.sxx),
        (&actual.sxy, &expected.sxy),
        (&actual.sxz, &expected.sxz),
        (&actual.syy, &expected.syy),
        (&actual.syz, &expected.syz),
        (&actual.szz, &expected.szz),
        (&actual.div_x, &expected.div_x),
        (&actual.div_y, &expected.div_y),
        (&actual.div_z, &expected.div_z),
    ] {
        assert_eq!(actual, expected);
    }
}

#[test]
fn mismatched_equal_length_stress_shape_rejects_before_mutation() {
    let (nx, ny, nz) = (2, 3, 4);
    let grid = Grid::new(nx, ny, nz, 0.7e-3, 1.1e-3, 1.3e-3).expect("grid");
    let lambda = Array3::from_elem((nx, ny, nz), 2.0e6);
    let mu = Array3::from_elem((nx, ny, nz), 0.8e6);
    let field = ElasticWaveField::new(nx, ny, nz);
    let mut scratch = ElasticStepScratch::new(nx, ny, nz);
    scratch.sxx.fill(1.0);
    scratch.syy.fill(2.0);
    scratch.szz.fill(3.0);
    scratch.sxy = Array3::from_elem((4, 3, 2), 4.0);
    scratch.sxz.fill(5.0);
    scratch.syz.fill(6.0);
    scratch.div_x.fill(7.0);
    scratch.div_y.fill(8.0);
    scratch.div_z.fill(9.0);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stress_divergence_into(&grid, &lambda, &mu, &field, &mut scratch);
    }))
    .expect_err("an equal-length scratch shape mismatch must be rejected");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&'static str>().copied())
        .expect("shape rejection must use a string panic payload");
    assert_eq!(
        message,
        "invariant: scratch.sxy shape [4, 3, 2] must match grid shape [2, 3, 4]"
    );

    assert_eq!(scratch.sxx, Array3::from_elem((nx, ny, nz), 1.0));
    assert_eq!(scratch.syy, Array3::from_elem((nx, ny, nz), 2.0));
    assert_eq!(scratch.szz, Array3::from_elem((nx, ny, nz), 3.0));
    assert_eq!(scratch.sxy, Array3::from_elem((4, 3, 2), 4.0));
    assert_eq!(scratch.sxz, Array3::from_elem((nx, ny, nz), 5.0));
    assert_eq!(scratch.syz, Array3::from_elem((nx, ny, nz), 6.0));
    assert_eq!(scratch.div_x, Array3::from_elem((nx, ny, nz), 7.0));
    assert_eq!(scratch.div_y, Array3::from_elem((nx, ny, nz), 8.0));
    assert_eq!(scratch.div_z, Array3::from_elem((nx, ny, nz), 9.0));
}
