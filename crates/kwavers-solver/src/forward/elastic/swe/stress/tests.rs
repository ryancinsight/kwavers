use super::super::{scratch::ElasticStepScratch, types::ElasticWaveField};
use super::*;
use kwavers_grid::Grid;
use leto::Array3;

pub(super) fn from_shape_fn_fortran<F>(shape: [usize; 3], mut f: F) -> Array3<f64>
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
    for (nx, ny) in [(11, 9)]
        .into_iter()
        .chain((1..=8).flat_map(|nx| (1..=8).map(move |ny| (nx, ny))))
    {
        let grid = Grid::new(nx, ny, 1, 0.7e-3, 1.3e-3, 2.0e-3).expect("grid");
        let lambda =
            Array3::from_shape_fn((nx, ny, 1), |[i, j, _]| 2.0e6 + (i * 37 + j * 11) as f64);
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
        let expected = pointwise_stress_divergence(&grid, &lambda, &mu, &field);
        assert_stress_scratch_eq(&spatial, &expected);
        assert_eq!(plane.sxx, expected.sxx);
        assert_eq!(plane.syy, expected.syy);
        assert_eq!(plane.sxy, expected.sxy);
        assert_eq!(plane.div_z, Array3::zeros((nx, ny, 1)));
    }
}

/// The stress tensor and its divergence assembled point by point from
/// derivatives evaluated from the stencil table: the kernel must
/// reproduce (which derivative of which field feeds which component, and the
/// order terms are summed in).
fn pointwise_stress_divergence(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
) -> ElasticStepScratch {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    // Independent pointwise evaluation of the documented stencil table.
    // Match its arithmetic order so differences identify wiring or closure
    // defects, without a tolerance hiding cancellation at the second sweep.
    let d = |axis: usize, f: &Array3<f64>| {
        let h = [grid.dx, grid.dy, grid.dz][axis];
        let n = f.shape()[axis];
        Array3::from_shape_fn((nx, ny, nz), |p| {
            let c = p[axis];
            let at = |coordinate| {
                let mut q = p;
                q[axis] = coordinate;
                f[q]
            };
            if n == 1 {
                0.0
            } else if c == 0 {
                (at(1) - at(0)) * (1.0 / h)
            } else if c == n - 1 {
                (at(c) - at(c - 1)) * (1.0 / h)
            } else if c == 1 || c == n - 2 {
                (at(c + 1) - at(c - 1)) * (1.0 / (2.0 * h))
            } else {
                ((-8.0 * at(c - 1)) + (8.0 * at(c + 1)) + (-at(c + 2)) + at(c - 2))
                    * (1.0 / (12.0 * h))
            }
        })
    };
    let mut scratch = ElasticStepScratch::new(nx, ny, nz);

    let [exx, eyy, ezz] = [d(0, &field.ux), d(1, &field.uy), d(2, &field.uz)];
    let [ux_y, uy_x, ux_z, uz_x, uy_z, uz_y] = [
        d(1, &field.ux),
        d(0, &field.uy),
        d(2, &field.ux),
        d(0, &field.uz),
        d(2, &field.uy),
        d(1, &field.uz),
    ];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = [i, j, k];
                let (la, mv) = (lambda[p], mu[p]);
                let la2mu = 2.0f64.mul_add(mv, la);
                scratch.sxx[p] = la2mu.mul_add(exx[p], la * (eyy[p] + ezz[p]));
                scratch.syy[p] = la2mu.mul_add(eyy[p], la * (exx[p] + ezz[p]));
                scratch.szz[p] = la2mu.mul_add(ezz[p], la * (exx[p] + eyy[p]));
                scratch.sxy[p] = mv * (ux_y[p] + uy_x[p]);
                scratch.sxz[p] = mv * (ux_z[p] + uz_x[p]);
                scratch.syz[p] = mv * (uy_z[p] + uz_y[p]);
            }
        }
    }

    let terms = [
        [d(0, &scratch.sxx), d(1, &scratch.sxy), d(2, &scratch.sxz)],
        [d(0, &scratch.sxy), d(1, &scratch.syy), d(2, &scratch.syz)],
        [d(0, &scratch.sxz), d(1, &scratch.syz), d(2, &scratch.szz)],
    ];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = [i, j, k];
                scratch.div_x[p] = terms[0][0][p] + terms[0][1][p] + terms[0][2][p];
                scratch.div_y[p] = terms[1][0][p] + terms[1][1][p] + terms[1][2][p];
                scratch.div_z[p] = terms[2][0][p] + terms[2][1][p] + terms[2][2][p];
            }
        }
    }
    scratch
}

fn assert_stress_scratch_eq(actual: &ElasticStepScratch, expected: &ElasticStepScratch) {
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

/// Sizes below and past the parallel floors of the sweeps and traversals,
/// and with short and singleton axes that take only the wall closures.
#[test]
fn stress_divergence_matches_its_pointwise_assembly() {
    let shapes = [(7, 6, 5), (13, 10, 9), (32, 30, 28), (4, 3, 1), (2, 6, 3)]
        .into_iter()
        .chain((1..=8).flat_map(|n| [(n, 7, 6), (7, n, 6), (7, 6, n)]));
    for (nx, ny, nz) in shapes {
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

        let expected = pointwise_stress_divergence(&grid, &lambda, &mu, &field);
        let mut swept = ElasticStepScratch::new(nx, ny, nz);
        stress_divergence_into(&grid, &lambda, &mu, &field, &mut swept);
        assert_stress_scratch_eq(&swept, &expected);
    }
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

    assert_stress_scratch_eq(&actual, &expected);
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
