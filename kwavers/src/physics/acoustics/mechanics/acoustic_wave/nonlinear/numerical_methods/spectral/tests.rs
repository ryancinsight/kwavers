use super::super::super::wave_model::NonlinearWave;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use ndarray::Array3;
use std::f64::consts::PI;

/// A spatially uniform field has zero spectral gradient in every direction.
///
/// Physics: DC-only spectrum; i·kx·F(k) = 0 for all non-zero k.
/// Tolerance: N·ε_mach·10 = 512·2.2e-16·10 ≈ 1.1e-12.
#[test]
fn compute_spectral_gradient_zero_for_constant_field() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let w = NonlinearWave::new(&grid, 1e-7);
    let field = Array3::<f64>::from_elem((8, 8, 8), 42.0);

    let (gx, gy, gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

    let tol = 512.0 * f64::EPSILON * 10.0;
    for &v in gx.iter().chain(gy.iter()).chain(gz.iter()) {
        assert!(
            v.abs() < tol,
            "gradient of constant field must be zero (got {v:.3e})"
        );
    }
}

/// A spatially uniform field has zero spectral Laplacian.
///
/// Physics: −k²·F(k) = 0 for the DC bin (k=0); all other modes are zero.
#[test]
fn compute_spectral_laplacian_zero_for_constant_field() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let mut w = NonlinearWave::new(&grid, 1e-7);
    w.precompute_k_squared(&grid);
    let field = Array3::<f64>::from_elem((8, 8, 8), 100.0);

    let lap = w.compute_spectral_laplacian(&field, &grid).unwrap();

    let tol = 512.0 * f64::EPSILON * 10.0;
    for &v in lap.iter() {
        assert!(
            v.abs() < tol,
            "Laplacian of constant field must be zero (got {v:.3e})"
        );
    }
}

/// For f[i,j,k] = sin(2π·i/N), the x-gradient is (2π/(N·dx))·cos(2π·i/N).
///
/// Mathematical proof:
///   F(k₁,0,0) = -(iN/2), F(-k₁,0,0) = (iN/2)  (DFT of sin)
///   df/dx |_spectral = IFFT(i·kx·F) = (2π/(N·dx))·cos(2πi/N)
///   where kx[1] = 2π/(N·dx).
#[test]
fn compute_spectral_gradient_x_analytical_for_single_mode_sinusoid() {
    let n = 8_usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let w = NonlinearWave::new(&grid, 1e-7);

    let mut field = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let v = (2.0 * PI * i as f64 / n as f64).sin();
        for j in 0..n {
            for k in 0..n {
                field[[i, j, k]] = v;
            }
        }
    }

    let (grad_x, _gy, _gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

    let k1x = 2.0 * PI / (n as f64 * dx);
    let tol = 1e-9 * k1x;
    for i in 0..n {
        let expected = k1x * (2.0 * PI * i as f64 / n as f64).cos();
        let got = grad_x[[i, 0, 0]];
        assert!(
            (got - expected).abs() < tol,
            "grad_x at i={i}: got {got:.6e} expected {expected:.6e} (tol {tol:.3e})"
        );
    }
}

/// For f[i,j,k] = sin(2π·i/N), the spectral Laplacian equals −k1x²·f.
///
/// Mathematical proof:
///   ∇²f|_spectral = IFFT(−k²·F) = −k1x²·sin(2π·i/n)
///   where k1x = 2π/(N·dx).
#[test]
fn compute_spectral_laplacian_negative_definite_for_single_mode_sinusoid() {
    let n = 8_usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let mut w = NonlinearWave::new(&grid, 1e-7);
    w.precompute_k_squared(&grid);

    let mut field = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let v = (2.0 * PI * i as f64 / n as f64).sin();
        for j in 0..n {
            for k in 0..n {
                field[[i, j, k]] = v;
            }
        }
    }

    let lap = w.compute_spectral_laplacian(&field, &grid).unwrap();

    let k1x = 2.0 * PI / (n as f64 * dx);
    let factor = -(k1x * k1x);
    let tol = 1e-9 * k1x * k1x;
    for i in 0..n {
        let expected = factor * field[[i, 0, 0]];
        let got = lap[[i, 0, 0]];
        assert!(
            (got - expected).abs() < tol,
            "Laplacian at i={i}: got {got:.6e} expected {expected:.6e}"
        );
    }
}

/// y-gradient of a y-only sinusoid satisfies the same analytical formula.
#[test]
fn compute_spectral_gradient_y_analytical_for_single_mode_sinusoid() {
    let n = 8_usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let w = NonlinearWave::new(&grid, 1e-7);

    let mut field = Array3::<f64>::zeros((n, n, n));
    for j in 0..n {
        let v = (2.0 * PI * j as f64 / n as f64).sin();
        for i in 0..n {
            for k in 0..n {
                field[[i, j, k]] = v;
            }
        }
    }

    let (_gx, grad_y, _gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

    let k1 = 2.0 * PI / (n as f64 * dx);
    let tol = 1e-9 * k1;
    for j in 0..n {
        let expected = k1 * (2.0 * PI * j as f64 / n as f64).cos();
        let got = grad_y[[0, j, 0]];
        assert!(
            (got - expected).abs() < tol,
            "grad_y at j={j}: got {got:.6e} expected {expected:.6e}"
        );
    }
}

/// `apply_k_space_correction` on a zero-pressure field returns a zero field.
///
/// Trivial null case: FFT(0) = 0; any linear operator on 0 = 0; IFFT(0) = 0.
#[test]
fn apply_k_space_correction_zero_field_returns_zero() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let mut w = NonlinearWave::new(&grid, 1e-7);
    w.precompute_k_squared(&grid);

    let pressure = Array3::<f64>::zeros((8, 8, 8));
    let corrected = w
        .apply_k_space_correction(&pressure, &medium, &grid)
        .unwrap();

    let tol = 512.0 * f64::EPSILON * 10.0;
    for &v in corrected.iter() {
        assert!(
            v.abs() < tol,
            "k-space correction of zero field must be zero (got {v:.3e})"
        );
    }
}
