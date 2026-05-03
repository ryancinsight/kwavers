//! Tests for `ConvergentBornSolver`.

use super::solver::ConvergentBornSolver;
use crate::domain::grid::Grid;
use crate::solver::forward::helmholtz::BornConfig;

#[test]
fn test_convergent_born_creation() {
    let config = BornConfig::default();
    let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1).unwrap();

    let solver = ConvergentBornSolver::new(config, grid);
    assert_eq!(solver.config.max_iterations, 50);
    assert_eq!(solver.grid.nx, 32);
}

#[test]
fn test_inverse_fft_normalization() {
    use approx::assert_relative_eq;
    use ndarray::Array3;
    use num_complex::Complex64;

    let config = BornConfig {
        use_fft_green: true,
        ..Default::default()
    };
    let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();
    let solver = ConvergentBornSolver::new(config, grid);

    let mut input = Array3::<Complex64>::zeros((16, 16, 16));
    input[[8, 8, 8]] = Complex64::new(1.0, 0.0);

    let mut fft_output = Array3::<Complex64>::zeros((16, 16, 16));
    let mut ifft_output = Array3::<Complex64>::zeros((16, 16, 16));

    solver
        .forward_fft_3d(&input.view(), &mut fft_output)
        .unwrap();
    solver
        .inverse_fft_3d(&fft_output.view(), &mut ifft_output)
        .unwrap();

    for ((i, j, k), &val) in ifft_output.indexed_iter() {
        assert_relative_eq!(val.re, input[[i, j, k]].re, epsilon = 1e-10);
        assert_relative_eq!(val.im, input[[i, j, k]].im, epsilon = 1e-10);
    }
    assert_relative_eq!(ifft_output[[8, 8, 8]].re, 1.0, epsilon = 1e-10);
}
