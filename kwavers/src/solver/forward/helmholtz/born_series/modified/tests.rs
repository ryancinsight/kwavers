use super::*;
use crate::domain::grid::Grid;
use crate::solver::forward::helmholtz::BornConfig;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_modified_born_creation() {
    let config = BornConfig::default();
    let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();

    let solver = ModifiedBornSolver::new(config, grid);
    assert_eq!(solver.config.max_iterations, 50);
    assert_eq!(solver.grid.nx, 16);
}

#[test]
fn test_parallel_green_consistency() {
    let config = BornConfig::default();
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut solver = ModifiedBornSolver::new(config, grid.clone());

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                solver.workspace.heterogeneity_workspace[[i, j, k]] =
                    Complex64::new((i + j + k) as f64, (i * j) as f64 * 0.1);
                solver.absorption_field[[i, j, k]] = Complex64::new(0.0, 0.01 * (k as f64));
            }
        }
    }

    let mut expected_green = Array3::<Complex64>::zeros((nx, ny, nz));
    let wavenumber = 10.0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let source_val = solver.workspace.heterogeneity_workspace[[i, j, k]];
                let absorption = solver.absorption_field[[i, j, k]];

                let k_complex = Complex64::new(wavenumber, absorption.im);
                let self_green = Complex64::new(0.5, 0.0) / k_complex.norm_sqr();
                expected_green[[i, j, k]] += self_green * source_val;

                let neighbors = [
                    (i.saturating_sub(1), j, k),
                    ((i + 1).min(nx - 1), j, k),
                    (i, j.saturating_sub(1), k),
                    (i, (j + 1).min(ny - 1), k),
                    (i, j, k.saturating_sub(1)),
                    (i, j, (k + 1).min(nz - 1)),
                ];

                for (ni, nj, nk) in neighbors {
                    let dx = (ni as f64 - i as f64) * grid.dx;
                    let dy = (nj as f64 - j as f64) * grid.dy;
                    let dz = (nk as f64 - k as f64) * grid.dz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                    if r > 1e-12 {
                        let kr_real = wavenumber * r;
                        let kr_imag = absorption.im * r;
                        let exp_factor = Complex64::from_polar(1.0, kr_real)
                            * Complex64::exp(Complex64::new(0.0, -kr_imag));
                        let green_val = exp_factor / (4.0 * PI * r);
                        expected_green[[ni, nj, nk]] += green_val * source_val;
                    }
                }
            }
        }
    }

    solver.apply_viscoacoustic_green(wavenumber, 0.0).unwrap();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let diff = (solver.workspace.green_workspace[[i, j, k]]
                    - expected_green[[i, j, k]])
                .norm();
                assert!(
                    diff < 1e-10,
                    "Mismatch at {},{},{}: actual {:?}, expected {:?}, diff {}",
                    i,
                    j,
                    k,
                    solver.workspace.green_workspace[[i, j, k]],
                    expected_green[[i, j, k]],
                    diff
                );
            }
        }
    }
}
