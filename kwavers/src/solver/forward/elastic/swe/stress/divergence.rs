//! Two-pass elastic stress tensor divergence computation.

use super::super::types::ElasticWaveField;
use super::fd_stencils::{fd1_x, fd1_y, fd1_z};
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Compute the elastic stress tensor divergence ∇·σ.
///
/// Returns `(div_x, div_y, div_z)` where each element satisfies:
/// ```text
/// (∇·σ)_α = ∂σαx/∂x + ∂σαy/∂y + ∂σαz/∂z
/// ```
///
/// ## Pass 1 — stress construction
///
/// At each grid point, strain components are computed from displacements
/// via 4th-order centered FD:
/// ```text
/// εxx = ∂ux/∂x,  εyy = ∂uy/∂y,  εzz = ∂uz/∂z
/// σxx = (λ+2μ) εxx + λ(εyy+εzz)
/// σyy = (λ+2μ) εyy + λ(εxx+εzz)
/// σzz = (λ+2μ) εzz + λ(εxx+εyy)
/// σxy = σyx = μ(∂ux/∂y + ∂uy/∂x)
/// σxz = σzx = μ(∂ux/∂z + ∂uz/∂x)
/// σyz = σzy = μ(∂uy/∂z + ∂uz/∂y)
/// ```
///
/// ## Pass 2 — divergence of σ
///
/// Each divergence component is obtained by differentiating the stress
/// arrays from pass 1 using the same 4th-order FD stencils.
pub fn stress_divergence(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &ElasticWaveField,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = field.ux.dim();
    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    // --- Pass 1: stress tensor at every grid point ---
    let mut sxx = Array3::<f64>::zeros((nx, ny, nz));
    let mut syy = Array3::<f64>::zeros((nx, ny, nz));
    let mut szz = Array3::<f64>::zeros((nx, ny, nz));
    let mut sxy = Array3::<f64>::zeros((nx, ny, nz));
    let mut sxz = Array3::<f64>::zeros((nx, ny, nz));
    let mut syz = Array3::<f64>::zeros((nx, ny, nz));

    let ux = field.ux.view();
    let uy = field.uy.view();
    let uz = field.uz.view();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let exx = fd1_x(ux, i, j, k, nx, dx);
                let eyy = fd1_y(uy, i, j, k, ny, dy);
                let ezz = fd1_z(uz, i, j, k, nz, dz);

                let exy_2 = fd1_y(ux, i, j, k, ny, dy) + fd1_x(uy, i, j, k, nx, dx);
                let exz_2 = fd1_z(ux, i, j, k, nz, dz) + fd1_x(uz, i, j, k, nx, dx);
                let eyz_2 = fd1_z(uy, i, j, k, nz, dz) + fd1_y(uz, i, j, k, ny, dy);

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

    // --- Pass 2: ∇·σ ---
    let mut div_x = Array3::<f64>::zeros((nx, ny, nz));
    let mut div_y = Array3::<f64>::zeros((nx, ny, nz));
    let mut div_z = Array3::<f64>::zeros((nx, ny, nz));

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                div_x[[i, j, k]] = fd1_x(sxx.view(), i, j, k, nx, dx)
                    + fd1_y(sxy.view(), i, j, k, ny, dy)
                    + fd1_z(sxz.view(), i, j, k, nz, dz);
                div_y[[i, j, k]] = fd1_x(sxy.view(), i, j, k, nx, dx)
                    + fd1_y(syy.view(), i, j, k, ny, dy)
                    + fd1_z(syz.view(), i, j, k, nz, dz);
                div_z[[i, j, k]] = fd1_x(sxz.view(), i, j, k, nx, dx)
                    + fd1_y(syz.view(), i, j, k, ny, dy)
                    + fd1_z(szz.view(), i, j, k, nz, dz);
            }
        }
    }

    (div_x, div_y, div_z)
}
