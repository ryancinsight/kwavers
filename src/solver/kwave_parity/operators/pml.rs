//! Perfectly Matched Layer (PML) implementations for k-Wave solver
//!
//! Based on: Berenger, J. P. (1994). "A perfectly matched layer for the absorption
//! of electromagnetic waves." Journal of Computational Physics, 114(2), 185-200.

use ndarray::Array3;

/// PML absorption coefficients
#[derive(Debug, Clone)]
pub struct PMLCoefficients {
    pub alpha_x: Array3<f64>,
    pub alpha_y: Array3<f64>,
    pub alpha_z: Array3<f64>,
}

/// Compute PML absorption coefficients
#[must_use]
pub fn compute_pml_coefficients(
    nx: usize,
    ny: usize,
    nz: usize,
    pml_size: usize,
    pml_alpha: f64,
) -> PMLCoefficients {
    let mut alpha_x = Array3::zeros((nx, ny, nz));
    let mut alpha_y = Array3::zeros((nx, ny, nz));
    let mut alpha_z = Array3::zeros((nx, ny, nz));

    // X-direction PML
    for i in 0..pml_size {
        let sigma = pml_alpha * ((pml_size - i) as f64 / pml_size as f64).powi(2);
        for j in 0..ny {
            for k in 0..nz {
                alpha_x[[i, j, k]] = sigma;
                alpha_x[[nx - 1 - i, j, k]] = sigma;
            }
        }
    }

    // Y-direction PML
    for j in 0..pml_size {
        let sigma = pml_alpha * ((pml_size - j) as f64 / pml_size as f64).powi(2);
        for i in 0..nx {
            for k in 0..nz {
                alpha_y[[i, j, k]] = sigma;
                alpha_y[[i, ny - 1 - j, k]] = sigma;
            }
        }
    }

    // Z-direction PML
    for k in 0..pml_size {
        let sigma = pml_alpha * ((pml_size - k) as f64 / pml_size as f64).powi(2);
        for i in 0..nx {
            for j in 0..ny {
                alpha_z[[i, j, k]] = sigma;
                alpha_z[[i, j, nz - 1 - k]] = sigma;
            }
        }
    }

    PMLCoefficients {
        alpha_x,
        alpha_y,
        alpha_z,
    }
}

/// Apply PML absorption to a field
pub fn apply_pml_absorption(field: &mut Array3<f64>, coefficients: &PMLCoefficients, dt: f64) {
    field.zip_mut_with(&coefficients.alpha_x, |f, &alpha| {
        *f *= (-alpha * dt).exp();
    });
}
