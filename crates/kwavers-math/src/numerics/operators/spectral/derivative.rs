//! Pseudospectral derivative operator using FFT.
//!
//! # Wavenumber Convention
//!
//! For an N-point grid, wavenumbers are arranged as:
//! ```text
//! k = [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1] * (2π / L)
//! ```
//! where L = N * Δx is the domain length.
//!
//! # Reference
//!
//! - Liu, Q. H. (1997). Microwave Opt. Technol. Lett., 15(3), 158-165.

use super::trait_def::SpectralOperatorTrait;
use super::{SpectralFilter, SpectralFilterType};
use crate::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, Array3, ArrayView3};
use std::f64::consts::PI;

/// Pseudospectral derivative operator using FFT
///
/// Computes spatial derivatives using the Fourier differentiation theorem,
/// providing spectral accuracy (exponential convergence) for smooth functions.
#[derive(Debug)]
pub struct PseudospectralDerivative {
    /// Wavenumber grid in X direction (rad/m)
    kx: Array1<f64>,
    /// Wavenumber grid in Y direction (rad/m)
    ky: Array1<f64>,
    /// Wavenumber grid in Z direction (rad/m)
    kz: Array1<f64>,
    /// Grid spacing in X (m)
    dx: f64,
    /// Grid spacing in Y (m)
    dy: f64,
    /// Grid spacing in Z (m)
    dz: f64,
}

impl PseudospectralDerivative {
    /// Create a new pseudospectral derivative operator
    ///
    /// # Arguments
    ///
    /// * `nx/ny/nz` - Number of grid points per direction
    /// * `dx/dy/dz` - Grid spacings (meters)
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }

        Ok(Self {
            kx: Self::wavenumber_vector(nx, dx),
            ky: Self::wavenumber_vector(ny, dy),
            kz: Self::wavenumber_vector(nz, dz),
            dx,
            dy,
            dz,
        })
    }

    /// Generate wavenumber vector for FFT.
    ///
    /// k`i` = 2π·i / (N·d)        for i = 0..N/2
    /// k`i` = 2π·(i−N) / (N·d)   for i = N/2..N
    pub(super) fn wavenumber_vector(n: usize, d: f64) -> Array1<f64> {
        let mut k = Array1::zeros([n]);
        let dk = TWO_PI / ((n as f64) * d);

        for i in 0..n / 2 {
            k[i] = (i as f64) * dk;
        }
        for i in n / 2..n {
            k[i] = ((i as i64) - (n as i64)) as f64 * dk;
        }

        k
    }

    /// Compute spectral derivative in X direction: ∂u/∂x = F⁻¹{ik_x F{u}}
    ///
    /// # Spectral Accuracy
    ///
    /// For smooth periodic functions, achieves O(exp(-cN)) convergence.
    /// Validated: ∂(sin(kx))/∂x = k·cos(kx) with L∞ error < 1e-12.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn derivative_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = field.shape();
        let mut derivative = Array3::zeros(shape);
        let mut line_workspace = Array1::<Complex64>::zeros([shape[0]]);
        self.derivative_x_into(field, &mut line_workspace, &mut derivative)?;
        Ok(derivative)
    }

    /// Compute ∂u/∂x into caller-owned workspaces.
    ///
    /// The implementation reuses one complex line buffer across every `(y,z)`
    /// pencil. This preserves the Fourier differentiation theorem while
    /// eliminating the per-line `Array1` allocations used by the allocating
    /// wrapper.
    ///
    /// # Errors
    /// Returns [`KwaversError::DimensionMismatch`] when either workspace shape
    /// does not match `field`, or propagates invalid-grid diagnostics when the
    /// field axis length is inconsistent with the operator wavenumber vector.
    pub fn derivative_x_into(
        &self,
        field: ArrayView3<f64>,
        line_workspace: &mut Array1<Complex64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.derivative_axis_into::<0>(field, line_workspace, output)
    }

    /// Compute spectral derivative in Y direction: ∂u/∂y = F⁻¹{ik_y F{u}}
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn derivative_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = field.shape();
        let mut derivative = Array3::zeros(shape);
        let mut line_workspace = Array1::<Complex64>::zeros([shape[1]]);
        self.derivative_y_into(field, &mut line_workspace, &mut derivative)?;
        Ok(derivative)
    }

    /// Compute ∂u/∂y into caller-owned workspaces.
    ///
    /// The implementation reuses one complex line buffer across every `(x,z)`
    /// pencil. This is the same monomorphized Fourier kernel used by the
    /// allocating wrapper and differs only in workspace ownership.
    ///
    /// # Errors
    /// Returns [`KwaversError::DimensionMismatch`] when either workspace shape
    /// does not match `field`, or propagates invalid-grid diagnostics when the
    /// field axis length is inconsistent with the operator wavenumber vector.
    pub fn derivative_y_into(
        &self,
        field: ArrayView3<f64>,
        line_workspace: &mut Array1<Complex64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.derivative_axis_into::<1>(field, line_workspace, output)
    }

    /// Compute spectral derivative in Z direction: ∂u/∂z = F⁻¹{ik_z F{u}}
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn derivative_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = field.shape();
        let mut derivative = Array3::zeros(shape);
        let mut line_workspace = Array1::<Complex64>::zeros([shape[2]]);
        self.derivative_z_into(field, &mut line_workspace, &mut derivative)?;
        Ok(derivative)
    }

    /// Compute ∂u/∂z into caller-owned workspaces.
    ///
    /// The implementation reuses one complex line buffer across every `(x,y)`
    /// pencil. This is the same monomorphized Fourier kernel used by the
    /// allocating wrapper and differs only in workspace ownership.
    ///
    /// # Errors
    /// Returns [`KwaversError::DimensionMismatch`] when either workspace shape
    /// does not match `field`, or propagates invalid-grid diagnostics when the
    /// field axis length is inconsistent with the operator wavenumber vector.
    pub fn derivative_z_into(
        &self,
        field: ArrayView3<f64>,
        line_workspace: &mut Array1<Complex64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.derivative_axis_into::<2>(field, line_workspace, output)
    }

    /// Monomorphized Fourier differentiation kernel for one Cartesian axis.
    ///
    /// `AXIS` is a structural compile-time parameter, so the compiler emits
    /// separate x/y/z specializations and strips the inactive loop branches.
    /// Variation is encoded in the type system instead of three cloned
    /// algorithm bodies.
    fn derivative_axis_into<const AXIS: usize>(
        &self,
        field: ArrayView3<f64>,
        line_workspace: &mut Array1<Complex64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        validate_output_shape(output.shape(), field.shape())?;

        match AXIS {
            0 => {
                validate_axis_len(nx, self.kx.size(), self.dx, self.dy, self.dz)?;
                validate_line_workspace(line_workspace.size(), nx)?;
                for j in 0..ny {
                    for k in 0..nz {
                        for i in 0..nx {
                            line_workspace[i] = Complex64::new(field[[i, j, k]], 0.0);
                        }
                        differentiate_line(line_workspace, &self.kx);
                        for i in 0..nx {
                            output[[i, j, k]] = line_workspace[i].re;
                        }
                    }
                }
            }
            1 => {
                validate_axis_len(ny, self.ky.size(), self.dx, self.dy, self.dz)?;
                validate_line_workspace(line_workspace.size(), ny)?;
                for i in 0..nx {
                    for k in 0..nz {
                        for j in 0..ny {
                            line_workspace[j] = Complex64::new(field[[i, j, k]], 0.0);
                        }
                        differentiate_line(line_workspace, &self.ky);
                        for j in 0..ny {
                            output[[i, j, k]] = line_workspace[j].re;
                        }
                    }
                }
            }
            2 => {
                validate_axis_len(nz, self.kz.size(), self.dx, self.dy, self.dz)?;
                validate_line_workspace(line_workspace.size(), nz)?;
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            line_workspace[k] = Complex64::new(field[[i, j, k]], 0.0);
                        }
                        differentiate_line(line_workspace, &self.kz);
                        for k in 0..nz {
                            output[[i, j, k]] = line_workspace[k].re;
                        }
                    }
                }
            }
            _ => unreachable!("PseudospectralDerivative only supports axes 0, 1, and 2"),
        }

        Ok(())
    }
}

fn differentiate_line(line_workspace: &mut Array1<Complex64>, wavenumbers: &Array1<f64>) {
    fft_1d_complex_inplace(line_workspace);

    for (value, &k_axis) in line_workspace.iter_mut().zip(wavenumbers.iter()) {
        *value *= Complex64::new(0.0, k_axis);
    }

    // Apollo applies 1/N inverse normalisation; no extra scale needed.
    ifft_1d_complex_inplace(line_workspace);
}

fn validate_axis_len(
    field_len: usize,
    expected_len: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> KwaversResult<()> {
    if field_len != expected_len {
        return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
    }

    Ok(())
}

fn validate_line_workspace(actual: usize, expected: usize) -> KwaversResult<()> {
    if actual != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "PseudospectralDerivative line workspace length {actual} must match axis length {expected}"
        )));
    }

    Ok(())
}

fn validate_output_shape(actual: [usize; 3], expected: [usize; 3]) -> KwaversResult<()> {
    if actual != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "PseudospectralDerivative output shape {actual:?} must match field shape {expected:?}"
        )));
    }

    Ok(())
}

impl SpectralOperatorTrait for PseudospectralDerivative {
    fn apply_kspace(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.derivative_x(field)
    }

    fn wavenumber_grid(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (self.kx.clone(), self.ky.clone(), self.kz.clone())
    }

    fn nyquist_wavenumber(&self) -> (f64, f64, f64) {
        (PI / self.dx, PI / self.dy, PI / self.dz)
    }

    fn apply_antialias_filter(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        SpectralFilter::new(2.0 / 3.0, SpectralFilterType::SharpCutoff).apply(field)
    }
}
