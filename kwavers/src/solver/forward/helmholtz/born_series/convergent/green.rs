//! Green's function computation and application for `ConvergentBornSolver`.

use super::solver::ConvergentBornSolver;
use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;

impl ConvergentBornSolver {
    /// Compute Green's function in k-space.
    ///
    /// `Ĝ(k) = 1/(k² - k₀² + iε)` where k₀ = wavenumber.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_green_kspace(
        &self,
        green: &mut Array3<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        let k0_squared = wavenumber * wavenumber;
        let epsilon = 1e-10;

        Zip::indexed(green).par_for_each(|(i, j, k), g| {
            let kx = if i <= self.grid.nx / 2 {
                2.0 * PI * (i as f64) / (self.grid.nx as f64 * self.grid.dx)
            } else {
                2.0 * PI * ((i as f64) - self.grid.nx as f64) / (self.grid.nx as f64 * self.grid.dx)
            };
            let ky = if j <= self.grid.ny / 2 {
                2.0 * PI * (j as f64) / (self.grid.ny as f64 * self.grid.dy)
            } else {
                2.0 * PI * ((j as f64) - self.grid.ny as f64) / (self.grid.ny as f64 * self.grid.dy)
            };
            let kz = if k <= self.grid.nz / 2 {
                2.0 * PI * (k as f64) / (self.grid.nz as f64 * self.grid.dz)
            } else {
                2.0 * PI * ((k as f64) - self.grid.nz as f64) / (self.grid.nz as f64 * self.grid.dz)
            };
            let k_squared = kx * kx + ky * ky + kz * kz;
            let denominator = k_squared - k0_squared + Complex64::new(0.0, epsilon);
            *g = Complex64::new(1.0, 0.0) / denominator;
        });

        Ok(())
    }

    /// Apply Green's operator (FFT path when available; direct otherwise).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_green_operator(&mut self) -> KwaversResult<()> {
        if self.green_fft.is_some() {
            self.apply_green_fft()
        } else {
            self.apply_green_direct()
        }
    }

    /// Apply Green's function via FFT-based 3D convolution: O(N³ log N).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_green_fft(&mut self) -> KwaversResult<()> {
        if self.workspace.fft_temp.is_empty() {
            let temp1 = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
            let temp2 = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
            self.workspace.fft_temp.push(temp1);
            self.workspace.fft_temp.push(temp2);
        }

        let mut source_fft = self.workspace.fft_temp[0].clone();
        self.forward_fft_3d(
            &self.workspace.heterogeneity_workspace.view(),
            &mut source_fft,
        )?;

        {
            let mut result_fft = self.workspace.fft_temp[1].view_mut();
            Zip::indexed(&mut result_fft).and(&source_fft).par_for_each(
                |(i, j, k), result_val, &source_val| {
                    if let Some(green_fft) = &self.green_fft {
                        *result_val = source_val * green_fft[[i, j, k]];
                    } else {
                        *result_val = source_val * Complex64::new(0.1, 0.0);
                    }
                },
            );
        }

        let fft_processor = self.fft_processor.as_ref();
        let workspace = &mut self.workspace;
        Self::perform_inverse_fft(
            fft_processor,
            &workspace.fft_temp[1].view(),
            &mut workspace.green_workspace,
        );

        Ok(())
    }

    /// Perform inverse FFT (static helper to avoid split borrow).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn perform_inverse_fft(
        fft_processor: Option<&crate::math::fft::Fft3d>,
        input: &ArrayView3<Complex64>,
        output: &mut Array3<Complex64>,
    ) {
        output.assign(input);
        if let Some(fft) = fft_processor {
            fft.inverse_complex_inplace(output);
        }
    }

    /// 3D forward FFT.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn forward_fft_3d(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut Array3<Complex64>,
    ) -> KwaversResult<()> {
        output.assign(input);
        if let Some(fft) = &self.fft_processor {
            fft.forward_complex_inplace(output);
        }
        Ok(())
    }

    /// 3D inverse FFT.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(test)]
    pub(crate) fn inverse_fft_3d(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut Array3<Complex64>,
    ) -> KwaversResult<()> {
        Self::perform_inverse_fft(self.fft_processor.as_ref(), input, output);
        Ok(())
    }

    /// Apply Green's function via direct spatial-domain stencil.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_green_direct(&mut self) -> KwaversResult<()> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        self.workspace
            .green_workspace
            .fill(Complex64::new(0.0, 0.0));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let source_val = self.workspace.heterogeneity_workspace[[i, j, k]];
                    let self_contribution = source_val * Complex64::new(0.5, 0.0);
                    self.workspace.green_workspace[[i, j, k]] += self_contribution;

                    let neighbors = [
                        (i.saturating_sub(1), j, k),
                        (i.min(nx - 1), j, k),
                        (i, j.saturating_sub(1), k),
                        (i, j.min(ny - 1), k),
                        (i, j, k.saturating_sub(1)),
                        (i, j, k.min(nz - 1)),
                    ];
                    for (ni, nj, nk) in neighbors {
                        let neighbor_contribution = source_val * Complex64::new(1.0 / 6.0, 0.0);
                        self.workspace.green_workspace[[ni, nj, nk]] += neighbor_contribution;
                    }
                }
            }
        }

        Ok(())
    }
}
