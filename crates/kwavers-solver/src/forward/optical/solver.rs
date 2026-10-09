//! Optical diffusion solver implementation.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use leto::Array3;

use std::sync::Arc;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_medium::Medium;
use leto::ArrayView3;
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};




fn k_vector(n: usize, spacing: f64) -> Vec<f64> {
    let dk = TWO_PI / (n as f64 * spacing);
    (0..n)
        .map(|i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
        .collect()
}

/// FFT-based optical diffusion solver for absorbed-energy generation.
#[derive(Debug)]
pub struct OpticalDiffusionSolver {
    /// Absorption coefficient `μ_a` [m⁻¹].
    mu_a: Array3<f64>,
    /// Reduced scattering coefficient `μ_s'` [m⁻¹].
    mu_s_prime: Array3<f64>,
    /// Effective attenuation `μ_eff = √(3μ_a(μ_a+μ_s'))`.
    mu_eff: Array3<f64>,
    /// Fluence `Φ(r)` [W/m²].
    fluence: Array3<f64>,
    /// Absorbed energy density `H = μ_a Φ` [W/m³].
    absorbed_energy: Array3<f64>,
    grid: Grid,
    fft: Arc<Fft3d>,
    k_squared: Array3<f64>,
    spectral_field: Array3<Complex64>,
    spectral_scratch: Array3<Complex64>,
    laplacian: Array3<f64>,
}

impl OpticalDiffusionSolver {
    /// Build the optical solver from explicit material fields.
    ///
    /// # Errors
    /// Returns [`Err`] when the input shapes do not match the grid.
    #[must_use]
    pub fn new(grid: Grid, mu_a: Array3<f64>, mu_s_prime: Array3<f64>) -> KwaversResult<Self> {
        if mu_a.shape() != [grid.nx, grid.ny, grid.nz]
            || mu_s_prime.shape() != [grid.nx, grid.ny, grid.nz]
        {
            return Err(KwaversError::DimensionMismatch(
                "optical coefficient fields must match the simulation grid".to_owned(),
            ));
        }

        let mut mu_eff = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for ((mu_eff_value, &mu_a_value), &mu_s_prime_value) in mu_eff
            .iter_mut()
            .zip(mu_a.iter())
            .zip(mu_s_prime.iter())
        {
            *mu_eff_value = (3.0 * mu_a_value * (mu_a_value + mu_s_prime_value)).sqrt();
        }

        let kx = k_vector(grid.nx, grid.dx);
        let ky = k_vector(grid.ny, grid.dy);
        let kz = k_vector(grid.nz, grid.dz);
        let mut k_squared = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    k_squared[[i, j, k]] =
                        kx[i].mul_add(kx[i], ky[j].mul_add(ky[j], kz[k] * kz[k]));
                }
            }
        }

        Ok(Self {
            mu_a,
            mu_s_prime,
            mu_eff,
            fluence: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            absorbed_energy: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            fft: get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            k_squared,
            spectral_field: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            spectral_scratch: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            laplacian: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            grid,
        })
    }

    /// Build the optical solver from the medium's optical coefficients.
    ///
    /// # Errors
    /// Returns [`Err`] when a medium-supplied optical coefficient is invalid.
    #[must_use]
    pub fn from_medium(grid: Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        let mut mu_a = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut mu_s_prime = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    mu_a[[i, j, k]] = medium.optical_absorption_coefficient(x, y, z, &grid);
                    mu_s_prime[[i, j, k]] = medium
                        .optical_reduced_scattering_coefficient(x, y, z, &grid)
                        .map_err(|error| {
                            KwaversError::InvalidInput(format!(
                                "invalid reduced scattering coefficient at ({i},{j},{k}): {error}"
                            ))
                        })?
                        .in_unit::<aequitas::systems::si::units::PerMeter>();
                }
            }
        }

        Self::new(grid, mu_a, mu_s_prime)
    }

    fn mu_eff_mean_squared(&self) -> f64 {
        let sum: f64 = self.mu_eff.iter().map(|value| value * value).sum();
        sum / self.mu_eff.len() as f64
    }

    fn compute_laplacian(&mut self, field: &Array3<f64>) {
        self.fft.forward_into(field, &mut self.spectral_field);
        for (value, &k_sq) in self.spectral_field.iter_mut().zip(self.k_squared.iter()) {
            *value *= -k_sq;
        }
        self.fft.inverse_into(
            &self.spectral_field,
            &mut self.laplacian,
            &mut self.spectral_scratch,
        );
    }

    /// Solve the steady-state diffusion equation for a given optical source.
    ///
    /// # Errors
    /// Returns [`Err`] when the source shape does not match the solver grid.
    pub fn solve(&mut self, source: &Array3<f64>) -> KwaversResult<&Array3<f64>> {
        if source.shape() != [self.grid.nx, self.grid.ny, self.grid.nz] {
            return Err(KwaversError::DimensionMismatch(
                "optical source shape must match the solver grid".to_owned(),
            ));
        }

        let mu_eff_mean_sq = self.mu_eff_mean_squared().max(1.0e-12);

        self.fft.forward_into(source, &mut self.spectral_field);
        for (value, &k_sq) in self.spectral_field.iter_mut().zip(self.k_squared.iter()) {
            *value /= k_sq + mu_eff_mean_sq;
        }
        self.fft.inverse_into(
            &self.spectral_field,
            &mut self.fluence,
            &mut self.spectral_scratch,
        );

        let relaxation = 0.6_f64;
        let mut residual = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        for _ in 0..6 {
            let current = self.fluence.clone();
            self.compute_laplacian(&current);
            for ((((residual_value, &source_value), &laplacian_value), &mu_eff_value), fluence) in
                residual
                    .iter_mut()
                    .zip(source.iter())
                    .zip(self.laplacian.iter())
                    .zip(self.mu_eff.iter())
                    .zip(self.fluence.iter_mut())
            {
                let operator_phi = laplacian_value - mu_eff_value * mu_eff_value * *fluence;
                *residual_value = source_value - operator_phi;
                *fluence += relaxation * *residual_value / mu_eff_mean_sq;
            }
        }

        for ((absorbed, &mu_a_value), &fluence_value) in self
            .absorbed_energy
            .iter_mut()
            .zip(self.mu_a.iter())
            .zip(self.fluence.iter())
        {
            *absorbed = mu_a_value * fluence_value;
        }

        Ok(&self.fluence)
    }

    #[must_use]
    pub fn absorbed_energy_density(&self) -> ArrayView3<'_, f64> {
        self.absorbed_energy.view()
    }

    #[must_use]
    pub fn fluence(&self) -> ArrayView3<'_, f64> {
        self.fluence.view()
    }

    #[must_use]
    pub fn reduced_scattering(&self) -> ArrayView3<'_, f64> {
        self.mu_s_prime.view()
    }
}