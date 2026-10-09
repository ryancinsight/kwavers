//! Optical fluence solver for photoacoustic tomography.
//!
//! Solves the steady-state optical diffusion equation via spectral (FFT)
//! methods, yielding the fluence distribution `Φ(r)` and the absorbed energy
//! `H = μ_a Φ(r)` that acts as the photoacoustic source term.
//!
//! ## Diffusion approximation validity
//!
//! Valid when scattering dominates: `μ_s' >> μ_a` (typically satisfied in soft
//! tissue at NIR wavelengths: `μ_s' ≈ 10 cm⁻¹`, `μ_a ≈ 0.1 cm⁻¹`).
//!
//! ## References
//!
//! - Arridge SR (1999). "Optical tomography in medical imaging."
//!   Inverse Problems 15(2), R41. DOI: 10.1088/0266-5611/15/2/022
//! - Wang LV, Wu H (2007). "Biomedical Optics." Wiley-Interscience.

use std::any::Any;
use std::sync::Arc;

use leto::{Array3, ArrayView3, Array4};

use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};
use kwavers_medium::Medium;

pub mod diffusion;

pub use diffusion::{DiffusionSolver, DiffusionSolverConfig};

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

/// Plugin that computes optical fluence and exposes absorbed energy for
/// photoacoustic source initialization.
#[derive(Debug)]
pub struct OpticalDiffusionPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<OpticalDiffusionSolver>,
    initialized_source: bool,
}

impl Default for OpticalDiffusionPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticalDiffusionPlugin {
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "optical_diffusion".to_owned(),
                name: "Optical Diffusion".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Steady-state optical diffusion for photoacoustic source generation"
                    .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
            initialized_source: false,
        }
    }
}

impl Plugin for OpticalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        Vec::new()
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::LightFluence]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let mut solver = OpticalDiffusionSolver::from_medium(grid.clone(), medium)?;
        let source = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
        let _ = solver.solve(&source)?;
        self.solver = Some(solver);
        self.initialized_source = true;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            KwaversError::InternalError("OpticalDiffusionPlugin updated before initialize()".to_owned())
        })?;

        let light_index = UnifiedFieldType::LightFluence.index();
        let pressure_index = UnifiedFieldType::Pressure.index();
        let source_view = fields
            .index_axis::<3>(0, light_index)
            .expect("invariant: light fluence field exists");

        let has_external_source = source_view.iter().any(|value| value.abs() > 0.0);
        if has_external_source || !self.initialized_source {
            let source = source_view.to_contiguous();
            let _ = solver.solve(&source)?;
            self.initialized_source = true;
        }

        fields
            .index_axis_mut::<3>(0, light_index)
            .expect("invariant: light fluence field exists")
            .assign(&solver.fluence());
        fields
            .index_axis_mut::<3>(0, pressure_index)
            .expect("invariant: pressure field exists")
            .assign(&solver.absorbed_energy_density());

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
