//! `HybridAngularSpectrum` propagator facade.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::grid::Grid;
use ndarray::Array3;

use super::config::HASConfig;
use super::solver::HybridAngularSpectrumSolver;
use kwavers_core::constants::numerical::TWO_PI;

/// Hybrid Angular Spectrum propagator.
///
/// Combines FFT-based diffraction with time-domain nonlinearity via operator splitting.
/// See Christopher & Parker (1991) *JASA* 90, 507–521.
#[derive(Debug)]
pub struct HybridAngularSpectrum {
    config: HASConfig,
    solver: HybridAngularSpectrumSolver,
}

impl HybridAngularSpectrum {
    /// Create a new HAS propagator.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by the solver constructor.
    pub fn new(grid: &Grid, config: HASConfig) -> KwaversResult<Self> {
        let solver = HybridAngularSpectrumSolver::new(grid, &config)?;
        Ok(Self { config, solver })
    }

    /// Propagate `pressure` field over `distance` metres.
    ///
    /// Splits the distance into `ceil(distance / config.dz)` equal steps.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] when `distance < 0`.
    pub fn propagate(&self, pressure: &Array3<f64>, distance: f64) -> KwaversResult<Array3<f64>> {
        if distance < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Propagation distance must be non-negative".to_owned(),
            ));
        }
        if distance == 0.0 {
            return Ok(pressure.clone());
        }
        let num_steps = (distance / self.config.dz).ceil() as usize;
        let actual_dz = distance / num_steps as f64;
        self.solver.propagate_steps(pressure, num_steps, actual_dz)
    }

    /// Shock formation distance z_shock = ρ₀c₀³ / (β·ω·p₀) (Hamilton & Blackstock 1998 §4.3).
    ///
    /// `config.nonlinearity` stores B/A; β = 1 + B/(2A) is derived here.
    pub fn shock_formation_distance(&self, p0: f64) -> f64 {
        let beta = 1.0 + self.config.nonlinearity / 2.0; // β = 1 + B/(2A)
        let c0 = self.config.sound_speed;
        let rho0 = self.config.density;
        let omega = TWO_PI * self.config.reference_frequency;
        (rho0 * c0.powi(3)) / (beta * omega * p0)
    }

    /// Rayleigh (near-field) distance z_R = a² / λ (Goodman 2005 §4.2).
    pub fn rayleigh_distance(&self, aperture_radius: f64) -> f64 {
        let wavelength = self.config.sound_speed / self.config.reference_frequency;
        aperture_radius * aperture_radius / wavelength
    }
}
