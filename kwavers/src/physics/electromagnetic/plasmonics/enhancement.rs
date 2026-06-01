//! Plasmonic field enhancement and effective medium theories.
//!
//! # Effective-medium theorems
//!
//! ## Maxwell-Garnett dilute limit
//!
//! For spherical inclusions with permittivity `eps_p`, host permittivity
//! `eps_h`, and inclusion volume fraction `f`, the Maxwell-Garnett relation is
//!
//! ```text
//! (eps_eff - eps_h)/(eps_eff + 2 eps_h)
//!     = f (eps_p - eps_h)/(eps_p + 2 eps_h).
//! ```
//!
//! Solving for `eps_eff` gives
//!
//! ```text
//! eps_eff = eps_h (eps_p + 2 eps_h + 2 f (eps_p - eps_h))
//!                 / (eps_p + 2 eps_h - f (eps_p - eps_h)).
//! ```
//!
//! Proof: multiply both sides by the denominators and collect the single
//! unknown `eps_eff`; the resulting linear equation has the closed form above.
//! At `f = 0`, `eps_eff = eps_h`, and at `eps_p = eps_h`, `eps_eff = eps_h`.
//!
//! ## Bruggeman symmetric mixture
//!
//! Dense quasi-static mixtures solve
//!
//! ```text
//! f (eps_p - eps_eff)/(eps_p + 2 eps_eff)
//! + (1 - f) (eps_h - eps_eff)/(eps_h + 2 eps_eff) = 0.
//! ```
//!
//! Expanding gives `2 eps_eff^2 - C eps_eff - eps_p eps_h = 0`, where
//! `C = (2 eps_h - eps_p) + 3 f (eps_p - eps_h)`. The physical branch is
//! `(C + sqrt(C^2 + 8 eps_p eps_h))/4`, because it returns `eps_h` at `f = 0`
//! and `eps_p` at `f = 1` for real positive permittivities.
//!
//! # References
//!
//! - Maxwell Garnett, J.C. (1904). Colours in metal glasses and in metallic films.
//!   *Philosophical Transactions of the Royal Society A*, 203, 385-420.
//! - Bruggeman, D.A.G. (1935). Berechnung verschiedener physikalischer Konstanten
//!   von heterogenen Substanzen. *Annalen der Physik*, 416, 636-664.
//! - Khlebtsov, N.G. (2008). Optics and biophotonics of nanoparticles with a
//!   plasmon resonance. *Quantum Electronics*, 38, 504-529.
//! - Recent photothermal-plasmonic modeling literature continues to use Mie theory
//!   and Johnson-Christy or Drude-Lorentz dispersive permittivity data for gold
//!   nanoparticle optical cross-sections and heating efficiency.

use super::mie_theory::MieTheory;
use super::types::CouplingModel;
use crate::core::constants::fundamental::VACUUM_PERMITTIVITY;
use crate::core::constants::numerical::FOUR_PI;
use num_complex::Complex;
use std::f64::consts::PI;

/// Plasmonic enhancement calculator for homogeneous nanoparticle dispersions
#[derive(Debug)]
pub struct PlasmonicEnhancementCalculator {
    /// Mie theory calculator describing the individual nanoparticles
    pub mie_theory: MieTheory,
    /// Nanoparticle concentration (particles/m³)
    pub concentration: f64,
    /// Inter-particle coupling model for dense media
    pub coupling_model: CouplingModel,
}

impl PlasmonicEnhancementCalculator {
    /// Create new plasmonic enhancement calculator
    #[must_use]
    pub fn new(mie_theory: MieTheory, concentration: f64) -> Self {
        Self {
            mie_theory,
            concentration,
            coupling_model: CouplingModel::DipoleDipole,
        }
    }

    /// Compute local electromagnetic field enhancement factor at a specific position
    /// relative to the nanoparticle center.
    #[must_use]
    pub fn field_enhancement_factor(&self, wavelength: f64, position: &[f64; 3]) -> f64 {
        // For single particle, enhancement is approximately |E_local|/|E_incident| ≈ 1 + α/(4π ε₀ r³)
        // where r is distance from particle center

        let distance_from_center = position[2]
            .mul_add(
                position[2],
                position[1].mul_add(position[1], position[0].powi(2)),
            )
            .sqrt();
        let min_distance = self.mie_theory.radius * 1.1; // Just outside particle boundary
        let effective_distance = distance_from_center.max(min_distance);

        let alpha = self.mie_theory.polarizability(wavelength);

        // Dipole field enhancement computation
        let enhancement =
            1.0 + alpha / (FOUR_PI * VACUUM_PERMITTIVITY * effective_distance.powi(3));
        enhancement.norm() // Magnitude of complex enhancement vector
    }

    /// Compute effective medium dielectric function incorporating plasmonic nanoparticles
    #[must_use]
    pub fn effective_dielectric(
        &self,
        wavelength: f64,
        host_dielectric: f64,
    ) -> num_complex::Complex<f64> {
        let volume_fraction =
            self.concentration * (4.0 / 3.0) * PI * self.mie_theory.radius.powi(3);
        let host = Complex::new(host_dielectric, 0.0);

        match self.coupling_model {
            CouplingModel::None => {
                // Maxwell-Garnett closed form for dilute spherical inclusions.
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                maxwell_garnett_effective_dielectric(eps_particle, host, volume_fraction)
            }
            CouplingModel::DipoleDipole => {
                // First-order Lorentz-Lorenz (Clausius-Mossotti) correction for
                // dipole-dipole coupling in a dilute dispersion:
                //
                //   eps_eff ≈ eps_h * (1 + 3 f * (eps_p − eps_h)/(eps_p + 2 eps_h))
                //
                // This follows from polarizability α = 4π ε₀ ε_h R³ (eps_p−eps_h)/(eps_p+2eps_h)
                // and eps_eff = eps_h + N α/ε₀.
                //
                // Previous code used 3 ε_h / (eps_p + 2 ε_h) as the Lorentz factor,
                // which is the wrong numerator (should be eps_p − eps_h, not ε_h).
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let contrast = eps_particle - Complex::new(host_dielectric, 0.0);
                let lorentz_factor = 3.0 * contrast / (eps_particle + 2.0 * host_dielectric);

                Complex::new(host_dielectric, 0.0) * (1.0 + volume_fraction * lorentz_factor)
            }
            CouplingModel::QuasiStatic => {
                // Bruggeman symmetric effective-medium solution for dense mixtures.
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                bruggeman_effective_dielectric(eps_particle, host, volume_fraction)
            }
        }
    }

    /// Compute specific surface plasmon resonance enhancement spectral profile
    #[must_use]
    pub fn surface_plasmon_enhancement(&self, wavelength: f64) -> f64 {
        if let Some(resonance_wavelength) = self.mie_theory.plasmon_resonance_wavelength() {
            // Lorentzian enhancement line shape profile
            let delta_lambda = 50e-9; // FWHM spectral width ≈ 50 nm
            let detuning = wavelength - resonance_wavelength;

            // Modeled peak near-field enhancement of 11x
            1.0 + 10.0 / (detuning / delta_lambda).mul_add(detuning / delta_lambda, 1.0)
        } else {
            1.0 // Base transmission (no enhancement)
        }
    }
}

/// Maxwell-Garnett effective permittivity for spherical inclusions.
///
/// The input `volume_fraction` is dimensionless. Values outside `[0, 1]` are
/// clamped to preserve the theorem's physical admissibility domain.
#[must_use]
pub(crate) fn maxwell_garnett_effective_dielectric(
    eps_particle: Complex<f64>,
    eps_host: Complex<f64>,
    volume_fraction: f64,
) -> Complex<f64> {
    let f = volume_fraction.clamp(0.0, 1.0);
    let contrast = eps_particle - eps_host;
    eps_host * (eps_particle + 2.0 * eps_host + 2.0 * f * contrast)
        / (eps_particle + 2.0 * eps_host - f * contrast)
}

/// Bruggeman symmetric effective permittivity for two-component mixtures.
///
/// Selects the quadratic branch that returns the host at `f = 0` and the
/// particle at `f = 1` for real positive permittivities.
#[must_use]
pub(crate) fn bruggeman_effective_dielectric(
    eps_particle: Complex<f64>,
    eps_host: Complex<f64>,
    volume_fraction: f64,
) -> Complex<f64> {
    let f = volume_fraction.clamp(0.0, 1.0);
    let c = (2.0 * eps_host - eps_particle) + 3.0 * f * (eps_particle - eps_host);
    (c + (c * c + 8.0 * eps_particle * eps_host).sqrt()) / 4.0
}
