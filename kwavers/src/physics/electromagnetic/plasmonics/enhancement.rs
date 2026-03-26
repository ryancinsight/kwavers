//! Plasmonic field enhancement and effective medium theories

use super::mie_theory::MieTheory;
use super::types::CouplingModel;
use std::f64::consts::PI;

/// Plasmonic enhancement calculator for homogeneous nanoparticle dispersions
#[derive(Debug)]
pub struct PlasmonicEnhancement {
    /// Mie theory calculator describing the individual nanoparticles
    pub mie_theory: MieTheory,
    /// Nanoparticle concentration (particles/m³)
    pub concentration: f64,
    /// Inter-particle coupling model for dense media
    pub coupling_model: CouplingModel,
}

impl PlasmonicEnhancement {
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

        let distance_from_center =
            (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
        let min_distance = self.mie_theory.radius * 1.1; // Just outside particle boundary
        let effective_distance = distance_from_center.max(min_distance);

        let alpha = self.mie_theory.polarizability(wavelength);
        let epsilon0 = 8.854e-12;

        // Dipole field enhancement computation
        let enhancement = 1.0 + alpha / (4.0 * PI * epsilon0 * effective_distance.powi(3));
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

        match self.coupling_model {
            CouplingModel::None => {
                // Maxwell-Garnett effective medium approximation (dilute limit coupling)
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let numerator = 3.0 * host_dielectric * (eps_particle - host_dielectric);
                let denominator = eps_particle + 2.0 * host_dielectric;

                host_dielectric * (1.0 + volume_fraction * numerator / denominator)
            }
            CouplingModel::DipoleDipole => {
                // Dipole-dipole coupling
                // TODO_AUDIT: P2 - Advanced Plasmonics - Implement full electromagnetic plasmonics with quantum corrections and non-local effects
                // DEPENDS ON: physics/electromagnetic/plasmonics/quantum.rs, physics/electromagnetic/plasmonics/nonlocal.rs, physics/electromagnetic/plasmonics/hydrodynamic.rs
                // MISSING: Quantum plasmonics with electron spill-out and image charges
                // MISSING: Non-local hydrodynamic Drude model for high frequencies
                // MISSING: Surface plasmon polariton dispersion with retardation effects
                // MISSING: Plasmon-exciton coupling in hybrid nanostructures
                // MISSING: Finite element method for complex geometries beyond Mie theory
                // THEOREM: Drude model: ε(ω) = ε_∞ - ω_p²/(ω² + jγω) with quantum corrections
                // THEOREM: Plasmon dispersion: ω(k) = ω_p/√(1 + α k + β k²) with non-local parameter α
                // REFERENCES: Maier (2007) Plasmonics; Novotny & Hecht (2006) Principles of Nano-Optics
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let lorentz_factor =
                    (3.0 * host_dielectric) / (eps_particle + 2.0 * host_dielectric);

                host_dielectric * (1.0 + volume_fraction * lorentz_factor)
            }
            CouplingModel::QuasiStatic => {
                // Bruggeman effective medium approximation for dense/percolative media
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let f = volume_fraction;

                // Simple linear approximation of Bruggeman symmetric mixing formula
                f * eps_particle + (1.0 - f) * num_complex::Complex::new(host_dielectric, 0.0)
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
            1.0 + 10.0 / (1.0 + (detuning / delta_lambda).powi(2))
        } else {
            1.0 // Base transmission (no enhancement)
        }
    }
}
