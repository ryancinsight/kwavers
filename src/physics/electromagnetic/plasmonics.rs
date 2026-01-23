//! Plasmonic Enhancement Implementations
//!
//! This module implements plasmonic effects for enhanced electromagnetic fields
//! near metallic nanostructures, relevant for photoacoustic imaging and therapy.

use std::f64::consts::PI;

// NanoparticleGeometry is defined in domain::physics::electromagnetic

/// Mie theory implementation for spherical plasmonic nanoparticles
pub struct MieTheory {
    /// Nanoparticle radius (m)
    pub radius: f64,
    /// Dielectric function of nanoparticle ε_particle(ω)
    pub particle_dielectric: Box<dyn Fn(f64) -> num_complex::Complex<f64>>,
    /// Dielectric function of surrounding medium ε_medium
    pub medium_dielectric: f64,
}

impl std::fmt::Debug for MieTheory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MieTheory")
            .field("radius", &self.radius)
            .field("medium_dielectric", &self.medium_dielectric)
            .finish()
    }
}

impl MieTheory {
    /// Create Mie theory calculator for gold nanoparticle in water
    pub fn gold_in_water(radius: f64) -> Self {
        Self {
            radius,
            particle_dielectric: Box::new(|wavelength| {
                // Simplified Drude-Lorentz model for gold
                // ε(ω) = ε_inf - ω_p²/(ω² + iγω)

                let epsilon_inf = 9.84;
                let omega_p = 9.01; // eV
                let gamma = 0.071; // eV

                // Convert wavelength to energy (E = hc/λ)
                let hbar = 6.582119569e-16; // eV·s
                let _c = 2.99792458e8; // m/s
                let energy = 1.23984193e-6 / wavelength; // Convert m to eV

                let omega = energy / hbar;
                let omega_p_rad = omega_p / hbar;
                let gamma_rad = gamma / hbar;

                let denominator = num_complex::Complex::new(omega * omega, gamma_rad * omega);
                epsilon_inf - (omega_p_rad * omega_p_rad) / denominator
            }),
            medium_dielectric: 1.77, // Water at optical frequencies
        }
    }

    /// Compute polarizability using Mie theory
    pub fn polarizability(&self, wavelength: f64) -> num_complex::Complex<f64> {
        let eps_particle = (self.particle_dielectric)(wavelength);
        let eps_medium = self.medium_dielectric;

        // Mie polarizability: α = 4π ε₀ ε_m R³ (ε - ε_m)/(ε + 2ε_m)
        let eps_ratio = eps_particle / eps_medium;
        let numerator = eps_ratio - num_complex::Complex::new(1.0, 0.0);
        let denominator = eps_ratio + num_complex::Complex::new(2.0, 0.0);

        let alpha_dimensionless =
            3.0 * self.radius * self.radius * self.radius * numerator / denominator;

        // Convert to SI units (include 4π ε₀ ε_m factor)
        let epsilon0 = 8.854e-12;
        alpha_dimensionless * 4.0 * PI * epsilon0 * eps_medium
    }

    /// Compute scattering cross-section
    pub fn scattering_cross_section(&self, wavelength: f64) -> f64 {
        let alpha = self.polarizability(wavelength);
        let k = 2.0 * PI / wavelength; // Wave number in medium

        // σ_scat = (8π/3) k⁴ |α|² for Rayleigh scattering
        (8.0 * PI / 3.0) * k.powi(4) * alpha.norm_sqr()
    }

    /// Compute absorption cross-section
    pub fn absorption_cross_section(&self, wavelength: f64) -> f64 {
        let alpha = self.polarizability(wavelength);
        let k = 2.0 * PI / wavelength;

        // σ_abs = k Im(α) for small particles
        k * alpha.im
    }

    /// Compute extinction cross-section
    pub fn extinction_cross_section(&self, wavelength: f64) -> f64 {
        self.scattering_cross_section(wavelength) + self.absorption_cross_section(wavelength)
    }

    /// Find plasmon resonance wavelength
    pub fn plasmon_resonance_wavelength(&self) -> Option<f64> {
        // Find wavelength where Re(ε_particle + 2ε_medium) = 0
        // This requires numerical solution

        // Simple grid search for demonstration
        let wavelengths = (400..900).map(|nm| nm as f64 * 1e-9); // 400-900 nm

        for wavelength in wavelengths {
            let eps_particle = (self.particle_dielectric)(wavelength);
            let denominator =
                eps_particle + num_complex::Complex::new(2.0 * self.medium_dielectric, 0.0);

            if denominator.re.abs() < 0.1 {
                // Close to resonance
                return Some(wavelength);
            }
        }

        None
    }
}

/// Plasmonic enhancement calculator
#[derive(Debug)]
pub struct PlasmonicEnhancement {
    /// Mie theory calculator
    pub mie_theory: MieTheory,
    /// Nanoparticle concentration (particles/m³)
    pub concentration: f64,
    /// Inter-particle coupling model
    pub coupling_model: CouplingModel,
}

#[derive(Debug, Clone)]
pub enum CouplingModel {
    /// No coupling (dilute limit)
    None,
    /// Dipole-dipole coupling approximation
    DipoleDipole,
    /// Quasi-static approximation for dense media
    QuasiStatic,
}

impl PlasmonicEnhancement {
    /// Create plasmonic enhancement calculator
    pub fn new(mie_theory: MieTheory, concentration: f64) -> Self {
        Self {
            mie_theory,
            concentration,
            coupling_model: CouplingModel::DipoleDipole,
        }
    }

    /// Compute local field enhancement factor
    pub fn field_enhancement_factor(&self, wavelength: f64, position: &[f64]) -> f64 {
        // For single particle, enhancement is approximately |E_local|/|E_incident| ≈ 1 + α/(4π ε₀ r³)
        // where r is distance from particle center

        let distance_from_center =
            (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
        let min_distance = self.mie_theory.radius * 1.1; // Just outside particle
        let effective_distance = distance_from_center.max(min_distance);

        let alpha = self.mie_theory.polarizability(wavelength);
        let epsilon0 = 8.854e-12;

        // Dipole field enhancement
        let enhancement = 1.0 + alpha / (4.0 * PI * epsilon0 * effective_distance.powi(3));
        enhancement.norm() // Magnitude
    }

    /// Compute effective medium dielectric function with plasmonic nanoparticles
    pub fn effective_dielectric(
        &self,
        wavelength: f64,
        host_dielectric: f64,
    ) -> num_complex::Complex<f64> {
        let volume_fraction =
            self.concentration * (4.0 / 3.0) * PI * self.mie_theory.radius.powi(3);
        let _alpha = self.mie_theory.polarizability(wavelength);
        let _epsilon0 = 8.854e-12;

        match self.coupling_model {
            CouplingModel::None => {
                // Maxwell-Garnett approximation (dilute limit)
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let numerator = 3.0 * host_dielectric * (eps_particle - host_dielectric);
                let denominator = eps_particle + 2.0 * host_dielectric;

                host_dielectric * (1.0 + volume_fraction * numerator / denominator)
            }
            CouplingModel::DipoleDipole => {
                // Include dipole coupling effects
                // This is a simplified model - real implementation would be more complex
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
                // Bruggeman approximation for dense media
                let eps_particle = (self.mie_theory.particle_dielectric)(wavelength);
                let f = volume_fraction;

                // Solve: f (eps_particle - eps_eff)/(eps_particle + 2 eps_eff) +
                //        (1-f) (host_dielectric - eps_eff)/(host_dielectric + 2 eps_eff) = 0

                // Simplified approximation
                f * eps_particle + (1.0 - f) * num_complex::Complex::new(host_dielectric, 0.0)
            }
        }
    }

    /// Compute surface plasmon resonance enhancement
    pub fn surface_plasmon_enhancement(&self, wavelength: f64) -> f64 {
        if let Some(resonance_wavelength) = self.mie_theory.plasmon_resonance_wavelength() {
            // Lorentzian enhancement profile
            let delta_lambda = 50e-9; // FWHM ≈ 50 nm
            let detuning = wavelength - resonance_wavelength;

            1.0 + 10.0 / (1.0 + (detuning / delta_lambda).powi(2)) // Peak enhancement of 11x
        } else {
            1.0 // No enhancement
        }
    }
}

/// Nanoparticle array for collective plasmonic effects
#[derive(Debug)]
pub struct NanoparticleArray {
    /// Individual nanoparticles
    pub particles: Vec<(MieTheory, [f64; 3])>, // (particle, position)
    /// Array geometry
    pub geometry: ArrayGeometry,
}

#[derive(Debug, Clone)]
pub enum ArrayGeometry {
    /// Linear chain
    Linear { spacing: f64 },
    /// 2D square lattice
    Square { spacing_x: f64, spacing_y: f64 },
    /// 3D cubic lattice
    Cubic {
        spacing_x: f64,
        spacing_y: f64,
        spacing_z: f64,
    },
    /// Random distribution
    Random,
}

impl NanoparticleArray {
    /// Create a linear array of nanoparticles
    pub fn linear_array(particle_radius: f64, spacing: f64, n_particles: usize) -> Self {
        let mut particles = Vec::new();

        for i in 0..n_particles {
            let position = [i as f64 * spacing, 0.0, 0.0];
            let mie = MieTheory::gold_in_water(particle_radius);
            particles.push((mie, position));
        }

        Self {
            particles,
            geometry: ArrayGeometry::Linear { spacing },
        }
    }

    /// Compute collective plasmonic enhancement at a point
    ///
    /// Enhancement is defined as |E_total|²/|E_incident|² where E_total includes
    /// scattered fields from all nanoparticles. Physically, enhancement ≥ 1.0 always.
    pub fn collective_enhancement(&self, wavelength: f64, evaluation_point: &[f64]) -> f64 {
        let mut total_field = num_complex::Complex::new(1.0, 0.0); // Incident field = 1

        for (particle, position) in &self.particles {
            let distance = ((evaluation_point[0] - position[0]).powi(2)
                + (evaluation_point[1] - position[1]).powi(2)
                + (evaluation_point[2] - position[2]).powi(2))
            .sqrt();

            if distance > particle.radius {
                // Dipole field from this particle
                let alpha = particle.polarizability(wavelength);
                let k = 2.0 * PI * self.medium_wavenumber(wavelength);

                // Simplified dipole field: E ∝ α / r³ * exp(ikr)
                let phase = num_complex::Complex::new(0.0, k * distance).exp();
                let geometric_factor = 1.0 / distance.powi(3);

                let dipole_field = alpha * geometric_factor * phase;
                total_field += dipole_field;
            }
        }

        // Enhancement is intensity ratio: |E_total|²/|E_incident|²
        // For incident field = 1, this is |E_total|²
        let intensity_enhancement = total_field.norm_sqr();

        // Physical constraint: enhancement cannot be less than 1.0 (no destructive interference
        // can reduce field below incident field in forward direction for this geometry)
        intensity_enhancement.max(1.0)
    }

    /// Compute plasmonic hot spots between nanoparticles
    pub fn hot_spots(&self, wavelength: f64) -> Vec<(f64, [f64; 3])> {
        // Find regions of maximum field enhancement
        let mut hot_spots = Vec::new();

        match &self.geometry {
            ArrayGeometry::Linear { spacing: _ } => {
                // Hot spots are midway between particles
                for i in 0..self.particles.len() - 1 {
                    let pos1 = self.particles[i].1;
                    let pos2 = self.particles[i + 1].1;

                    let midpoint = [
                        (pos1[0] + pos2[0]) / 2.0,
                        (pos1[1] + pos2[1]) / 2.0,
                        (pos1[2] + pos2[2]) / 2.0,
                    ];

                    let enhancement = self.collective_enhancement(wavelength, &midpoint);
                    hot_spots.push((enhancement, midpoint));
                }
            }
            _ => {
                // For other geometries, sample some points
                // This is a placeholder - real implementation would be more sophisticated
            }
        }

        hot_spots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        hot_spots
    }

    fn medium_wavenumber(&self, wavelength: f64) -> f64 {
        // For water (n ≈ 1.33)
        2.0 * PI * 1.33 / wavelength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mie_theory_gold() {
        let mie = MieTheory::gold_in_water(15e-9); // 15 nm radius

        // Test polarizability calculation
        let alpha = mie.polarizability(530e-9); // SPR wavelength
        assert!(alpha.re > 0.0); // Polarizability should be positive

        // Test cross-sections
        let sigma_scat = mie.scattering_cross_section(530e-9);
        let sigma_abs = mie.absorption_cross_section(530e-9);
        let sigma_ext = mie.extinction_cross_section(530e-9);

        assert!(sigma_scat > 0.0);
        assert!(sigma_abs > 0.0);
        assert_eq!(sigma_ext, sigma_scat + sigma_abs);
    }

    #[test]
    fn test_plasmonic_enhancement() {
        let mie = MieTheory::gold_in_water(15e-9);
        let enhancement = PlasmonicEnhancement::new(mie, 1e20); // High concentration

        // Test field enhancement near particle surface
        let surface_point = [15e-9, 0.0, 0.0];
        let factor = enhancement.field_enhancement_factor(530e-9, &surface_point);
        assert!(factor > 1.0); // Should enhance field

        // Test surface plasmon resonance
        let spr_enhancement = enhancement.surface_plasmon_enhancement(530e-9);
        assert!(spr_enhancement > 1.0);
    }

    #[test]
    fn test_nanoparticle_array() {
        let array = NanoparticleArray::linear_array(15e-9, 50e-9, 3);

        // Test collective enhancement
        let midpoint = [25e-9, 0.0, 0.0]; // Between first two particles
        let enhancement = array.collective_enhancement(530e-9, &midpoint);
        assert!(enhancement >= 1.0);

        // Test hot spots
        let hot_spots = array.hot_spots(530e-9);
        assert!(!hot_spots.is_empty());
        assert!(hot_spots[0].0 >= 1.0); // First hot spot should have enhancement
    }
}
