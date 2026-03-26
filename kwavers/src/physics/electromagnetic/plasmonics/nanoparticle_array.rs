//! Coherent nanoparticle arrays and collective plasmonic effects

use super::mie_theory::MieTheory;
use super::types::ArrayGeometry;
use std::f64::consts::PI;

/// Array of nanoparticles for collective enhancement computations
#[derive(Debug)]
pub struct NanoparticleArray {
    /// Individual nanoparticles with position coordinates
    pub particles: Vec<(MieTheory, [f64; 3])>, // (particle, position)
    /// Geometric layout of the array
    pub geometry: ArrayGeometry,
}

impl NanoparticleArray {
    /// Create a linear 1D uniform array of identical nanoparticles
    #[must_use]
    pub fn linear_array(particle_radius: f64, spacing: f64, n_particles: usize) -> Self {
        let mut particles = Vec::with_capacity(n_particles);

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

    /// Compute collective plasmonic field enhancement at a specific evaluation point
    ///
    /// Enhancement is defined analytically as |E_total|² / |E_incident|².
    /// Inherently restricted to ≥ 1.0 (no macroscopic destructive cancellation).
    #[must_use]
    pub fn collective_enhancement(&self, wavelength: f64, evaluation_point: &[f64; 3]) -> f64 {
        let mut total_field = num_complex::Complex::new(1.0, 0.0); // Baseline incident field = 1

        for (particle, position) in &self.particles {
            let distance = ((evaluation_point[0] - position[0]).powi(2)
                + (evaluation_point[1] - position[1]).powi(2)
                + (evaluation_point[2] - position[2]).powi(2))
            .sqrt();

            if distance > particle.radius {
                // Determine coherent dipole radiating field
                let alpha = particle.polarizability(wavelength);
                let k = self.medium_wavenumber(wavelength);

                // Near-field dominant term: E ∝ α / r³ · exp(ikr)
                let phase = num_complex::Complex::new(0.0, k * distance).exp();
                let geometric_factor = 1.0 / distance.powi(3);

                let dipole_field = alpha * geometric_factor * phase;
                total_field += dipole_field;
            }
        }

        // Enhancement factor is the normalized intensity ratio
        let intensity_enhancement = total_field.norm_sqr();

        intensity_enhancement.max(1.0)
    }

    /// Determine geometric locations of maximum field enhancement (plasmonic hot spots)
    #[must_use]
    pub fn hot_spots(&self, wavelength: f64) -> Vec<(f64, [f64; 3])> {
        let mut hot_spots = Vec::new();

        match &self.geometry {
            ArrayGeometry::Linear { spacing: _ } => {
                // Inter-particle gap centers exhibit maximum field confinement
                for i in 0..self.particles.len().saturating_sub(1) {
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
                // TODO_AUDIT: P2 - Implement rigorous 3D spatial field sampling or optimization
                // routine to systematically locate maximum field amplitude coordinates
                // for generalized nanoparticle distribution matrices.
            }
        }

        hot_spots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hot_spots
    }

    /// Helper to determine the physical wavenumber inside the host dielectric
    #[must_use]
    fn medium_wavenumber(&self, wavelength: f64) -> f64 {
        // Assume water host matrix (n ≈ 1.33)
        let refractive_index = 1.33;
        2.0 * PI * refractive_index / wavelength
    }
}
