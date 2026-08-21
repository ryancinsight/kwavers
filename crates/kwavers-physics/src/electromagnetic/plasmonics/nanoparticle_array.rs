//! Coherent nanoparticle arrays and collective plasmonic effects

use super::mie_theory::MieTheory;
use super::types::PlasmonicArrayGeometry;
use aequitas::systems::si::quantities::{Dimensionless, Length, ReciprocalLength};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::optical::REFRACTIVE_INDEX_WATER;

/// Array of nanoparticles for collective enhancement computations
#[derive(Debug)]
pub struct NanoparticleArray {
    /// Individual nanoparticles with position coordinates
    pub particles: Vec<(MieTheory, [Length; 3])>, // (particle, position)
    /// Geometric layout of the array
    pub geometry: PlasmonicArrayGeometry,
}

impl NanoparticleArray {
    /// Create a linear 1D uniform array of identical nanoparticles
    #[must_use]
    pub fn linear_array(particle_radius: Length, spacing: Length, n_particles: usize) -> Self {
        let mut particles = Vec::with_capacity(n_particles);

        for i in 0..n_particles {
            let position = [
                Length::from_base(i as f64 * spacing.into_base()),
                Length::from_base(0.0),
                Length::from_base(0.0),
            ];
            let mie = MieTheory::gold_in_water(particle_radius);
            particles.push((mie, position));
        }

        Self {
            particles,
            geometry: PlasmonicArrayGeometry::Linear { spacing },
        }
    }

    /// Compute collective plasmonic field enhancement at a specific evaluation point
    ///
    /// Enhancement is defined analytically as |E_total|² / |E_incident|².
    /// Inherently restricted to ≥ 1.0 (no macroscopic destructive cancellation).
    #[must_use]
    pub fn collective_enhancement(
        &self,
        wavelength: Length,
        evaluation_point: &[Length; 3],
    ) -> Dimensionless {
        let mut total_field = eunomia::Complex::new(1.0, 0.0); // Baseline incident field = 1

        for (particle, position) in &self.particles {
            let distance = (evaluation_point[2].into_base() - position[2].into_base())
                .mul_add(
                    evaluation_point[2].into_base() - position[2].into_base(),
                    (evaluation_point[1].into_base() - position[1].into_base()).mul_add(
                        evaluation_point[1].into_base() - position[1].into_base(),
                        (evaluation_point[0].into_base() - position[0].into_base()).powi(2),
                    ),
                )
                .sqrt();

            if distance > particle.radius.into_base() {
                // Determine coherent dipole radiating field
                let alpha = particle.polarizability(wavelength).into_base();
                let k = Self::medium_wavenumber(wavelength).into_base();

                // Near-field dominant term: E ∝ α / r³ · exp(ikr)
                let phase = eunomia::Complex::new(0.0, k * distance).exp();
                let geometric_factor = 1.0 / distance.powi(3);

                let dipole_field = alpha * geometric_factor * phase;
                total_field += dipole_field;
            }
        }

        // Enhancement factor is the normalized intensity ratio
        let intensity_enhancement = total_field.norm_sqr();

        Dimensionless::from_base(intensity_enhancement.max(1.0))
    }

    /// Determine pair-gap candidates ordered by field enhancement.
    #[must_use]
    pub fn hot_spots(&self, wavelength: Length) -> Vec<(Dimensionless, [Length; 3])> {
        let mut hot_spots = Vec::new();

        // Pair midpoints are the closed-form candidate gap locations available
        // for every explicit particle layout. A full spatial optimizer is a
        // separate sampling problem; returning no candidates for non-linear
        // layouts would silently discard their real pair interactions.
        for (i, (_, pos1)) in self.particles.iter().enumerate() {
            for (_, (_, pos2)) in self.particles.iter().enumerate().skip(i + 1) {
                let midpoint = [
                    (pos1[0] + pos2[0]) / 2.0,
                    (pos1[1] + pos2[1]) / 2.0,
                    (pos1[2] + pos2[2]) / 2.0,
                ];

                let enhancement = self.collective_enhancement(wavelength, &midpoint);
                hot_spots.push((enhancement, midpoint));
            }
        }

        hot_spots.sort_by(|a, b| b.0.into_base().total_cmp(&a.0.into_base()));
        hot_spots
    }

    /// Helper to determine the physical wavenumber inside the host dielectric
    #[must_use]
    fn medium_wavenumber(wavelength: Length) -> ReciprocalLength {
        // Assume water host matrix
        let refractive_index = REFRACTIVE_INDEX_WATER;
        ReciprocalLength::from_base(TWO_PI * refractive_index / wavelength.into_base())
    }
}
