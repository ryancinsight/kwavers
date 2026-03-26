use super::super::types::NanoparticleGeometry;
use super::maxwell::ElectromagneticWaveEquation;

/// Plasmonic enhancement trait for surface plasmon effects
///
/// Models enhanced electromagnetic fields near metallic nanostructures
pub trait PlasmonicEnhancement: ElectromagneticWaveEquation {
    /// Surface plasmon resonance frequency ω_res (rad/s)
    fn plasmon_resonance_frequency(
        &self,
        _nanoparticle_radius: f64,
        dielectric_constant: f64,
    ) -> f64 {
        // Drude model for spherical nanoparticles
        // ω_res² = ω_p² / (1 + 2ε_m / ε_d) where ε_m, ε_d are dielectric constants
        // This is a simplified approximation

        let omega_p = 1.2e16; // Plasma frequency for gold (approximate)
        let eps_m = -2.0; // Metal dielectric constant (approximate for visible)
        let eps_d = dielectric_constant;

        omega_p * (1.0 / (1.0 + 2.0 * eps_m / eps_d)).sqrt()
    }

    /// Local field enhancement factor |E_local|/|E_incident|
    fn field_enhancement_factor(
        &self,
        _position: &[f64],
        nanoparticle_geometry: &NanoparticleGeometry,
    ) -> f64 {
        // Quasistatic enhancement with simple dielectric contrast and shape factors.
        // NOTE: This is a geometry-only approximation without wavelength dependence.
        let eps_medium = 1.77; // Water at optical frequencies (approximate)
        let eps_particle = -2.0; // Gold dielectric constant (approximate in visible)

        let enhancement_for_l = |l: f64| -> f64 {
            let denom = eps_particle + l * (eps_medium - eps_particle) + (1.0 - l) * eps_medium;
            if denom.abs() > 0.0 {
                (eps_medium / denom).abs()
            } else {
                1.0
            }
        };

        let enhancement_sphere = || {
            let denom = eps_particle + 2.0 * eps_medium;
            if denom.abs() > 0.0 {
                (3.0 * eps_medium / denom).abs()
            } else {
                1.0
            }
        };

        match nanoparticle_geometry {
            NanoparticleGeometry::Sphere { .. } => enhancement_sphere(),
            NanoparticleGeometry::Ellipsoid { a, b, c } => {
                let (lx, ly, lz) = ellipsoid_depolarization_factors(*a, *b, *c);
                enhancement_for_l(lx)
                    .max(enhancement_for_l(ly))
                    .max(enhancement_for_l(lz))
            }
            NanoparticleGeometry::Nanorod { radius, length } => {
                let a = *radius;
                let c = 0.5 * length.max(2.0 * *radius);
                let (lx, ly, lz) = ellipsoid_depolarization_factors(a, a, c);
                enhancement_for_l(lx)
                    .max(enhancement_for_l(ly))
                    .max(enhancement_for_l(lz))
            }
            NanoparticleGeometry::Nanoshell {
                core_radius,
                shell_thickness,
            } => {
                let shell_ratio = if *core_radius > 0.0 {
                    shell_thickness / core_radius
                } else {
                    0.0
                };
                enhancement_sphere() * (1.0 + shell_ratio)
            }
        }
    }

    /// Near-field coupling between nanoparticles
    fn near_field_coupling(
        &self,
        particle1_pos: &[f64],
        particle2_pos: &[f64],
        wavelength: f64,
    ) -> f64 {
        // Dipole-dipole coupling approximation
        let distance = ((particle1_pos[0] - particle2_pos[0]).powi(2)
            + (particle1_pos[1] - particle2_pos[1]).powi(2)
            + (particle1_pos[2] - particle2_pos[2]).powi(2))
        .sqrt();

        if distance > 0.0 {
            let k = 2.0 * std::f64::consts::PI / wavelength;
            // Coupling strength ∝ 1/r³ exp(ikr)
            (1.0 / distance.powi(3)) * (k * distance).cos()
        } else {
            0.0
        }
    }

    /// Purcell factor for enhanced emission rates
    fn purcell_factor(&self, position: &[f64], wavelength: f64) -> f64 {
        // Purcell factor quantifies enhancement of spontaneous emission
        // F = (3/2π²) (λ/n)³ (Q/V) where Q is quality factor, V mode volume

        let enhancement = self
            .field_enhancement_factor(position, &NanoparticleGeometry::Sphere { radius: 15e-9 });
        let quality_factor = 10.0; // Typical plasmonic Q-factor
        let mode_volume = 15e-9_f64.powi(3) * 10.0;

        let lambda_over_n = wavelength / 1.5; // Effective wavelength in medium
        let base_factor = 3.0 / (2.0 * std::f64::consts::PI * std::f64::consts::PI)
            * (lambda_over_n / wavelength).powi(3)
            * (quality_factor / mode_volume);

        base_factor * enhancement * enhancement // |E|⁴ dependence
    }
}

pub(crate) fn ellipsoid_depolarization_factors(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let r = 0.5 * (a + b);
    if r <= 0.0 || c <= 0.0 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    if (c - r).abs() <= f64::EPSILON {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    if c > r {
        // Prolate spheroid
        let e = (1.0 - (r * r) / (c * c)).sqrt();
        let denom = 2.0 * e * e * e;
        let lz = if denom > 0.0 {
            let term = ((1.0 + e) / (1.0 - e)).ln() - 2.0 * e;
            (1.0 - e * e) / denom * term
        } else {
            1.0 / 3.0
        };
        let lx = 0.5 * (1.0 - lz);
        (lx, lx, lz)
    } else {
        // Oblate spheroid
        let e = (1.0 - (c * c) / (r * r)).sqrt();
        let denom = e * e * e;
        let lz = if denom > 0.0 {
            (1.0 + e * e) / denom * (e - e.atan())
        } else {
            1.0 / 3.0
        };
        let lx = 0.5 * (1.0 - lz);
        (lx, lx, lz)
    }
}
