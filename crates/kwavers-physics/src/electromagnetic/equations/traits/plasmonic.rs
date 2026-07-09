use super::super::types::NanoparticleGeometry;
use super::maxwell::ElectromagneticWaveEquation;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::optical::{
    GOLD_DRUDE_DAMPING_RAD_S, GOLD_EPS_INF, GOLD_PLASMA_FREQUENCY_RAD_S,
};
use eunomia::Complex;

mod geometry;

use geometry::ellipsoid_depolarization_factors;

/// Plasmonic enhancement trait for surface plasmon effects
///
/// Models enhanced electromagnetic fields near metallic nanostructures
pub trait PlasmonicEnhancementEquation: ElectromagneticWaveEquation {
    /// Surface plasmon resonance frequency ω_res (rad/s)
    ///
    /// # Theorem: Fröhlich resonance for a Drude nanosphere
    ///
    /// In the quasistatic limit, the dipole polarizability denominator of a
    /// sphere is `eps_p(omega) + 2 eps_d`. A resonance occurs when its real part
    /// vanishes:
    ///
    /// ```text
    /// Re eps_p(omega_res) + 2 eps_d = 0.
    /// ```
    ///
    /// For a damped Drude metal with
    /// `Re eps_p = eps_inf - omega_p^2/(omega^2 + gamma^2)`, solving gives
    ///
    /// ```text
    /// omega_res = sqrt(omega_p^2/(eps_inf + 2 eps_d) - gamma^2).
    /// ```
    ///
    /// Proof: substitute the real Drude dielectric into the Fröhlich condition,
    /// move terms, invert the positive denominator, and take the positive square
    /// root because optical frequency is nonnegative.
    fn plasmon_resonance_frequency(
        &self,
        nanoparticle_radius: f64,
        dielectric_constant: f64,
    ) -> f64 {
        if nanoparticle_radius <= 0.0 {
            return 0.0;
        }
        frohlich_drude_resonance_frequency(dielectric_constant)
    }

    /// Local field enhancement factor |E_local|/|E_incident|
    ///
    /// # Theorem: ellipsoidal quasistatic field factor
    ///
    /// Along a principal axis with depolarization factor `L`, the complex
    /// internal electrostatic field of an ellipsoid in a uniform host medium is
    ///
    /// ```text
    /// E_axis / E0 = eps_m / (eps_m + L (eps_p - eps_m)).
    /// ```
    ///
    /// For a sphere `L=1/3`, this reduces to
    /// `3 eps_m / (eps_p + 2 eps_m)`.
    ///
    /// The returned scalar is the complex magnitude. Drude damping contributes
    /// `Im eps_p > 0`, so the resonant denominator is finite.
    ///
    /// Proof: the ellipsoidal inclusion solution has uniform internal field;
    /// enforcing tangential-field continuity and normal-displacement continuity
    /// on a confocal ellipsoidal surface yields the denominator above. Setting
    /// all axes equal gives `L_x=L_y=L_z=1/3` and the spherical formula. A
    /// positive imaginary dielectric component makes the denominator norm
    /// nonzero at resonance.
    fn field_enhancement_factor(
        &self,
        _position: &[f64],
        nanoparticle_geometry: &NanoparticleGeometry,
    ) -> f64 {
        let eps_medium_re = 1.77;
        let eps_medium = Complex::new(eps_medium_re, 0.0);
        let omega_res = frohlich_drude_resonance_frequency(eps_medium_re);
        let eps_particle = Complex::new(
            -2.0 * eps_medium_re,
            drude_imaginary_permittivity(omega_res),
        );

        let enhancement_for_l = |l: f64| -> f64 {
            let denom = eps_medium + l * (eps_particle - eps_medium);
            if denom.norm() > 0.0 {
                (eps_medium / denom).norm()
            } else {
                1.0
            }
        };

        let enhancement_sphere = || {
            let denom = eps_particle + 2.0 * eps_medium;
            if denom.norm() > 0.0 {
                (3.0 * eps_medium / denom).norm()
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
        let distance = (particle1_pos[2] - particle2_pos[2])
            .mul_add(
                particle1_pos[2] - particle2_pos[2],
                (particle1_pos[1] - particle2_pos[1]).mul_add(
                    particle1_pos[1] - particle2_pos[1],
                    (particle1_pos[0] - particle2_pos[0]).powi(2),
                ),
            )
            .sqrt();

        if distance > 0.0 {
            let k = TWO_PI / wavelength;
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
        let base_factor = 3.0 / (TWO_PI * std::f64::consts::PI)
            * (lambda_over_n / wavelength).powi(3)
            * (quality_factor / mode_volume);

        base_factor * enhancement * enhancement // |E|⁴ dependence
    }
}

#[must_use]
pub(crate) fn frohlich_drude_resonance_frequency(dielectric_constant: f64) -> f64 {
    if dielectric_constant <= 0.0 {
        return 0.0;
    }

    let omega_squared = GOLD_DRUDE_DAMPING_RAD_S.mul_add(
        -GOLD_DRUDE_DAMPING_RAD_S,
        GOLD_PLASMA_FREQUENCY_RAD_S.powi(2) / 2.0f64.mul_add(dielectric_constant, GOLD_EPS_INF),
    );

    if omega_squared > 0.0 {
        omega_squared.sqrt()
    } else {
        0.0
    }
}

#[must_use]
fn drude_imaginary_permittivity(omega: f64) -> f64 {
    if omega <= 0.0 {
        return 0.0;
    }

    GOLD_PLASMA_FREQUENCY_RAD_S.powi(2) * GOLD_DRUDE_DAMPING_RAD_S
        / (omega * GOLD_DRUDE_DAMPING_RAD_S.mul_add(GOLD_DRUDE_DAMPING_RAD_S, omega.powi(2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frohlich_drude_resonance_is_positive_for_water() {
        let omega = frohlich_drude_resonance_frequency(1.77);
        assert!(
            omega > 3.0e15 && omega < 4.5e15,
            "gold-in-water Fröhlich resonance must be optical; omega={omega:e}"
        );
    }

    #[test]
    fn frohlich_drude_resonance_satisfies_real_denominator_condition() {
        let eps_d = 1.77;
        let omega = frohlich_drude_resonance_frequency(eps_d);
        let eps_real = GOLD_EPS_INF
            - GOLD_PLASMA_FREQUENCY_RAD_S.powi(2)
                / (omega.powi(2) + GOLD_DRUDE_DAMPING_RAD_S.powi(2));
        let residual = eps_real + 2.0 * eps_d;
        assert!(
            residual.abs() < 1e-12,
            "Fröhlich condition residual must vanish; residual={residual:e}"
        );
    }

    #[test]
    fn drude_imaginary_permittivity_is_positive_at_resonance() {
        let omega = frohlich_drude_resonance_frequency(1.77);
        let eps_im = drude_imaginary_permittivity(omega);
        assert!(
            eps_im > 0.0 && eps_im.is_finite(),
            "Drude damping must produce finite optical loss; eps_im={eps_im:e}"
        );
    }
}

