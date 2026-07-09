//! Mie theory implementation for spherical plasmonic nanoparticles.
//!
//! Describes scattering and absorption of electromagnetic radiation by spherical
//! particles.
//!
//! # Gold dielectric theorem
//!
//! For nonmagnetic noble metals, the relative complex permittivity used by
//! optical Mie theory is
//!
//! ```text
//! eps_r(lambda) = (n(lambda) + i k(lambda))^2
//!               = n(lambda)^2 - k(lambda)^2 + i 2 n(lambda) k(lambda).
//! ```
//!
//! The gold-in-water constructor uses the Johnson-Christy room-temperature
//! optical constants tabulated over `0.1879-1.9370 um`. Within any tabulated
//! interval `[lambda_j, lambda_{j+1}]`, `n(lambda)` and `k(lambda)` are the
//! unique affine interpolants through the adjacent measured values. The
//! interpolant is endpoint exact and continuous because both adjacent interval
//! formulas evaluate to the shared tabulated value at each internal knot.
//!
//! Proof: every interval has two distinct wavelengths, so the affine
//! interpolant exists and is unique. Substituting either endpoint into
//! `y_j + t (y_{j+1} - y_j)` with
//! `t = (lambda - lambda_j)/(lambda_{j+1} - lambda_j)` returns the measured
//! value at that endpoint. Squaring the resulting complex refractive index
//! gives the permittivity identity by complex multiplication.
//!
//! References:
//! - Johnson, P.B. and Christy, R.W. (1972). Optical constants of the noble
//!   metals. *Physical Review B*, 6, 4370-4379.
//! - Polyanskiy, M.N. (2024). Refractiveindex.info database of optical
//!   constants. *Scientific Data*, 11, 94.

use kwavers_core::constants::fundamental::VACUUM_PERMITTIVITY;
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use eunomia::Complex;
use std::f64::consts::PI;

/// Johnson-Christy gold optical constants `(wavelength_um, n, k)`.
const JOHNSON_CHRISTY_GOLD: &[(f64, f64, f64)] = &[
    (0.1879, 1.28, 1.188),
    (0.1916, 1.32, 1.203),
    (0.1953, 1.34, 1.226),
    (0.1993, 1.33, 1.251),
    (0.2033, 1.33, 1.277),
    (0.2073, 1.30, 1.304),
    (0.2119, 1.30, 1.350),
    (0.2164, 1.30, 1.387),
    (0.2214, 1.30, 1.427),
    (0.2262, 1.31, 1.460),
    (0.2313, 1.30, 1.497),
    (0.2371, 1.32, 1.536),
    (0.2426, 1.32, 1.577),
    (0.2490, 1.33, 1.631),
    (0.2551, 1.33, 1.688),
    (0.2616, 1.35, 1.749),
    (0.2689, 1.38, 1.803),
    (0.2761, 1.43, 1.847),
    (0.2844, 1.47, 1.869),
    (0.2924, 1.49, 1.878),
    (0.3009, 1.53, 1.889),
    (0.3107, 1.53, 1.893),
    (0.3204, 1.54, 1.898),
    (0.3315, 1.48, 1.883),
    (0.3425, 1.48, 1.871),
    (0.3542, 1.50, 1.866),
    (0.3679, 1.48, 1.895),
    (0.3815, 1.46, 1.933),
    (0.3974, 1.47, 1.952),
    (0.4133, 1.46, 1.958),
    (0.4305, 1.45, 1.948),
    (0.4509, 1.38, 1.914),
    (0.4714, 1.31, 1.849),
    (0.4959, 1.04, 1.833),
    (0.5209, 0.62, 2.081),
    (0.5486, 0.43, 2.455),
    (0.5821, 0.29, 2.863),
    (0.6168, 0.21, 3.272),
    (0.6595, 0.14, 3.697),
    (0.7045, 0.13, 4.103),
    (0.7560, 0.14, 4.542),
    (0.8211, 0.16, 5.083),
    (0.8920, 0.17, 5.663),
    (0.9840, 0.22, 6.350),
    (1.0880, 0.27, 7.150),
    (1.2160, 0.35, 8.145),
    (1.3930, 0.43, 9.519),
    (1.6100, 0.56, 11.21),
    (1.9370, 0.92, 13.78),
];

/// Mie theory calculator for spherical plasmonic nanoparticles
pub struct MieTheory {
    /// Nanoparticle radius (m)
    pub radius: f64,
    /// Dielectric function of nanoparticle ε_particle(ω)
    pub particle_dielectric: Box<dyn Fn(f64) -> eunomia::Complex64>,
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
    ///
    /// Uses Johnson-Christy tabulated gold optical constants with affine
    /// interpolation in wavelength and `eps=(n+ik)^2`.
    #[must_use]
    pub fn gold_in_water(radius: f64) -> Self {
        Self {
            radius,
            particle_dielectric: Box::new(gold_dielectric_johnson_christy),
            // Water at optical frequencies
            medium_dielectric: 1.77,
        }
    }

    /// Compute polarizability (α) using Mie theory
    #[must_use]
    pub fn polarizability(&self, wavelength: f64) -> eunomia::Complex64 {
        let eps_particle = (self.particle_dielectric)(wavelength);
        let eps_medium = self.medium_dielectric;

        // Mie polarizability (quasistatic limit): α = 4π ε₀ ε_m R³ (ε - ε_m)/(ε + 2ε_m)
        let eps_ratio = eps_particle / eps_medium;
        let numerator = eps_ratio - eunomia::Complex::new(1.0, 0.0);
        let denominator = eps_ratio + eunomia::Complex::new(2.0, 0.0);

        let alpha_dimensionless = self.radius * self.radius * self.radius * numerator / denominator;

        // Convert to SI units (include 4π ε₀ ε_m factor)
        alpha_dimensionless * FOUR_PI * VACUUM_PERMITTIVITY * eps_medium
    }

    /// Compute scattering cross-section (σ_scat)
    #[must_use]
    pub fn scattering_cross_section(&self, wavelength: f64) -> f64 {
        let alpha_si = self.polarizability(wavelength);
        // Convert SI polarizability to polarizability volume α_vol = R³·K [m³]
        let alpha_vol = alpha_si / (FOUR_PI * VACUUM_PERMITTIVITY * self.medium_dielectric);
        let n_medium = self.medium_dielectric.sqrt();
        let k = TWO_PI * n_medium / wavelength; // medium wavenumber k_m = n_m·ω/c

        // σ_scat = (8π/3) k_m⁴ |α_vol|² (quasistatic Mie; Bohren & Huffman 4.61)
        (8.0 * PI / 3.0) * k.powi(4) * alpha_vol.norm_sqr()
    }

    /// Compute absorption cross-section (σ_abs)
    #[must_use]
    pub fn absorption_cross_section(&self, wavelength: f64) -> f64 {
        let alpha_si = self.polarizability(wavelength);
        // Convert SI polarizability to polarizability volume α_vol = R³·K [m³]
        let alpha_vol = alpha_si / (FOUR_PI * VACUUM_PERMITTIVITY * self.medium_dielectric);
        let n_medium = self.medium_dielectric.sqrt();
        let k = TWO_PI * n_medium / wavelength; // medium wavenumber k_m = n_m·ω/c

        // σ_abs = 4π k_m Im(α_vol) (quasistatic Mie; van de Hulst / Bohren & Huffman)
        FOUR_PI * k * alpha_vol.im
    }

    /// Compute extinction cross-section (σ_ext = σ_scat + σ_abs)
    #[must_use]
    pub fn extinction_cross_section(&self, wavelength: f64) -> f64 {
        self.scattering_cross_section(wavelength) + self.absorption_cross_section(wavelength)
    }

    /// Find plasmon resonance wavelength by minimizing the denominator Re(ε_particle + 2ε_medium)
    #[must_use]
    pub fn plasmon_resonance_wavelength(&self) -> Option<f64> {
        // Simple grid search
        let wavelengths = (400..900).map(|nm| f64::from(nm) * 1e-9); // 400-900 nm

        for wavelength in wavelengths {
            let eps_particle = (self.particle_dielectric)(wavelength);
            let denominator =
                eps_particle + eunomia::Complex::new(2.0 * self.medium_dielectric, 0.0);

            if denominator.re.abs() < 0.1 {
                // Close to resonance
                return Some(wavelength);
            }
        }

        None
    }
}

#[must_use]
pub(crate) fn gold_dielectric_johnson_christy(wavelength_m: f64) -> Complex<f64> {
    let wavelength_um = wavelength_m * 1.0e6;
    let (n, k) = interpolate_gold_nk(wavelength_um);
    Complex::new(n.mul_add(n, -(k * k)), 2.0 * n * k)
}

#[must_use]
fn interpolate_gold_nk(wavelength_um: f64) -> (f64, f64) {
    let first = JOHNSON_CHRISTY_GOLD[0];
    if wavelength_um <= first.0 {
        return (first.1, first.2);
    }

    for pair in JOHNSON_CHRISTY_GOLD.windows(2) {
        let (lambda0, n0, k0) = pair[0];
        let (lambda1, n1, k1) = pair[1];
        if wavelength_um <= lambda1 {
            let t = (wavelength_um - lambda0) / (lambda1 - lambda0);
            return (t.mul_add(n1 - n0, n0), t.mul_add(k1 - k0, k0));
        }
    }

    let last = JOHNSON_CHRISTY_GOLD[JOHNSON_CHRISTY_GOLD.len() - 1];
    (last.1, last.2)
}

